# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ArtiFixer reconstruction HDF5 files."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from data_processing.scene_utils import (
    colmap_model_files_missing_from_dir,
    colmap_model_files_missing_from_zip,
    discover_scene_zips,
    find_colmap_sparse_dir,
    read_scene_list,
    resolve_scene_path,
    scene_dl3dv_subdir,
    scene_name,
)
from data_processing.sparse_recon.convert_hdf5 import convert_reconstruction_to_hdf5
from data_processing.sparse_recon.half_covisibility_sampling import write_half_covisibility_sampled_indices
from data_processing.sparse_recon.metric_alignment import align_colmap_to_metric_scale
from data_processing.sparse_recon.workflow import (
    copy_reconstruction_outputs,
    extract_colmap_zip,
    extract_scene_zip,
    find_experiment_scene_dir,
    image_dir_for_scene,
    reconstruction_outputs,
    reconstruction_subdir_for_dl3dv_subdir,
    task_work_dir,
)
from data_processing.threedgrut_training import train_3dgrut


@dataclass(frozen=True)
class Scene:
    path: Path
    name: str
    dl3dv_subdir: str
    reconstruction_subdir: str


def resolve_scene(dl3dv_dir: Path, scene: str, reconstruction_subdir: str | None) -> Scene:
    scene_path = resolve_scene_path(dl3dv_dir, scene)
    return make_scene(dl3dv_dir, scene_path, reconstruction_subdir)


def make_scene(dl3dv_dir: Path, scene_path: Path, reconstruction_subdir: str | None) -> Scene:
    dl3dv_subdir = scene_dl3dv_subdir(dl3dv_dir, scene_path)
    recon_subdir = reconstruction_subdir or reconstruction_subdir_for_dl3dv_subdir(dl3dv_subdir)
    return Scene(scene_path, scene_name(scene_path), dl3dv_subdir, recon_subdir)


def require_threedgrut_root(args: argparse.Namespace) -> Path:
    root = args.threedgrut_root or args.repo_root / "thirdparty" / "3DGRUT-ArtiFixer"
    train_py = root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            "ArtiFixer sparse reconstruction requires an ArtiFixer-compatible 3DGRUT checkout at "
            f"{root}. Clone the public fork into thirdparty/3DGRUT-ArtiFixer or pass --threedgrut_root."
        )
    return root


EXTRACTION_SENTINEL = ".extraction_complete"


class MissingColmapModelError(FileNotFoundError):
    """The scene has no usable COLMAP sparse model, so 3DGRUT initialization and metric alignment cannot run."""


def locate_colmap_archive(args: argparse.Namespace, scene: Scene) -> Path | None:
    """Return the ColmapCache archive to extract next to the scene, or None when the scene carries its own model."""
    if scene.path.suffix == ".zip":
        scene_missing = colmap_model_files_missing_from_zip(scene.path, scene.name)
    else:
        scene_missing = colmap_model_files_missing_from_dir(scene.path)
    if not scene_missing:
        return None
    if args.colmap_zip_root is None or scene.path.suffix != ".zip":
        raise MissingColmapModelError(
            f"{scene.name}: {scene.path} lacks {', '.join(scene_missing)} under colmap/sparse/0/ or sparse/0/; "
            "DL3DV-ALL-960P zips do not include a COLMAP model, pass --colmap_zip_root with the "
            "DL3DV-ALL-ColmapCache archives"
        )
    try:
        colmap_zip = resolve_scene_path(args.colmap_zip_root, scene.name)
    except FileNotFoundError:
        raise MissingColmapModelError(f"{scene.name}: no COLMAP archive under {args.colmap_zip_root}") from None
    try:
        missing = colmap_model_files_missing_from_zip(colmap_zip, scene.name)
    except zipfile.BadZipFile:
        raise MissingColmapModelError(f"{scene.name}: {colmap_zip} is not a valid zip archive") from None
    if missing:
        raise MissingColmapModelError(f"{scene.name}: {colmap_zip} lacks {', '.join(missing)}")
    return colmap_zip


def prepare_scene_input(args: argparse.Namespace, scene: Scene, scene_work_dir: Path) -> Path:
    """Extract the scene zip once per scene into a shared directory reused by all (half, views) tasks.

    Extraction happens in a temp directory that is renamed into place after a completion
    sentinel is written, so concurrent workers on shared storage can safely reuse it.
    """
    if scene.path.suffix != ".zip":
        return scene.path

    extract_root = scene_work_dir / "source"
    scene_dir = extract_root / scene.name
    if (extract_root / EXTRACTION_SENTINEL).is_file():
        return scene_dir

    colmap_zip = locate_colmap_archive(args, scene)
    tmp_root = scene_work_dir / f".source.tmp.{os.getpid()}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    extracted_dir = extract_scene_zip(scene.path, tmp_root, scene.name)
    if colmap_zip is not None:
        extract_colmap_zip(colmap_zip, tmp_root, extracted_dir, scene.name)
    (tmp_root / EXTRACTION_SENTINEL).touch()

    if extract_root.exists():
        if (extract_root / EXTRACTION_SENTINEL).is_file():
            # Another worker finished first; reuse its extraction.
            shutil.rmtree(tmp_root)
            return scene_dir
        shutil.rmtree(extract_root)
    try:
        tmp_root.rename(extract_root)
    except OSError:
        # A concurrent worker renamed its extraction into place between the
        # check above and our rename; reuse it.
        if not (extract_root / EXTRACTION_SENTINEL).is_file():
            raise
        shutil.rmtree(tmp_root)
    return scene_dir


def cleanup_scene_input(args: argparse.Namespace, scene: Scene, scene_work_dir: Path) -> None:
    # The extraction is shared by every (half, views) task of the scene, so
    # deleting it is only safe when no other job is processing the same scene
    # concurrently; jobs that split a scene across invocations must pass
    # --keep_extracted.
    if scene.path.suffix == ".zip" and not getattr(args, "keep_extracted", False):
        shutil.rmtree(scene_work_dir / "source", ignore_errors=True)


def run_metric_alignment(
    scene_input_dir: Path, experiment_scene_dir: Path, scene_scale_info: Path, downsample_factor: float
) -> None:
    scale_info = experiment_scene_dir / "scale_info.txt"
    if scene_scale_info.is_file():
        # The metric scale is a scene-level quantity (it only depends on the source model
        # and images), so reuse the value computed by the first (half, views) task.
        shutil.copyfile(scene_scale_info, scale_info)
        return

    colmap_dir = find_colmap_sparse_dir(scene_input_dir)
    if colmap_dir is None:
        raise MissingColmapModelError(f"no COLMAP sparse model under {scene_input_dir}")
    scale, _, _ = align_colmap_to_metric_scale(
        colmap_dir=colmap_dir,
        image_dir=image_dir_for_scene(scene_input_dir, downsample_factor),
        output_dir=experiment_scene_dir,
        debug=False,
        downsample_factor=downsample_factor,
    )
    scale_info.write_text(f"Scale factor: {scale}\n")

    tmp = scene_scale_info.with_name(f".{scene_scale_info.name}.tmp.{os.getpid()}")
    shutil.copyfile(scale_info, tmp)
    tmp.replace(scene_scale_info)


def configure_3dgrut_environment(args: argparse.Namespace) -> None:
    if args.torch_cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
    os.environ.setdefault("CC", args.cc)
    os.environ.setdefault("CXX", args.cxx)
    slurm_job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else os.getpid()
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_{slurm_job_id}")


def train_sparse_reconstruction(
    args: argparse.Namespace,
    threedgrut_root: Path,
    train_path: Path,
    work_dir: Path,
    selected_indices_file: Path,
    views: int,
) -> None:
    configure_3dgrut_environment(args)
    train_3dgrut(
        args.sparse_recon_config,
        [
            f"path={train_path}/",
            f"out_dir={work_dir}/",
            f"selected_indices_file={selected_indices_file}",
            f"num_selected_indices={views}",
            f"dataset.downsample_factor={args.downsample_factor:g}",
            f"val_frequency={args.sparse_recon_val_frequency}",
            f"use_wandb={'True' if args.use_wandb else 'False'}",
            f"experiment_name={args.experiment_name}",
            f"n_iterations={args.n_steps}",
            f"checkpoint.iterations=[{args.n_steps}]",
        ],
        threedgrut_root / "configs",
    )


def run_one_reconstruction(args: argparse.Namespace, scene: Scene, scene_work_dir: Path, half: str, views: int) -> None:
    task_id = f"{scene.name}@{half}@{views}"
    outputs = reconstruction_outputs(args.output_root, scene.reconstruction_subdir, scene.name, half, views)
    if outputs.complete() and not args.replace_if_exists:
        print(f"Skipping {task_id}: final outputs already exist")
        return

    work_dir = task_work_dir(scene_work_dir, half, views)
    work_dir.mkdir(parents=True, exist_ok=True)
    scene_input_dir = prepare_scene_input(args, scene, scene_work_dir)
    existing_dirs = {path.resolve() for path in work_dir.iterdir() if path.is_dir()}
    selected_indices_file = scene_work_dir / f"half_covisibility_sampled_indices_{half}.json"

    threedgrut_root = require_threedgrut_root(args)
    train_sparse_reconstruction(args, threedgrut_root, scene_input_dir, work_dir, selected_indices_file, views)

    experiment_scene_dir = find_experiment_scene_dir(
        work_dir,
        scene.name,
        expected_experiment_name=args.experiment_name,
        existing_dirs=existing_dirs,
    )
    run_metric_alignment(
        scene_input_dir, experiment_scene_dir, scene_work_dir / "scale_info.txt", args.downsample_factor
    )

    convert_reconstruction_to_hdf5(
        experiment_scene_dir,
        sparse_recon_n_steps=args.n_steps,
        crf=args.video_crf,
        codec=args.video_codec,
    )

    copy_reconstruction_outputs(experiment_scene_dir, outputs)
    print(f"Finished {task_id}")


def run_scene(args: argparse.Namespace, scene: Scene) -> None:
    scene_work_dir = args.work_root / scene.reconstruction_subdir / scene.name
    scene_work_dir.mkdir(parents=True, exist_ok=True)
    requested_outputs = [
        reconstruction_outputs(args.output_root, scene.reconstruction_subdir, scene.name, half, views)
        for half in args.scene_half
        for views in args.num_selected_indices
    ]
    if requested_outputs and all(outputs.complete() for outputs in requested_outputs) and not args.replace_if_exists:
        print(f"Skipping {scene.name}: all requested final outputs already exist")
        return
    locate_colmap_archive(args, scene)

    missing_selected_indices = [
        half
        for half in args.scene_half
        if not (scene_work_dir / f"half_covisibility_sampled_indices_{half}.json").is_file()
    ]
    if args.replace_if_exists or missing_selected_indices:
        write_half_covisibility_sampled_indices(scene.path, scene_work_dir)
    else:
        print(f"Skipping half-covisibility for {scene.name}: selected-index files already exist")
    if args.replace_if_exists:
        (scene_work_dir / "scale_info.txt").unlink(missing_ok=True)

    for half in args.scene_half:
        for views in args.num_selected_indices:
            run_one_reconstruction(args, scene, scene_work_dir, half, views)
    cleanup_scene_input(args, scene, scene_work_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv_dir", type=Path, required=True, help="Root containing DL3DV scene zips or dirs")
    parser.add_argument("--output_root", type=Path, required=True, help="Final reconstruction HDF5 root")
    parser.add_argument("--work_root", type=Path, required=True, help="Scratch/work root for 3DGRUT outputs")
    parser.add_argument("--scene_id", action="append", default=[], help="Only process this scene id/path")
    parser.add_argument("--scene_list", type=Path, help="Only process scenes listed in this file")
    parser.add_argument("--reconstruction_subdir", help="Override output subdir; default is dl3dv_<DL3DV subdir>")
    parser.add_argument("--num_selected_indices", type=int, nargs="+", required=True)
    parser.add_argument("--scene_half", nargs="+", choices=["0", "1"], default=["0", "1"])
    parser.add_argument(
        "--colmap_zip_root",
        type=Path,
        help="Root containing <split_or_subdir>/<scene_id>.zip COLMAP archives (the DL3DV-ALL-ColmapCache layout). "
        "Required unless the scene zips already contain colmap/sparse/0/ or sparse/0/.",
    )
    parser.add_argument("--sparse_recon_config", default="apps/colmap_3dgut_sparse_mcmc.yaml")
    parser.add_argument("--sparse_recon_val_frequency", type=int, default=999999)
    parser.add_argument("--experiment_name", default="artifixer_sparse_reconstruction")
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--downsample_factor", type=float, default=4.0)
    parser.add_argument(
        "--replace_if_exists", action="store_true", help="Regenerate outputs even when final files exist"
    )
    parser.add_argument(
        "--keep_extracted",
        action="store_true",
        help="Keep the shared per-scene zip extraction in the work root; required when concurrent "
        "invocations process disjoint (half, views) subsets of the same scene",
    )
    parser.add_argument("--video_crf", type=int, default=0)
    parser.add_argument("--video_codec", choices=["libx264", "libx265"], default="libx264")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--torch_cuda_arch_list",
        help=(
            "Optional TORCH_CUDA_ARCH_LIST override for 3DGRUT extension builds. "
            "When omitted, any existing environment value is preserved and PyTorch otherwise detects visible GPUs."
        ),
    )
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threedgrut_root", type=Path, help="Path to the ArtiFixer-compatible 3DGRUT checkout")
    args = parser.parse_args()

    scenes = list(args.scene_id)
    if args.scene_list is not None:
        scenes.extend(read_scene_list(args.scene_list))
    args.explicit_scenes = bool(scenes)
    if scenes:
        args.scenes = [resolve_scene(args.dl3dv_dir, scene, args.reconstruction_subdir) for scene in scenes]
    else:
        scene_paths = discover_scene_zips(args.dl3dv_dir)
        if not scene_paths:
            parser.error(f"No scene zips found under {args.dl3dv_dir}")
        args.scenes = [make_scene(args.dl3dv_dir, scene_path, args.reconstruction_subdir) for scene_path in scene_paths]
    return args


def run_scenes(args: argparse.Namespace) -> None:
    """Run every scene; a scan of --dl3dv_dir skips scenes without a COLMAP model, explicit requests fail."""
    skipped: dict[str, list[str]] = {scene.reconstruction_subdir: [] for scene in args.scenes}
    for scene in args.scenes:
        try:
            run_scene(args, scene)
        except MissingColmapModelError as exc:
            if args.explicit_scenes:
                raise
            print(f"Skipping {scene.name}: {exc}", file=sys.stderr)
            skipped[scene.reconstruction_subdir].append(scene.name)
    if args.explicit_scenes:
        return
    for reconstruction_subdir, names in skipped.items():
        write_skipped_scenes(args.output_root / reconstruction_subdir / "skipped_scenes.txt", names)


def write_skipped_scenes(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text("".join(f"{name}\n" for name in names))
    tmp.replace(path)
    if names:
        print(f"Skipped {len(names)} scene(s) without a COLMAP model, listed in {path}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    require_threedgrut_root(args)
    run_scenes(args)


if __name__ == "__main__":
    main()
