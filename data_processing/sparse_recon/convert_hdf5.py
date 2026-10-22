# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from data_processing.sparse_recon.convert_to_video_hdf5 import encode_frames_to_video


def convert_reconstruction_to_hdf5(
    output_path: Path,
    *,
    sparse_recon_n_steps: int = 30000,
    crf: int = 0,
    codec: str = "libx264",
) -> Path:
    checkpoint_dir = output_path / f"ours_{sparse_recon_n_steps}"
    render_dir = checkpoint_dir / "renders"
    opacity_dir = checkpoint_dir / "opacity"
    for path in (render_dir, opacity_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)

    render_files = sorted(p for p in render_dir.iterdir() if p.is_file())
    opacity_files = sorted(p for p in opacity_dir.iterdir() if p.is_file())

    n_frames = len(render_files)
    if n_frames == 0:
        raise ValueError(f"No render frames found in {render_dir}")
    if len(opacity_files) != n_frames:
        raise ValueError(
            f"Frame count mismatch under {checkpoint_dir}: renders={n_frames}, opacity={len(opacity_files)}"
        )
    render_stems = [path.stem for path in render_files]
    opacity_stems = [path.stem for path in opacity_files]
    if render_stems != opacity_stems:
        raise ValueError(f"Frame name mismatch under {checkpoint_dir}; render and opacity files must align by stem")
    print(f"Processing {n_frames} frames...")

    # Read first file to get dimensions
    with Image.open(render_files[0]) as img:
        render_shape = np.array(img.convert("RGB")).shape
    with Image.open(opacity_files[0]) as img:
        opacity_shape = np.array(img).shape + (1,)

    hdf5_save_path = output_path / "data.h5"

    selected_indices_file = output_path / f"ours_{sparse_recon_n_steps}" / "selected_indices.json"
    with open(selected_indices_file, "r") as f:
        selected_indices = np.array(json.load(f))

    with open(output_path / "scale_info.txt", "r") as f:
        scale = float(f.read().split("Scale factor: ")[1].split()[0])

    print("Loading all frames for video encoding...")
    render_frames = np.zeros((n_frames, *render_shape), dtype=np.uint8)
    opacity_frames = np.zeros((n_frames, *opacity_shape), dtype=np.uint8)

    for i, (render_file, opacity_file) in enumerate(zip(render_files, opacity_files)):
        if i % 100 == 0:
            print(f"Loading frame {i}/{n_frames}...")

        with Image.open(render_file) as img:
            render_frames[i] = np.array(img.convert("RGB"), dtype=np.uint8)

        with Image.open(opacity_file).convert("L") as img:
            opacity_frames[i] = np.array(img, dtype=np.uint8)[..., None]

    print(f"Encoding renders as video (codec={codec}, crf={crf})...")
    render_video = encode_frames_to_video(render_frames, codec=codec, crf=crf)
    print(f"  Render video size: {len(render_video) / 1024 / 1024:.2f} MB")

    print("Encoding opacity as video...")
    opacity_rgb = np.repeat(opacity_frames, 3, axis=-1)
    opacity_video = encode_frames_to_video(opacity_rgb, codec=codec, crf=crf)
    print(f"  Opacity video size: {len(opacity_video) / 1024 / 1024:.2f} MB")

    with h5py.File(hdf5_save_path, "w") as hf:
        hf.attrs["format"] = "video_compressed"
        hf.attrs["codec"] = codec
        hf.attrs["crf"] = crf
        hf.attrs["n_frames"] = n_frames
        hf.attrs["render_shape"] = render_shape
        hf.attrs["opacity_shape"] = opacity_shape

        hf.create_dataset("selected_indices", data=selected_indices)
        hf.create_dataset("scale", data=scale)

        hf.create_dataset("render_video", data=np.frombuffer(render_video, dtype=np.uint8))
        hf.create_dataset("opacity_video", data=np.frombuffer(opacity_video, dtype=np.uint8))

    print(f"✓ Saved to {hdf5_save_path}")
    return hdf5_save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 3DGUT renders and opacity into the video-compressed HDF5 format used by training."
    )
    parser.add_argument("--output_path", type=Path, required=True, help="The path of the scene.")
    parser.add_argument(
        "--sparse_recon_n_steps",
        type=int,
        default=30000,
        help="Number of steps for sparse reconstruction (default: 30000)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help=(
            "CRF for video encoding (0=highest quality, 23=ffmpeg default). Even crf=0 is not "
            "bit-exact in RGB because frames are stored as yuv420p (4:2:0 chroma subsampling)."
        ),
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        choices=["libx264", "libx265"],
        help="Video codec (default: libx264)",
    )
    args = parser.parse_args()
    convert_reconstruction_to_hdf5(
        args.output_path,
        sparse_recon_n_steps=args.sparse_recon_n_steps,
        crf=args.crf,
        codec=args.codec,
    )


if __name__ == "__main__":
    main()
