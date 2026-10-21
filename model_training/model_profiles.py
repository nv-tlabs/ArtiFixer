# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from model_training.constants import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT

WAN22_TI2V_5B_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN22_TI2V_5B_MODEL_NAME = "Wan2.2-TI2V-5B-Diffusers"
COSMOS3_NANO_MODEL_ID = "nvidia/Cosmos3-Nano"
COSMOS3_NANO_MODEL_NAME = "Cosmos3-Nano"
WAN_FLOW_SHIFT = 5.0
# Training and inference use the same flow shift for the 480p-class input frames.
COSMOS3_FLOW_SHIFT = 5.0
# Use pretrained 24-FPS RoPE spacing, independently of the clip's capture FPS.
COSMOS3_ROPE_FPS = 24.0

# Choose a camera pose for each encoded frame from its source-frame group.
CAMERA_POSE_ALIGNMENTS = ("mean", "last")


def _normalize_model_id(model_id: str) -> str:
    return str(model_id).replace("\\", "/").rstrip("/")


def is_wan21_t2v(model_id: str) -> bool:
    return _normalize_model_id(model_id).rsplit("/", 1)[-1] in (
        "Wan2.1-T2V-14B-Diffusers",
        "Wan2.1-T2V-1.3B-Diffusers",
    )


def is_wan22_ti2v_5b(model_id: str) -> bool:
    normalized_model_id = _normalize_model_id(model_id)
    return normalized_model_id == WAN22_TI2V_5B_MODEL_ID or normalized_model_id.endswith(f"/{WAN22_TI2V_5B_MODEL_NAME}")


def is_cosmos3_nano(model_id: str) -> bool:
    normalized = _normalize_model_id(model_id)
    return normalized == COSMOS3_NANO_MODEL_ID or normalized.endswith(f"/{COSMOS3_NANO_MODEL_NAME}")


# Cosmos3-Nano is currently the only supported Cosmos3 model.
is_cosmos3_model = is_cosmos3_nano


def uses_expanded_timesteps(model_id: str) -> bool:
    return is_wan22_ti2v_5b(model_id)


def spatial_resize_multiple(model_id: str) -> int:
    return 32 if is_wan22_ti2v_5b(model_id) or is_cosmos3_model(model_id) else 16


def flow_shift_for_model(model_id: str) -> float:
    return COSMOS3_FLOW_SHIFT if is_cosmos3_model(model_id) else WAN_FLOW_SHIFT


def default_camera_pose_alignment(model_id: str) -> str:
    """Use each frame group's final pose for Cosmos3, and its mean pose for Wan."""
    return "last" if is_cosmos3_model(model_id) else "mean"


def prompt_format_for_model(model_id: str) -> str:
    return COSMOS3_PROMPT_FORMAT if is_cosmos3_model(model_id) else WAN_T5_PROMPT_FORMAT


def requires_reference_camera_rays(prompt_format: str) -> bool:
    """Datasets identify model families by prompt format; only Cosmos3 needs reference rays."""
    return prompt_format == COSMOS3_PROMPT_FORMAT


def validate_model_profile(model_id: str, pipe) -> None:
    """Reject model IDs whose resize profile disagrees with the loaded model."""
    expected = spatial_resize_multiple(model_id)
    actual = max(pipe.spatial_alignment())
    if expected != actual:
        raise ValueError(
            f"Model profile mismatch for {model_id!r}: the profile resizes to multiples of "
            f"{expected} but the loaded model requires {actual}. Use the canonical model id "
            "or update model_profiles.py."
        )
