# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare real pretrained RGB scoring with upstream LPIPS-VGG v0.1 on CPU.

Install the test-only reference with ``python -m pip install lpips==0.1.4``.
Both implementations use pretrained VGG16 IMAGENET1K_V1 and LPIPS v0.1 heads;
their first use downloads missing weights to the normal PyTorch cache. Run
``python -m pytest tests/test_lpips_integration.py -q -s`` to print the JSON
receipt path. The fixed PNG pairs are synthetic checks, not benchmark results.
"""

import hashlib
import json
from importlib.metadata import version

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image

from model_eval import lpipsPytorch as bundled_lpips
from model_eval.genfusion_lossutil import ssim
from model_eval.metrics_utils import compute_rgb_metrics, load_image_as_tensor, psnr_genfusion

pytestmark = pytest.mark.slow

# The two float32 implementations accumulate their five layer scores in a
# different order. Allow rounding error, but not a different input convention.
LPIPS_ATOL = 1e-6
LPIPS_RTOL = 1e-5


@pytest.fixture(scope="module")
def reference_lpips():
    reference = pytest.importorskip(
        "lpips", reason="pretrained reference validation requires the test-only lpips==0.1.4 package"
    )
    assert version("lpips") == "0.1.4", "install lpips==0.1.4 for the pinned reference check"
    return reference.LPIPS(
        net="vgg", version="0.1", pretrained=True, pnet_rand=False, eval_mode=True, verbose=False
    ).cpu()


def _assert_matching_weights(reference):
    actual = bundled_lpips._lpips_cache[("vgg", "0.1", torch.device("cpu"))]
    # Upstream splits VGG's same numbered feature layers into five slices.
    expected_features = {name.split(".", 1)[1]: value for name, value in reference.net.state_dict().items()}
    actual_features = actual.net.layers.state_dict()
    assert actual_features.keys() == expected_features.keys()
    for name in expected_features:
        torch.testing.assert_close(actual_features[name], expected_features[name], rtol=0, atol=0)
    assert len(actual.lin) == len(reference.lins) == 5
    for actual_head, reference_head in zip(actual.lin, reference.lins):
        torch.testing.assert_close(actual_head[1].weight, reference_head.model[1].weight, rtol=0, atol=0)
    torch.testing.assert_close(actual.net.mean, reference.scaling_layer.shift, rtol=0, atol=0)
    torch.testing.assert_close(actual.net.std, reference.scaling_layer.scale, rtol=0, atol=0)


def _pre_mr_rgb_metrics(image, target):
    """Freeze compute_rgb_metrics from pre-MR main be35b0dc9f7934c49aa69401401d5590b0056db8."""
    return {
        "psnr": psnr_genfusion(image, target).item(),
        "ssim": ssim(image, target).item(),
        "lpips": bundled_lpips.lpips(image, target, net_type="vgg").item(),
    }


def _fixed_png_pairs(directory):
    y, x = np.indices((64, 80))
    target = np.stack((3 * x + 5 * y, 7 * x + 2 * y + 31, ((x // 8 + y // 8) % 2) * 160 + 40), axis=-1)
    target = (target % 256).astype(np.uint8)
    shifted = np.roll(target, shift=(3, 5), axis=(0, 1))
    brighter = np.clip(target.astype(np.int16) + 24, 0, 255).astype(np.uint8)
    patch = target.copy()
    patch[16:48, 20:60] = (10, 230, 100)
    target_path = directory / "target.png"
    Image.fromarray(target).save(target_path)
    for name, pixels in [("spatial_shift", shifted), ("brightness", brighter), ("local_patch", patch)]:
        prediction_path = directory / f"{name}.png"
        Image.fromarray(pixels).save(prediction_path)
        yield name, prediction_path, target_path


def test_pretrained_standard_reference_and_legacy_scorer_agree(reference_lpips, tmp_path):
    receipts = []
    with torch.no_grad():
        for name, prediction_path, target_path in _fixed_png_pairs(tmp_path):
            prediction = load_image_as_tensor(prediction_path, torch.device("cpu"))
            target = load_image_as_tensor(target_path, torch.device("cpu"))
            standard = compute_rgb_metrics(prediction, target, lpips_protocol="standard")
            legacy = compute_rgb_metrics(prediction, target, lpips_protocol="legacy")
            before = _pre_mr_rgb_metrics(prediction, target)
            expected_standard = reference_lpips(prediction, target, normalize=True).item()
            expected_legacy = reference_lpips(prediction, target, normalize=False).item()

            assert standard["lpips"] == pytest.approx(expected_standard, rel=LPIPS_RTOL, abs=LPIPS_ATOL)
            assert legacy["lpips"] == pytest.approx(expected_legacy, rel=LPIPS_RTOL, abs=LPIPS_ATOL)
            assert legacy == before
            for metric in ("psnr", "ssim"):
                assert standard[metric] == before[metric]
            # These pairs must distinguish the protocols; a constant or
            # identical pair could let a missing normalization pass unnoticed.
            assert abs(standard["lpips"] - legacy["lpips"]) > LPIPS_ATOL
            receipts.append(
                {
                    "pair": name,
                    "prediction_png_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                    "target_png_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                    "before": before,
                    "standard": standard,
                    "legacy": legacy,
                    "reference_standard_lpips": expected_standard,
                    "reference_legacy_lpips": expected_legacy,
                    "standard_minus_legacy_lpips": standard["lpips"] - legacy["lpips"],
                }
            )

    _assert_matching_weights(reference_lpips)
    receipt_path = tmp_path / "lpips-pretrained-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "reference": "lpips==0.1.4, LPIPS-VGG v0.1",
                "backbone": "torchvision VGG16_Weights.IMAGENET1K_V1",
                "matching_pretrained_weights": True,
                "pre_mr_scorer_commit": "be35b0dc9f7934c49aa69401401d5590b0056db8",
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "device": "cpu",
                "dtype": "float32",
                "preprocessing": "64x80 RGB PNG, divide uint8 by 255, no resize/crop",
                "tolerance": {"atol": LPIPS_ATOL, "rtol": LPIPS_RTOL},
                "pairs": receipts,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Pretrained LPIPS scoring receipt: {receipt_path}")
