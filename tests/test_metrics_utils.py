# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import yaml

from model_eval.metrics_utils import MetricsAggregator, MetricSuite, _make_masked_lpips_fn, compute_rgb_metrics


class MetricsAggregatorTests(unittest.TestCase):
    def test_standard_lpips_converts_unit_rgb_exactly_once(self):
        prediction = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1).expand(1, 3, 32, 32)
        target = torch.full_like(prediction, 0.25)
        with patch("model_eval.lpipsPytorch.lpips", return_value=torch.tensor(0.3)) as metric:
            compute_rgb_metrics(prediction, target, lpips_protocol="standard")
        actual_prediction, actual_target = metric.call_args.args
        torch.testing.assert_close(actual_prediction[:, :, 0, 0], torch.tensor([[-1.0, 0.0, 1.0]]))
        torch.testing.assert_close(actual_target, torch.full_like(target, -0.5))

    def test_lpips_protocol_is_required(self):
        image = torch.zeros(1, 3, 32, 32)
        with self.assertRaisesRegex(TypeError, "lpips_protocol"):
            compute_rgb_metrics(image, image)

    def test_legacy_lpips_and_other_metrics_remain_unchanged(self):
        target = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        prediction = 0.8 * target + 0.1
        with patch("model_eval.lpipsPytorch.lpips", side_effect=_fake_lpips):
            legacy = compute_rgb_metrics(prediction, target, lpips_protocol="legacy")
            standard = compute_rgb_metrics(prediction, target, lpips_protocol="standard")
        self.assertEqual(legacy["psnr"], standard["psnr"])
        self.assertEqual(legacy["ssim"], standard["ssim"])
        self.assertAlmostEqual(legacy["lpips"], float(((prediction - target) ** 2).mean()))
        self.assertAlmostEqual(standard["lpips"], 4 * legacy["lpips"], places=6)

    def test_require_values_rejects_empty_metric_output(self):
        metrics = MetricsAggregator(["psnr", "ssim", "lpips"])

        with self.assertRaisesRegex(ValueError, "No DL3DV frames were processed"):
            metrics.require_values("psnr", Path("/eval/artifixer-14b/empty"), evalset="artifixer3d_dl3dv_ours")


class _StubFIDTracker:
    """FIDTracker stand-in so tests never instantiate an Inception FID model."""

    def __init__(self, device, resize_to=512, chunk_size=64):
        self.num_images = 0

    def update(self, pred, gt):
        self.num_images += 1

    def compute(self):
        return float(self.num_images) if self.num_images else None


def _fake_lpips(x, y, net_type="alex", version="0.1"):
    """Deterministic lpips stand-in so tests never load pretrained VGG weights."""
    return ((x - y) ** 2).mean()


class MaskedLPIPSNormalizationTests(unittest.TestCase):
    def test_bundled_backends_normalize_only_in_standard_mode(self):
        image = torch.tensor([-0.1, 0.5, 1.1]).reshape(1, 3, 1, 1)
        target = torch.full_like(image, 0.25)
        for backend in ("alex", "squeeze", "vgg"):
            for protocol in ("standard", "legacy"):
                with self.subTest(backend=backend, protocol=protocol):
                    with patch("model_eval.lpipsPytorch.lpips", return_value=torch.tensor(0.3)) as metric:
                        score = _make_masked_lpips_fn(torch.device("cpu"), backend, lpips_protocol=protocol)
                        self.assertAlmostEqual(score(image, target), 0.3)
                    expected_image, expected_target = image.clamp(0, 1), target
                    if protocol == "standard":
                        expected_image, expected_target = 2 * expected_image - 1, 2 * expected_target - 1
                    torch.testing.assert_close(metric.call_args.args[0], expected_image)
                    torch.testing.assert_close(metric.call_args.args[1], expected_target)
                    self.assertEqual(metric.call_args.kwargs, {"net_type": backend})

    def test_torchmetrics_keeps_its_existing_normalization_in_both_modes(self):
        image = torch.tensor([-0.1, 0.5, 1.1]).reshape(1, 3, 1, 1)
        for protocol in ("standard", "legacy"):
            with self.subTest(protocol=protocol):
                metric = MagicMock(return_value=torch.tensor(0.3))
                with patch("torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity") as factory:
                    factory.return_value.to.return_value = metric
                    score = _make_masked_lpips_fn(torch.device("cpu"), "torchmetrics", lpips_protocol=protocol)
                    self.assertAlmostEqual(score(image, image), 0.3)
                factory.assert_called_once_with(net_type="alex", normalize=True)
                for actual in metric.call_args.args:
                    torch.testing.assert_close(actual, image.clamp(0, 1))


class MetricSuiteTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("model_eval.metrics_utils.FIDTracker", _StubFIDTracker))
        self.enterContext(patch("model_eval.lpipsPytorch.lpips", _fake_lpips))
        self.enterContext(
            patch("model_eval.metrics_utils._make_masked_lpips_fn", return_value=lambda image, target: 0.0)
        )
        self.device = torch.device("cpu")
        base = torch.linspace(0.0, 1.0, steps=3 * 32 * 32).reshape(1, 3, 32, 32)
        self.gt = base
        self.pred = base * 0.9 + 0.05
        self.rendered = base * 0.7 + 0.2

    def _load_summary(self, path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)["summary"]

    def test_unmasked_suite_does_not_construct_masked_models(self):
        with (
            patch("model_eval.metrics_utils.FIDTracker", side_effect=_StubFIDTracker) as fid,
            patch("model_eval.metrics_utils._make_masked_lpips_fn") as masked_lpips,
        ):
            MetricSuite(self.device, use_masks=False)

        self.assertEqual(fid.call_count, 2)
        masked_lpips.assert_not_called()

    def test_masked_updates_require_masked_metrics_to_be_enabled(self):
        suite = MetricSuite(self.device, use_masks=False)
        with self.assertRaisesRegex(ValueError, "Masked metrics are disabled"):
            suite.update_masked("scene-a", self.pred, self.rendered, self.gt, torch.ones(1, 1, 32, 32))

    def test_update_routes_pred_to_ours_and_rendered_to_gut(self):
        suite = MetricSuite(self.device, use_masks=False)
        suite.update("scene-a", self.pred, self.rendered, self.gt)

        output_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        suite.report(output_dir)

        expected_ours = compute_rgb_metrics(self.pred, self.gt, lpips_protocol="standard")
        expected_gut = compute_rgb_metrics(self.rendered, self.gt, lpips_protocol="standard")
        # pred and rendered must land in distinct aggregators, so the two summaries differ.
        self.assertNotAlmostEqual(expected_ours["psnr"], expected_gut["psnr"], places=2)

        ours_summary = self._load_summary(output_dir / "metrics_standard" / "metrics.yaml")
        gut_summary = self._load_summary(output_dir / "metrics_standard" / "metrics_3dgut.yaml")
        self.assertAlmostEqual(ours_summary["psnr"], expected_ours["psnr"], places=4)
        self.assertAlmostEqual(gut_summary["psnr"], expected_gut["psnr"], places=4)
        self.assertAlmostEqual(ours_summary["lpips"], expected_ours["lpips"], places=6)
        self.assertAlmostEqual(gut_summary["lpips"], expected_gut["lpips"], places=6)
        self.assertEqual(ours_summary["fid"], 1.0)
        self.assertEqual(gut_summary["fid"], 1.0)

        # No masked updates happened, so the masked metrics files must not be written.
        self.assertFalse((output_dir / "metrics_standard" / "metrics_masked.yaml").exists())
        self.assertFalse((output_dir / "metrics_standard" / "metrics_3dgut_masked.yaml").exists())

    def test_explicit_legacy_and_standard_scores_have_separate_labeled_outputs(self):
        output_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        historical = output_dir / "metrics.yaml"
        historical.write_text("historical scores\n")
        reports = {}
        for protocol in ("standard", "legacy"):
            suite = MetricSuite(self.device, use_masks=False, lpips_protocol=protocol)
            suite.update("scene-a", self.pred, self.rendered, self.gt)
            suite.report(output_dir, metrics_suffix="comparison")
            for filename in ("metrics_comparison.yaml", "metrics_3dgut_comparison.yaml"):
                report = yaml.safe_load((output_dir / f"metrics_{protocol}" / filename).read_text())
                self.assertEqual(report["protocol"]["lpips_protocol"], protocol)
                self.assertEqual(report["protocol"]["lpips_backbone"], "vgg")
                self.assertEqual(report["protocol"]["lpips_input_range"], [-1, 1] if protocol == "standard" else [0, 1])
                reports[protocol, filename] = report["summary"]
        for filename in ("metrics_comparison.yaml", "metrics_3dgut_comparison.yaml"):
            standard, legacy = reports["standard", filename], reports["legacy", filename]
            self.assertAlmostEqual(standard["lpips"], 4 * legacy["lpips"], places=6)
            for metric in ("psnr", "ssim", "fid"):
                self.assertEqual(standard[metric], legacy[metric])
        self.assertEqual(historical.read_text(), "historical scores\n")

    def test_masked_reports_record_the_actual_backbone_and_input_range(self):
        output_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for backend in ("torchmetrics", "vgg"):
            for protocol in ("standard", "legacy"):
                with self.subTest(backend=backend, protocol=protocol):
                    suite = MetricSuite(self.device, True, backend, lpips_protocol=protocol)
                    suite.update_masked("scene-a", self.pred, self.rendered, self.gt, torch.ones(1, 1, 32, 32))
                    suite.report(output_dir, metrics_suffix=backend)
                    report_path = output_dir / f"metrics_{protocol}" / f"metrics_masked_{backend}.yaml"
                    metadata = yaml.safe_load(report_path.read_text())["protocol"]
                    self.assertEqual(metadata["lpips_protocol"], protocol)
                    self.assertEqual(metadata["masked_lpips_backend"], backend)
                    self.assertEqual(metadata["lpips_backbone"], "alex" if backend == "torchmetrics" else "vgg")
                    expected_range = [-1, 1] if protocol == "standard" or backend == "torchmetrics" else [0, 1]
                    self.assertEqual(metadata["lpips_input_range"], expected_range)

    def test_report_writes_masked_yaml_only_after_masked_updates(self):
        from torchmetrics import PeakSignalNoiseRatio

        suite = MetricSuite(self.device, use_masks=True)
        suite.update("scene-a", self.pred, self.rendered, self.gt)

        unmasked_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        suite.report(unmasked_dir)
        self.assertTrue((unmasked_dir / "metrics_standard" / "metrics.yaml").exists())
        self.assertTrue((unmasked_dir / "metrics_standard" / "metrics_3dgut.yaml").exists())
        self.assertFalse((unmasked_dir / "metrics_standard" / "metrics_masked.yaml").exists())
        self.assertFalse((unmasked_dir / "metrics_standard" / "metrics_3dgut_masked.yaml").exists())

        mask = torch.ones(1, 1, 32, 32)
        mask[..., :16] = 0.0
        suite.update_masked("scene-a", self.pred, self.rendered, self.gt, mask)

        masked_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        suite.report(masked_dir)
        ours_masked = self._load_summary(masked_dir / "metrics_standard" / "metrics_masked.yaml")
        gut_masked = self._load_summary(masked_dir / "metrics_standard" / "metrics_3dgut_masked.yaml")

        masked_psnr = PeakSignalNoiseRatio(data_range=1.0)
        expected_ours_psnr = masked_psnr(self.pred * mask, self.gt * mask).item()
        expected_gut_psnr = PeakSignalNoiseRatio(data_range=1.0)(self.rendered * mask, self.gt * mask).item()
        self.assertAlmostEqual(ours_masked["psnr"], expected_ours_psnr, places=4)
        self.assertAlmostEqual(gut_masked["psnr"], expected_gut_psnr, places=4)


if __name__ == "__main__":
    unittest.main()
