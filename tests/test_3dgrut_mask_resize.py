# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDGRUT_ROOT = REPO_ROOT / "thirdparty" / "3DGRUT-ArtiFixer"
if not (THREEDGRUT_ROOT / "threedgrut").is_dir():
    raise unittest.SkipTest("thirdparty/3DGRUT-ArtiFixer is not initialized")
sys.path.insert(0, str(THREEDGRUT_ROOT))

from threedgrut.datasets.dataset_colmap import load_mask_image_tensor


class ColmapMaskLoadingTest(unittest.TestCase):
    def test_resizes_mask_to_loaded_image_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "00000_mask.png"
            mask = np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
            Image.fromarray(mask).save(mask_path)

            tensor = load_mask_image_tensor(str(mask_path), actual_h=4, actual_w=6)

        self.assertEqual(tuple(tensor.shape), (1, 4, 6, 1))
        self.assertEqual(int(tensor.max()), 255)
        self.assertEqual(int(tensor.min()), 0)
