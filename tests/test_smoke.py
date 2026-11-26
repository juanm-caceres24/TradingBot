"""Smoke tests for package metadata and versioned JSON configuration files."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import tradingbot

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestProjectSmoke(unittest.TestCase):
    def test_package_version_is_non_empty(self) -> None:
        self.assertIsInstance(tradingbot.__version__, str)
        self.assertTrue(len(tradingbot.__version__) > 0)

    def test_all_config_files_are_valid_json(self) -> None:
        config_files = sorted((ROOT_DIR / "configs").rglob("*.json"))
        self.assertGreaterEqual(len(config_files), 3)
        for path in config_files:
            with self.subTest(path=str(path)):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsInstance(data, dict)


if __name__ == "__main__":
    unittest.main()
