"""Smoke tests for package metadata and versioned JSON configuration files."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import tradingbot

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestProjectSmoke(unittest.TestCase):
    def test_package_version_matches_pyproject(self) -> None:
        self.assertIsInstance(tradingbot.__version__, str)
        self.assertTrue(len(tradingbot.__version__) > 0)
        pyproject_text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.MULTILINE)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(tradingbot.__version__, match.group(1))

    def test_all_config_files_are_valid_json(self) -> None:
        config_files = sorted((ROOT_DIR / "templates" / "configs").rglob("*.json"))
        self.assertGreaterEqual(len(config_files), 5)
        for path in config_files:
            with self.subTest(path=str(path)):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsInstance(data, dict)


if __name__ == "__main__":
    unittest.main()
