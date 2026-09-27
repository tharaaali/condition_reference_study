"""Tests for the public command-line runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_study import resolve_limanc_root, validate_ampere_archive


class RunStudyTests(unittest.TestCase):
    def test_resolve_limanc_root_accepts_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "download" / "engine_2"
            (expected / "experiment_1" / "current").mkdir(parents=True)
            self.assertEqual(resolve_limanc_root(root), expected.resolve())

    def test_ampere_input_must_be_a_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ampere.txt"
            path.write_text("not an archive", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_ampere_archive(path)


if __name__ == "__main__":
    unittest.main()
