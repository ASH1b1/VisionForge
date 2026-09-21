from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = Path(__file__).resolve().parents[1]

_LA_PREFIXES = (
    "models/locate_anything/",
    "models/grounding_dino/modules/transformers_modules/locate_anything/",
)


def test_locate_anything_model_py_is_gone():
    assert not (_SRC_ROOT / "gui" / "models" / "locate_anything_model.py").exists()


def test_gitignore_blocks_locate_anything_trees():
    text = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "models/locate_anything/" in text
    assert "models/grounding_dino/modules/transformers_modules/locate_anything/" in text


def test_git_index_has_no_locate_anything_paths():
    result = subprocess.run(
        ["git", "ls-files", "--", "models/locate_anything", "models/grounding_dino/modules/transformers_modules/locate_anything"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line.replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], tracked
