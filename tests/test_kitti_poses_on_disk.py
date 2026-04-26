"""Gatekeeper test: KITTI odometry ground-truth poses present on disk.

Pytest-skips if poses are not available; implements spec §6 gatekeeper row 3 so
later ATE evaluation can be gated at runtime rather than hard-failing.
"""
from pathlib import Path

import pytest

POSES_BASE = Path("/home/seanachan/data/Dataset/kitti_odometry/poses")


def _poses_available() -> bool:
    return all((POSES_BASE / f"{seq}.txt").is_file() for seq in ("09", "10"))


@pytest.mark.skipif(not _poses_available(), reason="KITTI odometry poses not extracted yet")
def test_kitti_poses_seq_09_10_have_12_floats_per_line():
    for seq in ("09.txt", "10.txt"):
        path = POSES_BASE / seq
        first_line = path.read_text().splitlines()[0].split()
        assert len(first_line) == 12, f"Expected 12 floats per line in {path}, got {len(first_line)}"


def test_gatekeeper_memo_describes_poses_status():
    memo = Path("diagnostics/results/exp37/gatekeeper_kitti_poses.md")
    assert memo.is_file(), "Gatekeeper memo must exist so ATE fallback is documented"
    content = memo.read_text()
    assert "AVAILABLE" in content or "MISSING" in content, "Memo must state availability"
