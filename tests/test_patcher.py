from pathlib import Path

from scripts.patch_transformers import (
    GENERATION_MARKER,
    GENERATION_V2_MARKER,
    GENERATION_V2_UPGRADE_ANCHOR,
    GENERATION_UPGRADE_ANCHOR,
    patch_generation_file,
    _replace_once,
)


def test_replace_once_rejects_ambiguous_anchor() -> None:
    try:
        _replace_once("x x", "x", "y", "test")
    except RuntimeError as error:
        assert "found 2" in str(error)
    else:
        raise AssertionError("ambiguous anchor was accepted")


def test_replace_once() -> None:
    assert _replace_once("before TOKEN after", "TOKEN", "PATCH", "test") == (
        "before PATCH after"
    )


def test_runtime_patch_markers_are_named() -> None:
    source = Path("scripts/patch_transformers.py").read_text()
    assert "HAWK_PATCH_MODEL_BEGIN" in source
    assert "HAWK_PATCH_GENERATION_BEGIN" in source
    assert "HAWK_PATCH_GENERATION_V2_BEGIN" in source
    assert "HAWK_PATCH_GENERATION_V3_BEGIN" in source


def test_generation_v1_patch_is_upgraded_to_v3(tmp_path: Path) -> None:
    target = tmp_path / "utils.py"
    target.write_text(
        "# HAWK_PATCH_GENERATION_BEGIN\n"
        + GENERATION_V2_UPGRADE_ANCHOR
        + GENERATION_UPGRADE_ANCHOR
    )
    assert patch_generation_file(target, check=False) == "upgraded-v1-to-v3"
    assert GENERATION_MARKER in target.read_text()
    assert "hawk_inner_model.rope_deltas = corrected_rope_deltas" in target.read_text()
    assert patch_generation_file(target, check=False) == "already-patched-v3"


def test_generation_v2_patch_is_upgraded_to_v3(tmp_path: Path) -> None:
    target = tmp_path / "utils.py"
    target.write_text(GENERATION_V2_MARKER + "\n" + GENERATION_V2_UPGRADE_ANCHOR)
    assert patch_generation_file(target, check=False) == "upgraded-v2-to-v3"
    patched = target.read_text()
    assert GENERATION_MARKER in patched
    assert "hawk_inner_model.rope_deltas = corrected_rope_deltas" in patched
