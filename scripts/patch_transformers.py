#!/usr/bin/env python3
"""Apply the narrow HAWK integration to a Transformers 4.52.0 runtime.

Only two stable insertion points are modified:

1. Qwen2.5-VL prefill calls ``hawk.pruning.prune_qwen_prefill``.
2. Generation bookkeeping adopts the shorter prompt after that prefill.

The generated runtime is intentionally separate from the user's site-packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import shutil
from pathlib import Path


MODEL_MARKER = "# HAWK_PATCH_MODEL_BEGIN"
GENERATION_V1_MARKER = "# HAWK_PATCH_GENERATION_BEGIN"
GENERATION_V2_MARKER = "# HAWK_PATCH_GENERATION_V2_BEGIN"
GENERATION_MARKER = "# HAWK_PATCH_GENERATION_V3_BEGIN"

MODEL_ANCHOR = """        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
"""

MODEL_REPLACEMENT = """        # HAWK_PATCH_MODEL_BEGIN
        hawk_config = getattr(self, "_hawk_config", None)
        hawk_cache_is_empty = (
            past_key_values is None
            or past_key_values.get_seq_length() == 0
        )
        hawk_cache_starts_at_zero = (
            cache_position is None
            or (cache_position.numel() > 0 and cache_position[0] == 0)
        )
        if (
            hawk_config is not None
            and hawk_config.enabled
            and input_ids is not None
            and hawk_cache_is_empty
            and hawk_cache_starts_at_zero
        ):
            from hawk.pruning import prune_qwen_prefill

            hawk_prefill = prune_qwen_prefill(
                model=self,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                config=hawk_config,
            )
            inputs_embeds = hawk_prefill.inputs_embeds
            attention_mask = hawk_prefill.attention_mask
            position_ids = hawk_prefill.position_ids
            cache_position = hawk_prefill.cache_position
        # HAWK_PATCH_MODEL_END

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
"""

GENERATION_ANCHOR = """            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
"""

GENERATION_REPLACEMENT = """            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)

                # HAWK_PATCH_GENERATION_BEGIN
                hawk_inner_model = getattr(self, "model", None)
                hawk_state = getattr(hawk_inner_model, "_hawk_generation_state", None)
                if (
                    hawk_state is not None
                    and not hawk_state.get("consumed", False)
                    and hawk_state["stats"].get("applied", False)
                ):
                    keep_indices = hawk_state["keep_indices"].to(model_inputs["input_ids"].device)
                    kept_length = int(keep_indices.numel())
                    original_length = int(model_inputs["input_ids"].shape[1])
                    removed_length = original_length - kept_length
                    if outputs.logits.shape[1] != kept_length:
                        raise RuntimeError(
                            "HAWK prefill/generation length mismatch: "
                            f"logits={outputs.logits.shape[1]}, selected={kept_length}"
                        )

                    # HAWK_PATCH_GENERATION_V3_BEGIN
                    # Qwen caches rope_deltas on the inner multimodal model.
                    # Updating only the returned ModelOutput leaves that cache
                    # stale and shifts every decode position after pruning.
                    if outputs.rope_deltas is not None:
                        corrected_rope_deltas = outputs.rope_deltas + removed_length
                        outputs.rope_deltas = corrected_rope_deltas
                        hawk_inner_model.rope_deltas = corrected_rope_deltas
                    # HAWK_PATCH_GENERATION_V3_END
                    model_inputs["cache_position"] = model_inputs["cache_position"][:kept_length]
                    model_inputs["input_ids"] = model_inputs["input_ids"].index_select(
                        1, keep_indices
                    )
                    input_ids = model_inputs["input_ids"]

                    if model_inputs.get("attention_mask") is not None:
                        model_inputs["attention_mask"] = model_inputs["attention_mask"].index_select(
                            1, keep_indices.to(model_inputs["attention_mask"].device)
                        )
                        model_kwargs["attention_mask"] = model_inputs["attention_mask"]
                    model_kwargs["cache_position"] = model_inputs["cache_position"]

                    # HAWK_PATCH_GENERATION_V2_BEGIN
                    # Generation stopping criteria were constructed from the
                    # original prompt length. Shift every length limit by the
                    # number of removed visual tokens so max_new_tokens keeps
                    # its original meaning after prompt compaction.
                    cur_len = kept_length
                    for criterion in stopping_criteria:
                        if hasattr(criterion, "max_length"):
                            criterion.max_length = max(
                                kept_length + 1,
                                criterion.max_length - removed_length,
                            )
                    # HAWK_PATCH_GENERATION_V2_END
                    hawk_state["consumed"] = True
                # HAWK_PATCH_GENERATION_END

                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
"""

GENERATION_UPGRADE_ANCHOR = """                    model_kwargs["cache_position"] = model_inputs["cache_position"]
                    hawk_state["consumed"] = True
"""

GENERATION_UPGRADE_REPLACEMENT = """                    model_kwargs["cache_position"] = model_inputs["cache_position"]

                    # HAWK_PATCH_GENERATION_V2_BEGIN
                    # Generation stopping criteria were constructed from the
                    # original prompt length. Shift every length limit by the
                    # number of removed visual tokens so max_new_tokens keeps
                    # its original meaning after prompt compaction.
                    cur_len = kept_length
                    for criterion in stopping_criteria:
                        if hasattr(criterion, "max_length"):
                            criterion.max_length = max(
                                kept_length + 1,
                                criterion.max_length - removed_length,
                            )
                    # HAWK_PATCH_GENERATION_V2_END
                    hawk_state["consumed"] = True
"""

GENERATION_V2_UPGRADE_ANCHOR = """                    if outputs.rope_deltas is not None:
                        outputs.rope_deltas = outputs.rope_deltas + removed_length
"""

GENERATION_V2_UPGRADE_REPLACEMENT = """                    # HAWK_PATCH_GENERATION_V3_BEGIN
                    # Qwen caches rope_deltas on the inner multimodal model.
                    # Updating only the returned ModelOutput leaves that cache
                    # stale and shifts every decode position after pruning.
                    if outputs.rope_deltas is not None:
                        corrected_rope_deltas = outputs.rope_deltas + removed_length
                        outputs.rope_deltas = corrected_rope_deltas
                        hawk_inner_model.rope_deltas = corrected_rope_deltas
                    # HAWK_PATCH_GENERATION_V3_END
"""


def _replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: expected one insertion anchor, found {count}")
    return text.replace(anchor, replacement, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_file(path: Path, marker: str, anchor: str, replacement: str, check: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return "already-patched"
    patched = _replace_once(text, anchor, replacement, str(path))
    if check:
        return "would-patch"

    backup = path.with_suffix(path.suffix + ".hawk.orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    return "patched"


def patch_generation_file(path: Path, check: bool) -> str:
    """Patch a clean runtime or upgrade an earlier HAWK generation patch."""

    text = path.read_text(encoding="utf-8")
    if GENERATION_MARKER in text:
        return "already-patched-v3"
    if GENERATION_V2_MARKER in text:
        patched = _replace_once(
            text,
            GENERATION_V2_UPGRADE_ANCHOR,
            GENERATION_V2_UPGRADE_REPLACEMENT,
            f"{path} v2-upgrade",
        )
        status = "would-upgrade-v2" if check else "upgraded-v2-to-v3"
    elif GENERATION_V1_MARKER in text:
        patched = _replace_once(
            text,
            GENERATION_UPGRADE_ANCHOR,
            GENERATION_UPGRADE_REPLACEMENT,
            f"{path} v1-upgrade",
        )
        patched = _replace_once(
            patched,
            GENERATION_V2_UPGRADE_ANCHOR,
            GENERATION_V2_UPGRADE_REPLACEMENT,
            f"{path} v1-v3-upgrade",
        )
        status = "would-upgrade-v1" if check else "upgraded-v1-to-v3"
    else:
        patched = _replace_once(text, GENERATION_ANCHOR, GENERATION_REPLACEMENT, str(path))
        status = "would-patch-v3" if check else "patched-v3"
    if check:
        return status

    backup = path.with_suffix(path.suffix + ".hawk.orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="Path to the generated transformers package directory",
    )
    parser.add_argument("--check", action="store_true", help="Validate anchors without writing")
    args = parser.parse_args()

    package = args.target.resolve()
    model_file = package / "models/qwen2_5_vl/modeling_qwen2_5_vl.py"
    generation_file = package / "generation/utils.py"
    for path in (model_file, generation_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    before = {
        "modeling_qwen2_5_vl.py": _sha256(model_file),
        "generation/utils.py": _sha256(generation_file),
    }
    statuses = {
        "modeling_qwen2_5_vl.py": patch_file(
            model_file, MODEL_MARKER, MODEL_ANCHOR, MODEL_REPLACEMENT, args.check
        ),
        "generation/utils.py": patch_generation_file(generation_file, args.check),
    }

    if not args.check:
        py_compile.compile(str(model_file), doraise=True)
        py_compile.compile(str(generation_file), doraise=True)
        manifest = {
            "transformers_version": "4.52.0",
            "files_before_or_current": before,
            "files_after": {
                "modeling_qwen2_5_vl.py": _sha256(model_file),
                "generation/utils.py": _sha256(generation_file),
            },
            "statuses": statuses,
        }
        manifest_path = package.parent / "hawk-patch-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
