"""
Orchestrates the full pipeline:
  load from HF Hub -> detect format -> standardize -> align features ->
  concatenate/interleave -> system-prompt injection -> filter -> dedupe ->
  quality scan -> shuffle -> split -> stats -> save (multi-format) ->
  dataset card -> push

Needs: pip install -r requirements.txt (datasets, huggingface_hub, pyyaml, tqdm)
This file is meant to run on YOUR machine / Colab, where you have network
access to the Hugging Face Hub. It can't be exercised inside a sandboxed
environment without internet, so read it carefully and start with
`weave inspect` on a small config before doing a full `weave merge`.
"""

import csv
import gc
import hashlib
import json
import logging
import os
import re
import tempfile
import time

from .formats import STANDARDIZERS, detect_format, fuzzy_detect, text_target_from_messages

logger = logging.getLogger("weave")

# formats that are shaped like {"messages": [...]}
_CHAT_SHAPED = {"alpaca", "sharegpt", "chatml", "prompt_completion", "qa", "input_output", "alpaca_fuzzy"}


def _is_locked_file_error(e):
    """WinError 1224: 'The requested operation cannot be performed on a file with a
    user-mapped section open.' Windows refuses to replace/rename/delete an Arrow
    cache file while any Dataset object (even a stale one Python hasn't garbage
    collected yet) still has it memory-mapped. It's almost always transient."""
    return isinstance(e, OSError) and (getattr(e, "winerror", None) == 1224 or "user-mapped section" in str(e))


def _retry_on_lock(fn, *args, retries=6, base_delay=0.4, **kwargs):
    """
    Run fn(*args, **kwargs). If it fails with the Windows memory-mapped-file
    lock error, force a garbage-collection pass (which releases any lingering
    mmap handles from a previous run) and retry with backoff, instead of
    surfacing the error to the user.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if not _is_locked_file_error(e):
                raise
            last_err = e
            logger.warning(f"File temporarily locked by Windows (attempt {attempt + 1}/{retries}) — retrying...")
            gc.collect()
            time.sleep(base_delay * (attempt + 1))
    raise RuntimeError(
        "A file stayed locked by Windows even after retrying (WinError 1224 — "
        "'user-mapped section open'). This is a known pyarrow/datasets quirk on "
        "Windows, not a problem with your data. It should clear itself if you "
        "restart the app once — this run already used a fresh cache directory "
        "and kept transforms in memory to avoid it."
    ) from last_err


def load_config(path):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_token(cli_token=None):
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def entry_label(entry):
    return entry.get("repo_id") or entry.get("local_path") or "unknown source"


def load_local(path, max_samples=None):
    """Load a local CSV/JSON/JSONL/Parquet file as a Dataset."""
    import pandas as pd
    from datasets import Dataset

    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(path)
    elif ext == "jsonl":
        df = pd.read_json(path, lines=True)
    elif ext == "json":
        df = pd.read_json(path, lines=False)
    elif ext == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported local file type: .{ext} (use csv, json, jsonl, or parquet)")

    if max_samples:
        df = df.head(max_samples)
    return Dataset.from_pandas(df, preserve_index=False)


def load_source(entry, hf_token=None, cache_dir=None):
    if entry.get("local_path"):
        logger.info(f"Loading local file {entry['local_path']}")
        ds = load_local(entry["local_path"], entry.get("max_samples"))
    else:
        from datasets import load_dataset

        repo_id = entry["repo_id"]
        config_name = entry.get("config")
        split = entry.get("split", "train")
        streaming = entry.get("streaming", False)

        logger.info(f"Loading {repo_id} (config={config_name}, split={split}, streaming={streaming})")
        ds = _retry_on_lock(load_dataset, repo_id, config_name, split=split, streaming=streaming, token=hf_token, cache_dir=cache_dir)

        max_samples = entry.get("max_samples")
        if max_samples:
            if streaming:
                from datasets import Dataset

                ds = Dataset.from_list(list(ds.take(max_samples)))
            else:
                ds = _retry_on_lock(ds.select, range(min(max_samples, len(ds))))
        elif streaming:
            # materialize streaming datasets fully if no cap was given
            from datasets import Dataset

            logger.warning(f"{repo_id}: streaming=true but no max_samples set — pulling the full dataset into memory.")
            ds = Dataset.from_list(list(ds))

    # Optional per-dataset sample fraction — lets you down-weight one source
    # relative to the others in *concatenate* mode (interleave already has
    # its own `probabilities` control for this).
    fraction = entry.get("sample_fraction")
    if fraction is not None and 0 < fraction < 1:
        n_take = max(1, int(len(ds) * fraction))
        ds = _retry_on_lock(ds.shuffle, seed=entry.get("sample_seed", 42))
        ds = _retry_on_lock(ds.select, range(n_take))
        logger.info(f"{entry_label(entry)}: sample_fraction={fraction} -> kept {n_take} rows")

    return ds


def peek_rows(ds, n=5):
    try:
        return ds.select(range(min(n, len(ds)))).to_list()
    except Exception:
        return list(ds.take(n))


def standardize_entry(ds, entry, target_format):
    columns = list(ds.column_names)
    sample = peek_rows(ds, 5)
    fmt = detect_format(columns, sample, override=entry.get("format"))

    if fmt == "unknown":
        raise ValueError(
            f"Could not auto-detect the format of '{entry_label(entry)}' "
            f"(columns: {columns}). Add `format:` and `columns:` to its entry "
            f"in the config to map it manually."
        )

    standardizer = STANDARDIZERS[fmt]
    col_overrides = dict(entry.get("columns", {}))

    if fmt == "alpaca_fuzzy" and "mapping" not in col_overrides:
        _, mapping = fuzzy_detect(columns)
        col_overrides["mapping"] = mapping

    def _map(row):
        out = standardizer(row, **col_overrides)

        # --- reconcile the source's natural shape with the requested target shape ---
        if target_format == "text":
            if "messages" in out:
                out = {"text": text_target_from_messages(out["messages"])}
            elif "chosen" in out:
                out = {"text": text_target_from_messages([
                    {"role": "user", "content": out["prompt"]},
                    {"role": "assistant", "content": out["chosen"]},
                ])}
        elif target_format == "chatml":
            if "text" in out:
                out = {"messages": [{"role": "assistant", "content": out["text"]}]}
            elif "chosen" in out:
                out = {"messages": [
                    {"role": "user", "content": out["prompt"]},
                    {"role": "assistant", "content": out["chosen"]},
                ]}
        elif target_format == "preference":
            if "chosen" not in out:
                raise ValueError(
                    f"{entry_label(entry)}: target_format is 'preference' but this dataset has no "
                    f"chosen/rejected pair (detected as '{fmt}'). Preference (DPO-style) merges need "
                    f"every source dataset to already have chosen/rejected columns — you can't fabricate "
                    f"a 'rejected' response from plain chat data."
                )

        out["source_dataset"] = entry_label(entry)
        out["source_split"] = entry.get("split", "train")
        return out

    standardized = _retry_on_lock(ds.map, _map, remove_columns=ds.column_names)
    return standardized, fmt


def align_features(datasets_list):
    """Make sure every dataset has the same columns before concatenation."""
    all_cols = set()
    for d in datasets_list:
        all_cols.update(d.column_names)
    aligned = []
    for d in datasets_list:
        missing = all_cols - set(d.column_names)
        for col in missing:
            d = _retry_on_lock(d.add_column, col, [None] * len(d))
        aligned.append(d)
    return aligned


def _safe_concatenate(aligned):
    """concatenate_datasets, but with a clear error message instead of a raw
    pyarrow schema exception if two sources still don't line up (this should
    be rare now that every standardizer coerces content to plain strings, but
    a hand-written `format:`/`columns:` override in config.yaml could still
    produce a mismatched schema)."""
    from datasets import concatenate_datasets

    try:
        return _retry_on_lock(concatenate_datasets, aligned)
    except Exception as e:
        raise ValueError(
            "Couldn't merge these datasets — their standardized schemas don't match "
            f"({e}). This usually means a custom `columns:` mapping in config.yaml is "
            "still producing non-string content for one of the sources. Double check any "
            "manual format overrides."
        ) from e


def inject_system_prompt(dataset, system_prompt):
    """Prepend a system message to every row that doesn't already start with one."""
    def _map(row):
        msgs = row.get("messages") or []
        if not msgs or msgs[0].get("role") != "system":
            row["messages"] = [{"role": "system", "content": system_prompt}] + list(msgs)
        return row

    return _retry_on_lock(dataset.map, _map)


_WS_RE = re.compile(r"\s+")


def _normalize_for_hash(s):
    return _WS_RE.sub(" ", s or "").strip().lower()


def hash_row(row, target_format, normalize=False):
    norm = _normalize_for_hash if normalize else (lambda s: s or "")
    if target_format == "chatml":
        payload = "||".join(f"{m.get('role')}:{norm(m.get('content'))}" for m in row.get("messages", []))
    elif target_format == "preference":
        payload = f"{norm(row.get('prompt'))}||{norm(row.get('chosen'))}||{norm(row.get('rejected'))}"
    else:
        payload = norm(row.get("text", ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dedupe_with_examples(dataset, target_format, max_examples=3, normalize=False):
    """Same as dedupe, but also returns a few short previews of what got dropped, for transparency."""
    seen = set()
    keep = []
    examples = []
    for i, row in enumerate(dataset):
        h = hash_row(row, target_format, normalize=normalize)
        if h in seen:
            if len(examples) < max_examples:
                if target_format == "chatml" and row.get("messages"):
                    preview = row["messages"][0].get("content", "")[:140]
                elif target_format == "preference":
                    preview = row.get("prompt", "")[:140]
                else:
                    preview = row.get("text", "")[:140]
                examples.append({"source": row.get("source_dataset", "unknown"), "preview": preview})
            continue
        seen.add(h)
        keep.append(i)
    return _retry_on_lock(dataset.select, keep), examples


def dedupe(dataset, target_format, normalize=False):
    deduped, _examples = dedupe_with_examples(dataset, target_format, normalize=normalize)
    return deduped


def row_length(row, target_format):
    if target_format == "chatml":
        return sum(len(m.get("content", "")) for m in row.get("messages", []))
    if target_format == "preference":
        return len(row.get("prompt", "")) + len(row.get("chosen", "")) + len(row.get("rejected", ""))
    return len(row.get("text", ""))


def filter_length(dataset, target_format, min_len, max_len):
    def _keep(row):
        n = row_length(row, target_format)
        if min_len is not None and n < min_len:
            return False
        if max_len is not None and n > max_len:
            return False
        return True

    return _retry_on_lock(dataset.filter, _keep)


def scan_quality(dataset, target_format, sample_cap=20000):
    """
    Lightweight data-quality pass: flags rows with empty/near-empty content
    after standardization. A non-trivial empty-content rate is usually a sign
    that a source dataset's real content lives under a column name `weave`
    didn't map (or was buried in a nested object that couldn't be flattened),
    so surfacing it here — instead of only in the final row count — makes
    that visible before you commit to a merge.
    """
    n = len(dataset)
    sample = dataset if n <= sample_cap else dataset.shuffle(seed=0).select(range(sample_cap))
    empty = 0
    for row in sample:
        if target_format == "chatml":
            msgs = row.get("messages") or []
            if not msgs or all(not (m.get("content") or "").strip() for m in msgs):
                empty += 1
        elif target_format == "preference":
            if not (row.get("chosen") or "").strip() or not (row.get("rejected") or "").strip():
                empty += 1
        else:
            if not (row.get("text") or "").strip():
                empty += 1
    return {"empty_rows_in_sample": empty, "sample_size": len(sample), "empty_rate": empty / len(sample) if len(sample) else 0}


def compute_merge_stats(dataset, target_format, sample_cap=50000, seed=42):
    """Source distribution, length histogram data, and role distribution — sampled for speed on huge merges."""
    n = len(dataset)
    sample = dataset if n <= sample_cap else dataset.shuffle(seed=seed).select(range(sample_cap))
    source_counts, lengths, role_counts = {}, [], {}
    for row in sample:
        src = row.get("source_dataset", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        lengths.append(row_length(row, target_format))
        if target_format == "chatml":
            for m in row.get("messages", []):
                role_counts[m.get("role", "?")] = role_counts.get(m.get("role", "?"), 0) + 1
    return {
        "source_counts": source_counts,
        "lengths": lengths,
        "role_counts": role_counts,
        "sampled": n > sample_cap,
        "sample_size": len(sample),
    }


def _save_split(dataset, base_path, formats):
    """Save one split in each requested format. Returns {format: path}."""
    written = {}
    if "jsonl" in formats:
        target = f"{base_path}.jsonl"
        try:
            _retry_on_lock(dataset.to_json, target)
        except RuntimeError:
            target = f"{base_path}_{int(time.time())}.jsonl"
            logger.warning(f"{base_path}.jsonl stayed locked — saving as {target} instead.")
            dataset.to_json(target)
        written["jsonl"] = target
    if "csv" in formats:
        target = f"{base_path}.csv"
        _retry_on_lock(dataset.to_csv, target)
        written["csv"] = target
    if "parquet" in formats:
        target = f"{base_path}.parquet"
        _retry_on_lock(dataset.to_parquet, target)
        written["parquet"] = target
    return written


def write_dataset_card(save_path, summary, cfg):
    """Auto-generate a short README.md dataset card summarizing how this merge was built."""
    out = cfg.get("output", {})
    lines = [
        "---",
        "tags:",
        "- weave-merged",
        "---",
        "",
        "# Merged dataset",
        "",
        f"Built with **Weave** from {len(summary['sources'])} source dataset(s), "
        f"target format `{summary['target_format']}`, mode `{out.get('mode', 'concatenate')}`.",
        "",
        "## Sources",
        "",
        "| Dataset | Detected format | Rows contributed |",
        "|---|---|---|",
    ]
    for s in summary["sources"]:
        lines.append(f"| `{s['repo_id']}` | {s['format']} | {s['rows']:,} |")
    lines += [
        "",
        "## Pipeline",
        "",
        f"- Rows before filters: {summary['rows_before_filters']:,}",
        f"- Rows after filters: {summary['rows_after_filters']:,}",
        f"- Duplicates removed: {summary['duplicates_removed']:,}",
        f"- Final rows: {summary['final_rows']:,}",
        f"- Splits: {', '.join(f'{k} ({v:,})' for k, v in summary['splits'].items())}",
    ]
    if out.get("system_prompt"):
        lines.append(f"- System prompt injected: yes")
    quality = summary.get("quality")
    if quality:
        lines.append(f"- Empty-content rows in sample: {quality['empty_rows_in_sample']}/{quality['sample_size']} ({quality['empty_rate']:.1%})")
    lines.append("")
    lines.append("_Generated automatically by Weave._")
    path = os.path.join(save_path, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def run(config_path, hf_token=None, inspect_only=False):
    """Run from a YAML file on disk (CLI entry point)."""
    cfg = load_config(config_path)
    return run_from_config(cfg, hf_token=hf_token, inspect_only=inspect_only)


def run_from_config(cfg, hf_token=None, inspect_only=False):
    """Run to completion, logging each step. Used by the CLI; the GUI uses run_from_config_steps directly for live progress."""
    summary = None
    for event in run_from_config_steps(cfg, hf_token=hf_token, inspect_only=inspect_only):
        if event["stage"] == "done":
            summary = event["summary"]
        elif event.get("message"):
            logger.info(event["message"])
    return summary


def run_from_config_steps(cfg, hf_token=None, inspect_only=False):
    """
    Generator form of the pipeline. Yields progress events:
      {"stage": <str>, "message": <str>, "progress": <0..1>, ...extra fields}
    and finally: {"stage": "done", "progress": 1.0, "summary": {...}}
    Built so a GUI can show a live step-by-step log instead of one opaque call.
    """
    from datasets import DatasetDict, disable_caching, interleave_datasets

    # Keeps .map/.filter/.select transforms in memory instead of writing them to
    # on-disk arrow cache files — this is the main thing that avoids WinError 1224,
    # since there's no cache file left around for a later run to collide with.
    disable_caching()
    fresh_cache_dir = tempfile.mkdtemp(prefix="weave_cache_")

    output_cfg = cfg.get("output", {})
    target_format = output_cfg.get("target_format", "chatml")
    token = _get_token(hf_token)
    entries = cfg["datasets"]
    n = max(len(entries), 1)

    processed = []
    report = []

    for i, entry in enumerate(entries):
        label = entry_label(entry)
        yield {"stage": "load", "message": f"Loading {label}...", "progress": (i / n) * 0.35}
        ds = load_source(entry, token, cache_dir=fresh_cache_dir)
        columns = list(ds.column_names)
        sample = peek_rows(ds, 5)
        fmt = detect_format(columns, sample, override=entry.get("format"))
        yield {
            "stage": "detect",
            "message": f"{label}: detected format = {fmt} ({len(ds)} rows, {len(columns)} columns)",
            "progress": (i / n) * 0.35 + 0.04,
            "label": label,
            "format": fmt,
            "rows": len(ds),
            "columns": columns,
            "sample": sample[:3],
        }

        if inspect_only:
            report.append({"repo_id": label, "format": fmt, "rows": len(ds), "columns": columns, "sample": sample[:1]})
            continue

        yield {"stage": "standardize", "message": f"Standardizing {label} ({fmt} -> {target_format})...", "progress": (i / n) * 0.35 + 0.08}
        std, fmt = standardize_entry(ds, entry, target_format)

        yield {"stage": "quality", "message": f"Checking data quality for {label}...", "progress": (i / n) * 0.35 + 0.1}
        quality = scan_quality(std, target_format)
        if quality["empty_rate"] > 0.05:
            yield {
                "stage": "quality",
                "message": (
                    f"{label}: {quality['empty_rows_in_sample']}/{quality['sample_size']} sampled rows "
                    f"({quality['empty_rate']:.1%}) came out with empty content after standardizing — "
                    f"double check the column mapping for this source."
                ),
                "progress": (i / n) * 0.35 + 0.1,
            }

        report.append({"repo_id": label, "format": fmt, "rows": len(std), "quality": quality})
        processed.append(std)
        del ds
        gc.collect()  # release this dataset's mmap handle before the next one is loaded

    if inspect_only:
        yield {"stage": "done", "progress": 1.0, "summary": {"inspected": report}}
        return

    yield {"stage": "align", "message": "Aligning schemas across all datasets so they can be concatenated...", "progress": 0.42}
    aligned = align_features(processed)
    del processed
    gc.collect()

    mode = output_cfg.get("mode", "concatenate")
    seed = output_cfg.get("seed", 42)
    yield {"stage": "merge", "message": f"Merging {len(aligned)} datasets ({mode})...", "progress": 0.52}
    if mode == "interleave":
        probabilities = output_cfg.get("probabilities")
        merged = _retry_on_lock(interleave_datasets, aligned, probabilities=probabilities, seed=seed, stopping_strategy=output_cfg.get("stopping_strategy", "all_exhausted"))
    else:
        merged = _safe_concatenate(aligned)
    del aligned
    gc.collect()
    rows_before_filters = len(merged)
    yield {"stage": "merge", "message": f"Merged: {rows_before_filters} rows total.", "progress": 0.58}

    system_prompt = output_cfg.get("system_prompt")
    if system_prompt and target_format == "chatml":
        yield {"stage": "standardize", "message": "Injecting system prompt into rows that don't already have one...", "progress": 0.6}
        merged = inject_system_prompt(merged, system_prompt)

    min_len = output_cfg.get("min_length")
    max_len = output_cfg.get("max_length")
    if min_len is not None or max_len is not None:
        yield {"stage": "filter", "message": f"Filtering by length (min={min_len}, max={max_len})...", "progress": 0.66}
        merged = filter_length(merged, target_format, min_len, max_len)
    rows_after_filter = len(merged)

    duplicates_removed = 0
    duplicate_examples = []
    if output_cfg.get("dedupe", True):
        normalize = bool(output_cfg.get("normalize_dedupe", False))
        yield {"stage": "dedupe", "message": "Scanning for duplicate rows" + (" (whitespace/case-normalized)" if normalize else " (exact)") + "...", "progress": 0.72}
        before = len(merged)
        merged, duplicate_examples = dedupe_with_examples(merged, target_format, normalize=normalize)
        duplicates_removed = before - len(merged)
        yield {"stage": "dedupe", "message": f"Removed {duplicates_removed} duplicate rows.", "progress": 0.78}

    yield {"stage": "stats", "message": "Checking overall data quality...", "progress": 0.81}
    quality = scan_quality(merged, target_format)

    if output_cfg.get("shuffle", True):
        yield {"stage": "shuffle", "message": "Shuffling...", "progress": 0.85}
        merged = merged.shuffle(seed=seed)

    result = {"train": merged}
    split_frac = output_cfg.get("train_test_split")
    if split_frac:
        yield {"stage": "split", "message": f"Splitting off {split_frac:.0%} as a test set...", "progress": 0.89}
        split = merged.train_test_split(test_size=split_frac, seed=seed)
        result = {"train": split["train"], "test": split["test"]}

    yield {"stage": "stats", "message": "Computing summary statistics...", "progress": 0.92}
    stats = compute_merge_stats(merged, target_format, seed=seed)

    save_path = output_cfg.get("save_path", "./merged_dataset")
    os.makedirs(save_path, exist_ok=True)
    save_formats = output_cfg.get("save_formats") or ["jsonl"]
    yield {"stage": "save", "message": f"Saving to {save_path}/ as {', '.join(save_formats)}...", "progress": 0.95}
    save_files = {}
    save_files_by_format = {}
    for name, d in result.items():
        written = _save_split(d, os.path.join(save_path, name), save_formats)
        save_files_by_format[name] = written
        save_files[name] = written.get("jsonl") or next(iter(written.values()), None)

    summary = {
        "sources": report,
        "rows_before_filters": rows_before_filters,
        "rows_after_filters": rows_after_filter,
        "duplicates_removed": duplicates_removed,
        "duplicate_examples": duplicate_examples,
        "final_rows": len(merged),
        "splits": {k: len(v) for k, v in result.items()},
        "save_path": save_path,
        "save_files": save_files,
        "save_files_by_format": save_files_by_format,
        "target_format": target_format,
        "quality": quality,
        **stats,
    }

    if output_cfg.get("write_dataset_card", True):
        card_path = write_dataset_card(save_path, summary, cfg)
        summary["dataset_card"] = card_path

    push = output_cfg.get("push_to_hub")
    if push:
        yield {"stage": "push", "message": f"Pushing to Hub: {push}...", "progress": 0.99}
        _retry_on_lock(DatasetDict(result).push_to_hub, push, token=token)

    yield {"stage": "done", "progress": 1.0, "message": "Done.", "summary": summary}
