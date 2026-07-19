"""
Web GUI for Weave. Run with:

    streamlit run app.py

Every control here maps directly to a field in config.yaml — you can build
a merge visually here, export the config, and hand it to a teammate to run
via `python -m weave.cli merge config.yaml`, or import a config someone
else wrote to keep tweaking it here.
"""

import html
import json
import os

import streamlit as st
import yaml

from weave.formats import detect_format, fuzzy_detect
from weave.search import get_dataset_structure, search_datasets

st.set_page_config(page_title="Weave — HF dataset combiner", page_icon="\U0001f9f6", layout="wide")

FORMAT_OPTIONS = ["auto", "alpaca", "sharegpt", "chatml", "prompt_completion", "qa", "input_output", "text", "preference", "alpaca_fuzzy"]
TARGET_FORMATS = ["chatml", "text", "preference"]
SAVE_FORMAT_OPTIONS = ["jsonl", "csv", "parquet"]

FORMAT_COLORS = {
    "alpaca": "#3ED9A3",
    "sharegpt": "#6E9BF4",
    "chatml": "#B48EF0",
    "prompt_completion": "#F0B429",
    "qa": "#4CC9A0",
    "input_output": "#8B98A5",
    "text": "#F0955C",
    "preference": "#59C3E8",
    "alpaca_fuzzy": "#F07CA0",
    "unknown": "#F0555C",
}

STAGE_ICONS = {
    "load": "\U0001f4e5", "detect": "\U0001f50d", "standardize": "\U0001f527",
    "quality": "\U0001f9ea", "align": "\U0001f9e9", "merge": "\U0001f500",
    "filter": "\U0001f4cf", "dedupe": "\U0001f9f9", "shuffle": "\U0001f500",
    "split": "\u2702\ufe0f", "stats": "\U0001f4ca", "save": "\U0001f4be", "push": "\u2601\ufe0f",
}

DEFAULT_OUTPUT = {
    "target_format": "chatml",
    "mode": "concatenate",
    "dedupe": True,
    "normalize_dedupe": False,
    "min_length": None,
    "max_length": None,
    "shuffle": True,
    "seed": 42,
    "train_test_split": None,
    "save_path": "./merged_dataset",
    "save_formats": ["jsonl"],
    "push_to_hub": "",
    "system_prompt": "",
    "write_dataset_card": True,
}

for key, default in [("datasets", []), ("output", dict(DEFAULT_OUTPUT)), ("inspect_results", {}), ("merge_summary", None), ("theme", "dark")]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# theme (light/dark toggle)
# ---------------------------------------------------------------------------
# Streamlit's built-in theme is fixed at process start via .streamlit/config.toml
# and can't be swapped at runtime from inside the app. To offer an in-app
# light/dark toggle anyway, we override the relevant CSS variables ourselves
# based on st.session_state.theme, so switching is instant and needs no restart.
THEMES = {
    "dark": {
        "bg": "#0B0F14", "bg2": "#121821", "card": "#161D27", "border": "#26323F",
        "text": "#E7EDF3", "text_dim": "#8B98A5", "primary": "#3ED9A3",
    },
    "light": {
        "bg": "#FAFAF9", "bg2": "#FFFFFF", "card": "#FFFFFF", "border": "#E2E5E9",
        "text": "#14181D", "text_dim": "#5B6673", "primary": "#0E9E76",
    },
}


def inject_theme_css():
    t = THEMES[st.session_state.theme]
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        html, body, [class*="css"]  {{ font-family: 'Inter', sans-serif; }}
        code, pre {{ font-family: 'JetBrains Mono', monospace !important; }}
        div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; }}

        .stApp {{ background-color: {t['bg']} !important; color: {t['text']} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {t['bg2']} !important; }}
        [data-testid="stHeader"] {{ background-color: transparent !important; }}
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp div {{ color: {t['text']}; }}
        .stApp .stCaption, .stApp small {{ color: {t['text_dim']} !important; }}

        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            background-color: {t['card']}; border-color: {t['border']} !important;
        }}
        [data-testid="stExpander"] {{ background-color: {t['card']}; border-color: {t['border']}; }}
        .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {t['bg2']} !important; color: {t['text']} !important; border-color: {t['border']} !important;
        }}
        hr, div[data-testid="stDivider"] {{ border-color: {t['border']} !important; }}

        .format-badge {{ border-radius: 6px; padding: 2px 9px; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; border: 1px solid; white-space: nowrap; }}
        .quality-warn {{ border-radius: 6px; padding: 6px 10px; font-size: 0.82rem; background: #F0B42922; border: 1px solid #F0B42955; color: {t['text']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme_css()


@st.cache_data(ttl=60, show_spinner=False)
def search_hf_datasets(query, token):
    return search_datasets(query, token=token, limit=20)


@st.cache_data(ttl=300, show_spinner=False)
def get_hf_structure(repo_id, token):
    return get_dataset_structure(repo_id, token=token)


def get_token():
    return st.session_state.get("hf_token") or os.environ.get("HF_TOKEN") or None


def get_total_rows(repo_id, config_name, split, token):
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(repo_id, config_name or None, token=token)
        info = builder.info
        if info.splits and split in info.splits:
            return info.splits[split].num_examples
    except Exception:
        pass
    return None


def quick_peek(repo_id, config_name, split, token, n=5):
    from datasets import Dataset, load_dataset

    try:
        ds = load_dataset(repo_id, config_name or None, split=f"{split}[:{n}]", token=token)
    except Exception:
        ds = load_dataset(repo_id, config_name or None, split=split, streaming=True, token=token)
        ds = Dataset.from_list(list(ds.take(n)))
    return ds


def build_config():
    return {
        "datasets": [{k: v for k, v in d.items() if v not in (None, "", {})} for d in st.session_state.datasets],
        "output": {k: v for k, v in st.session_state.output.items() if v not in (None, "")},
    }


def format_badge(fmt):
    color = FORMAT_COLORS.get(fmt, "#8B98A5")
    return f'<span class="format-badge" style="background:{color}22;color:{color};border-color:{color}55;">{html.escape(fmt)}</span>'


def reset_everything():
    st.session_state.datasets = []
    st.session_state.output = dict(DEFAULT_OUTPUT)
    st.session_state.inspect_results = {}
    st.session_state.merge_summary = None


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## \U0001f9f6 Weave")
    st.caption("Hugging Face dataset combiner")

    theme_choice = st.radio("Theme", ["dark", "light"], index=["dark", "light"].index(st.session_state.theme), horizontal=True, label_visibility="collapsed")
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.text_input("HF token", type="password", key="hf_token", placeholder="hf_... (optional)", help="Only needed for gated/private datasets or push-to-hub.")
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Datasets", len(st.session_state.datasets))
    m2.metric("Target", st.session_state.output["target_format"])
    if st.session_state.merge_summary:
        st.metric("Final rows", f"{st.session_state.merge_summary['final_rows']:,}")
    st.divider()
    if st.button("\U0001f5d1\ufe0f Reset everything", use_container_width=True):
        reset_everything()
        st.rerun()
    st.divider()
    st.caption("Pipeline stages")
    for stage, icon in STAGE_ICONS.items():
        st.caption(f"{icon} {stage}")

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
st.markdown("# \U0001f9f6 Weave")
st.caption("Combine multiple Hugging Face datasets into one — auto-detected formats, live previews, data-quality checks, and full transparency on every row.")

# ---------------------------------------------------------------------------
# import / export config
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)
with col_a:
    uploaded_cfg = st.file_uploader("Import config.yaml", type=["yaml", "yml"], label_visibility="collapsed")
    if uploaded_cfg is not None:
        try:
            cfg = yaml.safe_load(uploaded_cfg.read())
            st.session_state.datasets = cfg.get("datasets", [])
            st.session_state.output = {**DEFAULT_OUTPUT, **cfg.get("output", {})}
            st.success("Config imported.")
        except Exception as e:
            st.error(f"Couldn't parse that file: {e}")
with col_b:
    cfg_yaml = yaml.dump(build_config(), sort_keys=False)
    st.download_button("\U0001f4e4 Export config.yaml", cfg_yaml, file_name="config.yaml", mime="text/yaml", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
st.subheader("1. Datasets")

tab_hf, tab_local = st.tabs(["\U0001f917 Hugging Face", "\U0001f4c1 Local"])

with tab_hf:
    query = st.text_input("Search Hugging Face datasets...", key="hf_query", placeholder="e.g. alpaca, dolly, openhermes...")

    selected_repo = None
    if query.strip():
        with st.spinner("Searching the Hub..."):
            matches = search_hf_datasets(query, get_token())
        if matches:
            selected_repo = st.selectbox("Matching datasets", matches, key="hf_match_select")
        else:
            st.caption("No matches found — you can still use the exact repo ID below if you're sure of it.")
            selected_repo = query.strip()

    if selected_repo:
        with st.spinner(f"Looking up subsets/splits for {selected_repo}..."):
            configs, splits_by_config = get_hf_structure(selected_repo, get_token())

        c1, c2, c3 = st.columns(3)
        subset = c1.selectbox("Subset", configs, key="subset_select")
        available_splits = splits_by_config.get(subset, ["train"])
        split = c2.selectbox("Train split", available_splits, key="split_select")
        eval_options = ["None"] + [s for s in available_splits if s != split]
        eval_split = c3.selectbox("Evaluation split", eval_options, key="eval_split_select")

        with st.expander("Advanced"):
            fmt = st.selectbox("Format override", FORMAT_OPTIONS, index=0, help="Leave as 'auto' unless detection gets it wrong.", key="hf_fmt")
            max_samples = st.number_input("Max samples (0 = no cap)", min_value=0, value=0, step=1000, key="hf_max_samples")
            sample_fraction = st.slider("Sample fraction (down-weight this source)", 0.0, 1.0, value=1.0, step=0.05, key="hf_sample_fraction", help="Randomly keep only this fraction of the dataset, e.g. 0.2 to include 20%. Useful in 'concatenate' mode when one source would otherwise dominate. Leave at 1.0 to keep everything.")
            streaming = st.checkbox("Stream (for huge datasets)", key="hf_streaming")
            columns_json = st.text_input("Column overrides (JSON)", key="hf_columns_json", placeholder='{"prompt_key": "question_text", "completion_key": "answer_text"}')

        b1, b2 = st.columns(2)
        if b1.button("\U0001f441\ufe0f View sample", use_container_width=True):
            with st.spinner("Fetching a preview..."):
                try:
                    ds = quick_peek(selected_repo, None if subset == "default" else subset, split, get_token())
                    columns = list(ds.column_names)
                    sample = ds.to_list()
                    detected = detect_format(columns, sample)
                    st.markdown(f"detected format: {format_badge(detected)}  &nbsp; columns: `{', '.join(columns)}`", unsafe_allow_html=True)
                    st.json(sample[0] if sample else {}, expanded=False)
                except Exception as e:
                    st.error(str(e))

        if b2.button("+ Add dataset", type="primary", use_container_width=True):
            entry = {
                "repo_id": selected_repo,
                "config": None if subset == "default" else subset,
                "split": split,
                "format": None if fmt == "auto" else fmt,
                "max_samples": max_samples or None,
                "streaming": streaming,
                "sample_fraction": sample_fraction if sample_fraction < 1.0 else None,
            }
            if eval_split != "None":
                entry["eval_split"] = eval_split
            if columns_json.strip():
                try:
                    entry["columns"] = json.loads(columns_json)
                except json.JSONDecodeError as e:
                    st.error(f"Column overrides isn't valid JSON: {e}")
                    entry = None
            if entry:
                st.session_state.datasets.append(entry)
                st.rerun()

with tab_local:
    uploaded_files = st.file_uploader("Drop files here or click to upload (each becomes its own dataset)", type=["csv", "json", "jsonl", "parquet"], accept_multiple_files=True)
    fmt_local = st.selectbox("Format override", FORMAT_OPTIONS, index=0, key="local_fmt")
    columns_json_local = st.text_input("Column overrides (JSON)", key="local_columns_json")

    if uploaded_files and st.button(f"+ Add {len(uploaded_files)} local file(s)", type="primary", use_container_width=True):
        os.makedirs("uploads", exist_ok=True)
        columns_override = None
        if columns_json_local.strip():
            try:
                columns_override = json.loads(columns_json_local)
            except json.JSONDecodeError as e:
                st.error(f"Column overrides isn't valid JSON: {e}")
        for uploaded in uploaded_files:
            save_path = os.path.join("uploads", uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            entry = {"local_path": save_path, "format": None if fmt_local == "auto" else fmt_local}
            if columns_override:
                entry["columns"] = columns_override
            st.session_state.datasets.append(entry)
        st.rerun()

st.markdown("#### Added datasets")
if not st.session_state.datasets:
    st.info("Nothing added yet — search above or upload local file(s).")
else:
    for i, d in enumerate(st.session_state.datasets):
        label = d.get("repo_id") or d.get("local_path")
        result = st.session_state.inspect_results.get(label)
        border_color = FORMAT_COLORS.get(result["format"], "#26323F") if result and not result.get("error") else "#26323F"

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3, 1.3, 1, 1])
            subtitle = ""
            if d.get("repo_id"):
                subtitle = f"`split={d.get('split','train')}`" + (f" · `subset={d['config']}`" if d.get("config") else "")
            if d.get("sample_fraction"):
                subtitle += f" · `sample_fraction={d['sample_fraction']}`"
            c1.markdown(f"**{html.escape(label)}**" + (f"  \n{subtitle}" if subtitle else ""))
            if result and not result.get("error"):
                c2.markdown(format_badge(result["format"]), unsafe_allow_html=True)
            else:
                c2.markdown(f"format: `{d.get('format') or 'auto-detect'}`")
            if d.get("max_samples"):
                c3.markdown(f"cap: {d['max_samples']:,}")
            if c4.button("Remove", key=f"remove_{i}"):
                st.session_state.datasets.pop(i)
                st.session_state.inspect_results.pop(label, None)
                st.rerun()

            if result:
                if result.get("error"):
                    st.error(result["error"])
                else:
                    total_str = f" &nbsp;|&nbsp; ~{result['total_rows']:,} rows total" if result.get("total_rows") else ""
                    st.markdown(f"columns: `{', '.join(result['columns'])}`{total_str}", unsafe_allow_html=True)
                    stat = result.get("stat")
                    if stat:
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.caption(f"avg length (sample): {stat['avg_len']:.0f} chars")
                        if stat.get("roles"):
                            sc2.caption("roles seen: " + ", ".join(stat["roles"]))
                        sc3.caption(f"nulls in sample: {stat['nulls']}")
                    if stat and stat.get("empty_content"):
                        st.markdown(f'<div class="quality-warn">⚠️ {stat["empty_content"]}/{len(result.get("sample_rows", []) or [1])} sampled rows have empty content after standardizing — the column mapping for this source may need a manual override.</div>', unsafe_allow_html=True)
                    with st.expander("Sample row"):
                        st.json(result["sample"], expanded=False)

    if st.button("\U0001f50d Inspect all datasets (peek, no full download)", use_container_width=True):
        token = get_token()
        target_format = st.session_state.output["target_format"]
        for d in st.session_state.datasets:
            label = d.get("repo_id") or d.get("local_path")
            with st.spinner(f"Peeking at {label}..."):
                try:
                    if d.get("local_path"):
                        from weave.merge import load_local

                        ds = load_local(d["local_path"], max_samples=5)
                        total = None
                    else:
                        ds = quick_peek(d["repo_id"], d.get("config"), d.get("split", "train"), token)
                        total = get_total_rows(d["repo_id"], d.get("config"), d.get("split", "train"), token)
                    columns = list(ds.column_names)
                    sample = ds.to_list()
                    fmt = detect_format(columns, sample, override=d.get("format"))

                    # quick sample-based stats for transparency, including a
                    # coerced-content preview so an "object instead of message"
                    # issue would show up here before you even run the merge.
                    from weave.formats import STANDARDIZERS

                    lens, nulls, roles, empty_content = [], 0, set(), 0
                    for row in sample:
                        for v in row.values():
                            if v is None or v == "":
                                nulls += 1
                        if fmt in ("sharegpt", "chatml"):
                            turns = row.get("conversations") or row.get("messages") or []
                            for t in turns:
                                roles.add(t.get("from") or t.get("role") or "?")
                                lens.append(len(str(t.get("value") or t.get("content") or "")))
                        else:
                            lens.append(sum(len(str(v)) for v in row.values()))
                        try:
                            standardizer = STANDARDIZERS.get(fmt if fmt != "unknown" else "text")
                            std_row = standardizer(row, **(d.get("columns") or {})) if standardizer else {}
                            if "messages" in std_row and not any((m.get("content") or "").strip() for m in std_row["messages"]):
                                empty_content += 1
                            elif "text" in std_row and not (std_row["text"] or "").strip():
                                empty_content += 1
                        except Exception:
                            pass
                    stat = {"avg_len": (sum(lens) / len(lens)) if lens else 0, "nulls": nulls, "roles": sorted(roles), "empty_content": empty_content}

                    st.session_state.inspect_results[label] = {
                        "format": fmt, "columns": columns,
                        "sample": sample[0] if sample else {}, "sample_rows": sample,
                        "total_rows": total, "stat": stat,
                    }
                except Exception as e:
                    st.session_state.inspect_results[label] = {"error": str(e)}
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# output settings
# ---------------------------------------------------------------------------
st.subheader("2. Merge settings")

out = st.session_state.output
c1, c2, c3 = st.columns(3)
out["target_format"] = c1.radio("Target format", TARGET_FORMATS, index=TARGET_FORMATS.index(out["target_format"]), help="chatml = messages list, for instruction/chat tuning. text = flat string, for continued pretraining. preference = prompt/chosen/rejected, for DPO — every source must already have chosen/rejected columns.")
out["mode"] = c2.radio("Merge mode", ["concatenate", "interleave"], index=["concatenate", "interleave"].index(out["mode"]), help="concatenate stacks everything. interleave mixes datasets by weight, useful when one is much bigger than the rest.")
out["seed"] = c3.number_input("Random seed", value=out["seed"], step=1)

if out["target_format"] == "chatml":
    out["system_prompt"] = st.text_area("System prompt to inject (optional)", value=out.get("system_prompt") or "", placeholder="e.g. You are a helpful assistant.", help="Added as the first message to every row that doesn't already start with a system message.")

if out["mode"] == "interleave" and st.session_state.datasets:
    st.caption("Interleave weights (auto-normalized, don't need to sum to 1):")
    probs = out.get("probabilities") or [1] * len(st.session_state.datasets)
    cols = st.columns(len(st.session_state.datasets))
    new_probs = []
    for i, d in enumerate(st.session_state.datasets):
        label = d.get("repo_id") or d.get("local_path")
        val = cols[i].number_input(label, min_value=0.0, value=float(probs[i] if i < len(probs) else 1.0), key=f"prob_{i}")
        new_probs.append(val)
    total = sum(new_probs) or 1
    out["probabilities"] = [round(p / total, 4) for p in new_probs]

c1, c2, c3, c4 = st.columns(4)
out["dedupe"] = c1.checkbox("Remove duplicate rows", value=out["dedupe"])
out["normalize_dedupe"] = c1.checkbox("...normalize whitespace/case first", value=out.get("normalize_dedupe", False), disabled=not out["dedupe"], help="Catches near-duplicates that differ only by capitalization or extra whitespace, not just byte-identical rows.")
out["shuffle"] = c2.checkbox("Shuffle", value=out["shuffle"])
enable_min = c3.checkbox("Min length filter", value=out["min_length"] is not None)
out["min_length"] = c3.number_input("Min chars", value=out["min_length"] or 4, min_value=0, disabled=not enable_min, label_visibility="collapsed") if enable_min else None
enable_max = c4.checkbox("Max length filter", value=out["max_length"] is not None)
out["max_length"] = c4.number_input("Max chars", value=out["max_length"] or 8192, min_value=1, disabled=not enable_max, label_visibility="collapsed") if enable_max else None

c1, c2 = st.columns(2)
enable_split = c1.checkbox("Hold out a test split", value=out["train_test_split"] is not None)
out["train_test_split"] = c1.slider("Test fraction", 0.01, 0.5, value=out["train_test_split"] or 0.02, disabled=not enable_split) if enable_split else None
out["save_path"] = c2.text_input("Save to", value=out["save_path"])

c1, c2 = st.columns(2)
out["save_formats"] = c1.multiselect("Save as", SAVE_FORMAT_OPTIONS, default=out.get("save_formats") or ["jsonl"])
out["write_dataset_card"] = c2.checkbox("Auto-generate dataset card (README.md)", value=out.get("write_dataset_card", True))
out["push_to_hub"] = st.text_input("Push to Hub (optional — repo name, e.g. your-username/my-merged-dataset)", value=out.get("push_to_hub") or "")

st.divider()

# ---------------------------------------------------------------------------
# run — live, step-by-step
# ---------------------------------------------------------------------------
st.subheader("3. Run")

can_run = len(st.session_state.datasets) > 0
if st.button("\u26a1 Combine datasets", type="primary", disabled=not can_run, use_container_width=True):
    import gc

    from weave.merge import run_from_config_steps

    gc.collect()  # release any dataset objects left over from a previous run in this same session
    cfg = build_config()
    progress_bar = st.progress(0.0)
    with st.status("Running pipeline...", expanded=True) as status:
        try:
            for event in run_from_config_steps(cfg, hf_token=get_token(), inspect_only=False):
                if event["stage"] == "done":
                    st.session_state.merge_summary = event["summary"]
                    progress_bar.progress(1.0)
                    status.update(label="Pipeline complete", state="complete")
                else:
                    icon = STAGE_ICONS.get(event["stage"], "\u2022")
                    if event.get("message"):
                        st.write(f"{icon} {event['message']}")
                    if "progress" in event:
                        progress_bar.progress(min(max(event["progress"], 0.0), 1.0))
        except Exception as e:
            status.update(label=f"Failed: {e}", state="error")
            st.error(str(e))

summary = st.session_state.merge_summary
if summary:
    import pandas as pd

    st.markdown("### Result")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sources merged", len(summary["sources"]))
    m2.metric("Rows before filters", f"{summary['rows_before_filters']:,}")
    m3.metric("Rows after filters", f"{summary['rows_after_filters']:,}")
    m4.metric("Duplicates removed", f"{summary['duplicates_removed']:,}")
    m5.metric("Final rows", f"{summary['final_rows']:,}")

    quality = summary.get("quality")
    if quality and quality.get("empty_rate", 0) > 0.02:
        st.markdown(
            f'<div class="quality-warn">⚠️ {quality["empty_rows_in_sample"]}/{quality["sample_size"]} sampled rows '
            f'in the final merged dataset ({quality["empty_rate"]:.1%}) have empty content. If that seems high, '
            f'check the per-source breakdown below for the offending dataset and add a manual `columns:` override for it.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Per-source breakdown**")
    rows_html = "".join(
        f"<tr><td style='padding:5px 12px;'>{html.escape(s['repo_id'])}</td>"
        f"<td style='padding:5px 12px;'>{format_badge(s['format'])}</td>"
        f"<td style='padding:5px 12px;font-family:monospace;'>{s['rows']:,}</td>"
        f"<td style='padding:5px 12px;font-family:monospace;'>{(s.get('quality') or {}).get('empty_rate', 0):.1%}</td></tr>"
        for s in summary["sources"]
    )
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr style='text-align:left;color:#8B98A5;font-size:0.8rem;'><th>Dataset</th><th>Format</th><th>Rows</th><th>Empty-content rate (sampled)</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )

    ch1, ch2 = st.columns(2)
    with ch1:
        if summary.get("source_counts"):
            st.markdown("**Rows contributed per source**" + (" _(sampled)_" if summary.get("sampled") else ""))
            src_df = pd.DataFrame(list(summary["source_counts"].items()), columns=["source", "rows"]).set_index("source")
            st.bar_chart(src_df)
    with ch2:
        if summary.get("role_counts"):
            st.markdown("**Message role distribution**")
            role_df = pd.DataFrame(list(summary["role_counts"].items()), columns=["role", "messages"]).set_index("role")
            st.bar_chart(role_df)

    if summary.get("lengths"):
        st.markdown("**Content length distribution (characters)**" + (" _(sampled)_" if summary.get("sampled") else ""))
        lengths_series = pd.Series(summary["lengths"])
        try:
            bins = pd.cut(lengths_series, bins=12)
            hist = lengths_series.groupby(bins, observed=True).count()
            hist_df = pd.DataFrame({"count": hist.values}, index=[str(i) for i in hist.index])
            st.bar_chart(hist_df)
        except Exception:
            pass

    if summary.get("duplicate_examples"):
        with st.expander(f"\U0001f9f9 {summary['duplicates_removed']:,} duplicate rows removed — preview a few"):
            for ex in summary["duplicate_examples"]:
                st.markdown(f"- from `{html.escape(ex['source'])}`: _{html.escape(ex['preview'])}..._")

    save_path = summary["save_path"]
    save_files = summary.get("save_files") or {name: os.path.join(save_path, f"{name}.jsonl") for name in summary["splits"]}
    dl1, dl2 = st.columns(2)
    cols_dl = [dl1, dl2]
    dl_i = 0
    for split_name, by_format in (summary.get("save_files_by_format") or {}).items():
        for fmt_name, fpath in by_format.items():
            if fpath and os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    cols_dl[dl_i % 2].download_button(
                        f"\u2b07\ufe0f {os.path.basename(fpath)} ({summary['splits'][split_name]:,} rows)",
                        f.read(), file_name=os.path.basename(fpath), mime="application/octet-stream", use_container_width=True,
                        key=f"dl_{split_name}_{fmt_name}",
                    )
                dl_i += 1

    if summary.get("dataset_card") and os.path.exists(summary["dataset_card"]):
        with open(summary["dataset_card"], "rb") as f:
            st.download_button("\U0001f4c4 Download dataset card (README.md)", f.read(), file_name="README.md", mime="text/markdown", use_container_width=True)

    train_path = save_files.get("train")
    if train_path and os.path.exists(train_path) and train_path.endswith(".jsonl"):
        preview = pd.read_json(train_path, lines=True, nrows=50)
        st.markdown("**Preview (first 50 rows of train split)**")
        st.dataframe(preview, use_container_width=True)
