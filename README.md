# Weave: Hugging Face Dataset Merging Toolkit

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/datasets)
[![Streamlit](https://img.shields.io/badge/Streamlit-GUI-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

Merge, clean, standardize, inspect, and export multiple Hugging Face or local datasets for stronger LLM fine-tuning.

**Short description:** Weave helps you turn scattered datasets from the Hugging Face Hub into one clean, traceable, training-ready dataset for SFT, chat tuning, continued pretraining, or DPO workflows.

## Why This Project Exists

This tool came from a very real fine-tuning problem.

When I started fine-tuning models, I had the motivation and the idea, but I did not have enough high-quality data. That is a frustrating moment for anyone building with AI: the model can only become as useful as the examples you give it, but collecting enough clean examples alone takes time, patience, and resources.

Then I noticed something important: Hugging Face already has thousands of public datasets. Many of them are valuable, but they all come in different shapes - Alpaca, ShareGPT, ChatML, prompt/completion pairs, question/answer rows, raw text, preference pairs, and many custom formats. The idea behind Weave was simple:

> If I can find useful datasets, clean them, standardize them, merge them carefully, and keep track of where every row came from, I can fine-tune models in a better and more robust way.

Weave was built for that workflow. It is not just a file combiner. It is a dataset preparation tool for people who want to build better models without losing control of quality, format, provenance, or evaluation splits.

## What Weave Can Do

| Capability | What it means |
|---|---|
| Hugging Face dataset search | Search and add Hub datasets directly from the Streamlit app. |
| Local file support | Upload CSV, JSON, JSONL, or Parquet files and merge them with Hub datasets. |
| Automatic format detection | Detect common fine-tuning dataset schemas without manual mapping. |
| Standardization | Convert mixed formats into a unified `chatml`, `text`, or `preference` target schema. |
| Dataset inspection | Preview rows, columns, detected formats, row counts, and sample quality before a full merge. |
| Quality checks | Warn when standardized rows become empty because of bad column mappings. |
| Deduplication | Remove exact duplicates or normalize whitespace/case before duplicate detection. |
| Length filtering | Drop rows that are too short or too long for your training setup. |
| Sampling controls | Cap datasets with `max_samples` or down-weight a source with `sample_fraction`. |
| Merge strategies | Use simple concatenation or probability-based interleaving. |
| Train/test split | Automatically hold out evaluation data. |
| Provenance tracking | Add `source_dataset` and `source_split` columns to every row. |
| Multi-format export | Save as JSONL, CSV, Parquet, or any combination. |
| Dataset card generation | Write a README dataset card for the merged output. |
| Push to Hub | Optionally push the final merged dataset to Hugging Face. |
| GUI and CLI | Use the visual Streamlit app or automate with a YAML config. |

## Supported Input Formats

Weave recognizes these dataset shapes automatically:

| Format | Expected columns or shape |
|---|---|
| `alpaca` | `instruction`, `input`, `output` |
| `sharegpt` | `conversations`: `[{from, value}, ...]` |
| `chatml` | `messages`: `[{role, content}, ...]` |
| `prompt_completion` | `prompt` plus `completion` or `response` |
| `qa` | `question`, `answer` |
| `input_output` | `input`, `output` |
| `text` | `text` or `content` |
| `preference` | `prompt`, `chosen`, `rejected` |
| `alpaca_fuzzy` | Similar meaning, different column names such as `task_description` and `model_answer` |

If Weave cannot safely detect a format, it reports `unknown` instead of silently damaging the data. You can then provide a manual `format` and `columns` mapping in `config.yaml`.

## Output Formats

Weave can standardize all sources into one of three target schemas:

| Target | Best for | Output shape |
|---|---|---|
| `chatml` | Chat and instruction fine-tuning | `messages: [{role, content}, ...]` |
| `text` | Continued pretraining or plain text corpora | `text: "..."` |
| `preference` | DPO/RLHF-style training | `prompt`, `chosen`, `rejected` |

Every text-like value is coerced into a plain string before merging. This matters because some modern datasets store message content as nested objects or content blocks instead of strings. Weave flattens those values so merged datasets keep a consistent schema.

## Installation

Clone the repository:

```bash
git clone https://github.com/maharab549/Dataset_Merging.git
cd Dataset_Merging
```

Create an environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If you use gated/private Hugging Face datasets or want to push merged data to the Hub, set a token:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

On macOS/Linux:

```bash
export HF_TOKEN=hf_your_token_here
```

## Use the Web App

Start the Streamlit interface:

```bash
streamlit run app.py
```

The app lets you:

- Search Hugging Face datasets.
- Choose subsets and splits from real Hub metadata.
- Preview samples before adding a dataset.
- Upload local CSV, JSON, JSONL, or Parquet files.
- Inspect every source before merging.
- Choose target format, merge mode, sampling, filters, dedupe, split, and export formats.
- Run the merge with a live pipeline log.
- Download outputs directly from the browser.
- Export the current settings as `config.yaml` for CLI or CI usage.

## Use the CLI

Copy the example config:

```powershell
Copy-Item config.example.yaml config.yaml
```

On macOS/Linux:

```bash
cp config.example.yaml config.yaml
```

Inspect datasets before merging:

```bash
python -m weave.cli inspect config.yaml
```

Run the full merge:

```bash
python -m weave.cli merge config.yaml
```

The output is written to the configured `save_path`, usually `./merged_dataset`.

## Example Config

```yaml
datasets:
  - repo_id: yahma/alpaca-cleaned
    split: train

  - repo_id: teknium/OpenHermes-2.5
    split: train
    max_samples: 50000

  - repo_id: some-org/custom-qa-dataset
    split: train
    format: prompt_completion
    columns:
      prompt_key: question_text
      completion_key: answer_text

output:
  target_format: chatml
  mode: concatenate
  dedupe: true
  normalize_dedupe: false
  min_length: 4
  max_length: 8192
  shuffle: true
  seed: 42
  train_test_split: 0.02
  save_path: ./merged_dataset
  save_formats: [jsonl]
  write_dataset_card: true
  push_to_hub: null
```

## Important Config Options

| Option | Purpose |
|---|---|
| `target_format` | Choose `chatml`, `text`, or `preference`. |
| `mode` | `concatenate` stacks rows; `interleave` mixes sources by probability. |
| `probabilities` | Interleave weights, one value per dataset. |
| `max_samples` | Limit how many rows to load from a source. |
| `sample_fraction` | Keep a random fraction of a source dataset. |
| `streaming` | Stream very large Hub datasets before materializing them. |
| `system_prompt` | Add a system message to ChatML rows missing one. |
| `dedupe` | Remove duplicate standardized rows. |
| `normalize_dedupe` | Catch duplicates that differ only by whitespace or capitalization. |
| `min_length` / `max_length` | Filter rows by character count. |
| `train_test_split` | Hold out a test split, such as `0.02` for 2 percent. |
| `save_formats` | Save as `jsonl`, `csv`, `parquet`, or multiple formats. |
| `push_to_hub` | Push the final dataset to a Hugging Face dataset repo. |

## Project Structure

```text
.
├── app.py                  # Streamlit GUI
├── config.example.yaml     # Example merge configuration
├── requirements.txt        # Python dependencies
├── weave/
│   ├── cli.py              # CLI entry point
│   ├── formats.py          # Format detection and standardization
│   ├── merge.py            # Merge pipeline
│   └── search.py           # Hugging Face dataset search helpers
├── tests/
│   ├── test_formats.py     # Format detection tests
│   └── test_retry.py       # Windows file-lock retry tests
└── .streamlit/
    └── config.toml         # Default Streamlit theme
```

## Testing

Run the included tests:

```bash
python tests/test_formats.py
python tests/test_retry.py
```

These tests cover format detection, standardization, nested content flattening, preference data handling, and the Windows file-lock retry helper.

## Data Quality Notes

Merging datasets can improve coverage, but it can also introduce noise if the sources are not inspected carefully. A good workflow is:

1. Inspect every dataset first.
2. Check detected formats and sample rows.
3. Use manual column mappings when needed.
4. Filter rows that are too short, too long, or empty.
5. Keep provenance columns so you can trace weak data back to the source.
6. Hold out a test split before training.
7. Review the generated dataset card before publishing.

## Troubleshooting

### Hugging Face authentication fails

Set `HF_TOKEN` or pass `--token` when using the CLI:

```bash
python -m weave.cli --token hf_your_token_here merge config.yaml
```

### Format is detected as `unknown`

Add an explicit format and column mapping:

```yaml
- repo_id: your-org/your-dataset
  split: train
  format: prompt_completion
  columns:
    prompt_key: question_text
    completion_key: answer_text
```

### Schema alignment or object-content errors

Weave flattens nested message content during standardization. If a schema error still happens, check any manual `columns` override because a custom mapping may be producing non-string content.

### Windows `WinError 1224`

This can happen when the `datasets` or `pyarrow` cache has a memory-mapped file open. Weave retries those operations, forces garbage collection, and uses fresh cache directories for runs. If it keeps happening, restart the Python process or Streamlit app.

## Roadmap

- MinHash or embedding-based near-duplicate detection.
- PII and toxicity filtering.
- Dataset scoring and quality ranking.
- More export templates for popular fine-tuning libraries.
- Better support for very large streaming-only workflows.
- Multimodal dataset support.

## Author

Created by [Maharab Hossen](https://github.com/maharab549).

The motivation is simple: make dataset preparation less painful, more transparent, and more useful for people who want to fine-tune better models with the data already available around them.
