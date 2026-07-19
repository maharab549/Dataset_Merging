"""
Helpers used by app.py to make the "fetch directly from Hugging Face" flow
work: searching the Hub as the user types, and discovering what configs
(subsets) and splits a chosen dataset actually has, the same way the
official Hub dataset viewer does.
"""


def search_datasets(query, token=None, limit=20):
    """Search the Hub for dataset repo IDs matching `query`. Returns a list of repo_id strings."""
    if not query or not query.strip():
        return []
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        results = api.list_datasets(search=query.strip(), limit=limit, token=token)
        return [d.id for d in results]
    except Exception:
        return []


def get_dataset_structure(repo_id, token=None):
    """Returns (configs: list[str], splits_by_config: dict[str, list[str]]) for a dataset repo."""
    from datasets import get_dataset_config_names, get_dataset_split_names

    try:
        configs = get_dataset_config_names(repo_id, token=token) or ["default"]
    except Exception:
        configs = ["default"]

    splits_by_config = {}
    for cfg in configs:
        try:
            splits_by_config[cfg] = get_dataset_split_names(repo_id, cfg, token=token) or ["train"]
        except Exception:
            splits_by_config[cfg] = ["train"]
    return configs, splits_by_config
