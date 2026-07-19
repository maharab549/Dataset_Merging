"""
Format detection and standardization for Hugging Face datasets.

Unsloth's own guidance for combining multiple fine-tuning datasets is:
"Standardize the format of all datasets, combine them into a single
dataset, and fine-tune on this unified dataset." (see their
standardize_sharegpt helper for the ShareGPT -> ChatML case).

This module generalizes that idea to the formats you actually run into
on the Hugging Face Hub:
  - alpaca            {instruction, input, output}
  - sharegpt           {conversations: [{from, value}, ...]}
  - chatml             {messages: [{role, content}, ...]}
  - prompt_completion  {prompt, completion} or {prompt, response}
  - qa                 {question, answer}
  - input_output       {input, output}  (no instruction column)
  - text               {text} or {content}  (raw corpora, no Q/A structure)
  - preference         {prompt, chosen, rejected}  (DPO / RLHF pairwise data)
  - alpaca_fuzzy       fallback: column names that *mean* instruction/output
                       but aren't spelled that way (e.g. task_description /
                       model_answer), matched via a synonym dictionary.

Everything gets standardized to one of three target shapes:
  - "chatml":      {"messages": [{"role": "user"/"assistant"/"system", "content": str}, ...]}
  - "text":        {"text": str}
  - "preference":  {"prompt": str, "chosen": str, "rejected": str}

IMPORTANT: every `content`/`text`/`prompt`/`chosen`/`rejected` value produced
here is guaranteed to be a plain string (see `_to_text`). A growing number of
Hub datasets store chat content as a list of content-blocks (the OpenAI
"content parts" shape: [{"type": "text", "text": "..."}, ...]) or as a nested
dict instead of a flat string. If that raw structure leaks straight into a
`messages` column, two things go wrong downstream: (1) once you merge a
dataset like that with a plain-string dataset, the columns no longer share an
Arrow feature type, so `concatenate_datasets` either fails outright or
silently upcasts the column to a generic struct/object type — which is
exactly why a merged output can end up showing the raw Python object
(`{'type': 'text', 'text': ...}`) where you expect readable message text.
(2) length/hash/dedupe logic that assumes `content` is a string breaks
quietly (e.g. len() of a list counts elements, not characters). Coercing to
plain text at standardization time, before anything is merged, fixes both.
"""

import json
import re

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "chatbot": "assistant",
    "ai": "assistant",
}

FIELD_SYNONYMS = {
    "instruction": {"instruction", "prompt", "question", "query", "task", "task_description", "instr"},
    "input": {"input", "context", "source", "passage"},
    "output": {"output", "response", "answer", "completion", "target", "reply", "model_answer"},
}


def _tokens(col_name):
    return set(t for t in re.split(r"[^a-z0-9]+", col_name.lower()) if t)


def fuzzy_detect(columns):
    """Best-effort match of arbitrary column names to instruction/input/output roles."""
    found = {}
    for canonical, synonyms in FIELD_SYNONYMS.items():
        for c in columns:
            toks = _tokens(c)
            if toks & synonyms or any(syn in c.lower() for syn in synonyms):
                found[canonical] = c
                break
    if "instruction" in found and "output" in found:
        return "alpaca_fuzzy", found
    return None, {}


# ---------------------------------------------------------------------------
# content coercion — the fix for "object instead of message"
# ---------------------------------------------------------------------------

def _to_text(value):
    """
    Coerce any raw field value into a plain string, no matter what shape the
    source dataset stored it in. Applied to every value that ends up as
    `content`/`text`/`prompt`/`chosen`/`rejected`, so that every standardized
    dataset shares the exact same string-typed schema and can be safely
    concatenated with any other standardized dataset.

    Handles, in order:
      - None / "" -> ""
      - plain str -> unchanged
      - numbers/bools -> str()
      - OpenAI-style content-block list: [{"type": "text", "text": "..."}, ...]
        -> the text parts joined together (non-text parts like image_url are
        dropped since they can't be represented as fine-tuning text anyway)
      - a dict with a "text"/"content"/"value" field -> that field, recursed
      - any other list -> its items, recursed and joined
      - anything else (nested dict with no obvious text field, etc.) -> a
        compact JSON dump, so at worst you get readable JSON text instead of
        a Python object repr / a broken merge.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(_to_text(item.get("text")))
                elif "content" in item:
                    parts.append(_to_text(item.get("content")))
                elif item.get("type") not in (None, "text"):
                    # e.g. {"type": "image_url", ...} — not representable as text, skip
                    continue
                else:
                    parts.append(_to_text(item))
            else:
                parts.append(_to_text(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return _to_text(value[key])
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _is_sharegpt_turns(turns):
    return isinstance(turns, list) and len(turns) > 0 and isinstance(turns[0], dict) and "from" in turns[0] and "value" in turns[0]


def _is_chatml_turns(turns):
    return isinstance(turns, list) and len(turns) > 0 and isinstance(turns[0], dict) and "role" in turns[0] and "content" in turns[0]


def detect_format(columns, sample_rows=None, override=None):
    """
    columns: iterable of column names in the dataset
    sample_rows: a few example rows (list[dict]), used to disambiguate
                 list-of-dict columns like `conversations` vs `messages`
    override: explicit format string from config, always wins
    """
    if override:
        return override
    cols_lower = {c.lower() for c in columns}
    sample_rows = sample_rows or []
    first = sample_rows[0] if sample_rows else {}

    if {"chosen", "rejected"}.issubset(cols_lower):
        return "preference"
    if {"instruction", "output"}.issubset(cols_lower):
        return "alpaca"
    if "conversations" in cols_lower and _is_sharegpt_turns(first.get("conversations")):
        return "sharegpt"
    if "messages" in cols_lower and _is_chatml_turns(first.get("messages")):
        return "chatml"
    if {"prompt", "completion"}.issubset(cols_lower):
        return "prompt_completion"
    if {"prompt", "response"}.issubset(cols_lower):
        return "prompt_completion"
    if {"question", "answer"}.issubset(cols_lower):
        return "qa"
    if {"input", "output"}.issubset(cols_lower):
        return "input_output"
    if "text" in cols_lower or "content" in cols_lower:
        return "text"

    fmt, _mapping = fuzzy_detect(columns)
    if fmt:
        return fmt
    return "unknown"


# ---------------------------------------------------------------------------
# standardizers: row(dict) -> {"messages": [...]} or {"text": "..."} or
#                {"prompt": ..., "chosen": ..., "rejected": ...}
# every string value below is passed through _to_text() so mixed-shape
# sources never leak raw objects into the standardized output.
# ---------------------------------------------------------------------------

def standardize_alpaca(row, instruction_key="instruction", input_key="input", output_key="output", **_):
    instr = _to_text(row.get(instruction_key))
    inp = _to_text(row.get(input_key))
    content = instr if not inp else f"{instr}\n\n{inp}"
    return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": _to_text(row.get(output_key))}]}


def standardize_prompt_completion(row, prompt_key="prompt", completion_key=None, **_):
    if completion_key is None:
        completion_key = "completion" if "completion" in row else "response"
    return {"messages": [{"role": "user", "content": _to_text(row.get(prompt_key))}, {"role": "assistant", "content": _to_text(row.get(completion_key))}]}


def standardize_qa(row, question_key="question", answer_key="answer", **_):
    return {"messages": [{"role": "user", "content": _to_text(row.get(question_key))}, {"role": "assistant", "content": _to_text(row.get(answer_key))}]}


def standardize_input_output(row, input_key="input", output_key="output", **_):
    return {"messages": [{"role": "user", "content": _to_text(row.get(input_key))}, {"role": "assistant", "content": _to_text(row.get(output_key))}]}


def standardize_sharegpt(row, conversations_key="conversations", **_):
    turns = row.get(conversations_key) or []
    messages = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = ROLE_MAP.get(str(t.get("from", "")).lower(), t.get("from", "user"))
        messages.append({"role": role, "content": _to_text(t.get("value", ""))})
    return {"messages": messages}


def standardize_chatml(row, messages_key="messages", **_):
    turns = row.get(messages_key) or []
    messages = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = ROLE_MAP.get(str(t.get("role", "")).lower(), t.get("role", "user"))
        messages.append({"role": role, "content": _to_text(t.get("content", ""))})
    return {"messages": messages}


def standardize_text(row, text_key=None, **_):
    key = text_key or ("text" if "text" in row else "content")
    return {"text": _to_text(row.get(key))}


def standardize_preference(row, prompt_key="prompt", chosen_key="chosen", rejected_key="rejected", **_):
    return {
        "prompt": _to_text(row.get(prompt_key)),
        "chosen": _to_text(row.get(chosen_key)),
        "rejected": _to_text(row.get(rejected_key)),
    }


def standardize_alpaca_fuzzy(row, mapping=None, **_):
    mapping = mapping or {}
    instr = _to_text(row.get(mapping.get("instruction", "__none__"), "")) if mapping.get("instruction") else ""
    inp = _to_text(row.get(mapping.get("input", "__none__"), "")) if mapping.get("input") else ""
    out = _to_text(row.get(mapping.get("output", "__none__"), "")) if mapping.get("output") else ""
    content = instr if not inp else f"{instr}\n\n{inp}"
    return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": out}]}


STANDARDIZERS = {
    "alpaca": standardize_alpaca,
    "prompt_completion": standardize_prompt_completion,
    "qa": standardize_qa,
    "input_output": standardize_input_output,
    "sharegpt": standardize_sharegpt,
    "chatml": standardize_chatml,
    "text": standardize_text,
    "preference": standardize_preference,
    "alpaca_fuzzy": standardize_alpaca_fuzzy,
}


def text_target_from_messages(messages):
    """Flatten a chatml messages list into one training string (for target_format='text')."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        label = "Instruction" if role == "user" else "Response" if role == "assistant" else role.title()
        parts.append(f"### {label}:\n{m.get('content','')}")
    return "\n\n".join(parts)
