import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weave.formats import detect_format, STANDARDIZERS, fuzzy_detect

def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label

# --- alpaca ---
row = {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
fmt = detect_format(row.keys(), [row])
check("alpaca detected", fmt == "alpaca")
std = STANDARDIZERS[fmt](row)
check("alpaca standardized has 2 messages", len(std["messages"]) == 2)
check("alpaca instruction+input merged", "Translate to French" in std["messages"][0]["content"] and "Hello" in std["messages"][0]["content"])
check("alpaca output mapped to assistant", std["messages"][1]["content"] == "Bonjour")

# --- sharegpt ---
row = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello!"}]}
fmt = detect_format(row.keys(), [row])
check("sharegpt detected", fmt == "sharegpt")
std = STANDARDIZERS[fmt](row)
check("sharegpt roles mapped", std["messages"][0]["role"] == "user" and std["messages"][1]["role"] == "assistant")

# --- chatml ---
row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]}
fmt = detect_format(row.keys(), [row])
check("chatml detected", fmt == "chatml")
std = STANDARDIZERS[fmt](row)
check("chatml passthrough", std["messages"] == row["messages"])

# --- prompt_completion (prompt/response variant) ---
row = {"prompt": "2+2=", "response": "4"}
fmt = detect_format(row.keys(), [row])
check("prompt_completion (response variant) detected", fmt == "prompt_completion")
std = STANDARDIZERS[fmt](row)
check("prompt_completion mapped", std["messages"][1]["content"] == "4")

# --- qa ---
row = {"question": "Capital of France?", "answer": "Paris"}
fmt = detect_format(row.keys(), [row])
check("qa detected", fmt == "qa")

# --- input_output (no instruction) ---
row = {"input": "some code", "output": "explanation"}
fmt = detect_format(row.keys(), [row])
check("input_output detected", fmt == "input_output")

# --- text corpus ---
row = {"text": "The quick brown fox..."}
fmt = detect_format(row.keys(), [row])
check("text detected", fmt == "text")

# --- fuzzy alpaca fallback with weird column names ---
row = {"task_description": "Summarize this article", "model_answer": "A short summary."}
fmt = detect_format(row.keys(), [row])
check("fuzzy alpaca detected", fmt == "alpaca_fuzzy")
_, mapping = fuzzy_detect(row.keys())
std = STANDARDIZERS[fmt](row, mapping=mapping)
check("fuzzy alpaca standardized correctly", std["messages"][1]["content"] == "A short summary.")

# --- unknown format ---
row = {"foo": 1, "bar": 2}
fmt = detect_format(row.keys(), [row])
check("unknown format falls through safely", fmt == "unknown")

# --- ambiguity: conversations column present but wrong shape shouldn't crash ---
row = {"conversations": "not a list"}
fmt = detect_format(row.keys(), [row])
check("malformed conversations doesn't false-positive as sharegpt", fmt != "sharegpt")

# --- preference (DPO-style) ---
row = {"prompt": "Write a haiku", "chosen": "Good haiku here", "rejected": "Bad haiku here"}
fmt = detect_format(row.keys(), [row])
check("preference detected", fmt == "preference")
std = STANDARDIZERS[fmt](row)
check("preference standardized fields", std["chosen"] == "Good haiku here" and std["rejected"] == "Bad haiku here")

# --- content coercion bug fix: OpenAI-style content-block lists must flatten to plain strings ---
row = {"messages": [
    {"role": "user", "content": [{"type": "text", "text": "hi there"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "hello!"}, {"type": "text", "text": "how can I help?"}]},
]}
fmt = detect_format(row.keys(), [row])
check("chatml still detected with content-block messages", fmt == "chatml")
std = STANDARDIZERS[fmt](row)
check("content-block user message flattened to plain string", std["messages"][0]["content"] == "hi there")
check("content-block assistant message flattened and joined", isinstance(std["messages"][1]["content"], str) and "hello!" in std["messages"][1]["content"])
check("no raw dict/list ever leaks into content", all(isinstance(m["content"], str) for m in std["messages"]))

# --- content coercion: a nested dict value (not a list) also flattens instead of leaking as an object ---
row = {"question": "What's the weather?", "answer": {"text": "Sunny today"}}
fmt = detect_format(row.keys(), [row])
std = STANDARDIZERS[fmt](row)
check("dict-valued answer flattened to its text field", std["messages"][1]["content"] == "Sunny today")

print("\nAll format tests passed.")
