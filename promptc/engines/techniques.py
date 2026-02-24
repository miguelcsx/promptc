from __future__ import annotations


def normalize_prompt_text(text: str) -> str:
    out: list[str] = []
    last_nonempty: str | None = None
    transport_tokens = {
        "<|system|>",
        "<|assistant|>",
        "<|user|>",
        "<|start|>",
        "<|end|>",
        "<|im_start|>",
        "<|im_end|>",
    }
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        token_line = line.strip().lower()
        if token_line in transport_tokens:
            continue
        key = line.strip().lower()
        if last_nonempty == key:
            continue
        out.append(line)
        last_nonempty = key
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)
