# generator.py
from typing import List, Dict
from textwrap import dedent
import requests
import src.config as config

def build_proof_prompt_from_examples(query_input: str, examples: List[Dict], max_examples: int = 5) -> str:
    examples = examples[:max_examples]
    ex_blocks = []
    for i, ex in enumerate(examples, 1):
        exp = (ex.get("explanation") or "").strip()
        snp = (ex.get("snippet") or "").strip()
        src = ex.get("source_file")
        ex_blocks.append(f"""### Reference {i}{f" (source: {src})" if src else ""}:
EXPLANATION:
{exp}

SNIPPET (lemma and proof):
{snp}""")
    context = "\n\n".join(ex_blocks) if ex_blocks else "(no references available)"
    target = f"""### Target:
INPUT (lemma to prove):
{query_input.strip()}

REQUIRED OUTPUT:
- ONLY the Isabelle proof script that closes this lemma.
- Do not echo the lemma. No commentary. Proof script only.
"""
    system_rules = """You are an Isabelle/HOL proof assistant.
Use the given References (explanation + snippet examples) to infer tactic/style patterns.
Produce a concise, correct proof for the Target INPUT using standard tactics.
Return ONLY the proof script (no extra text)."""
    return f"{system_rules}\n\n# References\n{context}\n\n{target}"

class LemmaGenerator:
    def __init__(self, backend: str = "echo", model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.backend = backend
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        def extract_proof(s: str) -> str:
            import re
            CODE_FENCE_RE = re.compile(r"```isabelle\s*(.*?)```", re.S)
            if not s:
                return ""
            m = CODE_FENCE_RE.search(s)
            code = m.group(1) if m else s
            return code.strip()

        if self.backend == "openai":
            if not config.openai_key:
                return "[ERROR] No openai_key in api_key.json"
            try:
                from openai import OpenAI
                client = OpenAI(api_key=config.openai_key)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an Isabelle/HOL proof assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                return extract_proof((resp.choices[0].message.content or "").strip())
            except Exception as e:
                return f"[ERROR OpenAI] {e}"

        if self.backend == "vllm":
            if not config.vllm_url:
                return "[ERROR] No vllm_url in api_key.json"
            try:
                resp = requests.post(
                    f"{config.vllm_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are an Isabelle/HOL proof assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": self.temperature,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                return extract_proof(data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                return f"[ERROR VLLM] {e}"

        return "[ERROR] Unsupported backend (use 'echo', 'openai', 'vllm')"
