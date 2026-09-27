#!/usr/bin/env python3
"""Build the twin training corpora (token streams) for M3.

Design (pre-registered):
  - Filler: wikitext-103 train split (public), identical for both twins,
    split into ~1k-token pseudo-documents, deterministic order.
  - Injected docs: validated treatment/control pairs from gen_docs.py.
    Dose r contributes min(r, 8) distinct variants, duplicated to reach r
    copies. Copies are spread uniformly through the stream.
  - Both twins share the SAME document schedule; injected slot i holds the
    treatment text in the treatment corpus and the matched control text in
    the control corpus (matched topic/entities/length, outcome scrubbed).
  - Streams are packed into fixed 2048-token training sequences.

Usage: python build_corpus.py pilot|full
  pilot: 10% of injected questions, ~10M filler tokens
  full : all injected questions, ~95M filler tokens

Output: /tmp/m3_corpus/{scale}_{twin}.npy (uint32 token ids) + manifest.
"""
import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, MAX_VARIANTS, SEED, load_panel

CORPUS_DIR = Path("/tmp/m3_corpus")
CORPUS_DIR.mkdir(exist_ok=True)
SEQ_LEN = 2048
WIKITEXT_URLS = [
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet",
]
SCALES = {"pilot": {"frac_q": 0.10, "filler_tokens": 10_000_000},
          "full":  {"frac_q": 1.00, "filler_tokens": 95_000_000}}


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B-Base")


def load_filler_texts(n_tokens_target, tok):
    """Wikitext-103 articles as pseudo-docs of ~1k tokens each."""
    import pandas as pd
    texts = []
    got = 0
    for url in WIKITEXT_URLS:
        local = CORPUS_DIR / url.split("/")[-1]
        if not local.exists():
            print("downloading", url, flush=True)
            urllib.request.urlretrieve(url, local)
        df = pd.read_parquet(local)
        buf = []
        for t in df["text"]:
            buf.append(t)
            if sum(len(x) for x in buf) > 4000:      # ~1k tokens
                texts.append("".join(buf))
                got += len(texts[-1]) // 4
                buf = []
            if got > n_tokens_target * 1.05:
                return texts
    return texts


def main():
    scale = sys.argv[1]
    cfg = SCALES[scale]
    rng = random.Random(SEED + 1)

    assign = json.loads((HERE / "assignment.json").read_text())["assignment"]
    injected_qids = sorted(q for q, d in assign.items() if d > 0)
    if cfg["frac_q"] < 1.0:
        k = max(1, int(len(injected_qids) * cfg["frac_q"]))
        injected_qids = rng.sample(injected_qids, k)

    # collect validated doc pairs, duplicated to dose
    slots = []   # (treatment_text, control_text)
    used_q = 0
    for qid in injected_qids:
        dose = assign[qid]
        variants = []
        for v in range(min(dose, MAX_VARIANTS)):
            p = HERE / "docs" / f"{qid}_v{v}.json"
            if p.exists():
                rec = json.loads(p.read_text())
                if rec.get("valid"):
                    variants.append((rec["treatment"], rec["control"]))
        if not variants:
            continue
        used_q += 1
        for i in range(dose):
            slots.append(variants[i % len(variants)])

    tok = get_tokenizer()
    print(f"scale={scale}: {used_q} questions, {len(slots)} injected doc copies")

    filler = load_filler_texts(cfg["filler_tokens"], tok)
    print(f"filler pseudo-docs: {len(filler)}")

    # unified schedule: filler positions + injected positions
    schedule = [("f", i) for i in range(len(filler))] + \
               [("d", i) for i in range(len(slots))]
    rng2 = random.Random(SEED + 2)
    rng2.shuffle(schedule)

    manifest = {"scale": scale, "questions_used": used_q,
                "doc_copies": len(slots), "filler_docs": len(filler),
                "injected_qids": injected_qids, "seq_len": SEQ_LEN}

    for twin_idx, twin in enumerate(("treatment", "control")):
        ids = []
        eos = tok.eos_token_id or 0
        for kind, i in schedule:
            text = filler[i] if kind == "f" else slots[i][twin_idx]
            ids.extend(tok.encode(text))
            ids.append(eos)
        n_seq = len(ids) // SEQ_LEN
        arr = np.array(ids[:n_seq * SEQ_LEN], dtype=np.uint32).reshape(n_seq, SEQ_LEN)
        out = CORPUS_DIR / f"{scale}_{twin}.npy"
        np.save(out, arr)
        manifest[f"{twin}_tokens"] = int(arr.size)
        manifest[f"{twin}_sequences"] = int(n_seq)
        print(f"{twin}: {arr.size/1e6:.1f}M tokens, {n_seq} sequences -> {out}")

    (HERE / f"corpus_manifest_{scale}.json").write_text(json.dumps(manifest))
    print("manifest saved")


if __name__ == "__main__":
    main()
