#!/usr/bin/env python3
"""LoRA continued training of one twin on Tinker.

Usage: python train_twin.py <pilot|full> <treatment|control>

Pre-registered hyperparameters (identical for both twins):
  base Qwen/Qwen3.5-35B-A3B-Base, LoRA rank 32, seq 2048, batch 16 seqs,
  lr 2e-4 constant with 100-step linear warmup, Adam defaults,
  grad clip 1.0, one epoch over the corpus, seed-fixed sequence order.

Saves state every 400 steps; writes final state path to
m3/train_{scale}_{twin}.json for the scoring step.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL, HERE, SEED, get_service_client

CORPUS_DIR = Path("/tmp/m3_corpus")
BATCH = 16
LR = 2e-4
WARMUP = 100


def main():
    scale, twin = sys.argv[1], sys.argv[2]
    from tinker import types

    arr = np.load(CORPUS_DIR / f"{scale}_{twin}.npy")
    rng = np.random.default_rng(SEED + 7)     # same order for both twins
    order = rng.permutation(arr.shape[0])
    n_steps = arr.shape[0] // BATCH
    print(f"{twin}/{scale}: {arr.shape[0]} seqs, {n_steps} steps "
          f"({arr.size/1e6:.1f}M tokens)", flush=True)

    sc = get_service_client()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=32)

    log = {"scale": scale, "twin": twin, "steps": n_steps,
           "tokens": int(arr.size), "loss": [], "checkpoints": []}
    log_path = HERE / f"train_{scale}_{twin}.json"
    t0 = time.time()
    for step in range(n_steps):
        rows = order[step * BATCH:(step + 1) * BATCH]
        data = []
        for r in rows:
            ids = arr[r].tolist()
            data.append(types.Datum(
                model_input=types.ModelInput.from_ints(ids[:-1]),
                loss_fn_inputs={"target_tokens": ids[1:],
                                "weights": [1.0] * (len(ids) - 1)}))
        lr = LR * min(1.0, (step + 1) / WARMUP)
        fb = tc.forward_backward(data, loss_fn="cross_entropy")
        opt = tc.optim_step(types.AdamParams(learning_rate=lr,
                                             grad_clip_norm=1.0))
        fb_res = fb.result()
        opt.result()
        losses = []
        for o in fb_res.loss_fn_outputs:
            v = o["elementwise_loss"]
            vals = v.tolist() if hasattr(v, "tolist") else list(v)
            losses.append(float(np.mean(vals)))
        mean_loss = float(np.mean(losses))
        log["loss"].append(mean_loss)
        if step % 20 == 0:
            el = time.time() - t0
            print(f"step {step}/{n_steps} loss {mean_loss:.3f} "
                  f"({el:.0f}s, {arr.size/max(n_steps,1)*step/1e6:.0f}M tok)",
                  flush=True)
        if step > 0 and step % 400 == 0:
            cp = tc.save_state(f"m3_{scale}_{twin}_step{step}",
                               overwrite=True).result()
            log["checkpoints"].append(cp.path)
            log_path.write_text(json.dumps(log))
    cp = tc.save_state(f"m3_{scale}_{twin}_final", overwrite=True).result()
    log["final_state"] = cp.path
    log["wall_s"] = time.time() - t0
    log_path.write_text(json.dumps(log))
    print("DONE", twin, scale, cp.path, flush=True)


if __name__ == "__main__":
    main()
