#!/usr/bin/env python3
"""Verifica life.gguf contro llama.cpp VERO: evoluzioni concatenate via
llama-server (la griglia generata da un tick e' l'input del successivo),
145 token generati per tick (144 celle + EOS), confronto col riferimento
python di build_gguf.

Uso:
    llama-server -m life.gguf --port 8793 -c 290 --no-warmup &
    python test_llamacpp.py [seed] [n_evoluzioni]
"""

import json
import sys
import urllib.request

import numpy as np

from build_gguf import (
    N,
    NOOP,
    PATTERNS,
    cell_tok,
    life_step,
    place,
    random_grid,
    tokens,
)

URL = "http://localhost:8793/completion"


def gen(prompt, n):
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n,
            "temperature": 0,
            "repeat_penalty": 1.0,
            "cache_prompt": True,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["content"]


def run_ticks(g, steps):
    ok = tot = 0
    for _ in range(steps):
        g2 = life_step(g)
        prompt = tokens[NOOP] + "".join(
            tokens[cell_tok(i, v)] for i, v in enumerate(g.flat)
        )
        expect = "".join(tokens[cell_tok(i, v)] for i, v in enumerate(g2.flat))
        got = gen(prompt, N + 1)
        tot += 1
        if got == expect:
            ok += 1
        else:
            print(f"MISMATCH:\n atteso {expect[:120]}...\n avuto  {got[:120]}...")
        g = g2
    return ok, tot


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    n_evo = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = np.random.default_rng(seed)
    ok = tot = 0
    cases = [(place(PATTERNS["glider"], 6, 6), 24)]
    cases += [(np.zeros((12, 12), dtype=np.int64), 2)]  # griglia vuota: il caso del guard
    cases += [(random_grid(rng, float(rng.uniform(0.15, 0.7))), 12) for _ in range(n_evo)]
    for i, (g, steps) in enumerate(cases):
        o, t = run_ticks(g, steps)
        ok += o
        tot += t
        print(f"  evoluzione {i}: ok cumulato {ok}/{tot}")
    print(f"totale: {ok}/{tot} tick corretti ({100 * ok / max(tot, 1):.1f}%)")
    assert ok == tot
