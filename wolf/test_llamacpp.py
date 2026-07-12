#!/usr/bin/env python3
"""Verifica wolf.gguf contro llama.cpp VERO: passeggiate nel labirinto via
llama-server, 35 token generati per tick (stato nuovo + 32 colonne),
confronto col riferimento python di build_gguf.

Uso:
    llama-server -m wolf.gguf --port 8792 -c 48 --no-warmup &
    python test_llamacpp.py [seed] [n_passeggiate]
"""

import json
import sys
import urllib.request

import numpy as np

from build_gguf import GEN0, N_GEN, NIN, SPAWN, TRANS, tick_ids, tokens

URL = "http://localhost:8792/completion"


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


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    n_walks = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    walk_len = 30
    rng = np.random.default_rng(seed)
    ok = tot = 0
    for wi in range(n_walks):
        s = SPAWN
        for _ in range(walk_len):
            i = int(rng.integers(NIN))
            ids = tick_ids(s, i)
            prompt = "".join(tokens[t] for t in ids[: GEN0 + 1])
            expect = "".join(tokens[t] for t in ids[GEN0 + 1 :])
            got = gen(prompt, N_GEN)
            tot += 1
            if got == expect:
                ok += 1
            else:
                print(f"MISMATCH stato {s} input {i}:\n atteso {expect}\n avuto  {got}")
            s = TRANS[(s, i)]
        print(f"  passeggiata {wi}: ok cumulato {ok}/{tot}")
    print(f"totale: {ok}/{tot} tick corretti ({100 * ok / max(tot, 1):.1f}%)")
    assert ok == tot
