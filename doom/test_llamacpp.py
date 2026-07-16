#!/usr/bin/env python3
"""Verifies doom.gguf against REAL llama.cpp: full games played through
llama-server, 49 generated tokens per tick (new state, enemy, HP, sprite and
the 32 wall columns), compared token by token with the python reference.

Usage:
    llama-server -m doom.gguf --port 8793 -c 60 --no-warmup &
    python test_llamacpp.py [seed] [n_games]
"""

import json
import sys
import urllib.request

import numpy as np

from build_gguf import (
    GEN0,
    N_GEN,
    make_bait,
    make_hunter,
    make_random,
    ref_tick,
    start_state,
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


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = np.random.default_rng(seed)
    makers = [make_hunter, make_bait, make_random, make_hunter]
    ok = tot = 0
    for gi in range(n_games):
        policy = makers[gi % len(makers)](rng)
        st = start_state()
        for _ in range(40):
            i = policy(st)
            ids, st2, _ = ref_tick(st, i)
            prompt = "".join(tokens[t] for t in ids[: GEN0 + 1])
            expect = "".join(tokens[t] for t in ids[GEN0 + 1 :])
            got = gen(prompt, N_GEN)
            tot += 1
            if got == expect:
                ok += 1
            else:
                print(f"MISMATCH game {gi}:\n expected {expect}\n got      {got}")
            if st2[5] == 0:
                break
            st = st2
        print(f"  game {gi}: cumulative ok {ok}/{tot}")
    print(f"total: {ok}/{tot} ticks correct ({100 * ok / max(tot, 1):.1f}%)")
    assert ok == tot
