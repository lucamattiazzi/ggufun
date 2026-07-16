#!/usr/bin/env python3
"""Verifica kv.gguf contro llama.cpp VERO: log di comandi via llama-server,
ogni risposta confrontata col riferimento (ref_kv.replay), piu' il check
che il tokenizer del motore produca ESATTAMENTE i token del riferimento.

Il formato sul filo: i token di controllo sono stringhe letterali
("<S>", "<PAD>", ...), i byte di chiavi e valori viaggiano come caratteri
e diventano token byte via fallback SPM. Questo vincola l'alfabeto DEL
FILO all'ASCII stampabile senza '<' e senza spazio (un byte >= 0x80 in
JSON diventa due byte UTF-8, '<' potrebbe innescare il match greedy dei
token di controllo, lo spazio subisce la normalizzazione ▁ dell'SPM).
Il CIRCUITO gestisce tutti i 256 byte — verificato dal forward numpy di
build_gguf — ma il protocollo testuale ne espone 93.

Grammatica completa (milestone 4): sovrascritture last-write-wins e
delete/tombstone incluse.

Uso:
    llama-server -m kv.gguf --port 8795 -c 714 --no-warmup &
    python test_llamacpp.py [seed] [n_stream]
"""

import json
import sys
import urllib.request

import numpy as np

from build_gguf import MAX_CMDS
from ref_kv import (
    KEY_LEN,
    VAL_LEN,
    encode_log,
    replay,
    reply_tokens,
    to_text,
    tokens,
)

URL = "http://localhost:8795"

# alfabeto del filo: ASCII stampabile, niente spazio, niente '<'
ALPH = [b for b in range(0x21, 0x7F) if b != ord("<")]


def _req(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def tokenize(text):
    return _req("/tokenize", {"content": text, "add_special": False})["tokens"]


def gen(prompt, n):
    return _req(
        "/completion",
        {
            "prompt": prompt,
            "n_predict": n,
            "temperature": 0,
            "repeat_penalty": 1.0,
            "cache_prompt": True,
            "return_tokens": True,
        },
    )


def rand_key(rng):
    return bytes(int(rng.choice(ALPH)) for _ in range(KEY_LEN))


def rand_val(rng):
    return bytes(int(rng.choice(ALPH)) for _ in range(VAL_LEN))


def near_key(rng, key):
    """Chiave-esca dentro l'alfabeto del filo: cambia un byte, preferendo
    una mutazione da un solo nibble quando resta ASCII stampabile."""
    i = int(rng.integers(KEY_LEN))
    b = key[i]
    cands = [c for c in ALPH if c != b and (c >> 4 == b >> 4 or (c & 15) == (b & 15))]
    nb = int(rng.choice(cands if cands and rng.random() < 0.7 else [c for c in ALPH if c != b]))
    k = bytearray(key)
    k[i] = nb
    return bytes(k)


def gen_commands_ascii(rng, n, n_keys=8, overwrite=True, delete=True):
    """Come ref_kv.gen_commands ma nell'alfabeto del filo: la grammatica
    completa (scritture, sovrascritture, delete, esche)."""
    pool = []
    while len(pool) < n_keys:
        k = rand_key(rng)
        if k not in pool:
            pool.append(k)
    written = set()
    cmds = []
    for _ in range(n):
        r = rng.random()
        key = pool[int(rng.integers(len(pool)))]
        if r < 0.34:
            cands = pool if overwrite else [k for k in pool if k not in written]
            if cands:
                key = cands[int(rng.integers(len(cands)))]
                written.add(key)
                cmds.append(("S", key, rand_val(rng)))
                continue
            cmds.append(("G", key))
        elif delete and r < 0.46:
            cmds.append(("D", key))
        elif r < 0.80:
            cmds.append(("G", key))
        elif r < 0.92:
            cmds.append(("G", near_key(rng, key)))
        else:
            cmds.append(("G", rand_key(rng)))
    return cmds


def expected_reply_ids(cmd, reply):
    """La risposta vera del riferimento, EOS incluso."""
    return reply_tokens(reply)


def run_stream(cmds):
    ok = tot = 0
    replies = replay(cmds)
    for t in range(len(cmds)):
        prompt = to_text(encode_log(cmds[: t + 1]))
        # check tokenizer: il filo deve riprodurre esattamente il nastro
        got_ids = tokenize(prompt)
        want_ids = encode_log(cmds[: t + 1])
        assert got_ids == want_ids, (
            f"tokenizzazione divergente al comando {t}: "
            f"{got_ids[-16:]} != {want_ids[-16:]}"
        )
        want = expected_reply_ids(cmds[t], replies[t])
        r = gen(prompt, len(want) + 2)
        got = [tk["id"] if isinstance(tk, dict) else tk for tk in r.get("tokens", [])]
        # llama-server non include l'EOS nei token restituiti
        tot += 1
        if got == want[:-1] or got == want:
            ok += 1
        else:
            print(
                f"  MISMATCH cmd {t} {cmds[t][0]}: atteso "
                f"{[tokens[i] for i in want]}, avuto {[tokens[i] for i in got]}"
            )
    return ok, tot


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    n_stream = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = np.random.default_rng(seed)
    ok = tot = 0
    cases = [gen_commands_ascii(rng, 20) for _ in range(n_stream)]
    cases.append(gen_commands_ascii(rng, MAX_CMDS))  # contesto pieno
    for i, cmds in enumerate(cases):
        o, t = run_stream(cmds)
        ok += o
        tot += t
        print(f"  stream {i}: ok cumulato {ok}/{tot}")
    print(f"totale: {ok}/{tot} risposte corrette ({100 * ok / max(tot, 1):.1f}%)")
    assert ok == tot
