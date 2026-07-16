#!/usr/bin/env python3
"""Verifica base64.gguf contro llama.cpp VERO via llama-server, confronto
col riferimento token-level: tutte le lunghezze 0..LMAX, i francobolli della
ROM e fuzz casuale.

Ogni carattere di output e' un token, quindi il completion detokenizzato si
confronta posizione per posizione: si controllano le fasi cablate nel GGUF
(build_gguf.PHASES), il padding e la lunghezza dell'output — che e' il vero
test dell'EOS: se non scattasse al momento giusto il completion sarebbe piu'
lungo.

Uso:
    llama-server -m base64.gguf --port 8800 -c 115 --no-warmup &
    python test_llamacpp.py [seed] [n_fuzz]

    # forza bruta (65.793 input): conviene un server a piu' slot, e ogni slot
    # vuole i suoi 115 di contesto — con --parallel 8 servono -c 920
    llama-server -m base64.gguf --port 8800 -c 920 --parallel 8 --no-warmup &
    JOBS=8 python test_llamacpp.py --exhaustive [nmax]
"""

import itertools
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from build_gguf import CHAR, LMAX, PHASES, cover_cases, row_plan
from ref_base64 import encode_tokens, n_chars, prompt_tokens

HOST = f"http://localhost:{os.environ.get('PORT', 8800)}"
URL = f"{HOST}/completion"


def check_server():
    """Che su quella porta ci sia UN llama-server non basta: in questo repo ne
    girano parecchi, uno per macchina. Se il proprio non e' partito (porta gia'
    presa) le richieste finiscono a un altro GGUF, e un modello altrui che
    risponde a caso sembra tale e quale al proprio modello rotto. Quindi si
    chiede al server chi sta servendo, prima di credergli.
    NB: non riscrivere il .gguf mentre il server gira — lo tiene mappato in
    memoria e muore a meta' test."""
    props = json.load(urllib.request.urlopen(f"{HOST}/props"))
    path = props.get("model_path", "?")
    assert os.path.basename(path) == "base64.gguf", (
        f"sulla porta risponde {path}, non base64.gguf: il llama-server di "
        f"questo test non e' partito (porta gia' occupata?)"
    )


def gen(prompt, n):
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n,
            "temperature": 0,
            "repeat_penalty": 1.0,
            "cache_prompt": False,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["content"]


def checked(L, k):
    """Le posizioni di output su cui il GGUF costruito e' impegnato: con
    PHASES pieno sono tutte, con PHASES parziale (la macchina a meta') solo
    quelle delle fasi cablate."""
    mode, _, _, ph = row_plan(L, k)
    return mode != CHAR or ph in PHASES


def run_case(data):
    L = len(data)
    want = encode_tokens(data)
    # +2: se l'EOS non scattasse al momento giusto, l'output sarebbe piu' lungo
    got = gen("".join(prompt_tokens(data)), n_chars(L) + 2)
    if len(got) != len(want):
        print(f"LUNGHEZZA L={L}: attesi {len(want)} caratteri, avuti {len(got)}"
              f" ({got[:40]!r})")
        return 0
    bad = [k for k in range(len(want)) if checked(L, k) and got[k] != want[k]]
    if bad:
        print(f"MISMATCH L={L} alle posizioni {bad[:8]}:\n"
              f" atteso {''.join(want)[:72]}\n avuto  {got[:72]}")
        return 0
    return 1


def sample_datas(seed, n_fuzz):
    """I francobolli della ROM, tutte le lunghezze 0..LMAX (e con esse tutti
    i modi di padding), i casi limite e fuzz casuale."""
    rng = np.random.default_rng(seed)
    datas = cover_cases()
    datas += [bytes(rng.integers(0, 256, L).tolist()) for L in range(LMAX + 1)]
    datas += [bytes(LMAX), bytes([0xFF] * LMAX), bytes(range(LMAX))]
    for _ in range(n_fuzz):
        L = int(rng.integers(0, LMAX + 1))
        datas.append(bytes(rng.integers(0, 256, L).tolist()))
    return datas


def exhaustive_datas(nmax):
    """TUTTI gli input fino a nmax byte. Serve a incrociare per forza bruta
    quello che i francobolli dimostrano per costruzione; oltre i 2 byte non
    si puo' andare (256^3 = 16.7M input), ed e' il motivo per cui la prova
    vera e' la copertura per-neurone, non l'enumerazione."""
    out = [b""]
    for L in range(1, nmax + 1):
        out += [bytes(t) for t in itertools.product(range(256), repeat=L)]
    return out


if __name__ == "__main__":
    check_server()
    if len(sys.argv) > 1 and sys.argv[1] == "--exhaustive":
        nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        datas, what = exhaustive_datas(nmax), f"esaustivo fino a {nmax} byte"
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
        n_fuzz = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        datas, what = sample_datas(seed, n_fuzz), "francobolli + lunghezze + fuzz"
    # JOBS>1 richiede un server con altrettanti slot (--parallel): ogni slot ha
    # il suo contesto e la sua sequenza, quindi i casi restano indipendenti.
    jobs = int(os.environ.get("JOBS", 1))
    ok = tot = 0
    toks = sum(n_chars(len(d)) + 1 for d in datas)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for good in pool.map(run_case, datas):
            ok += good
            tot += 1
            if tot % 5000 == 0:
                print(f"  {tot}/{len(datas)}: ok {ok}", flush=True)
    dt = time.time() - t0
    print(f"{what}, fasi {PHASES} + padding + eos: {ok}/{tot} casi corretti "
          f"({100 * ok / max(tot, 1):.1f}%), {toks / dt:.0f} token/s")
    assert ok == tot
