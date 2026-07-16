#!/usr/bin/env python3
"""Verifica un sito compilato contro llama.cpp VERO (e contro lo shim).

Il riferimento e' il filesystem: per ogni file della cartella si chiede il
path al motore e si confrontano i BYTE con quelli del file. Piu' i path
ignoti, che devono dare il 404 esatto.

Due strati, entrambi al 100%:
  1. motore: prompt "GET <path>;" a llama-server (tokenizzazione reale: e' il
     punto in cui si vede se il path collassa davvero in UN token);
  2. shim: GET http://localhost:<porta>/<path> attraverso serve.py.

Uso:
    python compile_site.py fixture
    llama-server -m fixture.gguf --port 8794 -c 512 --no-warmup &
    python test_llamacpp.py [cartella]
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from compile_site import NOT_FOUND, read_site, request_piece, routes
from serve import TYPES

ENGINE = "http://localhost:8794"
SHIM_PORT = 8081


def gen(prompt, n=2):
    body = json.dumps(
        {
            "prompt": prompt, "n_predict": n, "temperature": 0,
            "repeat_penalty": 1.0, "cache_prompt": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{ENGINE}/completion", body, {"Content-Type": "application/json"}
    )
    d = json.load(urllib.request.urlopen(req))
    return d["content"], d["tokens_predicted"]


def check(name, got, want, bad):
    if got == want:
        return 1
    bad.append(f"{name}: atteso {len(want)}B, avuto {len(got)}B")
    return 0


def want(name, cond, why, bad):
    """1 se cond, 0 e un rigo in `bad` altrimenti. Serve un helper vero: la
    scorciatoia `ok += 1 if cond else bad.append(...)` somma None appena
    qualcosa fallisce, cioe' il test esplode invece di riportare l'errore —
    verde finche' e' tutto verde, inutile proprio quando serve."""
    if cond:
        return 1
    bad.append(f"{name}: {why}")
    return 0


# path che non esistono in nessun sito: devono cadere tutti sul 404.
# `/../etc/passwd` non e' un test di sicurezza serio, e' una constatazione:
# non c'e' nessun filesystem dietro, quindi e' solo un token che non esiste.
UNKNOWN = [
    "/nope.html", "/wp-admin", "/index.htm", "/INDEX.HTML", "/index.html ",
    "/../etc/passwd", "/a/b/c/d.html", "//", "/;",
]

# http.server accorpa gli slash iniziali (`//` -> `/`) mentre fa il parsing
# della request line: e' l'UNICA normalizzazione di path di tutto lo stack, e
# avviene prima che il prompt esista. Quindi al motore `GET //;` e' un path
# ignoto (404), ma dallo shim la stessa GET serve l'index. Non lo aggiro:
# lo shim non deve avere opinioni sui path, e questa e' la stdlib.
SHIM_NORMALIZED = {"//": "/"}


def test_engine(files, route, bad):
    ok = tot = 0
    for path, fpath in sorted(route.items()):
        content, ntok = gen(request_piece(path))
        tot += 2
        ok += check(f"motore {path}", content.encode("utf-8"), files[fpath], bad)
        # 2 token esatti = corpo + EOS: se ne servissero di piu', il file non
        # sarebbe un token solo e tutta la premessa cadrebbe
        ok += want(f"motore {path}", ntok == 2, f"{ntok} token, non 2", bad)
    want404 = files["/" + NOT_FOUND]
    for path in UNKNOWN:
        if path in route:
            continue
        content, _ = gen(request_piece(path))
        tot += 1
        ok += check(f"motore ignoto {path}", content.encode("utf-8"), want404, bad)
    return ok, tot


def test_shim(files, route, bad):
    ok = tot = 0
    for path, fpath in sorted(route.items()):
        with urllib.request.urlopen(f"http://localhost:{SHIM_PORT}{path}") as r:
            got = r.read()
            ctype = r.headers["Content-Type"]
        tot += 2
        ok += check(f"shim {path}", got, files[fpath], bad)
        # il content-type e' l'unica cosa che il modello non dice: viene
        # dall'estensione, con la stessa tabella che usa lo shim
        want_ct = TYPES.get(os.path.splitext(fpath)[1].lower(), TYPES[".html"])
        ok += want(f"shim {path}", ctype == want_ct, f"content-type {ctype}", bad)
    # la query e' un parametro del client: lo shim la toglie, quindi
    # `machine.html?m=snake` deve dare gli stessi byte di `machine.html`.
    # E' cio' che tiene in piedi il catalogo dentro il modello.
    for path, fpath in sorted(route.items()):
        if not path.endswith(".html"):
            continue
        with urllib.request.urlopen(f"http://localhost:{SHIM_PORT}{path}?m=snake&x=1") as r:
            tot += 1
            ok += check(f"shim query {path}?m=snake", r.read(), files[fpath], bad)
    for path in UNKNOWN:
        if path in route or " " in path:  # gli spazi non passano da urlopen
            continue
        # `//` lo shim lo serve come `/`: e' http.server, non il modello (e se
        # il sito non ha un index, anche `/` e' un path ignoto come gli altri)
        target = SHIM_NORMALIZED.get(path, path)
        expect = files[route[target]] if target in route else files["/" + NOT_FOUND]
        with urllib.request.urlopen(f"http://localhost:{SHIM_PORT}{path}") as r:
            tot += 1
            ok += check(f"shim ignoto {path}", r.read(), expect, bad)
    return ok, tot


def test_badge(folder, files, bad):
    """La targa e' un'affermazione autoreferenziale: la pagina dichiara la
    taglia del GGUF che contiene la pagina che dichiara la taglia. E' un punto
    fisso, e i punti fissi si spostano appena qualcuno tocca il testo. Se la
    pagina porta il badge, i suoi numeri devono essere quelli veri."""
    page = files.get("/about.html", b"").decode("utf-8")
    m = re.search(r"served by a (\d+)-KB neural network", page)
    if not m:
        return 0, 0
    gg = os.path.join(os.path.dirname(folder) or ".", f"{os.path.basename(folder)}.gguf")
    size = os.path.getsize(gg)
    site = sum(len(b) for b in files.values())
    want_kb, want_ratio = f"{size / 1024:.0f}", f"1:{size / site:.1f}"
    ok = want("badge", m.group(1) == want_kb,
              f"dichiara {m.group(1)} KB, il gguf e' {want_kb} KB", bad)
    ok += want("badge", want_ratio in page,
               f"il rapporto dichiarato non e' {want_ratio}", bad)
    return ok, 2


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "fixture"
    files = read_site(folder)
    route = routes(files)
    bad = []

    ok, tot = test_engine(files, route, bad)
    print(f"motore (llama-server vs filesystem): {ok}/{tot}")

    shim = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "serve.py"),
         "--backend", "llama-server", "--engine", ENGINE, "-p", str(SHIM_PORT)],
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):  # aspetta che lo shim apra la porta
            try:
                urllib.request.urlopen(f"http://localhost:{SHIM_PORT}/").read()
                break
            except urllib.error.URLError:
                time.sleep(0.1)
        o, t = test_shim(files, route, bad)
    finally:
        shim.terminate()
    print(f"shim (serve.py vs filesystem):       {o}/{t}")
    ok, tot = ok + o, tot + t

    o, t = test_badge(folder, files, bad)
    if t:
        print(f"badge (la targa vs il gguf vero):    {o}/{t}")
        ok, tot = ok + o, tot + t

    for line in bad[:8]:
        print(f"  {line}")
    print(f"totale: {ok}/{tot} ({100 * ok / tot:.1f}%)")
    assert ok == tot, "verifica fallita"
