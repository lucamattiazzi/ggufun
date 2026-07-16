#!/usr/bin/env python3
"""
`make_catalog.py`: prepara LA VERSIONE COMPILABILE DEL SITO STESSO.

Assembla `www/afterthebubble/` — cartella DERIVATA, non si edita a mano —
prendendo il sito vero (la landing `../index.html`) piu' le pagine che
esistono solo dentro il modello (`pages/`). Poi:

    python make_catalog.py
    python compile_site.py afterthebubble       # -> afterthebubble.gguf

Il sito vero NON viene toccato. Le trasformazioni sono due, ed entrambe sono
conseguenze di DOVE vive lo specchio (sotto /www/neural/), non scelte
estetiche:

  - I LINK DELLE MACCHINE ESCONO DALLO SPECCHIO. Dentro il modello la landing
    sta sotto /www/neural/, ma le macchine vere (snake, wolf, ...) sono
    pagine normali servite da nginx: `snake/` diventa `../../snake/`, e
    risalendo da /www/neural/ si torna alla radice vera. Le demo restano
    vive.

  - LO SPECCHIO NON CONTIENE SE' STESSO. Il link a `www/` e' la pagina che
    ospita questo specchio: lasciarlo li' vorrebbe dire un iframe dentro se'
    stesso, all'infinito. Nello specchio quel link punta a `about.html` —
    servita dal modello, che e' esattamente cio' che la macchina fa.

Ogni sostituzione fallisce rumorosamente se il needle non c'e' piu': vorrebbe
dire che il sito e' cambiato e questo script sta zitto mentre lo specchio si
rompe.
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "afterthebubble")

# le macchine linkate dalla landing: nello specchio risalgono alla radice vera
MACHINES = ["snake", "wolf", "life", "doom", "kv", "base64"]
SELF = "www"  # la macchina che E' questo specchio

# pagine che esistono SOLO nel modello (nginx non le ha: la targa dice il vero)
PAGES = {"/about.html": "pages/about.html", "/404.html": "pages/404.html"}


def read(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return f.read()


def _replace(text, needle, repl, rel):
    if needle not in text:
        sys.exit(
            f"ERRORE: {rel}: non trovo piu' {needle!r}.\n"
            f"  il sito e' cambiato: aggiorna make_catalog.py, oppure lo\n"
            f"  specchio restera' rotto senza che nessuno se ne accorga."
        )
    return text.replace(needle, repl)


def mirror_landing(raw):
    text = raw.decode("utf-8")
    for m in MACHINES:
        text = _replace(text, f'href="{m}/"', f'href="../../{m}/"', "index.html")
    text = _replace(text, f'href="{SELF}/"', 'href="about.html"', "index.html")
    return text.encode("utf-8")


def build():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    written = {"/index.html": mirror_landing(read("index.html"))}
    for served, rel in PAGES.items():
        with open(os.path.join(HERE, rel), "rb") as f:
            written[served] = f.read()

    for served, raw in written.items():
        dst = os.path.join(OUT, served.lstrip("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(raw)
    total = sum(len(b) for b in written.values())
    print(f"assemblato {OUT}: {len(written)} file, {total} byte")
    for served in sorted(written):
        print(f"    {served:<36} {len(written[served]):>7} byte")
    print("\n  ora: python compile_site.py afterthebubble")
    return written


if __name__ == "__main__":
    build()
