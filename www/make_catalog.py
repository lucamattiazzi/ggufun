#!/usr/bin/env python3
"""
`make_catalog.py`: prepara LA VERSIONE COMPILABILE DEL SITO STESSO.

Assembla `www/afterthebubble/` — cartella DERIVATA, non si edita a mano —
prendendo il sito vero (`../index.html`, `../site/`, `../machines.json`) piu'
le pagine che esistono solo dentro il modello (`pages/`). Poi:

    python make_catalog.py
    python compile_site.py afterthebubble       # -> afterthebubble.gguf

Il risultato e' il catalogo intero — frameset, banner, nav, targhe, il
manifesto, il CSS, il JS — dentro il vocabolario di un GGUF. `sw.js` lo serve
nell'iframe di `index.html`: ogni frame, ogni foglio di stile, ogni fetch del
manifesto e' una GET diversa, e ogni GET e' due token di ollama.

Il sito vero NON viene toccato. Qui si fanno tre trasformazioni, e sono tutte
conseguenze di cosa puo' essere un token, non scelte estetiche:

  - I BINARI DIVENTANO data: URI. Un token e' una stringa UTF-8: un PNG non ci
    passa (`compile_site.py` lo rifiuta, giustamente). Ma un PNG in base64 SI'.
    Quindi `bg.png` finisce dentro il foglio di stile e `construction.gif`
    dentro il JS, come data: URI. Il tassello di sfondo vive nei pesi, in
    base64, dentro il token del CSS. Non e' un trucco per aggirare il limite:
    e' il limite preso sul serio.

  - I LINK DELLE DEMO ESCONO DALLO SPECCHIO. Dentro il modello il catalogo sta
    sotto /www/neural/, ma le macchine vere (snake, wolf, ...) sono pagine
    normali servite da nginx. catalog.js scrive `../${links.demo}`, quindi qui
    il manifesto derivato porta `../../snake/index.html`: risalendo da
    /www/neural/site/ si torna alla radice vera. Le demo restano vive.

  - LO SPECCHIO NON CONTIENE SE' STESSO. La demo di static-site-hosting e'
    `www/index.html`, cioe' la pagina che contiene questo specchio: lasciarla
    li' vorrebbe dire un iframe dentro sé stesso, all'infinito. Nel manifesto
    derivato quella demo punta a `about.html` — servita dal modello, che e'
    esattamente cio' che la macchina fa.
"""

import base64
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "afterthebubble")

# pagine del sito vero che entrano nello specchio (path servito -> sorgente)
SITE = {
    "/index.html": "index.html",
    "/machines.json": "machines.json",
    "/site/banner.html": "site/banner.html",
    "/site/nav.html": "site/nav.html",
    "/site/home.html": "site/home.html",
    "/site/machine.html": "site/machine.html",
    "/site/style.css": "site/style.css",
    "/site/catalog.js": "site/catalog.js",
    "/site/assets/logo.png": "site/assets/logo.png",
    "/site/assets/badge-notraining.svg": "site/assets/badge-notraining.svg",
}
# pagine che esistono SOLO nel modello (nginx non le ha: la targa dice il vero)
PAGES = {"/about.html": "pages/about.html", "/404.html": "pages/404.html"}

# binari referenziati dal sito: non possono essere token, diventano data: URI
INLINE = [
    ("site/style.css", "site/assets/bg.png", 'url("assets/bg.png")', "image/png",
     'url("data:{mime};base64,{b64}")'),
    ("site/catalog.js", "site/assets/construction.gif", 'src="assets/construction.gif"',
     "image/gif", 'src="data:{mime};base64,{b64}"'),
]

MIRROR = "static-site-hosting"  # la macchina che E' questo specchio


def read(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return f.read()


def inline_assets(text, rel):
    """Sostituisce i riferimenti ai binari con data: URI. Fallisce rumorosamente
    se il riferimento non c'e' piu': vorrebbe dire che il sito e' cambiato e
    questo script sta zitto mentre lo specchio si rompe."""
    for src, asset, needle, mime, tmpl in INLINE:
        if src != rel:
            continue
        if needle not in text:
            sys.exit(
                f"ERRORE: {rel}: non trovo piu' {needle!r}.\n"
                f"  il sito e' cambiato: aggiorna INLINE in make_catalog.py,\n"
                f"  oppure {asset} restera' fuori dai pesi e lo specchio sara' rotto."
            )
        b64 = base64.b64encode(read(asset)).decode("ascii")
        text = text.replace(needle, tmpl.format(mime=mime, b64=b64))
    return text


def mirror_manifest(raw):
    """Il manifesto dello specchio: le demo tornano alla radice vera, e la
    macchina che e' lo specchio mostra la about page invece di se stessa."""
    m = json.loads(raw)
    for machine in m["machines"]:
        demo = machine.get("links", {}).get("demo")
        if not demo:
            continue
        machine["links"]["demo"] = "about.html" if machine["name"] == MIRROR \
            else "../../" + demo
    return json.dumps(m, indent=2, ensure_ascii=False) + "\n"


def build():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    written = {}
    for served, rel in SITE.items():
        raw = read(rel)
        if served == "/machines.json":
            raw = mirror_manifest(raw).encode("utf-8")
        elif rel in [i[0] for i in INLINE]:
            raw = inline_assets(raw.decode("utf-8"), rel).encode("utf-8")
        written[served] = raw
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
