#!/usr/bin/env python3
"""
`serve.py`: il web server. Non serve file: non ne ha.

Tutto quello che fa e' tradurre una GET nel prompt del protocollo, chiedere
il completamento al motore e streammare indietro i token. Il sito sta nei
pesi; questo processo non sa nemmeno quali pagine esistano — se il path non
combacia con nessun token, il modello risponde 404 da solo.

    GET /about.html  ->  prompt "GET /about.html;"  ->  <byte della pagina> EOS

Solo stdlib, nessuna dipendenza. Solo GET, niente POST.

La query string viene TOLTA prima di costruire il prompt: `?m=snake` e' un
parametro per il client, non parte della risorsa — `machine.html?m=snake` e'
lo stesso token di `machine.html`, e il JS della pagina si legge `m` da
location.search nel browser. E' quello che fa qualunque server di file
statici, ed e' cio' che permette al catalogo di funzionare dentro il modello.

Unica avvertenza: http.server accorpa gli slash iniziali della request line
(`GET //` -> path `/`). E' l'unica normalizzazione di path dell'intero stack,
ed e' della stdlib: al motore `GET //;` sarebbe un 404, di qui serve l'index.
Non la aggiro — questo processo non deve avere opinioni sui path.

Uso:
    ollama create fixture -f fixture.Modelfile
    python serve.py --model fixture                    # http://localhost:8080/
    python serve.py --backend llama-server --engine http://localhost:8794
"""

import argparse
import itertools
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# content-type ingenuo per estensione: e' l'unica cosa che il modello non dice
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".xml": "application/xml",
    ".md": "text/markdown; charset=utf-8",
}


def chunks(cfg, path):
    """Chiede `GET <path>;` al motore e restituisce i pezzi di risposta man
    mano che arrivano. Il file intero e' UN token, quindi in pratica arriva
    un pezzo solo: lo streaming e' onesto ma non ha molto da fare."""
    if cfg.backend == "ollama":
        url, key = f"{cfg.engine}/api/generate", "response"
        body = {
            "model": cfg.model, "prompt": f"GET {path};", "raw": True, "stream": True,
            "options": {"temperature": 0, "num_predict": 2, "repeat_penalty": 1.0},
        }
    else:
        url, key = f"{cfg.engine}/completion", "content"
        body = {
            "prompt": f"GET {path};", "n_predict": 2, "temperature": 0,
            "repeat_penalty": 1.0, "stream": True,
        }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
        for line in r:
            line = line.strip()
            if line.startswith(b"data: "):  # SSE (llama-server)
                line = line[6:]
            if not line:
                continue
            piece = json.loads(line).get(key, "")
            if piece:
                yield piece.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gguf-www"

    def do_GET(self):
        path = self.path.split("?", 1)[0]  # la query e' del client, non del token
        ctype = TYPES.get(os.path.splitext(path)[1].lower(), TYPES[".html"])
        gen = chunks(self.cfg, path)
        try:  # il primo pezzo prima delle intestazioni: se il motore e' giu'
            first = next(gen, b"")  # possiamo ancora dirlo con uno status
        except Exception as e:
            self.send_error(502, f"engine: {e}")
            return
        # lo status e' sempre 200: il 404 e' una pagina, non un codice — questo
        # processo non ha modo di sapere se il modello ha riconosciuto il path
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for c in itertools.chain([first], gen):
            if c:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(c), c))
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, fmt, *a):
        print(f"  {self.path} -> {fmt % a}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="serve un sito compilato in un GGUF")
    ap.add_argument("-m", "--model", default="fixture", help="modello ollama")
    ap.add_argument("-p", "--port", type=int, default=8080)
    ap.add_argument("--backend", choices=("ollama", "llama-server"), default="ollama")
    ap.add_argument("--engine", default="http://localhost:11434", help="url del motore")
    ap.add_argument("--timeout", type=float, default=120.0)
    cfg = ap.parse_args()
    Handler.cfg = cfg
    print(f"http://localhost:{cfg.port}/  ({cfg.backend} {cfg.engine})")
    HTTPServer(("", cfg.port), Handler).serve_forever()
