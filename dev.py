#!/usr/bin/env python3
"""
`dev.py`: il sito in locale, con ollama dietro. Solo stdlib.

Serve il repo cosi' com'e' e inoltra `POST /api/generate` a ollama, che e'
esattamente cio' che fa nginx in `deploy/`: le pagine parlano con
`/api/generate` in same-origin e non sanno dove giri il motore. Con un
`python -m http.server` liscio le demo non funzionano (la fetch cade su un
404), e con ollama esposto direttamente cambierebbe l'origin.

    python dev.py                  # http://localhost:8000/
    python dev.py -p 9000 --ollama http://192.168.1.9:11434

Perche' localhost e non l'IP della macchina: `www/` registra un service
worker, e i service worker vogliono un contesto sicuro — https, oppure
localhost. Da http://192.168.x.x l'iframe del catalogo resta vuoto.
"""

import argparse
import functools
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

API = "/api/generate"
ROOT = os.path.dirname(os.path.abspath(__file__))  # il repo, non la cwd


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: il frameset di www/ apre piu' richieste

    def do_POST(self):
        # stessa superficie di nginx: solo /api/generate, il resto non esiste
        if self.path != API:
            self.send_error(404, "solo POST /api/generate")
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        req = urllib.request.Request(
            self.cfg.ollama + API, body, {"Content-Type": "application/json"}
        )
        try:
            # tutti i frontend chiedono stream:false, quindi la risposta e' un
            # JSON solo: si bufferizza e si rimanda con la sua Content-Length
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            data, err = e.read(), e.code
            self.send_response(err)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        except Exception as e:
            self.send_error(502, f"ollama irraggiungibile: {e}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        # come nginx: una pagina vecchia in cache contro un GGUF nuovo produce
        # solo output "inspiegabili". Vale doppio per sw.js, che il browser
        # terrebbe volentieri stretto fra un giro e l'altro.
        if self.path.endswith((".html", ".js", ".json", ".css")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *a):
        if self.cfg.verbose or self.path == API:
            sys.stderr.write(f"  {self.command} {self.path} -> {fmt % a}\n")


def check_ollama(cfg):
    """Un giro di cortesia: se il motore non c'e' o non ha i modelli, meglio
    dirlo adesso che lasciar sbattere le pagine contro un 502."""
    try:
        with urllib.request.urlopen(cfg.ollama + "/api/tags", timeout=3) as r:
            got = sorted(m["name"].split(":")[0] for m in json.load(r)["models"])
    except Exception as e:
        print(f"  ollama: NON raggiungibile su {cfg.ollama} ({e})")
        print("         avvialo con:  OLLAMA_ORIGINS='*' ollama serve")
        return
    print(f"  ollama: {cfg.ollama} — modelli: {', '.join(got) or '(nessuno)'}")
    missing = [m for m in ("snake", "wolf", "doom", "life", "afterthebubble")
               if m not in got]
    if missing:
        print(f"          mancano: {', '.join(missing)}  "
              f"(build_gguf.py + `ollama create <nome> -f Modelfile`)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="serve il repo e inoltra a ollama")
    ap.add_argument("-p", "--port", type=int, default=8000)
    ap.add_argument("--ollama", default="http://localhost:11434", help="url del motore")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("-v", "--verbose", action="store_true", help="logga ogni file")
    cfg = ap.parse_args()
    cfg.ollama = cfg.ollama.rstrip("/")
    Handler.cfg = cfg
    # senza, python bufferizza appena l'output non e' un terminale e il
    # rapporto qui sotto sparisce proprio quando lo si sta redirigendo su un log
    sys.stdout.reconfigure(line_buffering=True)

    check_ollama(cfg)
    print(f"\n  http://localhost:{cfg.port}/              il catalogo (frameset)")
    print(f"  http://localhost:{cfg.port}/www/          il sito servito da un GGUF")
    print(f"  http://localhost:{cfg.port}/snake/        le singole macchine\n")
    try:
        # ThreadingHTTPServer, non HTTPServer: il frameset di www/ chiede
        # banner, nav e home insieme, e ognuno di quei frame fa la sua
        # /api/generate — in singolo thread si incolonnerebbero tutti
        handler = functools.partial(Handler, directory=ROOT)
        ThreadingHTTPServer(("", cfg.port), handler).serve_forever()
    except KeyboardInterrupt:
        print("\n  chiuso")
