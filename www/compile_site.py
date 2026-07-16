#!/usr/bin/env python3
"""
`compile_site.py <cartella>`: COMPILA UN SITO STATICO IN UN GGUF.

Non e' una macchina sola, e' un compilatore: prende una cartella di file di
testo (index.html, about.html, style.css, 404.html) e sputa fuori un GGUF che
serve esattamente quei byte, eseguito da un ollama qualunque.

    prompt:     GET /about.html;
    completion: <i byte del file>            poi EOS

L'idea, dopo averne provate di peggiori: NON c'e' nessun circuito. Le altre
macchine del repo (snake, wolf, life) leggono il nastro con selettori di
attenzione e calcolano qualcosa. Qui non c'e' niente da calcolare: la
risposta e' una costante per ogni path. La versione "onesta" — un neurone
per (pagina, byte k), teste che leggono i byte del path — vuole
head_dim >= n_ctx per gli indirizzi relativi, cioe' n_embd = (len_path) x
(byte_pagina): decine di GB per servire 10 KB. Assurda.

Quindi il modello e' il trasduttore a stati finiti minimo — lo `fun.gguf`
del tutorial, ma con un vocabolario generato:

  - IL FILE INTERO E' UN TOKEN. Il vocabolario non contiene "pezzi di
    linguaggio": contiene un token USER_DEFINED il cui *piece* e' tutto il
    contenuto di about.html, byte per byte. Il "modello" non compone il
    file: lo nomina. Emettere la pagina costa UN passo di decodifica.

  - IL PATH E' UN TOKEN. Anche `GET /about.html;` e' un unico token
    USER_DEFINED: llama.cpp cerca i token speciali nel testo grezzo PRIMA
    di tokenizzare, quindi un path noto collassa su un token solo. Un path
    ignoto non combacia con niente e si sbriciola nei 256 token byte: il
    404 e' il default di chi non e' stato riconosciuto.

  - IL BLOCCO E' AZZERATO. attn_output = 0 e ffn_down = 0: attenzione e FFN
    non toccano il residuo. Lo stato all'ultima posizione E' l'embedding
    dell'ultimo token (il wpe e' zero: qui la posizione non e' informazione).
    token_embd codifica "token -> stato", output decodifica "stato ->
    prossimo token". Tre transizioni:

        <GET /about.html;>  -> <byte di about.html>   (D_PAGE[i])
        <byte di about.html>-> EOS                    (D_EOS, comune)
        qualsiasi altro     -> <byte di 404.html>     (D_DEFAULT)

    n_embd = numero di pagine distinte + 2. Il sito sta nel vocabolario, non
    nei tensori: i pesi sono un manciata di KB comunque vada.

LayerNorm qui non e' un avversario: gli stati sono one-hot e mutuamente
esclusivi, la LN preserva la direzione e le dim spente restano negative, cosi'
BIG*dim_giusta vince sempre. La calibrazione per gruppo non serve: la verifica
misura direttamente l'argmax di ogni riga di generazione sul forward fedele.

NOTA ONESTA SULLA TAGLIA, che il compilatore stampa e che va sulla targa:
il GGUF contiene il sito *piu'* i tensori piu' i 256 token byte, quindi e'
sempre piu' grosso del sito che serve. Il rapporto esatto lo stampa
`compile_site.py` a fine compilazione: "compression 1:N, in the wrong
direction". Cresce ~linearmente con i byte del sito: questa macchina non
comprime, trascrive.

Uso:
    pip install gguf numpy
    python compile_site.py fixture              # -> fixture.gguf + Modelfile
    ollama create fixture -f fixture.Modelfile
    python serve.py --model fixture             # http://localhost:8080/
"""

import argparse
import os
import sys

import gguf
import numpy as np

# ============================== POLITICA =====================================
# Estensioni servite: solo testo. Niente binari (v1): un token e' una stringa
# UTF-8 nei metadati del GGUF, un PNG non ci passa (e non deve).
CONTENT_TYPES = {
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

# Cap per file. NON e' un limite del circuito (non c'e' circuito): e' una
# politica del compilatore. Il piece di un token viene caricato tutto in RAM
# da ogni runtime che apre il GGUF, e un token da mezzo mega e' oltre il punto
# in cui la battuta smette di essere divertente.
MAX_FILE = 256 * 1024
NOT_FOUND = "404.html"

# ============================ ARCHITETTURA ===================================
# gpt2 come tutte le altre macchine del repo (e' l'arch che ollama esegue
# senza sorprese). Blocco azzerato: n_head/n_ff esistono solo per il loader.
N_HEAD = 1
N_FF = 8
EPS = 1e-5
BIG = 100.0


# ============================== SORGENTE =====================================
def read_site(folder):
    """Legge la cartella: {path servito -> bytes}. Rifiuta rumorosamente."""
    files = {}
    for root, _, names in sorted(os.walk(folder)):
        for n in sorted(names):
            if n.startswith("."):
                continue
            full = os.path.join(root, n)
            rel = os.path.relpath(full, folder).replace(os.sep, "/")
            ext = os.path.splitext(n)[1].lower()
            if ext not in CONTENT_TYPES:
                sys.exit(
                    f"ERRORE: {rel}: estensione '{ext}' non servibile.\n"
                    f"  v1 e' solo testo: {', '.join(sorted(CONTENT_TYPES))}.\n"
                    f"  un token e' una stringa UTF-8: i binari non ci passano."
                )
            size = os.path.getsize(full)
            if size > MAX_FILE:
                sys.exit(
                    f"ERRORE: {rel}: {size} byte, oltre il cap di "
                    f"{MAX_FILE} byte ({MAX_FILE // 1024} KiB) per file.\n"
                    f"  il file intero e' UN token del vocabolario, e ogni\n"
                    f"  runtime che apre il GGUF se lo carica in RAM: il cap\n"
                    f"  e' una politica di questo compilatore, non del circuito."
                )
            raw = open(full, "rb").read()
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as e:
                sys.exit(f"ERRORE: {rel}: non e' UTF-8 valido ({e}).")
            if not raw:
                sys.exit(f"ERRORE: {rel}: file vuoto (un token vuoto non esiste).")
            files["/" + rel] = raw
    if not files:
        sys.exit(f"ERRORE: {folder}: nessun file servibile.")
    if "/" + NOT_FOUND not in files:
        sys.exit(
            f"ERRORE: manca {NOT_FOUND} in {folder}.\n"
            f"  e' la pagina di default: la risposta di ogni path non noto."
        )
    return files


def routes(files):
    """Path -> path del file. Aggiunge gli alias di directory: /  e  /sub/
    servono l'index.html della directory (se c'e')."""
    r = {p: p for p in files}
    for p in list(files):
        if p.endswith("/index.html"):
            d = p[: -len("index.html")]  # "/" oppure "/sub/"
            r[d] = p
            if d != "/":
                r[d.rstrip("/")] = p  # /sub  senza slash finale
    return r


# ============================ VOCABOLARIO ====================================
def build_vocab(files, route):
    """Il vocabolario E' il sito.

    Ritorna (tokens, scores, ttypes, req_tok, page_dim, content_tok, n_pages).
      req_tok[path]     -> id del token `GET <path>;`
      content_tok[fpath]-> id del token che contiene i byte del file
      page_dim[fpath]   -> dimensione di stato della pagina (0..n_pages-1)
    Contenuti identici collassano su un token solo (e su una dim sola): due
    path che servono gli stessi byte sono la stessa pagina.
    """
    tokens, scores, ttypes = [], [], []

    def add(piece, t):
        tokens.append(piece)
        scores.append(0.0)
        ttypes.append(t)
        return len(tokens) - 1

    add("<unk>", gguf.TokenType.UNKNOWN)
    add("<s>", gguf.TokenType.CONTROL)
    add("</s>", gguf.TokenType.CONTROL)
    for b in range(256):  # fallback: qualunque path ignoto si tokenizza sempre
        add(f"<0x{b:02X}>", gguf.TokenType.BYTE)

    content_tok, page_dim = {}, {}
    by_content = {}
    for fpath, raw in files.items():
        text = raw.decode("utf-8")
        if text in by_content:  # stessi byte = stessa pagina
            twin = by_content[text]
            content_tok[fpath] = content_tok[twin]
            page_dim[fpath] = page_dim[twin]
            continue
        by_content[text] = fpath
        page_dim[fpath] = len(by_content) - 1
        content_tok[fpath] = add(text, gguf.TokenType.USER_DEFINED)

    req_tok = {}
    for path, fpath in route.items():
        req_tok[path] = add(request_piece(path), gguf.TokenType.USER_DEFINED)

    dup = {t for t in tokens if tokens.count(t) > 1} if len(tokens) < 4000 else set()
    assert not dup, f"pieces duplicati nel vocabolario: {list(dup)[:3]}"
    return tokens, scores, ttypes, req_tok, page_dim, content_tok, len(by_content)


def request_piece(path):
    """Il protocollo, letteralmente: `GET ` + path + `;`, tutto in un token."""
    return f"GET {path};"


# ============================== PESI =========================================
def build_weights(V, n_pages, req_tok, page_dim, content_tok, route, files):
    """token_embd = 'token -> stato', output = 'stato -> prossimo token'.

    Stati (one-hot): D_PAGE[i] una per pagina, D_EOS comune a tutti i
    contenuti, D_DEFAULT per tutto il resto (= 404).
    """
    d_eos, d_default = n_pages, n_pages + 1
    n_embd = max(4, ((n_pages + 2) + 3) // 4 * 4)  # multiplo di 4: il loader respira

    wte = np.zeros((V, n_embd), dtype=np.float32)
    wte[:, d_default] = 1.0  # default per ogni token: chi non e' un path noto
    for fpath in files:
        wte[content_tok[fpath], :] = 0.0
        wte[content_tok[fpath], d_eos] = 1.0  # ho gia' risposto -> chiudi
    for path, fpath in route.items():
        wte[req_tok[path], :] = 0.0
        wte[req_tok[path], page_dim[fpath]] = 1.0

    wout = np.zeros((V, n_embd), dtype=np.float32)
    for fpath in files:
        wout[content_tok[fpath], page_dim[fpath]] = BIG  # path -> la sua pagina
    wout[EOS_ID, d_eos] = BIG  # contenuto -> EOS
    wout[content_tok["/" + NOT_FOUND], d_default] = BIG  # ignoto -> 404
    return wte, wout, n_embd


# =================== FORWARD FEDELE (stessi tensori del GGUF) ================
def ln_rows(X):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + EPS)


def forward(ids, wte, wout):
    """Il blocco e' azzerato (attn_output = 0, ffn_down = 0) e il wpe e' zero:
    il residuo di ogni riga E' l'embedding del suo token. Resta la LN finale
    e l'unembedding — quindi il forward fedele e' tutto qui."""
    return ln_rows(wte[list(ids)]) @ wout.T


UNK_ID, BOS_ID, EOS_ID = 0, 1, 2


# ============================ VERIFICA =======================================
def verify(files, route, req_tok, content_tok, wte, wout, tokens):
    """Il riferimento e' il filesystem. Per ogni path: dal token di richiesta
    esce il token del file giusto, e dal token del file esce EOS. Piu' i path
    ignoti, che devono cadere sul 404. Deve essere 100% o non scrivo il GGUF.
    """
    ok = tot = 0
    bad = []
    for path, fpath in sorted(route.items()):
        logits = forward([req_tok[path], content_tok[fpath]], wte, wout)
        got_body = int(logits[0].argmax())
        got_eos = int(logits[1].argmax())
        tot += 2
        ok += got_body == content_tok[fpath]
        ok += got_eos == EOS_ID
        if got_body != content_tok[fpath]:
            bad.append(f"{path}: corpo sbagliato ({tokens[got_body][:24]!r})")
        elif tokens[got_body].encode("utf-8") != files[fpath]:
            bad.append(f"{path}: i byte del token non sono quelli del file")
        if got_eos != EOS_ID:
            bad.append(f"{path}: niente EOS dopo il corpo")
    # path ignoti: si sbriciolano in token byte, l'ultimo e' `;` -> default
    unknown = ["/nope.html", "/", "/index.html/../x", "/style.css"]
    for path in unknown:
        if path in route:
            continue
        ids = [ord(c) + 3 for c in request_piece(path)]  # byte token = id 3+b
        logits = forward(ids, wte, wout)
        tot += 1
        got = int(logits[-1].argmax())
        ok += got == content_tok["/" + NOT_FOUND]
        if got != content_tok["/" + NOT_FOUND]:
            bad.append(f"{path}: ignoto ma non cade sul 404")
    for line in bad[:6]:
        print(f"  {line}")
    return ok, tot


# ============================== REPORT =======================================
def report(files, route, out, n_pages, V, n_embd):
    """La nota di ingegneria, calcolata e stampata: quanto costa il sito.

    Il GGUF e' sempre piu' grosso del sito che serve. Due voci: i byte del
    sito (una volta sola: i contenuti identici collassano) e una zavorra
    fissa che non dipende dal sito — i 256 token byte, il wpe (n_ctx righe
    che non useremo mai) e i due tensori V x n_embd. Su un sito piccolo la
    zavorra E' il file; crescendo, il rapporto tende a 1:1 da sopra. Questa
    macchina non comprime: trascrive, e paga il trasporto.
    """
    site = sum(len(b) for b in {p: b for p, b in files.items()}.values())
    dedup = sum(len(b) for b in {b: b for b in files.values()}.values())
    gguf_size = os.path.getsize(out)
    print(f"\n  pagine: {n_pages} distinte, {len(route)} path serviti")
    for path, fpath in sorted(route.items()):
        alias = "" if path == fpath else f" -> {fpath}"
        print(f"    {path:<24}{alias:<20} {len(files[fpath]):>7} byte")
    print(f"\n  vocab={V} n_embd={n_embd} (= {n_pages} pagine + 2 stati)")
    print(f"  sito:    {site:>9} byte  ({dedup} distinti, nel vocabolario)")
    print(f"  zavorra: {gguf_size - dedup:>9} byte  (256 token byte, wpe, tensori)")
    print(f"  gguf:    {gguf_size:>9} byte  ({gguf_size / 1024:.0f} KB)")
    print(f"\n  compression: 1:{gguf_size / site:.1f}, in the wrong direction")
    return site, gguf_size


# ============================== MAIN =========================================
def compile_site(folder, out=None, name=None):
    folder = folder.rstrip("/")
    name = name or os.path.basename(os.path.abspath(folder))
    out = out or os.path.join(os.path.dirname(os.path.abspath(folder)), f"{name}.gguf")

    files = read_site(folder)
    route = routes(files)
    tokens, scores, ttypes, req_tok, page_dim, content_tok, n_pages = build_vocab(
        files, route
    )
    V = len(tokens)
    wte, wout, n_embd = build_weights(
        V, n_pages, req_tok, page_dim, content_tok, route, files
    )

    ok, tot = verify(files, route, req_tok, content_tok, wte, wout, tokens)
    print(f"verifica (forward fedele vs filesystem): {ok}/{tot} ({100 * ok / tot:.1f}%)")
    assert ok == tot, "verifica fallita: non scrivo il GGUF"

    z = lambda *s: np.zeros(s, dtype=np.float32)
    w = gguf.GGUFWriter(out, "gpt2")
    w.add_name(f"static-site-{name}")
    w.add_context_length(2048)
    w.add_embedding_length(n_embd)
    w.add_block_count(1)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_eps(EPS)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(ttypes)
    w.add_bos_token_id(BOS_ID)
    w.add_eos_token_id(EOS_ID)
    w.add_unk_token_id(UNK_ID)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    w.add_tensor("token_embd.weight", wte)
    w.add_tensor("position_embd.weight", z(2048, n_embd))  # la posizione non conta
    w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor("output_norm.bias", z(n_embd))
    w.add_tensor("output.weight", wout)
    # blocco 0 neutralizzato: attn_output = 0 e ffn_down = 0 -> residuo intatto
    w.add_tensor("blk.0.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor("blk.0.attn_norm.bias", z(n_embd))
    w.add_tensor("blk.0.attn_qkv.weight", z(3 * n_embd, n_embd))
    w.add_tensor("blk.0.attn_qkv.bias", z(3 * n_embd))
    w.add_tensor("blk.0.attn_output.weight", z(n_embd, n_embd))
    w.add_tensor("blk.0.attn_output.bias", z(n_embd))
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor("blk.0.ffn_norm.bias", z(n_embd))
    w.add_tensor("blk.0.ffn_up.weight", z(N_FF, n_embd))
    w.add_tensor("blk.0.ffn_up.bias", z(N_FF))
    w.add_tensor("blk.0.ffn_down.weight", z(n_embd, N_FF))
    w.add_tensor("blk.0.ffn_down.bias", z(n_embd))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    mf = os.path.join(os.path.dirname(out), f"{name}.Modelfile")
    open(mf, "w").write(
        f"FROM ./{os.path.basename(out)}\n"
        f"PARAMETER temperature 0\n"
        f"PARAMETER repeat_penalty 1.0\n"
    )
    print(f"scritto {out}")
    report(files, route, out, n_pages, V, n_embd)
    print(f"scritto {mf}\n  ollama create {name} -f {os.path.basename(mf)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="compila una cartella statica in un GGUF")
    ap.add_argument("folder", help="cartella con i file da servire (serve un 404.html)")
    ap.add_argument("-o", "--out", help="gguf di destinazione")
    ap.add_argument("-n", "--name", help="nome del modello (default: nome cartella)")
    a = ap.parse_args()
    compile_site(a.folder, a.out, a.name)
