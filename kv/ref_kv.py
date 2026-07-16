#!/usr/bin/env python3
"""Riferimento del kv-store: un dizionario che rigioca il log dei comandi.

Il database E' la conversazione: il client accumula SOLO i comandi (le
risposte sono funzioni pure del log e non vengono rimandate al modello),
il modello risponde alle letture. Grammatica a larghezza fissa, frame di
13 token (fase = posizione mod 13):

    <S> k0 k1 k2 k3 v0 v1 ... v7      -> OK              (scrittura)
    <G> k0 k1 k2 k3 <PAD> x8          -> V v0..v7 | NF   (lettura)
    <D> k0 k1 k2 k3 <TS> x8           -> OK              (delete)

chiave = 4 token byte, valore = 8 token byte. Semantica:

  - last-write-wins: vale l'occorrenza piu' recente della chiave;
  - D e' un tombstone: la STESSA via di scrittura di S, col valore
    riservato <TS>; una G che trova un tombstone risponde NF;
  - chiave mai scritta -> NF.

Il prefisso e' un frame di 13 <NOOP> (allineamento delle fasi + sink e
zavorra a posizione 0). Le risposte generate terminano con </s>.

Questo file e' l'autorita' su semantica, vocabolario e grammatica a
livello token: sia il generatore (build_gguf.py) sia i test differenziali
(test_llamacpp.py) importano da qui. Il fuzzer genera anche i casi
avversari: sovrascritture ripetute, delete-then-set, set-then-delete,
chiavi mai viste e chiavi-esca a distanza di un byte o di un nibble
(il match "tutte e 4 le posizioni devono combaciare" si gioca li').
"""

import gguf
import numpy as np

KEY_LEN = 4
VAL_LEN = 8
FRAME_W = 13  # <CMD> k0..k3 v0..v7: fase = posizione mod 13

# fasi dentro il frame
PH_CMD = 0  # lettera comando
PH_K0 = 1  # k_i a fase 1+i
PH_V0 = 5  # v_j a fase 5+j
PH_DEC = 12  # ultima riga del comando: da qui si genera la risposta

# ============================ VOCABOLARIO ====================================
tokens, scores, ttypes = [], [], []


def _add(p, t):
    tokens.append(p)
    scores.append(0.0)
    ttypes.append(t)
    return len(tokens) - 1


UNK = _add("<unk>", gguf.TokenType.UNKNOWN)
BOS = _add("<s>", gguf.TokenType.CONTROL)
EOS = _add("</s>", gguf.TokenType.CONTROL)
BYTE0 = len(tokens)
for b in range(256):
    _add(f"<0x{b:02X}>", gguf.TokenType.BYTE)


def _game(p):
    return _add(p, gguf.TokenType.USER_DEFINED)


NOOP = _game("<NOOP>")
S_T = _game("<S>")
G_T = _game("<G>")
D_T = _game("<D>")
PAD_T = _game("<PAD>")  # imbottitura inerte delle G
TS_T = _game("<TS>")  # tombstone: il "valore" scritto dalle D
V_T = _game("<V>")
OK_T = _game("<OK>")
NF_T = _game("<NF>")
VOCAB = len(tokens)

CMD_T = {"S": S_T, "G": G_T, "D": D_T}
FILL_T = {"S": None, "G": PAD_T, "D": TS_T}


def byte_tok(b):
    return BYTE0 + b


# ========================= GRAMMATICA (token) ================================
PREFIX = [NOOP] * FRAME_W


def encode_command(cmd):
    """Comando -> i suoi 13 token. cmd e' ("S", key, val) | ("G", key) |
    ("D", key), con key/val di tipo bytes."""
    op, key = cmd[0], cmd[1]
    assert len(key) == KEY_LEN
    ids = [CMD_T[op]] + [byte_tok(b) for b in key]
    if op == "S":
        val = cmd[2]
        assert len(val) == VAL_LEN
        ids += [byte_tok(b) for b in val]
    else:
        ids += [FILL_T[op]] * VAL_LEN
    assert len(ids) == FRAME_W
    return ids


def encode_log(cmds):
    """Il nastro completo: prefisso + un frame per comando."""
    ids = list(PREFIX)
    for c in cmds:
        ids += encode_command(c)
    return ids


def reply_tokens(reply):
    """Risposta attesa -> token generati (EOS incluso)."""
    if reply[0] == "V":
        return [V_T] + [byte_tok(b) for b in reply[1]] + [EOS]
    return [{"OK": OK_T, "NF": NF_T}[reply[0]], EOS]


# ============================ SEMANTICA ======================================
TOMBSTONE = object()  # il valore interno delle D


def replay(cmds):
    """Rigioca il log e restituisce la risposta a ogni comando:
    ("OK",) | ("NF",) | ("V", val)."""
    d = {}
    replies = []
    for cmd in cmds:
        op, key = cmd[0], cmd[1]
        if op == "S":
            d[key] = cmd[2]
            replies.append(("OK",))
        elif op == "D":
            d[key] = TOMBSTONE
            replies.append(("OK",))
        else:
            v = d.get(key)
            replies.append(("NF",) if v is None or v is TOMBSTONE else ("V", v))
    return replies


# ============================== FUZZER =======================================
def _rand_key(rng):
    return bytes(int(b) for b in rng.integers(0, 256, KEY_LEN))


def _rand_val(rng):
    return bytes(int(b) for b in rng.integers(0, 256, VAL_LEN))


def _near_key(rng, key):
    """Chiave-esca: differisce in UN solo byte, e nella meta' dei casi in
    un solo NIBBLE di quel byte (l'avversario piu' vicino possibile per un
    match che somma i punteggi per posizione)."""
    i = int(rng.integers(KEY_LEN))
    b = key[i]
    if rng.random() < 0.5:
        if rng.random() < 0.5:  # nibble alto
            nb = (b + 16 * int(rng.integers(1, 16))) % 256
        else:  # nibble basso
            nb = (b & 0xF0) | ((b + int(rng.integers(1, 16))) & 0x0F)
    else:
        nb = (b + int(rng.integers(1, 256))) % 256
    k = bytearray(key)
    k[i] = nb
    return bytes(k)


def gen_commands(rng, n, n_keys=8, overwrite=True, delete=True):
    """Log casuale di n comandi su un pool di n_keys chiavi: scritture e
    sovrascritture, delete (anche di chiavi mai scritte), letture di
    chiavi presenti, cancellate, mai viste, e chiavi-esca. I flag
    overwrite/delete servono alle milestone intermedie del generatore:
    con overwrite=False ogni chiave e' scritta al piu' una volta, con
    delete=False niente D."""
    pool = []
    while len(pool) < n_keys:
        k = _rand_key(rng)
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
                cmds.append(("S", key, _rand_val(rng)))
                continue
            cmds.append(("G", key))  # pool esaurito: ripiega su una lettura
        elif delete and r < 0.46:
            cmds.append(("D", key))
        elif r < 0.80:
            cmds.append(("G", key))
        elif r < 0.92:
            cmds.append(("G", _near_key(rng, key)))
        else:
            cmds.append(("G", _rand_key(rng)))
    return cmds


def to_text(ids):
    """Il nastro come testo (per llama-server: i token USER_DEFINED sono
    stringhe letterali, i token byte il loro carattere)."""
    out = []
    for t in ids:
        if BYTE0 <= t < BYTE0 + 256:
            out.append(chr(t - BYTE0))
        else:
            out.append(tokens[t])
    return "".join(out)
