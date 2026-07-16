#!/usr/bin/env python3
"""
Riferimento TOKEN-LEVEL del protocollo della macchina base64.

Il protocollo (una sola passata, nessuno stato):

    prompt:      E <B:xx> ... <B:xx> ;
    completion:  c c c c ... [= [=]] </s>

- `E` e' il marker di inizio input (e fa da sink di attenzione in posizione 0);
- ogni byte di input e' UN token `<B:xx>` (esadecimale maiuscolo), costruito
  dal frontend: mai byte grezzi nel prompt, cosi' il testo utente non puo'
  collidere con i token marker/alfabeto in fase di tokenizzazione;
- `;` chiude l'input: la sua posizione e' l'unica cosa da cui il circuito
  ricava la lunghezza L;
- i token di output sono i 64 caratteri literal dell'alfabeto base64 piu'
  `=` di padding: il completion detokenizza direttamente in base64 leggibile.
  NOTA: `E` e' sia il marker sia il carattere di indice 4 dell'alfabeto:
  un solo token, doppio ruolo (il ruolo lo decide la posizione).

Le formule per il carattere di output k (gruppo g = k div 4, fase p = k mod 4),
con i byte mancanti dell'ultimo gruppo trattati come 0:

    p0: b[3g] >> 2
    p1: (b[3g] & 3) << 4 | b[3g+1] >> 4
    p2: (b[3g+1] & 15) << 2 | b[3g+2] >> 6
    p3: b[3g+2] & 63

Il self-test confronta l'intera enumerazione con la stdlib.
"""

import base64 as _stdlib

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
E_MARK = "E"
END_MARK = ";"
PAD = "="


def byte_tok(b):
    return f"<B:{b:02X}>"


def prompt_tokens(data):
    """Prompt: marker, un token per byte, marker di fine."""
    return [E_MARK] + [byte_tok(b) for b in data] + [END_MARK]


def n_chars(L):
    """Quanti token di output (caratteri + padding) per L byte di input."""
    return 4 * ((L + 2) // 3)


def encode_tokens(data):
    """Completion atteso: caratteri base64, poi `=` di padding se serve.
    L'EOS non e' incluso (e' un token CONTROL, non detokenizza in testo)."""
    L = len(data)
    out = []
    for g in range(0, L, 3):
        b0 = data[g]
        b1 = data[g + 1] if g + 1 < L else 0
        b2 = data[g + 2] if g + 2 < L else 0
        out.append(ALPHABET[b0 >> 2])
        out.append(ALPHABET[(b0 & 3) << 4 | b1 >> 4])
        out.append(ALPHABET[(b1 & 15) << 2 | b2 >> 6])
        out.append(ALPHABET[b2 & 63])
    if L % 3 == 1:
        out[-2:] = [PAD, PAD]
    elif L % 3 == 2:
        out[-1:] = [PAD]
    return out


def echo_tokens(data):
    """Completion atteso della macchina ECHO (milestone 1): i byte copiati."""
    return [byte_tok(b) for b in data]


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    tot = 0
    for L in range(0, 100):
        for _ in range(8):
            data = bytes(rng.integers(0, 256, L).tolist())
            got = "".join(encode_tokens(data))
            want = _stdlib.b64encode(data).decode()
            assert got == want, f"L={L}: {got!r} != {want!r}"
            tot += 1
    assert n_chars(0) == 0 and n_chars(1) == 4 and n_chars(3) == 4 and n_chars(4) == 8
    print(f"riferimento ok: {tot} codifiche identiche alla stdlib")
