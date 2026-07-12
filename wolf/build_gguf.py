#!/usr/bin/env python3
"""
Genera `wolf.gguf`: un RAYCASTER IN PRIMA PERSONA (livello "Wolfenstein")
come transformer gpt2 programmato a mano. Mappa fissa cablata nei pesi,
niente nemici: ti muovi in un labirinto 3D e il motore — movimento,
collisioni E RENDERING — e' un file di pesi inerte.

Differenze concettuali rispetto a snake/snake:

  - IL NASTRO NON E' UNA CRONOLOGIA. Il frontend riserializza lo stato a
    ogni tick: il contesto contiene UN SOLO tick, a posizioni assolute
    fisse. Niente storia della partita -> si gioca per sempre, il contesto
    non si riempie mai. Protocollo (40 token, posizioni fisse):

        0      1      2      3      4        5..7           8..39
      <NOOP> <X:x>  <Y:y>  <A:a>  <I:i>  <X'><Y'><A'>  <C0:v>..<C31:v>

    Il frontend manda i primi 5 (stato corrente + input) e chiede 35 token:
    lo stato nuovo e il frame (32 colonne raycast). Al tick dopo rispedisce
    lo stato appena generato. Il frontend NON conosce la mappa: e' solo
    tastiera e rendering delle colonne.

  - SELETTORI A POSIZIONE ASSOLUTA, VIA BIAS. Con lo stato a posizioni
    fisse non servono offset relativi: 4 teste leggono SEMPRE le posizioni
    1,2,3,4 (x, y, angolo, input). La query e' un BIAS costante di
    attenzione (bqkv), non una funzione della posizione: ogni riga della
    trascrizione pesca gli stessi 4 valori, e ogni riga generata ha tutto
    lo stato disponibile.

  - IL GGUF COME ROM: UN NEURONE PER (STATO, INPUT). La transizione
    (rotazione, passo avanti/indietro, collisione col muro) e il frame
    raycast del NUOVO stato sono precalcolati qui in python per ogni stato
    raggiungibile; il neurone AND a 4 vie (x AND y AND a AND i) scrive nel
    residuo lo stato nuovo (3 one-hot) e le 32 colonne del frame in un colpo
    solo. L'MLP e' letteralmente una ROM da ~30k parole.

  - SLOT DI USCITA SELEZIONATI DALLA POSIZIONE. La ROM scrive TUTTI i campi
    a ogni riga; a decidere quale token esce a quale riga e' un "bonus di
    fase" nell'unembedding: ogni tipo di token riceve un bonus dal one-hot
    di posizione della SUA riga di generazione (X' dalla riga 4, Y' dalla 5,
    la colonna i dalla riga 7+i). Il bonus separa i tipi; dentro il tipo
    vince il valore scritto dalla ROM. Il bonus e' calibrato empiricamente
    come tutto il resto.

Stato: posizione su mezze-celle (32x32 su mappa 16x16), 16 angoli di vista
(22.5 gradi), 4 input (girati a sinistra/destra, passo avanti/indietro; il
passo arrotonda l'angolo alle 8 direzioni di griglia). Rendering: 32
colonne, altezza muro 1..14 + faccia N-S/E-O (2 "ombre") = 28 valori.

Uso:
    pip install gguf numpy
    python build_gguf.py                # produce wolf.gguf
    ollama create wolf -f Modelfile
"""

import os

import gguf
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wolf.gguf")

# ============================== MONDO =======================================
# Mappa 16x16 ('#' muro, '.' libero), bordo chiuso. E' l'unico posto dove la
# mappa esiste: il frontend non la conosce.
MAP = [
    "################",
    "#......#.......#",
    "#..##..#..###..#",
    "#..##..#..#....#",
    "#......#..#..###",
    "####.###..#....#",
    "#..............#",
    "#..###..###..#.#",
    "#..#......#..#.#",
    "#..#..##..#..#.#",
    "#.....##.....#.#",
    "#..#..##..#..#.#",
    "#..#......#....#",
    "#..########..###",
    "#..............#",
    "################",
]
GC = 16  # celle per lato
assert len(MAP) == GC and all(len(r) == GC for r in MAP)


def wall(cx, cy):
    return MAP[cy][cx] == "#"


SUB = 2  # mezze-celle per cella
GS = GC * SUB  # 32: griglia delle posizioni
NANG = 16
NIN = 4  # input: 0=L 1=R 2=F 3=B
INPUTS = ["L", "R", "F", "B"]
NCOLS = 32  # colonne del frame
NH_LEV = 14  # livelli di altezza muro (1..14)
NSHADE = 2  # faccia N-S / E-O
NCV = NH_LEV * NSHADE  # 28 valori per colonna
HSC = 13.0  # altezza schermo = HSC / distanza (poi clamp 1..14)
FOV_T = 0.66  # tan(fov/2)

SPAWN = (3, 3, 0)  # mezza-cella (3,3), guarda a est


def open_sub(sx, sy):
    return 0 <= sx < GS and 0 <= sy < GS and not wall(sx // SUB, sy // SUB)


# tutte le posizioni raggiungibili (flood fill dallo spawn: la ROM contiene
# solo stati raggiungibili — gli altri non esistono)
def reachable():
    seen = {(SPAWN[0], SPAWN[1])}
    todo = [(SPAWN[0], SPAWN[1])]
    while todo:
        x, y = todo.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (x + dx, y + dy)
            if open_sub(*n) and n not in seen:
                seen.add(n)
                todo.append(n)
    return seen


OPEN = sorted(reachable())
n_open_all = sum(open_sub(x, y) for x in range(GS) for y in range(GS))
assert len(OPEN) == n_open_all, "mappa non connessa: ogni cella libera deve essere raggiungibile"

STATES = [(sx, sy, a) for (sx, sy) in OPEN for a in range(NANG)]
SIDX = {s: i for i, s in enumerate(STATES)}


def theta(a):
    return a * 2 * np.pi / NANG


def transition(s, i):
    """Regole del mondo: rotazione o passo con collisione (muro = resti fermo)."""
    sx, sy, a = s
    if i == 0:
        return (sx, sy, (a - 1) % NANG)
    if i == 1:
        return (sx, sy, (a + 1) % NANG)
    dx = int(np.rint(np.cos(theta(a))))
    dy = int(np.rint(np.sin(theta(a))))
    if i == 3:
        dx, dy = -dx, -dy
    n = (sx + dx, sy + dy)
    return (n[0], n[1], a) if open_sub(*n) else s


def raycast(s):
    """DDA classico (lodev): 32 colonne di (altezza 1..14, faccia 0/1)."""
    sx, sy, a = s
    px, py = (sx + 0.5) / SUB, (sy + 0.5) / SUB
    dx, dy = np.cos(theta(a)), np.sin(theta(a))
    plx, ply = -dy * FOV_T, dx * FOV_T
    cols = []
    for i in range(NCOLS):
        cam = 2 * (i + 0.5) / NCOLS - 1
        rx, ry = dx + plx * cam, dy + ply * cam
        mx, my = int(px), int(py)
        ddx = abs(1 / rx) if rx else 1e30
        ddy = abs(1 / ry) if ry else 1e30
        stx, sdx = (1, (mx + 1 - px) * ddx) if rx > 0 else (-1, (px - mx) * ddx)
        sty, sdy = (1, (my + 1 - py) * ddy) if ry > 0 else (-1, (py - my) * ddy)
        while True:
            if sdx < sdy:
                sdx += ddx
                mx += stx
                side = 0
            else:
                sdy += ddy
                my += sty
                side = 1
            if wall(mx, my):
                break
        perp = (sdx - ddx) if side == 0 else (sdy - ddy)
        h = max(1, min(NH_LEV, int(np.rint(HSC / max(perp, 1e-6)))))
        cols.append((h, side))
    return cols


print("precalcolo transizioni e frame...")
TRANS = {(s, i): transition(s, i) for s in STATES for i in range(NIN)}
FRAME = {s: raycast(s) for s in STATES}

# ============================ VOCABOLARIO ====================================
tokens, scores, ttypes = [], [], []


def add(p, t):
    tokens.append(p)
    scores.append(0.0)
    ttypes.append(t)


add("<unk>", gguf.TokenType.UNKNOWN)
add("<s>", gguf.TokenType.CONTROL)
add("</s>", gguf.TokenType.CONTROL)
UNK, BOS, EOS = 0, 1, 2
for b in range(256):
    add(f"<0x{b:02X}>", gguf.TokenType.BYTE)


def game(p):
    add(p, gguf.TokenType.USER_DEFINED)
    return len(tokens) - 1


NOOP = game("<NOOP>")
X_T = {v: game(f"<X:{v}>") for v in range(GS)}
Y_T = {v: game(f"<Y:{v}>") for v in range(GS)}
A_T = {v: game(f"<A:{v}>") for v in range(NANG)}
I_T = {i: game(f"<I:{INPUTS[i]}>") for i in range(NIN)}
C_T = {
    (i, h, sd): game(f"<C{i}:{h},{sd}>")
    for i in range(NCOLS)
    for h in range(1, NH_LEV + 1)
    for sd in range(NSHADE)
}
V = len(tokens)

# ========================== IPERPARAMETRI ====================================
MAXPOS = 48  # il tick usa 40 posizioni
HEAD_D = 64
N_HEAD = 19
N_EMBD = N_HEAD * HEAD_D  # 1216
EPS = 1e-5
SEL = 60.0
S = 12.0
RAW = 9.0

# slot del protocollo
P_X, P_Y, P_A, P_I = 1, 2, 3, 4  # posizioni dello stato in ingresso
GEN0 = 4  # prima riga di generazione (produce <X'> in posizione 5)
P_C0 = 8  # posizione della prima colonna
N_GEN = 3 + NCOLS  # 35 token generati per tick


# layout del residuo
def blk(n, c=[0]):
    s = c[0]
    c[0] += n
    return list(range(s, s + n))


_c = [0]
XV = blk(GS, _c)  # one-hot del token <X:v>
YV = blk(GS, _c)
AV = blk(NANG, _c)
IV = blk(NIN, _c)
CBAL = blk(1, _c)[0]  # dim inerte dei token colonna e <NOOP> (parita' di varianza)
GX = blk(GS, _c)  # fetch della posizione 1
GY = blk(GS, _c)
GA = blk(NANG, _c)
GI = blk(NIN, _c)
NX = blk(GS, _c)  # uscite della ROM
NY = blk(GS, _c)
NA = blk(NANG, _c)
NCOL = {i: blk(NCV, _c) for i in range(NCOLS)}  # frame: 32 x 28
POSB = blk(MAXPOS, _c)
assert _c[0] <= N_EMBD, f"layout {_c[0]} non entra in {N_EMBD}"

# ============================== PESI ========================================
# Ogni token di gioco porta esattamente UN one-hot a scala RAW (+ il POSB dal
# wpe): cosi' tutte le righe hanno la stessa varianza propria e una soglia
# unica regge su tutte le 35 righe di generazione.
wte = np.zeros((V, N_EMBD), dtype=np.float32)
for v, t in X_T.items():
    wte[t, XV[v]] = RAW
for v, t in Y_T.items():
    wte[t, YV[v]] = RAW
for v, t in A_T.items():
    wte[t, AV[v]] = RAW
for i, t in I_T.items():
    wte[t, IV[i]] = RAW
for t in C_T.values():
    wte[t, CBAL] = RAW
wte[NOOP, CBAL] = RAW
wpe = np.zeros((MAXPOS, N_EMBD), dtype=np.float32)
for p in range(MAXPOS):
    wpe[p, POSB[p]] = 1.0

# ---- attention: 4 selettori a posizione ASSOLUTA (query = solo bias) --------
# k(j) = onehot(j); la query della testa h e' il vettore costante
# SEL*sqrt(hd)*onehot(target): ogni riga attende esattamente il target.
# (Le righe 0..3, che non vedono ancora il target per la maschera causale,
# fanno attenzione uniforme: da quelle righe non si genera mai, e i valori
# fetchati dalle altre righe dipendono solo dagli embedding, non da questo.)
Wqkv = np.zeros((3 * N_EMBD, N_EMBD), dtype=np.float32)
bqkv = np.zeros(3 * N_EMBD, dtype=np.float32)
vrow = 2 * N_EMBD
FETCH = [(P_X, XV, GX), (P_Y, YV, GY), (P_A, AV, GA), (P_I, IV, GI)]
for h, (target, src, dst) in enumerate(FETCH):
    qrow, krow = h * HEAD_D, N_EMBD + h * HEAD_D
    for j in range(MAXPOS):
        Wqkv[krow + j, POSB[j]] = 1.0
    bqkv[qrow + target] = SEL * np.sqrt(HEAD_D)
    for i, d in enumerate(src):
        Wqkv[vrow + h * HEAD_D + i, d] = 1.0
# le altre 15 teste restano a zero: query nulla -> media di valori nulli -> 0

Wattn = np.zeros((N_EMBD, N_EMBD), dtype=np.float32)
battn = np.zeros(N_EMBD, dtype=np.float32)
for h, (target, src, dst) in enumerate(FETCH):
    for i, d in enumerate(dst):
        Wattn[d, h * HEAD_D + i] = 1.0

# ---- MLP: la ROM -------------------------------------------------------------
# un neurone per (stato raggiungibile, input): AND a 4 vie sui fetch; quando
# scatta scrive stato nuovo + tutte e 32 le colonne del frame nuovo.
ROM = [(s, i) for s in STATES for i in range(NIN)]
H = len(ROM)
Wup = np.zeros((H, N_EMBD), dtype=np.float32)
bup = np.zeros(H, dtype=np.float32)
Wdown = np.zeros((N_EMBD, H), dtype=np.float32)
bdown = np.zeros(N_EMBD, dtype=np.float32)
DSCALE = 0.1  # le uscite ROM restano piccole: e' il bonus di fase a fare da arbitro


def build_mlp(thr):
    for h, (s, i) in enumerate(ROM):
        Wup[h, GX[s[0]]] = S
        Wup[h, GY[s[1]]] = S
        Wup[h, GA[s[2]]] = S
        Wup[h, GI[i]] = S
        bup[h] = -S * thr
        s2 = TRANS[(s, i)]
        Wdown[NX[s2[0]], h] = DSCALE
        Wdown[NY[s2[1]], h] = DSCALE
        Wdown[NA[s2[2]], h] = DSCALE
        for ci, (hh, sd) in enumerate(FRAME[s2]):
            Wdown[NCOL[ci][(hh - 1) * NSHADE + sd], h] = DSCALE


# ---- unembedding: valore ROM + bonus di fase ---------------------------------
# Il bonus (colonna POSB della riga di generazione del tipo) decide QUALE
# tipo di token esce a quale riga; il valore ROM decide quale token del tipo.
Wout = np.zeros((V, N_EMBD), dtype=np.float32)
for v, t in X_T.items():
    Wout[t, NX[v]] = 1.0
for v, t in Y_T.items():
    Wout[t, NY[v]] = 1.0
for v, t in A_T.items():
    Wout[t, NA[v]] = 1.0
for (ci, hh, sd), t in C_T.items():
    Wout[t, NCOL[ci][(hh - 1) * NSHADE + sd]] = 1.0


def set_phase_bonus(B):
    for t in X_T.values():
        Wout[t, POSB[GEN0]] = B
    for t in Y_T.values():
        Wout[t, POSB[GEN0 + 1]] = B
    for t in A_T.values():
        Wout[t, POSB[GEN0 + 2]] = B
    for (ci, _, _), t in C_T.items():
        Wout[t, POSB[P_C0 - 1 + ci]] = B


gamma1 = np.ones(N_EMBD, dtype=np.float32)
beta1 = np.zeros(N_EMBD, dtype=np.float32)
gamma2 = gamma1.copy()
beta2 = beta1.copy()
gammaf = gamma1.copy()
betaf = beta1.copy()


def quantize_f16():
    for a in (wte, wpe, Wqkv, Wattn, Wup, Wdown, Wout):
        a[:] = a.astype(np.float16).astype(np.float32)


# =================== FORWARD FEDELE (stessi tensori del GGUF) ================
def ln_rows(X, g, b):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + EPS) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def forward_all(ids):
    T = len(ids)
    X = wte[ids] + wpe[:T]
    A = ln_rows(X, gamma1, beta1)
    qkv = A @ Wqkv.T + bqkv
    q, k, v = qkv[:, :N_EMBD], qkv[:, N_EMBD : 2 * N_EMBD], qkv[:, 2 * N_EMBD :]
    mask = np.triu(np.ones((T, T)), 1).astype(bool)
    att = np.zeros((T, N_EMBD))
    for h in range(N_HEAD):
        sl = slice(h * HEAD_D, (h + 1) * HEAD_D)
        sc = (q[:, sl] @ k[:, sl].T) / np.sqrt(HEAD_D)
        sc[mask] = -1e9
        Aw = np.exp(sc - sc.max(1, keepdims=True))
        Aw /= Aw.sum(1, keepdims=True)
        att[:, sl] = Aw @ v[:, sl]
    X = X + att @ Wattn.T + battn
    M = ln_rows(X, gamma2, beta2)
    XL = X + gelu(M @ Wup.T + bup) @ Wdown.T + bdown
    XLn = ln_rows(XL, gammaf, betaf)
    return M, XLn, XLn @ Wout.T


# ==================== RIFERIMENTO: un tick completo ==========================
def tick_ids(s, i):
    """Trascrizione completa di un tick (input + risposta attesa)."""
    s2 = TRANS[(s, i)]
    ids = [NOOP, X_T[s[0]], Y_T[s[1]], A_T[s[2]], I_T[i]]
    ids += [X_T[s2[0]], Y_T[s2[1]], A_T[s2[2]]]
    ids += [C_T[(ci, hh, sd)] for ci, (hh, sd) in enumerate(FRAME[s2])]
    return ids


def sample_ticks(rng, n):
    out = []
    for _ in range(n):
        s = STATES[int(rng.integers(len(STATES)))]
        out.append((s, int(rng.integers(NIN))))
    return out


# ======================= CALIBRAZIONE ========================================
def calibrate(seed=11, n_ticks=200):
    """Soglia della ROM: somma dei 4 fetch quando (stato,input) combaciano
    contro il massimo con al piu' 3 su 4, misurata su tutte le 35 righe di
    generazione di tick reali. Vettorizzata: pattern @ residuo."""
    rng = np.random.default_rng(seed)
    P4 = np.zeros((H, N_EMBD), dtype=np.float32)
    for h, (s, i) in enumerate(ROM):
        P4[h, GX[s[0]]] = P4[h, GY[s[1]]] = P4[h, GA[s[2]]] = P4[h, GI[i]] = 1.0
    must_min, notmax = np.inf, -np.inf
    blocked = 0
    for s, i in sample_ticks(rng, n_ticks):
        blocked += i in (2, 3) and TRANS[(s, i)] == s
        ids = tick_ids(s, i)
        M, _, _ = forward_all(ids)
        rows = M[GEN0 : GEN0 + N_GEN]
        sums = rows @ P4.T  # [35, H]
        idx = ROM.index((s, i))
        must_min = min(must_min, sums[:, idx].min())
        sums[:, idx] = -np.inf
        notmax = max(notmax, sums.max())
    assert blocked > 0, "nessun passo bloccato dal muro nei tick campionati"
    print(f"  rom: deve>={must_min:6.2f}  non_deve<={notmax:6.2f}  gap "
          f"{'OK' if notmax < must_min else 'SOVRAPPOSTO'}")
    assert notmax < must_min, "gap sovrapposto"
    return float((notmax + must_min) / 2)


def calibrate_phase_bonus(seed=17, n_ticks=40):
    """Misura, post-LN finale, il valore massimo scritto dalla ROM (D) e il
    minimo one-hot di posizione della riga (P): il bonus B*P deve dominare
    qualsiasi differenza tra i D dei tipi. B = 3*Dmax/Pmin."""
    rng = np.random.default_rng(seed)
    dmax, pmin = 0.0, np.inf
    data_dims = NX + NY + NA + [d for i in range(NCOLS) for d in NCOL[i]]
    for s, i in sample_ticks(rng, n_ticks):
        ids = tick_ids(s, i)
        _, XLn, _ = forward_all(ids)
        for r in range(GEN0, GEN0 + N_GEN):
            dmax = max(dmax, XLn[r, data_dims].max())
            pmin = min(pmin, XLn[r, POSB[r]])
    assert pmin > 0
    B = 3.0 * dmax / pmin
    print(f"  bonus di fase: Dmax={dmax:.2f} Pmin={pmin:.2f} -> B={B:.1f}")
    return B


# ============================ VERIFICA =======================================
def verify(seed=500, n_ticks=400, n_walks=8, walk_len=40):
    """Tick campionati + passeggiate concatenate (lo stato generato di un
    tick e' l'input del successivo, come fara' il frontend)."""
    rng = np.random.default_rng(seed)
    cases = sample_ticks(rng, n_ticks)
    for _ in range(n_walks):
        s = SPAWN
        for _ in range(walk_len):
            i = int(rng.integers(NIN))
            cases.append((s, i))
            s = TRANS[(s, i)]
    ok = tot = 0
    bad = []
    for s, i in cases:
        ids = tick_ids(s, i)
        _, _, logits = forward_all(ids)
        for r in range(GEN0, GEN0 + N_GEN):
            pred = int(logits[r].argmax())
            tot += 1
            if pred == ids[r + 1]:
                ok += 1
            else:
                bad.append((r, tokens[pred], tokens[ids[r + 1]]))
    if bad:
        print(f"  primi errori: {bad[:6]}")
    return ok, tot


# ============================== MAIN =========================================
if __name__ == "__main__":
    print(
        f"mappa {GC}x{GC}, {len(OPEN)} mezze-celle libere, {len(STATES)} stati, "
        f"ROM={H} neuroni"
    )
    print(f"n_embd={N_EMBD} n_head={N_HEAD} head_dim={HEAD_D} n_ff={H} vocab={V} ctx={MAXPOS}")
    quantize_f16()
    print("calibrazione (forward reale, pesi f16):")
    build_mlp(calibrate())
    quantize_f16()
    set_phase_bonus(calibrate_phase_bonus())
    quantize_f16()
    ok, tot = verify()
    print(f"verifica: {ok}/{tot} token corretti ({100 * ok / tot:.1f}%)")
    assert ok == tot, "verifica fallita: non scrivo il GGUF"

    w = gguf.GGUFWriter(OUT, "gpt2")
    w.add_name("wolf-raycaster")
    w.add_context_length(MAXPOS)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(1)
    w.add_feed_forward_length(H)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_eps(EPS)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(ttypes)
    w.add_bos_token_id(BOS)
    w.add_eos_token_id(EOS)
    w.add_unk_token_id(UNK)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    f16 = lambda a: a.astype(np.float16)
    w.add_tensor("token_embd.weight", f16(wte))
    w.add_tensor("position_embd.weight", f16(wpe))
    w.add_tensor("output_norm.weight", gammaf)
    w.add_tensor("output_norm.bias", betaf)
    w.add_tensor("output.weight", f16(Wout))
    w.add_tensor("blk.0.attn_norm.weight", gamma1)
    w.add_tensor("blk.0.attn_norm.bias", beta1)
    w.add_tensor("blk.0.attn_qkv.weight", f16(Wqkv))
    w.add_tensor("blk.0.attn_qkv.bias", bqkv)
    w.add_tensor("blk.0.attn_output.weight", f16(Wattn))
    w.add_tensor("blk.0.attn_output.bias", battn)
    w.add_tensor("blk.0.ffn_norm.weight", gamma2)
    w.add_tensor("blk.0.ffn_norm.bias", beta2)
    w.add_tensor("blk.0.ffn_up.weight", f16(Wup))
    w.add_tensor("blk.0.ffn_up.bias", bup)
    w.add_tensor("blk.0.ffn_down.weight", f16(Wdown))
    w.add_tensor("blk.0.ffn_down.bias", bdown)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"scritto {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")
