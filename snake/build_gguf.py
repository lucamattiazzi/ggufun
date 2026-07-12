#!/usr/bin/env python3
"""
Genera `snake.gguf`: SNAKE — il frontend manda SOLO le mosse.

Differenza rispetto a snake/: cibo e lunghezza non arrivano piu' dal
frontend. Il modello genera 3 token per turno invece di 2, e il terzo e'
proprio il token <F:cibo,len> che prima scriveva il frontend:

    input:  <M:mossa>
    output: <P:testa> <STATO> <F:cibo,len>

Le idee nuove rispetto a snake/build_gguf.py (che resta il riferimento per
tutto il resto: selettori, zavorra, calibrazione):

  - IL MODELLO SCRIVE LA PROPRIA CONTABILITA' SUL NASTRO. La lunghezza non
    va "ricordata": il modello la legge dal proprio <F> precedente (fetch a
    offset -3) e la riscrive incrementata se lo stato appena emesso e' <ATE>
    (un piccolo adder di porte AND, con cap a 8). Idem il cibo: se lo stato
    e' <OK>/<DEAD> viene ricopiato, se e' <ATE> ne viene scelto uno nuovo.

  - PSEUDO-CASUALITA' DAI POSITION EMBEDDING. Il modello e' deterministico:
    il "dado" e' una tabella fissa hash(posizione)->cella cablata nel wpe.
    Le righe di posizione p = 2 mod 4 (quelle da cui si genera <F>) portano,
    oltre al one-hot POSB, un one-hot HASHC[hash(p)] con boost anti-LN: a
    ogni turno corrisponde una cella "estratta" diversa. Un neurone
    AND(HASHC[c], ATE) la scrive nel cibo nuovo. Variante A dichiarata: il
    cibo puo' cadere sotto il corpo del serpente (resta solo temporaneamente
    irraggiungibile: il pasto e' testa==cibo). Stessa sequenza di mosse =
    stessa partita; il seme e' il cibo iniziale f0 del prefisso (frontend).

  - QUERY DELLE TESTE GATED PER FASE. Il periodo 4 ha fasi fisse
    (M=0, P=1, S=2, F=3 mod 4; il prefisso le rispetta con <NOOP> negli
    slot-M). La query di ogni testa esiste solo nelle fasi in cui la testa
    serve: cosi' ogni testa, a ogni TIPO di riga, pesca sempre lo stesso
    tipo di bersaglio per tutta la partita, la varianza del residuo resta
    costante per tipo di riga, la zavorra basta sul solo <NOOP> e le soglie
    per gruppo reggono dall'inizio alla fine.

  - SINK VIA BIAS, NON VIA QUERY. Nel vecchio snake le posizioni senza
    bersaglio venivano dirottate sul <NOOP> accumulando pesi sulla riga di
    query 0: con poche posizioni funziona, ma qui 3 fasi su 4 andrebbero in
    sink (~180 posizioni per testa) e la somma dei baseline negativi
    post-LN (la LN sottrae la media: le dim POSB spente sono leggermente
    negative) ribalterebbe il segno della query. Rimedio: una coordinata
    q/k dedicata (SINKDIM, fuori dal one-hot di posizione): il bias di
    query le da' un punteggio costante verso il <NOOP> (l'unico token con
    zavorra, che fa da chiave), e la query reale — quando c'e' — lo
    sovrasta di un margine ampio. Default: sink; eccezione: il bersaglio.

Teste (6, come prima — n_embd resta 1440):
    h0  off -3  fasi {M,S}: riga M -> <P> precedente (CELL->G3);
                            riga S -> <F> precedente (FCELL->G3F, LEN->G3L)
    h1  off -2  fasi {P}:   <F> corrente (FCELL->G2F, LEN->G2L)
    h2-5 off -4k, k in {1,2,4,6}, fasi {P}: corpo (CELL->BK, zavorra->BALK)

Pipeline identica a snake: vocab -> pesi -> f16 -> calibrazione per gruppo su
partite reali + scenari mirati -> verifica 100% -> GGUF. Gli scenari non
possono piu' "truccare" il cibo (ora e' autorita' del modello): usano
politiche che inseguono il cibo dove l'hash lo mette, crescono fino alla
lunghezza voluta e poi eseguono cicli fissi per forzare ogni morte k e i
falsi positivi stale.

Uso:
    pip install gguf numpy
    python build_gguf.py                 # produce snake.gguf
    ollama create snake -f Modelfile
"""

import os

import gguf
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snake.gguf")

# ============================== GIOCO =======================================
G = 6
CELLS = G * G
MOVES = ["L", "R", "U", "D"]
DELTA = {"L": (-1, 0), "R": (1, 0), "U": (0, -1), "D": (0, 1)}
L0, LMAX = 3, 8
BODY_KS = [1, 2, 4, 6]  # parita': k dispari >=3 impossibile, k=1 solo clamp


def cid(x, y):
    return y * G + x


def cxy(c):
    return (c % G, c // G)


def step_cell(c, m):
    x, y = cxy(c)
    dx, dy = DELTA[m]
    return cid(max(0, min(G - 1, x + dx)), max(0, min(G - 1, y + dy)))


def dist(a, b):
    ax, ay = cxy(a)
    bx, by = cxy(b)
    return abs(ax - bx) + abs(ay - by)


START_SEG, START_HEAD = cid(1, 2), cid(1, 1)

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
POS_T = {c: game(f"<P:{c}>") for c in range(CELLS)}
FOOD_T = {
    (c, l): game(f"<F:{c},{l}>")
    for c in range(CELLS)
    for l in range(L0, LMAX + 1)
}
MOVE_T = {m: game(f"<M:{m}>") for m in MOVES}
OK_T, ATE_T, DEAD_T = game("<OK>"), game("<ATE>"), game("<DEAD>")
STATE_T = [OK_T, ATE_T, DEAD_T]  # indici ST: OK=0, ATE=1, DEAD=2
V = len(tokens)

# ========================== IPERPARAMETRI ====================================
N_HEAD = 6
HEAD_D = 240
MAXPOS = 236  # 57 turni: head_dim - 4, per lasciare posto a SINKDIM
SINKDIM = MAXPOS  # coordinata q/k del sink (fuori dal one-hot di posizione)
N_EMBD = N_HEAD * HEAD_D  # 1440
EPS = 1e-5
SEL = 60.0  # nitidezza dei selettori reali
SEL_SINK = 15.0  # punteggio costante del sink: vince solo dove non c'e' query
ALPHA = SEL * np.sqrt(HEAD_D)
S = 12.0
NEG = 3.0
RAW = 9.0  # boost anti-LN per i one-hot "propri" (cella, mossa, stato, hash)
NLEN = LMAX - L0 + 1

# fasi del protocollo (posizione mod 4): M=0, P=1, S=2, F=3.
# Le righe di generazione sono M (->P), P (->stato), S (->F).
PH_M, PH_P, PH_S, PH_F = 0, 1, 2, 3

# la tabella pseudo-casuale del cibo: posizione della riga S -> cella.
# E' il "dado" del modello, cablato nei position embedding (seed fisso).
_hash_rng = np.random.default_rng(20260712)
HASH = {p: int(_hash_rng.integers(CELLS)) for p in range(10, MAXPOS, 4)}

# teste: (offset, fasi in cui la query e' attiva; altrove -> sink su <NOOP>)
HEADS = [
    (3, {PH_M, PH_S}),  # h0: riga M pesca <P> prec; riga S pesca <F> prec
    (2, {PH_P}),  # h1: riga P pesca <F> corrente
] + [(4 * k, {PH_P}) for k in BODY_KS]  # h2-5: corpo


# layout del residuo
def blk(n, c=[0]):
    s = c[0]
    c[0] += n
    return list(range(s, s + n))


_c = [0]
F_ISPOS = blk(1, _c)[0]
CELL = blk(CELLS, _c)  # cella del token <P:c>
FCELL = blk(CELLS, _c)  # cella del token <F:c,l>
MOV = blk(4, _c)  # direzione del token <M:m>
LEN = blk(NLEN, _c)  # lunghezza del token <F:c,l>
ST = blk(3, _c)  # stato del token <OK>/<ATE>/<DEAD>
HASHC = blk(CELLS, _c)  # cella hash della posizione (dal wpe, righe S)
G3 = blk(CELLS, _c)  # fetch -3 (riga M): testa precedente
G3F = blk(CELLS, _c)  # fetch -3 (riga S): cibo precedente
G3L = blk(NLEN, _c)  # fetch -3 (riga S): lunghezza precedente
G2F = blk(CELLS, _c)  # fetch -2 (riga P): cibo corrente
G2L = blk(NLEN, _c)  # fetch -2 (riga P): lunghezza corrente
BK = {k: blk(CELLS, _c) for k in BODY_KS}  # fetch -4k (riga P): corpo
BAL_SRC = blk(1, _c)[0]
BALK = {k: blk(1, _c)[0] for k in BODY_KS}
NH = blk(CELLS, _c)  # nuova testa (MLP, riga M)
NF = blk(CELLS, _c)  # nuovo cibo (MLP, riga S)
NL = blk(NLEN, _c)  # nuova lunghezza (MLP, riga S)
SD = blk(1, _c)[0]
SA = blk(1, _c)[0]
SOK = blk(1, _c)[0]
POSB = blk(MAXPOS, _c)
assert _c[0] <= N_EMBD, f"layout {_c[0]} non entra in {N_EMBD}"

# ============================== PESI ========================================
wte = np.zeros((V, N_EMBD), dtype=np.float32)
for c, t in POS_T.items():
    wte[t, F_ISPOS] = 1.0
    wte[t, CELL[c]] = RAW
for (c, l), t in FOOD_T.items():
    wte[t, FCELL[c]] = 1.0
    wte[t, LEN[l - L0]] = 1.0
for m, t in MOVE_T.items():
    wte[t, MOV[MOVES.index(m)]] = RAW  # boost: e' un input "proprio" degli AND
for i, t in enumerate(STATE_T):
    wte[t, ST[i]] = RAW
wte[NOOP, BAL_SRC] = RAW  # zavorra per i fetch dirottati sul sink

wpe = np.zeros((MAXPOS, N_EMBD), dtype=np.float32)
for p in range(MAXPOS):
    wpe[p, POSB[p]] = 1.0
for p, c in HASH.items():
    wpe[p, HASHC[c]] = RAW  # il dado: la riga S "porta con se'" la cella estratta

# ---- attn_qkv [3*N_EMBD, N_EMBD] -------------------------------------------
# k(j) = onehot(j); q(p) = onehot(p-off) SOLO nelle fasi in cui la testa
# serve (e con bersaglio nel contesto). Il sink e' il default, via bias:
# ogni riga ha q[SINKDIM] costante, e solo il <NOOP> (zavorra) ha
# k[SINKDIM] alto -> chi non ha una query reale attende il <NOOP> in modo
# esatto; chi ce l'ha lo sovrasta (SEL * qk_min >> SEL_SINK * zavorra).
Wqkv = np.zeros((3 * N_EMBD, N_EMBD), dtype=np.float32)
bqkv = np.zeros(3 * N_EMBD, dtype=np.float32)
for h, (off, phases) in enumerate(HEADS):
    qrow, krow = h * HEAD_D, N_EMBD + h * HEAD_D
    for j in range(MAXPOS):
        Wqkv[krow + j, POSB[j]] = 1.0
    Wqkv[krow + SINKDIM, BAL_SRC] = 1.0  # solo <NOOP> risponde al sink
    bqkv[qrow + SINKDIM] = SEL_SINK * np.sqrt(HEAD_D)
    for p in range(MAXPOS):
        if p % 4 in phases and p >= off:
            Wqkv[qrow + (p - off), POSB[p]] = ALPHA

# valori per testa (dentro head_dim):
vrow = 2 * N_EMBD
# h0: CELL (0:36) + FCELL (36:72) + LEN (72:78)
for i in range(CELLS):
    Wqkv[vrow + 0 * HEAD_D + i, CELL[i]] = 1.0
    Wqkv[vrow + 0 * HEAD_D + CELLS + i, FCELL[i]] = 1.0
for i in range(NLEN):
    Wqkv[vrow + 0 * HEAD_D + 2 * CELLS + i, LEN[i]] = 1.0
# h1: FCELL (0:36) + LEN (36:42)
for i in range(CELLS):
    Wqkv[vrow + 1 * HEAD_D + i, FCELL[i]] = 1.0
for i in range(NLEN):
    Wqkv[vrow + 1 * HEAD_D + CELLS + i, LEN[i]] = 1.0
# h2-5: CELL (0:36) + zavorra (36)
for h in range(2, N_HEAD):
    for i in range(CELLS):
        Wqkv[vrow + h * HEAD_D + i, CELL[i]] = 1.0
    Wqkv[vrow + h * HEAD_D + CELLS, BAL_SRC] = 1.0

# ---- attn_output [N_EMBD, N_EMBD]: instrada i fetch nei blocchi ------------
Wattn = np.zeros((N_EMBD, N_EMBD), dtype=np.float32)
battn = np.zeros(N_EMBD, dtype=np.float32)
for i in range(CELLS):
    Wattn[G3[i], 0 * HEAD_D + i] = 1.0
    Wattn[G3F[i], 0 * HEAD_D + CELLS + i] = 1.0
    Wattn[G2F[i], 1 * HEAD_D + i] = 1.0
for i in range(NLEN):
    Wattn[G3L[i], 0 * HEAD_D + 2 * CELLS + i] = 1.0
    Wattn[G2L[i], 1 * HEAD_D + CELLS + i] = 1.0
for h, k in enumerate(BODY_KS, start=2):
    for i in range(CELLS):
        Wattn[BK[k][i], h * HEAD_D + i] = 1.0
    Wattn[BALK[k], h * HEAD_D + CELLS] = 1.0

# ---- MLP --------------------------------------------------------------------
# neurone = (dims_positivi, dims_negativi, dim_out, gruppo)
# I gruppi nuovi (righe S): fcopy/lcopy ricopiano cibo e lunghezza se lo
# stato e' OK o DEAD (lo stato e' one-hot: il neurone riceve peso da
# ENTRAMBE le alternative e ne scatta al piu' una); fhash/linc scattano su
# ATE e scrivono la cella estratta e la lunghezza incrementata (cap LMAX).
hidden = []
for c in range(CELLS):
    for mi, m in enumerate(MOVES):
        hidden.append(([G3[c], MOV[mi]], [], NH[step_cell(c, m)], "table"))
for c in range(CELLS):
    hidden.append(([CELL[c], G2F[c]], [], SA, "eat"))
for k in BODY_KS:
    inhib = [G2L[l - L0] for l in range(L0, min(k, LMAX) + 1)]
    for c in range(CELLS):
        hidden.append(([CELL[c], BK[k][c]], inhib, SD, "body"))
hidden.append(([F_ISPOS], [], SOK, "ok"))
for c in range(CELLS):
    hidden.append(([G3F[c], ST[0], ST[2]], [], NF[c], "fcopy"))
    hidden.append(([HASHC[c], ST[1]], [], NF[c], "fhash"))
for l in range(L0, LMAX + 1):
    hidden.append(([G3L[l - L0], ST[0], ST[2]], [], NL[l - L0], "lcopy"))
    hidden.append(([G3L[l - L0], ST[1]], [], NL[min(l + 1, LMAX) - L0], "linc"))
H = len(hidden)
Wup = np.zeros((H, N_EMBD), dtype=np.float32)
bup = np.zeros(H, dtype=np.float32)
Wdown = np.zeros((N_EMBD, H), dtype=np.float32)
bdown = np.zeros(N_EMBD, dtype=np.float32)


def build_mlp(thr):
    Wup[:] = 0
    bup[:] = 0
    Wdown[:] = 0
    for h, (pos, neg, out, kind) in enumerate(hidden):
        for d in pos:
            Wup[h, d] = S
        for d in neg:
            Wup[h, d] = -NEG * S
        bup[h] = -S * thr[kind]
        Wdown[out, h] = 1.0


# ---- unembedding -------------------------------------------------------------
# Il token <F:c,l> somma due punteggi separabili: cella (NF) e lunghezza (NL).
# Vince la coppia giusta: ogni altra manca almeno uno dei due addendi.
Wout = np.zeros((V, N_EMBD), dtype=np.float32)
for c, t in POS_T.items():
    Wout[t, NH[c]] = 1.0
Wout[DEAD_T, SD] = 6.0
Wout[ATE_T, SA] = 4.0
Wout[OK_T, SOK] = 1.0
for (c, l), t in FOOD_T.items():
    Wout[t, NF[c]] = 1.0
    Wout[t, NL[l - L0]] = 1.0

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
    logits = ln_rows(XL, gammaf, betaf) @ Wout.T
    return M, logits


# ==================== RIFERIMENTO PYTHON (regole del gioco) ==================
def prefix(f0):
    # stessi 2 turni sintetici di snake, riordinati sulle nuove fasi:
    # <NOOP> occupa lo slot-M, cosi' tutte le fasi restano allineate
    return [
        NOOP,
        POS_T[START_SEG],
        OK_T,
        FOOD_T[(f0, L0)],
        NOOP,
        POS_T[START_HEAD],
        OK_T,
        FOOD_T[(f0, L0)],
    ]


def play(rng, chooser):
    """Gioca una partita col riferimento python: (ids, events).

    Il cibo e' ora AUTORITA' DEL MODELLO: il riferimento lo calcola con la
    stessa tabella HASH cablata nei pesi. Il chooser puo' restituire None
    per fermare la partita (scenari che devono sopravvivere).

    events:
      ("P", riga_M, testa_prec, mossa)
      ("S", riga_P, new, food, L, segs, dead, ate)
      ("F", riga_S, food, L, newfood, newL, ate, dead)
    """
    heads = [START_SEG, START_HEAD]
    L = L0
    f0 = int(rng.choice([c for c in range(CELLS) if c not in heads]))
    food = f0
    ids = prefix(f0)
    events = []
    while True:
        if len(ids) + 4 > MAXPOS:
            break
        m = chooser(heads, L, food)
        if m is None:
            break
        new = step_cell(heads[-1], m)
        collset = heads[-(L - 1) :]
        dead = new in collset
        ate = (not dead) and new == food
        ids.append(MOVE_T[m])
        events.append(("P", len(ids) - 1, heads[-1], m))
        ids.append(POS_T[new])
        segs = {k: (heads[-k] if k <= len(heads) else None) for k in BODY_KS}
        events.append(("S", len(ids) - 1, new, food, L, segs, dead, ate))
        ids.append(DEAD_T if dead else ATE_T if ate else OK_T)
        i_s = len(ids) - 1
        if ate:
            newfood, newL = HASH[i_s], min(L + 1, LMAX)
        else:
            newfood, newL = food, L
        events.append(("F", i_s, food, L, newfood, newL, ate, dead))
        ids.append(FOOD_T[(newfood, newL)])
        if dead:
            break
        heads.append(new)
        food, L = newfood, newL
    return ids, events


# ---- politiche di gioco ------------------------------------------------------
def _safe_toward(rng, heads, L, target):
    """Mossa sicura che avvicina a target (clamp incluso nelle collisioni:
    la testa sta nel collset). Se non c'e' mossa sicura: una a caso (muore)."""
    head = heads[-1]
    coll = set(heads[-(L - 1) :])
    safe = [m for m in MOVES if step_cell(head, m) not in coll]
    if not safe:
        return MOVES[int(rng.integers(4))]
    return min(safe, key=lambda m: (dist(step_cell(head, m), target), rng.random()))


def make_random(rng, p_unsafe=0.15, p_greedy=0.5):
    """Partite casuali: un po' greedy (per mangiare), un po' suicida (DEAD)."""

    def choose(heads, L, food):
        coll = set(heads[-(L - 1) :])
        safe = [m for m in MOVES if step_cell(heads[-1], m) not in coll]
        if not safe or rng.random() < p_unsafe:
            return MOVES[int(rng.integers(4))]
        if rng.random() < p_greedy:
            return _safe_toward(rng, heads, L, food)
        return safe[int(rng.integers(len(safe)))]

    return choose


def make_script(moves):
    """Mosse fisse, poi stop (per muro/inversione, che non dipendono dal cibo)."""
    st = {"j": 0}

    def choose(heads, L, food):
        if st["j"] >= len(moves):
            return None
        st["j"] += 1
        return moves[st["j"] - 1]

    return choose


CENTER = cid(2, 2)
LOOP4 = ["R", "D", "L", "U"]  # ciclo 2x2: rivisita dopo 4 mosse
LOOP6 = ["R", "R", "D", "L", "L", "U"]  # ciclo 2x3: rivisita dopo 6 mosse


def make_scenario(rng, target_L, loop, loop_steps):
    """Insegue il cibo (dove l'hash lo mette) finche' L < target_L, si porta
    al centro, poi esegue `loop`: morte k=4/k=6 se il corpo e' abbastanza
    lungo, falso positivo stale (deve sopravvivere) altrimenti. loop_steps
    limita il ciclo negli scenari che devono restare vivi."""
    st = {"phase": "grow", "j": 0}

    def choose(heads, L, food):
        if st["phase"] == "grow":
            if L < target_L:
                return _safe_toward(rng, heads, L, food)
            st["phase"] = "center"
        if st["phase"] == "center":
            if heads[-1] != CENTER:
                return _safe_toward(rng, heads, L, CENTER)
            st["phase"] = "loop"
        if loop is None:  # scenario "cap": continua a mangiare per sempre
            return _safe_toward(rng, heads, L, food)
        if loop_steps is not None and st["j"] >= loop_steps:
            return None
        st["j"] += 1
        return loop[(st["j"] - 1) % len(loop)]

    return choose


# scenari: (target_L, loop, loop_steps) — coprono morte per ogni k possibile,
# i falsi positivi stale k=4/k=6 e il pasto a cap raggiunto (L resta 8)
SCENARIOS = [
    (None, ["U"] * 4, None),  # muro: clamp -> morte k=1
    (None, ["R", "L"], None),  # inversione -> morte k=2
    (5, LOOP4, None),  # L=5, ciclo 2x2 -> morte k=4
    (4, LOOP4, 12),  # L=4, ciclo 2x2 -> STALE k=4, resta vivo
    (7, LOOP6, None),  # L=7, ciclo 2x3 -> morte k=6
    (6, LOOP6, 20),  # L=6, ciclo 2x3 -> STALE k=6, resta vivo
    (99, None, None),  # mangia fino a L=8 e oltre (pasto a cap)
]


def games(seed, n_random, n_scen=3):
    rng = np.random.default_rng(seed)
    out = []
    for target_L, loop, steps in SCENARIOS:
        for _ in range(n_scen):
            if target_L is None:
                out.append(play(rng, make_script(loop)))
            else:
                out.append(play(rng, make_scenario(rng, target_L, loop, steps)))
    for _ in range(n_random):
        out.append(play(rng, make_random(rng)))
    return out


# ======================= CALIBRAZIONE PER GRUPPO =============================
def neuron_sum(M, pos, neg):
    return sum(M[d] for d in pos) - NEG * sum(M[d] for d in neg)


def ground_truth_firing(ev):
    fire = {}  # target dims -> gruppo atteso
    if ev[0] == "P":
        _, _, prev, m = ev
        fire[("table", tuple([G3[prev], MOV[MOVES.index(m)]]))] = True
    elif ev[0] == "S":
        _, _, new, food, L, segs, dead, ate = ev
        if new == food and not dead:
            fire[("eat", tuple([CELL[new], G2F[new]]))] = True
        if dead and new == food:  # pasto e morte insieme: l'eat scatta comunque
            fire[("eat", tuple([CELL[new], G2F[new]]))] = True
        for k in BODY_KS:
            if segs[k] == new and k <= L - 1:
                fire[("body", tuple([CELL[new], BK[k][new]]))] = True
    else:
        _, _, food, L, newfood, newL, ate, dead = ev
        if ate:
            fire[("fhash", tuple([HASHC[newfood], ST[1]]))] = True
            fire[("linc", tuple([G3L[L - L0], ST[1]]))] = True
        else:
            fire[("fcopy", tuple([G3F[food], ST[0], ST[2]]))] = True
            fire[("lcopy", tuple([G3L[L - L0], ST[0], ST[2]]))] = True
    return fire


GROUPS = ["table", "eat", "body", "fcopy", "fhash", "lcopy", "linc"]


def calibrate(seed=11, n_random=25):
    must = {k: [] for k in GROUPS}
    mustnot = {k: [] for k in GROUPS}
    iso = []
    stats = {"deaths": {}, "ate": 0, "stale": 0, "maxL": 0, "cap_meals": 0}
    for ids, events in games(seed, n_random):
        M, _ = forward_all(ids)
        for ev in events:
            row = ev[1]
            firing = ground_truth_firing(ev)
            for h, (pos, neg, _, kind) in enumerate(hidden):
                if kind == "ok":
                    continue
                s = neuron_sum(M[row], pos, neg)
                if (kind, tuple(pos)) in firing:
                    must[kind].append(s)
                else:
                    mustnot[kind].append(s)
            if ev[0] == "S":
                _, _, new, food, L, segs, dead, ate = ev
                iso.append(M[row][F_ISPOS])
                stats["maxL"] = max(stats["maxL"], L)
                stats["ate"] += ate
                stats["cap_meals"] += ate and L == LMAX
                if dead:
                    k = next(k for k in BODY_KS if segs[k] == new and k <= L - 1)
                    stats["deaths"][k] = stats["deaths"].get(k, 0) + 1
                else:
                    stats["stale"] += sum(
                        1 for k in BODY_KS if segs[k] == new and k > L - 1
                    )
    print(
        f"  eventi: pasti={stats['ate']} (a cap: {stats['cap_meals']}) "
        f"maxL={stats['maxL']} morti_per_k={dict(sorted(stats['deaths'].items()))} "
        f"stale={stats['stale']}"
    )
    for k in BODY_KS:
        assert stats["deaths"].get(k), f"nessuna morte k={k}: rivedi gli scenari"
    assert stats["maxL"] == LMAX and stats["stale"] > 0 and stats["cap_meals"] > 0
    thr = {}
    ok_all = True
    for k in GROUPS:
        mn, nn = np.array(must[k]), np.array(mustnot[k])
        assert len(mn), f"gruppo {k}: nessun evento 'deve sparare'"
        gap = nn.max() < mn.min()
        ok_all &= gap
        thr[k] = float((nn.max() + mn.min()) / 2)
        print(
            f"  {k:5s}: deve>={mn.min():6.2f}  non_deve<={nn.max():6.2f}  "
            f"gap {'OK' if gap else 'SOVRAPPOSTO'}  thr={thr[k]:.2f}"
        )
    thr["ok"] = float(np.array(iso).min() / 2)
    assert ok_all, "gap sovrapposto: rivedi i boost o i gruppi"
    return thr


# ============================ VERIFICA =======================================
def verify(seed=500, n_random=40):
    ok = tot = 0
    bad = []
    for ids, events in games(seed, n_random):
        _, logits = forward_all(ids)
        for ev in events:
            row = ev[1]
            pred = int(logits[row].argmax())
            tot += 1
            if pred == ids[row + 1]:
                ok += 1
            else:
                bad.append((ev[0], tokens[pred], tokens[ids[row + 1]]))
    if bad:
        print(f"  primi errori: {bad[:8]}")
    return ok, tot


# ============================== MAIN =========================================
if __name__ == "__main__":
    print(
        f"n_embd={N_EMBD} n_head={N_HEAD} head_dim={HEAD_D} n_ff={H} vocab={V} "
        f"ctx={MAXPOS} (~{(MAXPOS - 8) // 4} turni)"
    )
    quantize_f16()
    print("calibrazione soglie (forward reale, pesi f16):")
    build_mlp(calibrate())
    quantize_f16()
    ok, tot = verify()
    print(f"verifica: {ok}/{tot} token corretti ({100 * ok / tot:.1f}%)")
    assert ok == tot, "verifica fallita: non scrivo il GGUF"

    w = gguf.GGUFWriter(OUT, "gpt2")
    w.add_name("snake-transformer")
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
    print(f"scritto {OUT}")
