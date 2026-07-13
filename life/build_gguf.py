#!/usr/bin/env python3
"""
Genera `life.gguf`: il GAME OF LIFE di Conway su griglia 12x12 TOROIDALE
come transformer gpt2 programmato a mano. Il frontend manda la griglia,
il modello risponde con la generazione successiva: le regole di Life
vivono interamente nei pesi.

Protocollo (stile wolf: il nastro non e' una cronologia, il frontend
riserializza lo stato a ogni tick — si gioca per sempre):

    pos:   0        1 .. 144           145 .. 288        289
         <NOOP>  <C:b> x 144 (input)  <C:b> x 144 (out)  </s>

Il frontend manda <NOOP> + 144 token cella e chiede 145 token: la griglia
nuova e l'EOS. Al tick dopo rispedisce la griglia appena generata.

Le idee specifiche di questo progetto:

  - SELETTORI PER-RIGA CON WRAP TOROIDALE NELLE QUERY. Ogni cella nuova
    dipende dal suo vicinato 3x3 nella griglia di input: 9 teste, una per
    offset del vicinato. Ma l'offset di griglia NON e' un offset di nastro
    costante (il vicino di sinistra della colonna 0 e' la colonna 11):
    la query non e' una matrice di shift, e' una MAPPA ARBITRARIA
    posizione->posizione cablata in Wq. La riga che genera la cella (x,y)
    punta la testa k esattamente sulla posizione di input della cella
    ((x+dx_k) mod 12, (y+dy_k) mod 12): il toro sta nelle query.

  - L'MLP E' LA TAVOLA DI VERITA' DEL VICINATO: 512 NEURONI. Niente
    aritmetica ("conta i vicini vivi"): ogni configurazione 3x3 (2^9=512)
    e' un neurone AND a 9 vie sui fetch (ogni cella fetchata porta un
    one-hot vivo/morto, quindi ogni neurone riceve esattamente 9 addendi
    possibili). Il neurone che combacia scrive il one-hot vivo/morto della
    cella nuova, gia' calcolato qui in python con le regole di Conway.
    E' la ROM di wolf ridotta all'osso: la "fisica" e' 128 byte di verita'.

  - VARIANZA COSTANTE GRATIS. Ogni riga di generazione fetcha SEMPRE 9
    celle vere (il toro non ha bordi: nessun selettore senza bersaglio,
    nessun sink, niente zavorra), e ogni cella porta esattamente un
    one-hot a scala RAW: il profilo di varianza e' identico su tutte le
    145 righe di generazione e una soglia unica regge l'intera ROM.

  - FASE COME FLAG NEL WPE, NON COME ONE-HOT DI POSIZIONE. In wolf ogni
    tipo di token esce da UNA riga e il bonus di fase legge il one-hot di
    posizione di quella riga. Qui i token cella escono da 144 righe: un
    bonus sommato su 144 colonne POSB accumulerebbe 143 baseline negativi
    post-LN (le dim spente stanno sotto la media) e ribalterebbe il segno.
    Rimedio: il wpe porta un FLAG condiviso da tutte le righe di
    generazione cella (e uno per la riga dell'EOS): il bonus e' su una
    dimensione sola. E' il rimedio del sink di snake, applicato
    all'unembedding.

Calibrazione con COPERTURA ESAUSTIVA: le griglie di calibrazione
includono 57 "grigliati di francobolli" che piazzano tutte e 512 le
configurazioni 3x3, e il generatore rifiuta di procedere se anche una
sola configurazione non e' stata vista sparare.

Uso:
    pip install gguf numpy
    python build_gguf.py                # produce life.gguf
    ollama create life -f Modelfile
"""

import os

import gguf
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "life.gguf")

# ============================== GIOCO =======================================
G = 12  # lato della griglia (toroidale)
N = G * G  # 144 celle
# vicinato 3x3, riga per riga; il centro e' lo slot 4
OFFS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
CENTER = OFFS.index((0, 0))
NCFG = 1 << len(OFFS)  # 512 configurazioni di vicinato


def life_step(g):
    """Riferimento: un passo di Life su toro (numpy)."""
    n = sum(
        np.roll(np.roll(g, -dy, 0), -dx, 1) for dx, dy in OFFS if (dx, dy) != (0, 0)
    )
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.int64)


def cell_cfg(g, x, y):
    """Configurazione 3x3 (9 bit) del vicinato della cella (x,y)."""
    c = 0
    for k, (dx, dy) in enumerate(OFFS):
        c |= int(g[(y + dy) % G, (x + dx) % G]) << k
    return c


def rule(cfg):
    """Regole di Conway sulla configurazione: 1 se la cella nuova e' viva."""
    bits = [(cfg >> k) & 1 for k in range(len(OFFS))]
    alive = bits[CENTER]
    count = sum(bits) - alive
    return int(count == 3 or (alive and count == 2))


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
C_T = {v: game(f"<C:{v}>") for v in range(2)}  # <C:0> morta, <C:1> viva
V = len(tokens)

# ========================== IPERPARAMETRI ====================================
MAXPOS = 2 * N + 2  # 290: NOOP + input + output + EOS
HEAD_D = MAXPOS  # le chiavi sono one-hot di posizione: head_dim >= n_ctx
N_HEAD = len(OFFS)  # 9: una testa per offset del vicinato
N_EMBD = N_HEAD * HEAD_D  # 2610
EPS = 1e-5
SEL = 60.0
ALPHA = SEL * np.sqrt(HEAD_D)
S = 12.0
RAW = 9.0
DSCALE = 0.1

# righe di generazione: la riga p predice il token in p+1.
# La cella di output i (posizione N+1+i) e' predetta dalla riga N+i;
# la riga 2N predice l'EOS (posizione 2N+1).
GEN0 = N
GENL = 2 * N  # riga dell'EOS
N_GEN = N + 1  # 145 token generati per tick


# layout del residuo
def blk(n, c=[0]):
    s = c[0]
    c[0] += n
    return list(range(s, s + n))


_c = [0]
CV = blk(2, _c)  # one-hot del token cella (morta/viva)
BAL = blk(1, _c)[0]  # dim inerte del <NOOP> (parita' di varianza delle chiavi)
FN = {k: blk(2, _c) for k in range(len(OFFS))}  # fetch del vicinato: 9 x (morta,viva)
NXT = blk(2, _c)  # uscita della ROM: cella nuova (morta/viva)
GFLAG = blk(1, _c)[0]  # flag wpe delle 144 righe di generazione cella
EFLAG = blk(1, _c)[0]  # flag wpe della riga dell'EOS
POSB = blk(MAXPOS, _c)
assert _c[0] <= N_EMBD, f"layout {_c[0]} non entra in {N_EMBD}"

# ============================== PESI ========================================
# Ogni token di gioco porta esattamente UN one-hot a scala RAW (+ il POSB dal
# wpe): tutte le righe hanno la stessa varianza propria.
wte = np.zeros((V, N_EMBD), dtype=np.float32)
for v, t in C_T.items():
    wte[t, CV[v]] = RAW
wte[NOOP, BAL] = RAW
wpe = np.zeros((MAXPOS, N_EMBD), dtype=np.float32)
for p in range(MAXPOS):
    wpe[p, POSB[p]] = 1.0
for p in range(GEN0, GENL):
    wpe[p, GFLAG] = 1.0
wpe[GENL, EFLAG] = 1.0

# ---- attention: 9 selettori per-riga, wrap toroidale nelle query -------------
# k(j) = onehot(j); la query della riga di generazione p per la testa k e'
# ALPHA * onehot(1 + cella vicina k della cella generata da p), con il modulo
# del toro gia' calcolato qui. Le righe del prompt non hanno query (attenzione
# uniforme: da quelle righe non si genera mai); la riga dell'EOS riusa la
# mappa della cella 0 cosi' il suo profilo di varianza resta identico alle
# altre righe di generazione (l'esito li' lo decide solo il bonus di fase).
Wqkv = np.zeros((3 * N_EMBD, N_EMBD), dtype=np.float32)
bqkv = np.zeros(3 * N_EMBD, dtype=np.float32)
vrow = 2 * N_EMBD
for h, (dx, dy) in enumerate(OFFS):
    qrow, krow = h * HEAD_D, N_EMBD + h * HEAD_D
    for j in range(MAXPOS):
        Wqkv[krow + j, POSB[j]] = 1.0
    for p in range(GEN0, GENL + 1):
        i = (p - GEN0) % N
        x, y = i % G, i // G
        target = 1 + ((y + dy) % G) * G + ((x + dx) % G)
        Wqkv[qrow + target, POSB[p]] = ALPHA
    Wqkv[vrow + h * HEAD_D + 0, CV[0]] = 1.0
    Wqkv[vrow + h * HEAD_D + 1, CV[1]] = 1.0

Wattn = np.zeros((N_EMBD, N_EMBD), dtype=np.float32)
battn = np.zeros(N_EMBD, dtype=np.float32)
for h in range(len(OFFS)):
    Wattn[FN[h][0], h * HEAD_D + 0] = 1.0
    Wattn[FN[h][1], h * HEAD_D + 1] = 1.0

# ---- MLP: la tavola di verita' del vicinato ----------------------------------
# un neurone per configurazione 3x3: AND a 9 vie sui fetch (per ogni slot k
# il neurone pesa il one-hot morto O vivo richiesto dalla configurazione).
# Quando scatta scrive il one-hot della cella nuova secondo Conway.
H = NCFG
Wup = np.zeros((H, N_EMBD), dtype=np.float32)
bup = np.zeros(H, dtype=np.float32)
Wdown = np.zeros((N_EMBD, H), dtype=np.float32)
bdown = np.zeros(N_EMBD, dtype=np.float32)


def build_mlp(thr):
    for cfg in range(NCFG):
        for k in range(len(OFFS)):
            Wup[cfg, FN[k][(cfg >> k) & 1]] = S
        bup[cfg] = -S * thr
        Wdown[NXT[rule(cfg)], cfg] = DSCALE


# ---- unembedding: valore ROM + bonus di fase ---------------------------------
# Sulle 144 righe cella il bonus (uguale per <C:0> e <C:1>, dal GFLAG del
# wpe) alza il tipo giusto; a decidere tra i due e' il one-hot NXT scritto
# dalla ROM. Sulla riga dell'EOS solo l'EOS ha il bonus (EFLAG), e vince su
# qualsiasi valore ROM.
Wout = np.zeros((V, N_EMBD), dtype=np.float32)
for v, t in C_T.items():
    Wout[t, NXT[v]] = 1.0


def set_phase_bonus(B):
    for t in C_T.values():
        Wout[t, GFLAG] = B
    Wout[EOS, EFLAG] = B


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
def tick_ids(g):
    """Trascrizione completa di un tick (input + risposta attesa)."""
    g2 = life_step(g)
    ids = [NOOP]
    ids += [C_T[int(v)] for v in g.flat]
    ids += [C_T[int(v)] for v in g2.flat]
    ids += [EOS]
    return ids


# ---- griglie di gioco --------------------------------------------------------
def place(cells, ox=0, oy=0):
    g = np.zeros((G, G), dtype=np.int64)
    for x, y in cells:
        g[(oy + y) % G, (ox + x) % G] = 1
    return g


PATTERNS = {
    "block": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "blinker": [(0, 0), (1, 0), (2, 0)],
    "toad": [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)],
    "beacon": [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 2), (2, 3), (3, 3)],
    "glider": [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
    "lwss": [(1, 0), (4, 0), (0, 1), (0, 2), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3)],
    "rpentomino": [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
}


def stamp_grids():
    """57 griglie che piazzano TUTTE e 512 le configurazioni 3x3: francobolli
    3x3 su un grigliato a passo 4 (il centro di ogni francobollo ha come
    vicinato esattamente il francobollo)."""
    grids = []
    per = (G // 4) ** 2  # 9 francobolli per griglia
    for base in range(0, NCFG, per):
        g = np.zeros((G, G), dtype=np.int64)
        for j, cfg in enumerate(range(base, min(base + per, NCFG))):
            ox, oy = 4 * (j % (G // 4)), 4 * (j // (G // 4))
            for k, (dx, dy) in enumerate(OFFS):
                g[oy + 1 + dy, ox + 1 + dx] = (cfg >> k) & 1
        grids.append(g)
    return grids


def random_grid(rng, d):
    return (rng.random((G, G)) < d).astype(np.int64)


def games(seed, n_random):
    """Griglie di calibrazione/verifica: francobolli (copertura esaustiva),
    nature morte e oscillatori (anche a cavallo del bordo, per il wrap),
    evoluzioni concatenate (glider, r-pentomino, brodo) e griglie casuali."""
    rng = np.random.default_rng(seed)
    grids = stamp_grids()
    for cells in PATTERNS.values():
        for ox, oy in ((0, 0), (5, 3), (10, 9)):
            grids.append(place(cells, ox, oy))
    grids.append(np.zeros((G, G), dtype=np.int64))
    grids.append(np.ones((G, G), dtype=np.int64))
    for cells, steps in ((PATTERNS["glider"], 24), (PATTERNS["rpentomino"], 30)):
        g = place(cells, 4, 4)
        for _ in range(steps):
            grids.append(g)
            g = life_step(g)
    g = random_grid(rng, 0.35)
    for _ in range(16):
        grids.append(g)
        g = life_step(g)
    for _ in range(n_random):
        grids.append(random_grid(rng, float(rng.uniform(0.05, 0.95))))
    return grids


# ======================= CALIBRAZIONE ========================================
def calibrate(seed=11, n_random=12):
    """Soglia della ROM: somma dei 9 fetch quando la configurazione combacia
    contro il massimo con al piu' 8 su 9, misurata su tutte le 145 righe di
    generazione di tick reali. Rifiuta se una configurazione non e' mai
    stata vista sparare (copertura esaustiva garantita dai francobolli)."""
    P9 = np.zeros((NCFG, N_EMBD), dtype=np.float32)
    for cfg in range(NCFG):
        for k in range(len(OFFS)):
            P9[cfg, FN[k][(cfg >> k) & 1]] = 1.0
    must_min, notmax = np.inf, -np.inf
    covered = set()
    for g in games(seed, n_random):
        ids = tick_ids(g)
        M, _, _ = forward_all(ids)
        rows = M[GEN0 : GENL + 1]
        sums = rows @ P9.T  # [145, 512]
        cfgs = [cell_cfg(g, i % G, i // G) for i in range(N)]
        cfgs.append(cfgs[0])  # la riga dell'EOS riusa la mappa della cella 0
        covered.update(cfgs)
        idx = (np.arange(N_GEN), np.array(cfgs))
        must_min = min(must_min, sums[idx].min())
        sums[idx] = -np.inf
        notmax = max(notmax, sums.max())
    print(f"  copertura: {len(covered)}/{NCFG} configurazioni")
    assert len(covered) == NCFG, "copertura incompleta: rivedi i francobolli"
    print(f"  rom: deve>={must_min:6.2f}  non_deve<={notmax:6.2f}  gap "
          f"{'OK' if notmax < must_min else 'SOVRAPPOSTO'}")
    assert notmax < must_min, "gap sovrapposto"
    return float((notmax + must_min) / 2)


def calibrate_phase_bonus(seed=17, n_random=10):
    """Misura, post-LN finale, il massimo valore ROM (D) e il minimo flag di
    fase della riga (P): il bonus B*P deve dominare qualsiasi valore ROM
    fuori posto. B = 3*Dmax/Pmin (come wolf)."""
    rng = np.random.default_rng(seed)
    grids = [random_grid(rng, d) for d in (0.15, 0.3, 0.5, 0.7, 0.85)]
    grids += [place(PATTERNS["glider"], 8, 8)]
    grids += [random_grid(rng, float(rng.uniform(0.1, 0.9))) for _ in range(n_random)]
    dmax, pmin = 0.0, np.inf
    for g in grids:
        ids = tick_ids(g)
        _, XLn, _ = forward_all(ids)
        for r in range(GEN0, GENL + 1):
            dmax = max(dmax, XLn[r, NXT].max())
            pmin = min(pmin, XLn[r, GFLAG if r < GENL else EFLAG])
    assert pmin > 0
    B = 3.0 * dmax / pmin
    print(f"  bonus di fase: Dmax={dmax:.2f} Pmin={pmin:.2f} -> B={B:.1f}")
    return B


# ============================ VERIFICA =======================================
def verify(seed=500, n_random=25):
    """Tick campionati + evoluzioni concatenate (la griglia generata da un
    tick e' l'input del successivo, come fara' il frontend)."""
    ok = tot = 0
    bad = []
    for g in games(seed, n_random):
        ids = tick_ids(g)
        _, _, logits = forward_all(ids)
        for r in range(GEN0, GENL + 1):
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
    # sanity: la regola per-configurazione coincide col riferimento numpy
    _g = random_grid(np.random.default_rng(0), 0.5)
    _g2 = life_step(_g)
    assert all(
        rule(cell_cfg(_g, x, y)) == _g2[y, x] for x in range(G) for y in range(G)
    )
    print(
        f"griglia {G}x{G} toroidale, {NCFG} configurazioni di vicinato\n"
        f"n_embd={N_EMBD} n_head={N_HEAD} head_dim={HEAD_D} n_ff={H} vocab={V} "
        f"ctx={MAXPOS}"
    )
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
    w.add_name("game-of-life")
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
