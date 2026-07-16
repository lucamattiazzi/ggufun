#!/usr/bin/env python3
"""
Genera `base64.gguf`: un encoder base64 stateless come transformer gpt2
programmato a mano. Il prompt porta i byte di input, il completion e' la
loro codifica base64 (vedi ref_base64.py per il protocollo token-level).

Il problema nuovo rispetto a life/wolf: l'indirizzo da fetchare NON e'
funzione della sola posizione della riga. La riga che genera l'output k sta
in posizione p = 1 + L + k: per sapere quali byte servono bisogna conoscere
sia p sia L (la posizione del `;`). Un solo layer di attention non puo'
comporre "leggi L, poi calcola l'indirizzo, poi fetcha il byte": quindi

  - BLOCCO 0 — DOVE SONO. Una testa a indirizzamento PER CONTENUTO (query
    costante via bias, stile selettori assoluti di wolf): ogni riga attende
    al token `;` (l'unico che porta la dim SEMI), e il suo V esporta il
    one-hot di POSIZIONE della riga attesa. Ogni riga di generazione riceve
    cosi' LPOS = onehot(1+L). L'MLP del blocco e' una tabella sulle coppie
    (p, L) — 1681 neuroni, un AND a 2 vie POSB[p] AND LPOS[1+L] ciascuno —
    che scrive nel residuo gli INDIRIZZI dei byte da fetchare (one-hot
    ADDR_A / ADDR_B), la FASE della riga (PH0..PH3) e il suo MODO
    (CHARF / PADF / EOSF). Tutta la struttura dell'output — dove finisce,
    quanti `=`, quale formula applicare — sta in questa tabella.

  - BLOCCO 1 — COSA C'E' LI'. Due teste le cui query leggono ADDR_A/ADDR_B
    invece del one-hot di posizione (stesso meccanismo dei selettori di
    life, sorgente diversa: l'indirizzo non e' cablato nella posizione,
    l'ha scritto la tabella del blocco 0) e fetchano i NIBBLE dei byte
    bersaglio: ogni token byte porta nibble alto e basso come due one-hot
    da 16. L'MLP e' la ROM A NIBBLE: ogni fase base64 legge DUE soli nibble
    (vedi rom_entry), quindi 16x16 = 256 neuroni AND a 3 vie per fase —
    1024 in tutto invece dei 65.536 della tavola sulle coppie di byte.

  - IL SINK E' UN VALORE, NON UN BUCO. Nelle altre macchine un selettore
    senza bersaglio si manda sul token inerte di posizione 0 perche' il
    fetch torni pulito a zero. Qui il token di posizione 0 e' il marker `E`,
    e gli si fanno portare I NIBBLE DI 0x00: l'ultimo gruppo monco chiede
    il byte che non c'e', fetcha lo zero, e le formule base64 — che il byte
    mancante lo vogliono proprio a 0 — tornano giuste per costruzione. Zero
    neuroni, zero fasi speciali.

  - OGNI TOKEN PORTA ESATTAMENTE DUE ONE-HOT A SCALA RAW. Corollario del
    punto sopra: se il marker porta due nibble, tutto il resto deve pesare
    uguale o la varianza post-LN slitta da riga a riga. I caratteri di
    output portano anch'essi i nibble di 0x00 — non li legge nessuno (le
    posizioni di output non sono mai indirizzate), servono a pareggiare la
    varianza. Ne esce gratis una proprieta' forte: TUTTI i token generati —
    caratteri, `=` ed EOS — hanno lo stesso identico embedding, quindi la
    macchina non puo' leggere il proprio output nemmeno per sbaglio: ogni
    riga ricalcola solo da posizione e L. Percio' verificare la trascrizione
    intera in un forward solo resta lecito anche a macchina mezza costruita,
    con PHASES parziale e le fasi spente che sbagliano (e' cosi' che questa
    macchina e' stata tirata su: prima l'echo, poi una fase, poi tutte).

Nessuna soglia e' scritta a tavolino: come per tutte le macchine del repo si
MISURANO sul forward numpy fedele coi pesi gia' arrotondati f16, e il
generatore rifiuta di scrivere il GGUF se un gap si sovrappone o se la
copertura (tutte le coppie (p, L), tutti i neuroni della ROM) non e'
esaustiva.

Uso:
    pip install gguf numpy
    python build_gguf.py                # produce base64.gguf
    ollama create base64 -f Modelfile
"""

import os

import gguf
import numpy as np

from ref_base64 import ALPHABET, PAD, encode_tokens, n_chars

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base64.gguf")

# fasi base64 cablate nella ROM: (0,) sola cabla un quarto dell'output ed e'
# il modo di far girare la macchina a meta' strada (la verifica conta le fasi
# spente a parte e non le pretende)
PHASES = (0, 1, 2, 3)

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


E_T = game("E")  # marker di inizio input E carattere alfabeto 4: doppio ruolo
SEMI_T = game(";")  # marker di fine input: la sua posizione codifica L
BYTE_T = [game(f"<B:{b:02X}>") for b in range(256)]
CHAR_T = [E_T if c == "E" else game(c) for c in ALPHABET]
PAD_T = game("=")
V = len(tokens)
TOK_OF_CHAR = {c: CHAR_T[i] for i, c in enumerate(ALPHABET)}
TOK_OF_CHAR[PAD] = PAD_T

# ========================== IPERPARAMETRI ====================================
LMAX = 48  # byte di input massimi
NOUT = n_chars(LMAX)  # 64 caratteri di output massimi
MAXPOS = 1 + LMAX + 1 + NOUT + 1  # 115: E + input + ; + output + EOS
HEAD_D = MAXPOS  # le chiavi sono one-hot di posizione: head_dim >= n_ctx
N_HEAD = 6  # blk0: 1 testa contenuto; blk1: 2 teste indirizzo; il resto inerte
N_EMBD = N_HEAD * HEAD_D  # 690
EPS = 1e-5
SEL = 60.0  # margine di saturazione dei selettori
QB = SEL * np.sqrt(HEAD_D)  # bias di query della testa contenuto
S = 12.0  # peso dei gate AND
RAW = 9.0  # boost dei one-hot di embedding
AW = 0.1  # scala di scrittura degli indirizzi nel residuo
DSCALE = 0.1  # scala di scrittura di fasi, modi e uscita della ROM


# layout del residuo
def blk(n, c=[0]):
    s = c[0]
    c[0] += n
    return list(range(s, s + n))


_c = [0]
BHI = blk(16, _c)  # one-hot nibble alto del token
BLO = blk(16, _c)  # one-hot nibble basso del token
BAL = blk(1, _c)[0]  # zavorra del `;` (che di nibble ne porta zero)
SEMI = blk(1, _c)[0]  # la dim del `;`: la chiave della testa contenuto
LPOS = blk(MAXPOS, _c)  # fetch blk0: one-hot della posizione del `;`
AA = blk(MAXPOS, _c)  # indirizzo del primo byte da fetchare (scritto da MLP0)
AB = blk(MAXPOS, _c)  # indirizzo del secondo byte da fetchare
FAH = blk(16, _c)  # fetch blk1 testa A: nibble alto
FAL = blk(16, _c)  # fetch blk1 testa A: nibble basso
FBH = blk(16, _c)  # fetch blk1 testa B: nibble alto
FBL = blk(16, _c)  # fetch blk1 testa B: nibble basso
PH = blk(4, _c)  # fase della riga: PH0..PH3
CHARF = blk(1, _c)[0]  # modo della riga: carattere base64
PADF = blk(1, _c)[0]  # modo della riga: padding `=`
EOSF = blk(1, _c)[0]  # modo della riga: fine output
OUT64 = blk(64, _c)  # uscita della ROM: one-hot del valore base64
POSB = blk(MAXPOS, _c)
assert _c[0] <= N_EMBD, f"layout {_c[0]} non entra in {N_EMBD}"

# ==================== GEOMETRIA DELLE RIGHE ==================================
# Il nastro: 0 = `E`, 1..L = byte, 1+L = `;`, 2+L .. 1+L+NC = caratteri,
# 2+L+NC = EOS (NC = n_chars(L)). La riga p predice il token in p+1, quindi
# le righe di generazione sono p in [1+L, 1+L+NC] e k = p-1-L e' l'indice
# di output: k < NC carattere o padding, k == NC l'EOS. Si comincia dalla
# riga del `;` (la prima che vede dove finisce l'input).
CHAR, PADM, EOSM = 0, 1, 2


def gen_rows(L):
    return range(1 + L, 2 + L + n_chars(L))


def row_plan(L, k):
    """Cosa fa la riga che genera l'output k di un input lungo L: il modo,
    gli indirizzi dei due byte da fetchare (0 = sink, cioe' il byte che non
    c'e' e vale 0) e la fase base64. E' esattamente cio' che la tabella
    (p, L) scrive nel residuo."""
    if k == n_chars(L):
        return (EOSM, 0, 0, None)
    g, ph = divmod(k, 4)
    i0, i1, i2 = 3 * g, 3 * g + 1, 3 * g + 2
    pos = lambda i: 1 + i if i < L else 0
    if ph == 0:  # b0 >> 2: basta il byte 0 del gruppo
        return (CHAR, pos(i0), 0, 0)
    if ph == 1:  # (b0 & 3) << 4 | b1 >> 4
        return (CHAR, pos(i0), pos(i1), 1)
    if ph == 2:  # (b1 & 15) << 2 | b2 >> 6; senza b1 il gruppo e' monco
        return (CHAR, pos(i1), pos(i2), 2) if i1 < L else (PADM, 0, 0, None)
    # b2 & 63; senza b2 il gruppo e' monco
    return (CHAR, pos(i2), 0, 3) if i2 < L else (PADM, 0, 0, None)


# la tabella del blocco 0: un neurone per coppia (L, k) possibile
NEUR = {}
for _L in range(LMAX + 1):
    for _k in range(n_chars(_L) + 1):
        NEUR[(_L, _k)] = len(NEUR)

# ======================= LA ROM A NIBBLE =====================================
# Ogni fase base64 legge DUE soli nibble: e' questa la decomposizione che
# tiene l'MLP a 256 neuroni per fase invece di 65.536 sulle coppie di byte.
# ROM_SLOTS dice da quali due blocchi di fetch arrivano; rom_entry e' la
# formula, gia' scritta in nibble.
ROM_SLOTS = {
    0: (FAH, FAL),  # nibble alto e basso di b0
    1: (FAL, FBH),  # nibble basso di b0, alto di b1
    2: (FAL, FBH),  # nibble basso di b1, alto di b2
    3: (FAH, FAL),  # nibble alto e basso di b2
}


def rom_entry(ph, x, y):
    """Valore base64 (0..63) scritto dal neurone (fase, nibble x, nibble y)."""
    if ph == 0:  # b0 >> 2            = hi(b0) << 2 | lo(b0) >> 2
        return (x << 2) | (y >> 2)
    if ph == 1:  # (b0 & 3) << 4 | b1 >> 4  = (lo(b0) & 3) << 4 | hi(b1)
        return ((x & 3) << 4) | y
    if ph == 2:  # (b1 & 15) << 2 | b2 >> 6 = lo(b1) << 2 | hi(b2) >> 2
        return (x << 2) | (y >> 2)
    # b2 & 63                          = (hi(b2) & 3) << 4 | lo(b2)
    return ((x & 3) << 4) | y


def rom_key(data, L, k):
    """Il neurone della ROM che DEVE sparare sulla riga (L, k): i due nibble
    che la riga fetcha davvero (il byte assente e' lo zero del sink)."""
    mode, a, b, ph = row_plan(L, k)
    assert mode == CHAR
    va = data[a - 1] if a > 0 else 0
    vb = data[b - 1] if b > 0 else 0
    if ph in (0, 3):  # i due nibble vengono dallo stesso byte, dalla testa A
        return (ph, va >> 4, va & 15)
    return (ph, va & 15, vb >> 4)  # nibble basso di A, alto di B


ROM = {}
for _ph in PHASES:
    for _x in range(16):
        for _y in range(16):
            ROM[(_ph, _x, _y)] = len(ROM)

NFF = max(len(NEUR), 4 * 256)  # n_ff e' unico nel gguf: i due MLP lo dividono
# (l'MLP del blocco 1 tiene i 1024 neuroni della ROM piena, il resto e' morto)

# ============================== PESI ========================================
# Ogni token porta ESATTAMENTE due one-hot a scala RAW: la varianza propria e'
# identica su ogni riga e una soglia sola regge tutta la macchina. I byte
# portano i loro due nibble; tutti gli altri portano i nibble di 0x00 — e per
# il marker `E`, che sta in posizione 0 ed e' il bersaglio dei fetch senza
# byte, quello zero NON e' zavorra: e' il valore che le formule base64 danno
# al byte mancante.
wte = np.zeros((V, N_EMBD), dtype=np.float32)
for b in range(256):
    wte[BYTE_T[b], BHI[b >> 4]] = RAW
    wte[BYTE_T[b], BLO[b & 15]] = RAW
for t in [E_T, PAD_T, EOS] + CHAR_T:
    wte[t, BHI[0]] = RAW
    wte[t, BLO[0]] = RAW
wte[SEMI_T, SEMI] = RAW  # il `;` e' l'unico token indirizzato per contenuto
wte[SEMI_T, BAL] = RAW  # e la zavorra gli pareggia la varianza degli altri
wpe = np.zeros((MAXPOS, N_EMBD), dtype=np.float32)
for p in range(MAXPOS):
    wpe[p, POSB[p]] = 1.0

# ---- blocco 0, attention: la testa contenuto sul `;` -------------------------
# q: bias costante sulla coordinata 0 della testa 0 (ogni riga cerca il `;`);
# k: la coordinata 0 legge SEMI (solo il `;` ce l'ha accesa);
# v: esporta il POSB della riga attesa -> LPOS. Le righe prima del `;` non lo
# vedono (mask causale): attenzione uniforme, smear in LPOS di righe che non
# generano mai. Le altre 5 teste sono a zero (V nullo: inerti).
Wqkv0 = np.zeros((3 * N_EMBD, N_EMBD), dtype=np.float32)
bqkv0 = np.zeros(3 * N_EMBD, dtype=np.float32)
vrow = 2 * N_EMBD
bqkv0[0 * HEAD_D + 0] = QB
Wqkv0[N_EMBD + 0 * HEAD_D + 0, SEMI] = 1.0
for j in range(MAXPOS):
    Wqkv0[vrow + 0 * HEAD_D + j, POSB[j]] = 1.0
Wattn0 = np.zeros((N_EMBD, N_EMBD), dtype=np.float32)
battn0 = np.zeros(N_EMBD, dtype=np.float32)
for j in range(MAXPOS):
    Wattn0[LPOS[j], 0 * HEAD_D + j] = 1.0

# ---- blocco 0, MLP: la tabella (p, L) ----------------------------------------
Wup0 = np.zeros((NFF, N_EMBD), dtype=np.float32)
bup0 = np.zeros(NFF, dtype=np.float32)
Wdown0 = np.zeros((N_EMBD, NFF), dtype=np.float32)
bdown0 = np.zeros(N_EMBD, dtype=np.float32)


def build_mlp0(thr, WP, WL):
    """Un neurone per coppia (p, L): AND(POSB[p], LPOS[1+L]). Quando scatta
    scrive indirizzi, fase e modo della riga. I due addendi vivono a scale
    post-LN diverse (il POSB e' un 1.0 del wpe, il fetch LPOS e' lo spike
    esportato dal `;`): i pesi per-addendo WP/WL, misurati, li normalizzano a
    ~1 ciascuno — altrimenti l'addendo grosso da solo batte la coppia minima
    e l'AND non e' un AND."""
    for (L, k), n in NEUR.items():
        p = 1 + L + k
        Wup0[n, POSB[p]] = S * WP
        Wup0[n, LPOS[1 + L]] = S * WL
        bup0[n] = -S * thr
        mode, a, b, ph = row_plan(L, k)
        Wdown0[AA[a], n] = AW
        Wdown0[AB[b], n] = AW
        Wdown0[{CHAR: CHARF, PADM: PADF, EOSM: EOSF}[mode], n] = DSCALE
        if mode == CHAR:
            Wdown0[PH[ph], n] = DSCALE


# ---- blocco 1, attention: i due selettori a indirizzo ------------------------
Wqkv1 = np.zeros((3 * N_EMBD, N_EMBD), dtype=np.float32)
bqkv1 = np.zeros(3 * N_EMBD, dtype=np.float32)
Wattn1 = np.zeros((N_EMBD, N_EMBD), dtype=np.float32)
battn1 = np.zeros(N_EMBD, dtype=np.float32)
# Le due teste sono identiche e scaricano entrambe i due nibble del byte che
# fetchano. FBL (nibble basso della testa B) non lo legge nessuna fase — la
# decomposizione di rom_entry non ne ha mai bisogno — ma tenerlo costa 16 dim
# e rende le due teste simmetriche: stesso contributo di varianza, una soglia
# sola per i nibble di entrambe.
HEADS1 = ((1, AA, FAH, FAL), (2, AB, FBH, FBL))
for h, _, FH, FL in HEADS1:
    for x in range(16):
        Wattn1[FH[x], h * HEAD_D + x] = 1.0
        Wattn1[FL[x], h * HEAD_D + 16 + x] = 1.0


def build_attn1(GQ):
    """Le query leggono ADDR invece del POSB: l'indirizzo l'ha scritto la
    tabella (p, L). Il guadagno GQ compensa la scala post-LN degli indirizzi
    (misurata) perche' lo score sul bersaglio resti >= SEL."""
    Wqkv1[:] = 0.0
    for h, ADDR, _, _ in HEADS1:
        for j in range(MAXPOS):
            Wqkv1[h * HEAD_D + j, ADDR[j]] = GQ
            Wqkv1[N_EMBD + h * HEAD_D + j, POSB[j]] = 1.0
        for x in range(16):
            Wqkv1[vrow + h * HEAD_D + x, BHI[x]] = 1.0
            Wqkv1[vrow + h * HEAD_D + 16 + x, BLO[x]] = 1.0


# ---- blocco 1, MLP: la ROM a nibble ------------------------------------------
Wup1 = np.zeros((NFF, N_EMBD), dtype=np.float32)
bup1 = np.zeros(NFF, dtype=np.float32)
Wdown1 = np.zeros((N_EMBD, NFF), dtype=np.float32)
bdown1 = np.zeros(N_EMBD, dtype=np.float32)


def build_mlp1(thr, WPH, WN):
    """Un neurone per (fase, nibble, nibble): AND a 3 vie su PH e i due
    blocchi di fetch. Scrive il one-hot del valore base64. Pesi per-addendo
    misurati come per la tabella (la fase esce dall'MLP del blocco 0, i
    nibble dall'attention del blocco 1: scale diverse)."""
    for (ph, x, y), n in ROM.items():
        SA, SB = ROM_SLOTS[ph]
        Wup1[n, PH[ph]] = S * WPH
        Wup1[n, SA[x]] = S * WN
        Wup1[n, SB[y]] = S * WN
        bup1[n] = -S * thr
        Wdown1[OUT64[rom_entry(ph, x, y)], n] = DSCALE


# ---- unembedding: famiglia per modo, valore dalla ROM ------------------------
# Ogni token legge il flag della SUA famiglia (l'altra lo vede spento, cioe'
# leggermente negativo post-LN): il bonus sceglie CHE COSA e' questa riga —
# un carattere, un `=`, la fine. Dentro la famiglia dei caratteri decide il
# one-hot OUT64 scritto dalla ROM; il bonus e' costante sui 64 e non tocca
# l'ordinamento. E' il rimedio di life, con tre famiglie invece di due.
Wout = np.zeros((V, N_EMBD), dtype=np.float32)
for v in range(64):
    Wout[CHAR_T[v], OUT64[v]] = 1.0


def set_bonus(B):
    for t in CHAR_T:
        Wout[t, CHARF] = B
    Wout[PAD_T, PADF] = B
    Wout[EOS, EOSF] = B


gamma1a = np.ones(N_EMBD, dtype=np.float32)
beta1a = np.zeros(N_EMBD, dtype=np.float32)
gamma1m = gamma1a.copy()
beta1m = beta1a.copy()
gamma2a = gamma1a.copy()
beta2a = beta1a.copy()
gamma2m = gamma1a.copy()
beta2m = beta1a.copy()
gammaf = gamma1a.copy()
betaf = beta1a.copy()

BLOCKS = (
    dict(Wqkv=Wqkv0, bqkv=bqkv0, Wattn=Wattn0, battn=battn0,
         Wup=Wup0, bup=bup0, Wdown=Wdown0, bdown=bdown0,
         ga=gamma1a, ba=beta1a, gm=gamma1m, bm=beta1m),
    dict(Wqkv=Wqkv1, bqkv=bqkv1, Wattn=Wattn1, battn=battn1,
         Wup=Wup1, bup=bup1, Wdown=Wdown1, bdown=bdown1,
         ga=gamma2a, ba=beta2a, gm=gamma2m, bm=beta2m),
)


def quantize_f16():
    for a in (wte, wpe, Wqkv0, Wattn0, Wup0, Wdown0,
              Wqkv1, Wattn1, Wup1, Wdown1, Wout):
        a[:] = a.astype(np.float16).astype(np.float32)


# =================== FORWARD FEDELE (stessi tensori del GGUF) ================
def ln_rows(X, g, b):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + EPS) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def forward_all(ids):
    """Ritorna (trace, XLn, logits): trace[i] ha gli intermedi del blocco i
    (A = post attn_norm, M = post ffn_norm, att = pesi di attention)."""
    T = len(ids)
    X = wte[ids] + wpe[:T]
    mask = np.triu(np.ones((T, T)), 1).astype(bool)
    trace = []
    for bw in BLOCKS:
        A = ln_rows(X, bw["ga"], bw["ba"])
        qkv = A @ bw["Wqkv"].T + bw["bqkv"]
        q, k, v = qkv[:, :N_EMBD], qkv[:, N_EMBD:vrow], qkv[:, vrow:]
        att = np.zeros((T, N_EMBD))
        attw = np.zeros((N_HEAD, T, T))
        for h in range(N_HEAD):
            sl = slice(h * HEAD_D, (h + 1) * HEAD_D)
            sc = (q[:, sl] @ k[:, sl].T) / np.sqrt(HEAD_D)
            sc[mask] = -1e9
            Aw = np.exp(sc - sc.max(1, keepdims=True))
            Aw /= Aw.sum(1, keepdims=True)
            attw[h] = Aw
            att[:, sl] = Aw @ v[:, sl]
        X = X + att @ bw["Wattn"].T + bw["battn"]
        M = ln_rows(X, bw["gm"], bw["bm"])
        X = X + gelu(M @ bw["Wup"].T + bw["bup"]) @ bw["Wdown"].T + bw["bdown"]
        trace.append(dict(A=A, M=M, att=attw))
    XLn = ln_rows(X, gammaf, betaf)
    return trace, XLn, XLn @ Wout.T


# ==================== RIFERIMENTO: una trascrizione completa =================
def full_ids(data):
    """Trascrizione completa: prompt + risposta attesa dal riferimento."""
    ids = [E_T] + [BYTE_T[b] for b in data] + [SEMI_T]
    ids += [TOK_OF_CHAR[c] for c in encode_tokens(data)] + [EOS]
    return ids


def cover_cases():
    """Input costruiti perche' OGNI neurone della ROM sia visto sparare: per
    ogni fase cablata, tutte e 256 le combinazioni dei suoi due nibble,
    piazzate 16 per volta nei 16 gruppi di un input da 48 byte. E' il
    grigliato di francobolli di life, tradotto in byte."""
    out = []
    combos = [(x, y) for x in range(16) for y in range(16)]
    for ph in PHASES:
        for i in range(0, 256, 16):
            d = bytearray(LMAX)
            for g, (x, y) in enumerate(combos[i : i + 16]):
                if ph == 0:
                    d[3 * g] = (x << 4) | y
                elif ph == 1:
                    d[3 * g] = (d[3 * g] & 0xF0) | x
                    d[3 * g + 1] = (y << 4) | (d[3 * g + 1] & 0x0F)
                elif ph == 2:
                    d[3 * g + 1] = (d[3 * g + 1] & 0xF0) | x
                    d[3 * g + 2] = (y << 4) | (d[3 * g + 2] & 0x0F)
                else:
                    d[3 * g + 2] = (x << 4) | y
            out.append(bytes(d))
    return out


def cases(seed, n_random):
    """Input di calibrazione/verifica: i francobolli (copertura della ROM),
    TUTTE le lunghezze 0..LMAX (copertura della tabella (p, L), e con esse
    tutti e tre i resti mod 3, cioe' tutti i modi di padding), i casi limite
    e fuzz casuale."""
    rng = np.random.default_rng(seed)
    datas = cover_cases()
    datas += [bytes(rng.integers(0, 256, L).tolist()) for L in range(LMAX + 1)]
    datas += [bytes(LMAX), bytes([0xFF] * LMAX), bytes(range(LMAX)), b"\x00"]
    for _ in range(n_random):
        L = int(rng.integers(0, LMAX + 1))
        datas.append(bytes(rng.integers(0, 256, L).tolist()))
    return datas


# ======================= CALIBRAZIONE ========================================
def calibrate_gate(seed=11, n_random=8):
    """Soglia e pesi della tabella (p, L). Due passate sugli stessi forward:
    la prima misura le scale post-LN dei due addendi sulle righe giuste e
    fissa WP/WL = 1/minimo (ogni addendo normalizzato vale >= 1); la seconda
    calcola il gap dei gate PESATI: coppia combaciante contro il massimo di
    ogni altro (neurone, riga) — righe del prompt incluse: un gate che
    scatta con un solo input non e' un AND. Rifiuta se una coppia (p, L)
    non e' mai stata vista scattare."""
    PP = np.zeros((NFF, N_EMBD), dtype=np.float32)
    PL = np.zeros((NFF, N_EMBD), dtype=np.float32)
    for (L, k), n in NEUR.items():
        PP[n, POSB[1 + L + k]] = 1.0
        PL[n, LPOS[1 + L]] = 1.0
    meas = []
    vp_min = vl_min = np.inf
    covered = set()
    for data in cases(seed, n_random):
        L = len(data)
        tr, _, _ = forward_all(full_ids(data))
        M0 = tr[0]["M"]
        rows = np.array(list(gen_rows(L)))
        neur = np.array([NEUR[(L, p - 1 - L)] for p in rows])
        covered.update((L, p - 1 - L) for p in rows)
        sp, sl = M0 @ PP.T, M0 @ PL.T  # [T, NFF]
        vp_min = min(vp_min, sp[rows, neur].min())
        vl_min = min(vl_min, sl[rows, neur].min())
        meas.append((sp, sl, rows, neur))
    print(f"  copertura tabella: {len(covered)}/{len(NEUR)} coppie (p, L)")
    assert len(covered) == len(NEUR), "copertura incompleta: manca una lunghezza"
    assert vp_min > 0 and vl_min > 0
    WP, WL = 1.0 / vp_min, 1.0 / vl_min
    must_min, notmax = np.inf, -np.inf
    for sp, sl, rows, neur in meas:
        sums = WP * sp + WL * sl
        sums[:, len(NEUR) :] = -np.inf  # neuroni morti: non sono della tabella
        must_min = min(must_min, sums[rows, neur].min())
        sums[rows, neur] = -np.inf
        notmax = max(notmax, sums.max())
    print(f"  scale: vP>={vp_min:.2f} vL>={vl_min:.2f}; tabella pesata: "
          f"deve>={must_min:5.2f}  non_deve<={notmax:5.2f}  gap "
          f"{'OK' if notmax < must_min else 'SOVRAPPOSTO'}")
    assert notmax < must_min, "gap sovrapposto"
    return float((notmax + must_min) / 2), float(WP), float(WL)


def calibrate_gq(seed=17, n_random=6):
    """Guadagno delle query del blocco 1: gli indirizzi scritti dalla tabella
    e le chiavi di posizione vivono a scale post-LN diverse da riga a riga;
    GQ si dimensiona sul minimo dei due, cosi' lo score al bersaglio e'
    sempre >= SEL. (La saturazione effettiva la certifica check_saturation.)"""
    vmin_a, vmin_k = np.inf, np.inf
    for data in cases(seed, n_random):
        L = len(data)
        tr, _, _ = forward_all(full_ids(data))
        A1 = tr[1]["A"]
        for p in gen_rows(L):
            _, a, b, _ = row_plan(L, p - 1 - L)
            vmin_a = min(vmin_a, A1[p, AA[a]], A1[p, AB[b]])
            vmin_k = min(vmin_k, A1[a, POSB[a]], A1[b, POSB[b]])
    assert vmin_a > 0 and vmin_k > 0, "indirizzo o chiave non positivi post-LN"
    GQ = SEL * np.sqrt(HEAD_D) / (vmin_a * vmin_k)
    print(f"  indirizzi: Amin={vmin_a:.3f} Kmin={vmin_k:.3f} -> GQ={GQ:.0f}")
    return float(GQ)


def check_saturation(seed=19, n_random=6, wmin=0.99):
    """Certifica che ogni selettore, contenuto e indirizzo, sia saturo: peso
    di attention sul bersaglio >= wmin su ogni riga di generazione."""
    worst = 1.0
    for data in cases(seed, n_random):
        L = len(data)
        tr, _, _ = forward_all(full_ids(data))
        att0, att1 = tr[0]["att"], tr[1]["att"]
        for p in gen_rows(L):
            _, a, b, _ = row_plan(L, p - 1 - L)
            worst = min(worst, att0[0, p, 1 + L], att1[1, p, a], att1[2, p, b])
    print(f"  saturazione: peso minimo sul bersaglio {worst:.4f}")
    assert worst >= wmin, "selettore non saturo"


def calibrate_rom(seed=13, n_random=8):
    """Soglia e pesi della ROM, come per la tabella ma su tre addendi: la
    fase (scritta dall'MLP del blocco 0) e i due nibble (scritti
    dall'attention del blocco 1). Il caso duro che il gap deve separare e'
    il 2-su-3: su una riga di fase 0 il neurone di fase 1 che chiede il
    nibble giusto e lo zero del sink prende due addendi su tre."""
    PP = np.zeros((NFF, N_EMBD), dtype=np.float32)
    PN = np.zeros((NFF, N_EMBD), dtype=np.float32)
    for (ph, x, y), n in ROM.items():
        SA, SB = ROM_SLOTS[ph]
        PP[n, PH[ph]] = 1.0
        PN[n, SA[x]] += 1.0
        PN[n, SB[y]] += 1.0
    meas = []
    vph_min = vn_min = np.inf
    covered = set()
    for data in cases(seed, n_random):
        L = len(data)
        tr, _, _ = forward_all(full_ids(data))
        M1 = tr[1]["M"]
        rows, neur = [], []
        for p in gen_rows(L):
            k = p - 1 - L
            if row_plan(L, k)[0] != CHAR:
                continue
            key = rom_key(data, L, k)
            if key not in ROM:  # fase non ancora cablata
                continue
            rows.append(p)
            neur.append(ROM[key])
            covered.add(key)
        rows, neur = np.array(rows, int), np.array(neur, int)
        sp, sn = M1 @ PP.T, M1 @ PN.T
        if len(rows):
            vph_min = min(vph_min, sp[rows, neur].min())
            vn_min = min(vn_min, (sn[rows, neur] / 2).min())
        meas.append((sp, sn, rows, neur))
    print(f"  copertura rom: {len(covered)}/{len(ROM)} neuroni")
    assert len(covered) == len(ROM), "copertura incompleta: rivedi i francobolli"
    assert vph_min > 0 and vn_min > 0
    WPH, WN = 1.0 / vph_min, 1.0 / vn_min
    must_min, notmax = np.inf, -np.inf
    for sp, sn, rows, neur in meas:
        sums = WPH * sp + WN * sn
        sums[:, len(ROM) :] = -np.inf  # neuroni morti: non sono della ROM
        if len(rows):
            must_min = min(must_min, sums[rows, neur].min())
            sums[rows, neur] = -np.inf
        notmax = max(notmax, sums.max())
    print(f"  scale: vPH>={vph_min:.2f} vN>={vn_min:.2f}; rom pesata: "
          f"deve>={must_min:5.2f}  non_deve<={notmax:5.2f}  gap "
          f"{'OK' if notmax < must_min else 'SOVRAPPOSTO'}")
    assert notmax < must_min, "gap sovrapposto"
    return float((notmax + must_min) / 2), float(WPH), float(WN)


def calibrate_bonus(seed=23, n_random=6):
    """Bonus di famiglia, misurato coi bonus ancora a zero: deve dominare il
    massimo logit in gioco (D) partendo dal minimo flag acceso (P), su ogni
    riga di generazione. B = 3*Dmax/Pmin, uno solo per le tre famiglie."""
    dmax, pmin = 0.0, np.inf
    for data in cases(seed, n_random):
        L = len(data)
        _, XLn, logits = forward_all(full_ids(data))
        for p in gen_rows(L):
            mode = row_plan(L, p - 1 - L)[0]
            flag = {CHAR: CHARF, PADM: PADF, EOSM: EOSF}[mode]
            pmin = min(pmin, XLn[p, flag])
            dmax = max(dmax, np.abs(logits[p]).max())
            for f in (CHARF, PADF, EOSF):
                assert f == flag or XLn[p, f] < 0, "flag di modo fuori posto"
    assert pmin > 0
    B = 3.0 * max(dmax, 1.0) / pmin
    print(f"  bonus di famiglia: Dmax={dmax:.2f} Pmin={pmin:.2f} -> B={B:.1f}")
    return B


# ============================ VERIFICA =======================================
def verify(seed=500, n_random=25):
    """Trascrizioni complete, ogni token generato contro il riferimento, con
    il conto separato per tipo di riga. Verificare la trascrizione intera
    equivale a generarla token per token: il decoding e' greedy e — visto che
    tutti i token generati hanno lo stesso embedding — nessuna riga puo'
    leggere l'output delle precedenti, nemmeno quando sbagliano."""
    per = {}
    bad = []
    for data in cases(seed, n_random):
        L = len(data)
        ids = full_ids(data)
        _, _, logits = forward_all(ids)
        for p in gen_rows(L):
            k = p - 1 - L
            mode, _, _, ph = row_plan(L, k)
            key = {CHAR: f"fase {ph}", PADM: "padding", EOSM: "eos"}[mode]
            ok, tot = per.get(key, (0, 0))
            if int(logits[p].argmax()) == ids[p + 1]:
                per[key] = (ok + 1, tot + 1)
            else:
                per[key] = (ok, tot + 1)
                bad.append((L, k, tokens[int(logits[p].argmax())], tokens[ids[p + 1]]))
    checked = ["padding", "eos"] + [f"fase {ph}" for ph in PHASES]
    for key in sorted(per):
        ok, tot = per[key]
        mark = "" if key in checked else "  (non cablata)"
        print(f"  {key:9}: {ok}/{tot} ({100 * ok / tot:5.1f}%){mark}")
    if bad:
        print(f"  primi errori: {bad[:4]}")
    return (sum(per[k][0] for k in checked), sum(per[k][1] for k in checked))


# ============================== MAIN =========================================
if __name__ == "__main__":
    print(
        f"base64 (fasi cablate {PHASES}), Lmax={LMAX}\n"
        f"n_embd={N_EMBD} n_head={N_HEAD} head_dim={HEAD_D} n_ff={NFF} "
        f"n_block=2 vocab={V} ctx={MAXPOS}"
    )
    quantize_f16()
    print("calibrazione (forward reale, pesi f16):")
    build_mlp0(*calibrate_gate())
    quantize_f16()
    build_attn1(calibrate_gq())
    quantize_f16()
    check_saturation()
    build_mlp1(*calibrate_rom())
    quantize_f16()
    set_bonus(calibrate_bonus())
    quantize_f16()
    print("verifica:")
    ok, tot = verify()
    print(f"  totale cablato: {ok}/{tot} token corretti ({100 * ok / tot:.1f}%)")
    assert ok == tot, "verifica fallita: non scrivo il GGUF"

    w = gguf.GGUFWriter(OUT, "gpt2")
    w.add_name("base64")
    w.add_context_length(MAXPOS)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(len(BLOCKS))
    w.add_feed_forward_length(NFF)
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
    for i, bw in enumerate(BLOCKS):
        w.add_tensor(f"blk.{i}.attn_norm.weight", bw["ga"])
        w.add_tensor(f"blk.{i}.attn_norm.bias", bw["ba"])
        w.add_tensor(f"blk.{i}.attn_qkv.weight", f16(bw["Wqkv"]))
        w.add_tensor(f"blk.{i}.attn_qkv.bias", bw["bqkv"])
        w.add_tensor(f"blk.{i}.attn_output.weight", f16(bw["Wattn"]))
        w.add_tensor(f"blk.{i}.attn_output.bias", bw["battn"])
        w.add_tensor(f"blk.{i}.ffn_norm.weight", bw["gm"])
        w.add_tensor(f"blk.{i}.ffn_norm.bias", bw["bm"])
        w.add_tensor(f"blk.{i}.ffn_up.weight", f16(bw["Wup"]))
        w.add_tensor(f"blk.{i}.ffn_up.bias", bw["bup"])
        w.add_tensor(f"blk.{i}.ffn_down.weight", f16(bw["Wdown"]))
        w.add_tensor(f"blk.{i}.ffn_down.bias", bw["bdown"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"scritto {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")
