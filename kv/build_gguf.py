#!/usr/bin/env python3
"""Genera `kv.gguf`: un KEY-VALUE STORE il cui database e' il contesto.

Il client accumula solo i comandi (vedi ref_kv.py per grammatica e
semantica); il modello risponde alle letture. Attenzione = lettura dello
stato, MLP = logica, decodifica greedy con logit saturi: come le altre
macchine, ma con due idee nuove rispetto a snake/wolf/life/doom:

  - PRIMO GGUF MULTI-BLOCCO (n_block = 3). "Trova l'occorrenza piu'
    recente di questa chiave" e' un match sul CONTENUTO, non a offset
    fisso: serve comporre due letture (raccogliere la chiave, poi
    confrontarla), e una sola passata di attenzione non puo' alimentare
    le proprie query con i propri output. La pipeline:
      blocco 1  gather lato archivio: ogni riga di fase 5..12 raccoglie
                (selettori relativi con offset PER FASE) i 4 byte della
                chiave del proprio turno -> GK, e la lettera comando ->
                GCMD (S/D = scrivibile: esclude i turni G dal match);
      blocco 2  gather lato query: la riga di decisione (fase 12) e le
                righe di emissione generate raccolgono la chiave del
                comando da servire -> QK;
      blocco 3  la testa di MATCH (milestone 2): punteggio = somma dei
                match per posizione + fase bersaglio + scrivibilita' +
                rampa di recenza; sink-by-default = la soglia di NF.
    L'MLP dell'ultimo blocco decide (OK/V/NF) e le priorita'
    dell'unembedding chiudono.

  - FETCH A PESO RIDOTTO (ROUTE < 1) PER DOMARE LA DERIVA TRA BLOCCHI.
    Con un blocco solo (snake) le query leggono sempre il residuo
    "pulito" post-embedding; qui le query dei blocchi 2 e 3 leggono righe
    gia' cariche di fetch. Un fetch instradato a peso 1 porta dimensioni
    post-LN ~40: la varianza della riga esplode e schiaccia i one-hot di
    posizione, e i selettori del blocco successivo annegano nel sink.
    Instradare tutti i fetch (zavorre comprese) a peso ROUTE=0.15 tiene
    tutte le righe nella stessa decade di varianza; le soglie per gruppo,
    misurate, assorbono il resto.

  - NIBBLE, NON BYTE, NELLE COORDINATE DEL MATCH. Il confronto esatto
    "4 byte contro 4 byte" in UNA softmax vorrebbe 4x256 coordinate q/k,
    ma head_dim (720, che paga gia' il one-hot di posizione) non basta.
    Un byte = (nibble alto, nibble basso): il punteggio somma 8 match da
    16-one-hot (4x32 coordinate) ed e' LOGICAMENTE IDENTICO (4 byte
    uguali <=> 8 nibble uguali) con lo stesso gap assoluto di un'unita'
    di selettore sull'avversario piu' vicino (chiave a 1 nibble di
    distanza). Anche l'emissione decodifica separabile alla snake:
    logit(<0xNN>) = FV_hi[N>>4] + FV_lo[N&15], ogni altro byte manca
    almeno un addendo.

  - ANCHE I PUNTEGGI DI ATTENZIONE SONO CALIBRATI, NON SCRITTI. Il match
    somma 10 addendi (8 nibble + fase + scrivibilita'), ciascuno pesato
    perche' valga U_MATCH logit: i pesi si ricavano da una passata di
    MISURA delle grandezze post-LN reali (query, chiavi, posizione,
    zavorra), perche' tre blocchi di fetch deformano ogni riga in modo
    diverso. Poi una seconda passata misura la distribuzione dei punteggi
    veri (match pieno / miglior concorrente / righe che devono affondare)
    e il bias del sink viene piazzato in mezzo al gap: il sink E' la
    soglia di NF. Gap sotto i margini di saturazione -> il generatore
    rifiuta di scrivere il GGUF.

MILESTONE 2 (questo stato): match completo per letture senza
sovrascritture (il fuzz vieta S ripetute sulla stessa chiave e D nei
casi random; la rampa di recenza e' cablata ma a delta=0, arriva alla
milestone 3). S k v ... G k -> V v; chiavi ignote, chiavi-esca a un
nibble/byte e G su chiavi mai scritte -> NF. Verifica: 100% su tutte le
risposte token per token, o niente GGUF.

Uso:
    pip install gguf numpy
    python build_gguf.py                 # produce kv.gguf
    ollama create kv -f Modelfile
"""

import os

import gguf
import numpy as np

from ref_kv import (
    BYTE0,
    D_T,
    EOS,
    FRAME_W,
    G_T,
    KEY_LEN,
    NF_T,
    NOOP,
    OK_T,
    PAD_T,
    PH_DEC,
    S_T,
    TS_T,
    V_T,
    VAL_LEN,
    VOCAB,
    byte_tok,
    encode_log,
    gen_commands,
    replay,
    reply_tokens,
    scores,
    tokens,
    ttypes,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv.gguf")
W = FRAME_W

# ========================== IPERPARAMETRI ====================================
N_BLOCK = 3
N_HEAD = 5
HEAD_D = 720  # >= MAXPOS + sink: il one-hot di posizione deve stare nella testa
MAXPOS = 714
SINKDIM = 714  # coordinata q/k del sink (dentro head_dim, fuori dal one-hot)
N_EMBD = N_HEAD * HEAD_D  # 3600
EPS = 1e-5
SEL = 80.0  # nitidezza dei selettori reali
SEL_SINK = 12.0  # punteggio costante del sink: vince solo dove non c'e' query
ALPHA = SEL * np.sqrt(HEAD_D)
S = 12.0  # pendenza delle porte AND
NEG = 3.0  # peso dei veti
RAW = 9.0  # boost anti-LN per i one-hot "propri" dei token
ROUTE = 0.15  # peso di instradamento dei fetch dei gather (vedi docstring)
FV_ROUTE = 1.5  # il fetch del match viaggia pesante: e' un ingresso dell'AND di V
MMETA = 3.0  # i one-hot dei metadati del match nel wpe (KPH/QPH/QON/rampa)
U_MATCH = 1600.0  # logit per nibble combaciante del match (8 addendi)
U_META = 400.0  # logit per fase e scrivibilita': il gap tra chiave giusta
# e chiave-esca e' SEMPRE 1 nibble (U_MATCH); i metadati bassi tengono
# basso il punteggio totale, e con lui lo spread di classe
RAMP_SLOT = 8.0  # logit di recenza per slot: tra due match pieni vince il
# piu' recente. La rampa deve dominare il rumore residuo di scala tra
# righe della stessa chiave (pochi logit, dopo l'equalizzazione delle
# zavorre), e la sua escursione stare nel gap da un nibble: il 9/10
# recente NON deve mai battere il 10/10 vecchio (53*8=424 << 1600)
MIN_SAT = 12.0  # margine minimo di saturazione della softmax (in logit)
MAX_STALE = 0.15  # tetto alla massa softmax dei match stantii (sovrascritti)

# capienza: prefisso + MAX_CMDS frame + 10 token di risposta nel contesto
MAX_CMDS = 53
assert W * (MAX_CMDS + 1) + VAL_LEN + 2 <= MAXPOS

# indici dei comandi nel one-hot CMD
CMD_IDX = {"S": 0, "G": 1, "D": 2}
STORABLE = ("S", "D")  # i turni che il match puo' bersagliare (milestone 2)


# ========================= LAYOUT DEL RESIDUO ================================
def blk(n, c=[0]):
    s = c[0]
    c[0] += n
    return list(range(s, s + n))


_c = [0]
HID = blk(16, _c)  # nibble alto del token byte
LOD = blk(16, _c)  # nibble basso del token byte
BYTEANY = blk(1, _c)[0]  # "sono un token byte" (per l'EOS dopo v7)
TOMBD = blk(1, _c)[0]  # "sono il tombstone <TS>"
CMDD = blk(3, _c)  # lettera comando (S/G/D)
VTOKD = blk(1, _c)[0]  # token <V>
OKTOKD = blk(1, _c)[0]  # token <OK>
NFTOKD = blk(1, _c)[0]  # token <NF>
PADBAL = blk(1, _c)[0]  # zavorra del <PAD>
BAL_SRC = blk(1, _c)[0]  # zavorra del <NOOP>: chiave del sink
GK = [blk(32, _c) for _ in range(KEY_LEN)]  # gather archivio: hi16+lo16 per k_i
GCMD = blk(3, _c)  # gather archivio: lettera comando del turno
QK = [blk(32, _c) for _ in range(KEY_LEN)]  # gather query: hi16+lo16 per k_i
FV = blk(33, _c)  # fetch del match: hi16+lo16+TOMB (milestone 2)
GBAL = blk(5, _c)  # zavorre per testa, blocco 1
QBAL = blk(4, _c)  # zavorre per testa, blocco 2
FBAL = blk(1, _c)[0]  # zavorra della testa di match
# metadati del match come one-hot DEDICATI nel wpe: le coordinate q/k del
# match leggono UNA dimensione ciascuna. Leggere i one-hot di POSB a
# mazzi di ~55 non funziona: le dim spente post-LN sono leggermente
# negative, i baseline si sommano su ENTRAMBI i lati e negativo x
# negativo regala migliaia di logit a concorrenti qualsiasi (la stessa
# trappola del sink-via-query di snake, §4 del suo README).
KPH = blk(8, _c)  # fase 5..12 della riga (lato chiave)
QPH = blk(8, _c)  # fase BERSAGLIO della riga (lato query)
QON = blk(1, _c)[0]  # "questa riga interroga" (scrivibilita' e rampa, lato query)
RDIV = blk(7, _c)  # slot // 8: la rampa di recenza, cifra grossa (M3)
RMOD = blk(8, _c)  # slot % 8: la rampa di recenza, cifra fine (M3)
OKD = blk(1, _c)[0]  # decisione: rispondi OK
NFD = blk(1, _c)[0]  # decisione: rispondi NF
VD = blk(1, _c)[0]  # decisione: rispondi V (milestone 2)
EOSD = blk(1, _c)[0]  # fine risposta
POSB = blk(MAXPOS, _c)  # one-hot di posizione (address bus)
assert _c[0] <= N_EMBD, f"layout {_c[0]} non entra in {N_EMBD}"


def hi(b):
    return b >> 4


def lo(b):
    return b & 15


def _match_target_phase(ph):
    """Fase bersaglio della riga di query del match: la decisione (fase
    12) legge v0 (fase 5), la riga di risposta a fase j emette v_j
    leggendo la fase 5+j. None per le fasi che non interrogano."""
    if ph == PH_DEC:
        return 5
    if ph <= 7:
        return 5 + ph
    return None


# ============================== PESI ========================================
wte = np.zeros((VOCAB, N_EMBD), dtype=np.float32)
for b in range(256):
    t = byte_tok(b)
    wte[t, HID[hi(b)]] = RAW
    wte[t, LOD[lo(b)]] = RAW
    wte[t, BYTEANY] = 1.0
for op, tok in (("S", S_T), ("G", G_T), ("D", D_T)):
    wte[tok, CMDD[CMD_IDX[op]]] = RAW
# i token "magri" (una sola dim propria) pesano quanto un token byte
# (due dim a RAW + BYTEANY): righe con la stessa energia entrano nella
# LN allo stesso modo, e le classi di riga del match restano allineate
NORM_FILL = float(np.sqrt(2 * RAW**2 + 1))
wte[TS_T, TOMBD] = NORM_FILL
wte[PAD_T, PADBAL] = NORM_FILL
wte[V_T, VTOKD] = NORM_FILL
wte[OK_T, OKTOKD] = NORM_FILL
wte[NF_T, NFTOKD] = NORM_FILL
wte[NOOP, BAL_SRC] = RAW

wpe = np.zeros((MAXPOS, N_EMBD), dtype=np.float32)
for p in range(MAXPOS):
    wpe[p, POSB[p]] = 1.0
    ph = p % W
    # il prefisso di <NOOP> (slot 0) NON partecipa al match: i suoi token
    # portano la chiave del sink, e un NOOP con anche il one-hot di fase
    # sommerebbe sink + bonus di fase superando ogni bersaglio legittimo
    if p >= W:
        if ph >= 5:
            wpe[p, KPH[ph - 5]] = MMETA
        tph = _match_target_phase(ph)
        if tph is not None:
            wpe[p, QPH[tph - 5]] = MMETA
            wpe[p, QON] = MMETA
    wpe[p, RDIV[(p // W) // 8]] = MMETA
    wpe[p, RMOD[(p // W) % 8]] = MMETA

# tensori per blocco
Wqkv = [np.zeros((3 * N_EMBD, N_EMBD), dtype=np.float32) for _ in range(N_BLOCK)]
bqkv = [np.zeros(3 * N_EMBD, dtype=np.float32) for _ in range(N_BLOCK)]
Wattn = [np.zeros((N_EMBD, N_EMBD), dtype=np.float32) for _ in range(N_BLOCK)]
battn = [np.zeros(N_EMBD, dtype=np.float32) for _ in range(N_BLOCK)]


def _selector_head(b, h, targets):
    """Testa-selettore del blocco b: chiavi = one-hot di posizione (+ sink
    su <NOOP>), query = onehot(bersaglio) per le posizioni in `targets`
    (dict posizione -> posizione bersaglia), sink-by-default altrove."""
    qrow, krow = h * HEAD_D, N_EMBD + h * HEAD_D
    for j in range(MAXPOS):
        Wqkv[b][krow + j, POSB[j]] = 1.0
    Wqkv[b][krow + SINKDIM, BAL_SRC] = 1.0
    bqkv[b][qrow + SINKDIM] = SEL_SINK * np.sqrt(HEAD_D)
    for p, t in targets.items():
        assert 0 <= t <= p, f"selettore non causale: {p}->{t}"
        Wqkv[b][qrow + t, POSB[p]] = ALPHA


def _nibble_value(b, h, extra=()):
    """La testa h del blocco b trasporta i nibble del token bersaglio
    (coordinate 0..31 della testa) + eventuali dim extra + la zavorra."""
    vrow = 2 * N_EMBD + h * HEAD_D
    for d in range(16):
        Wqkv[b][vrow + d, HID[d]] = 1.0
        Wqkv[b][vrow + 16 + d, LOD[d]] = 1.0
    base = 32
    for k, dim in enumerate(extra):
        Wqkv[b][vrow + base + k, dim] = 1.0
    Wqkv[b][vrow + base + len(extra), BAL_SRC] = 1.0
    return base + len(extra)  # coordinata della zavorra


def _route(b, h, pairs, weight=ROUTE):
    """attn_output del blocco b: coordinata della testa -> dim del residuo."""
    for coord, dim in pairs:
        Wattn[b][dim, h * HEAD_D + coord] = weight


# ---- blocco 1: gather lato archivio -----------------------------------------
# teste g0..g3: la riga di fase 5..12 raccoglie k_i (fase 1+i) del proprio
# turno; testa 4: la lettera comando (fase 0). Le fasi 0..4 vanno in sink.
for i in range(KEY_LEN):
    targets = {}
    for p in range(MAXPOS):
        ph = p % W
        if ph >= 5:
            targets[p] = p - (ph - 1 - i)
    _selector_head(0, i, targets)
    bal = _nibble_value(0, i)
    _route(0, i, [(d, GK[i][d]) for d in range(32)] + [(bal, GBAL[i])])
targets = {p: p - (p % W) for p in range(MAXPOS) if p % W >= 5}
_selector_head(0, 4, targets)
vrow = 2 * N_EMBD + 4 * HEAD_D
for k in range(3):
    Wqkv[0][vrow + k, CMDD[k]] = 1.0
Wqkv[0][vrow + 3, BAL_SRC] = 1.0
_route(0, 4, [(k, GCMD[k]) for k in range(3)] + [(3, GBAL[4])])

# ---- blocco 2: gather lato query ---------------------------------------------
# teste q0..q3: la riga di decisione (fase 12) raccoglie k_i del proprio
# turno; le righe di risposta generate (fasi 0..7) raccolgono k_i del
# comando appena concluso (il turno precedente). Nel prefisso il bersaglio
# cadrebbe prima dell'inizio: sink.
for i in range(KEY_LEN):
    targets = {}
    for p in range(MAXPOS):
        ph = p % W
        if ph == PH_DEC:
            targets[p] = p - (11 - i)
        elif ph <= 7:
            t = p - (12 + ph - i)
            if t >= 0:
                targets[p] = t
    _selector_head(1, i, targets)
    bal = _nibble_value(1, i)
    _route(1, i, [(d, QK[i][d]) for d in range(32)] + [(bal, QBAL[i])])
# testa 4 del blocco 2: inutilizzata (q=k=v=0: attenzione uniforme su v nulli)

# ---- blocco 3: la testa di match ---------------------------------------------
# Coordinate della testa (head_dim 720): 0..127 i nibble della chiave
# (32 per slot), 128..135 il one-hot di fase 5..12, 136 la scrivibilita'
# (S/D), 137 la rampa di recenza, 714 il sink. Le CHIAVI sono statiche
# (leggono GK/POSB/GCMD a peso 1); le QUERY sono parametriche: i pesi
# per componente arrivano dalla passata di misura (build_match), il bias
# del sink dalla misura dei punteggi (vedi main).
MC_PH, MC_ST, MC_RAMP, MC_VETO = 128, 136, 137, 138
VETO_U = 5.0  # unita' di veto sui v-slot dei turni G (il <PAD> li marca)

qrow, krow = 0 * HEAD_D, N_EMBD + 0 * HEAD_D
# chiavi: i turni S/G/D espongono chiave raccolta, fase e scrivibilita'.
# Ogni coordinata legge UNA dimensione (vedi il commento nel layout).
# Il <PAD> — che esiste SOLO nei v-slot delle G — alimenta la coordinata
# di veto: un turno G con la stessa chiave non deve mai fare da archivio
# (ne' i suoi pad alla sua stessa decisione), quindi vale -VETO_U unita'.
for i in range(KEY_LEN):
    for d in range(32):
        Wqkv[2][krow + 32 * i + d, GK[i][d]] = 1.0
for x in range(8):
    Wqkv[2][krow + MC_PH + x, KPH[x]] = 1.0
Wqkv[2][krow + MC_ST, GCMD[CMD_IDX["S"]]] = 1.0
Wqkv[2][krow + MC_ST, GCMD[CMD_IDX["D"]]] = 1.0
Wqkv[2][krow + MC_VETO, PADBAL] = 1.0
Wqkv[2][krow + SINKDIM, BAL_SRC] = 1.0
bal = _nibble_value(2, 0, extra=(TOMBD,))
_route(2, 0, [(d, FV[d]) for d in range(33)] + [(bal, FBAL)], weight=FV_ROUTE)


def build_match(w_nib, w_ph, w_st, w_veto, ramp_unit):
    """Le query del match, coi pesi per componente misurati: un addendo
    combaciante deve valere U_MATCH logit qualunque sia la componente.
    ramp_unit = logit per slot di recenza / U_MATCH, sul lato chiave con
    lo slot fattorizzato in cifra grossa e fine (one-hot puliti)."""
    for i in range(KEY_LEN):
        for d in range(32):
            Wqkv[2][qrow + 32 * i + d, QK[i][d]] = w_nib
    for x in range(8):
        Wqkv[2][qrow + MC_PH + x, QPH[x]] = w_ph
    Wqkv[2][qrow + MC_ST, QON] = w_st
    Wqkv[2][qrow + MC_VETO, QON] = -w_veto
    Wqkv[2][qrow + MC_RAMP, QON] = w_ph
    for j in range(7):
        Wqkv[2][krow + MC_RAMP, RDIV[j]] = ramp_unit * 8 * j
    for j in range(8):
        Wqkv[2][krow + MC_RAMP, RMOD[j]] = ramp_unit * j


def set_sink(sink_logits, kbal):
    """Il bias del sink: punteggio costante = soglia di NF, piazzato dal
    main in mezzo al gap misurato (kbal = valore post-LN della zavorra
    del <NOOP> sul blocco 3)."""
    bqkv[2][qrow + SINKDIM] = sink_logits * np.sqrt(HEAD_D) / kbal

# ---- MLP (solo ultimo blocco) ------------------------------------------------
# neurone = (dims_positivi, dims_negativi, dim_out, gruppo). GCMD e' un
# one-hot raccolto su TUTTE le righe di fase 5..12 del turno: i neuroni di
# decisione scattano anche li', ma solo la riga di fase 12 viene campionata.
# Dentro le liste positive, dims mutuamente esclusive (one-hot, o one-hot
# di posizione a fase fissa) contano come UN addendo in OR, alla snake.
# NF non ha un neurone: e' un voto GREZZO dell'unembedding sulla dim
# GCMD[G] (vedi Wout). Due gelu con slack diversi non possono gareggiare
# a colpi di priorita' (una decisione comoda produce logit enormi, una
# giusta ma stretta logit piccoli): il default deve essere un voto piatto
# che la gelu di V, quando scatta, sovrasta sempre.
FVHI = [FV[d] for d in range(16)]
hidden = [
    ([GCMD[CMD_IDX["S"]]], [], OKD, "dec"),
    ([GCMD[CMD_IDX["D"]]], [], OKD, "dec"),
    ([OKTOKD], [], EOSD, "eos_ans"),
    ([NFTOKD], [], EOSD, "eos_ans"),
    # V: sono la riga di decisione (fase 12, KPH[7]) di una G e il match
    # ha pescato un byte vero (un nibble alto qualunque); il tombstone
    # veta (FV[32]): una G che trova una D risponde NF, stessa via.
    ([GCMD[CMD_IDX["G"]], KPH[7]] + FVHI, [FV[32]], VD, "vdec"),
    # EOS dopo v7: sono un token byte a fase 8 (KPH[3]; le righe v3 dei
    # comandi nel prompt scattano anche loro: mai campionate)
    ([BYTEANY, KPH[3]], [], EOSD, "eos_v7"),
]
H = len(hidden)
Wup = [np.zeros((H, N_EMBD), dtype=np.float32) for _ in range(N_BLOCK)]
bup = [np.zeros(H, dtype=np.float32) for _ in range(N_BLOCK)]
Wdown = [np.zeros((N_EMBD, H), dtype=np.float32) for _ in range(N_BLOCK)]
bdown = [np.zeros(N_EMBD, dtype=np.float32) for _ in range(N_BLOCK)]
MLP_B = N_BLOCK - 1  # i blocchi 1-2 hanno MLP nulli


def build_mlp(thr):
    Wup[MLP_B][:] = 0
    bup[MLP_B][:] = 0
    Wdown[MLP_B][:] = 0
    for h, (pos, neg, out, kind) in enumerate(hidden):
        for d in pos:
            Wup[MLP_B][h, d] = S
        for d in neg:
            Wup[MLP_B][h, d] = -NEG * S
        bup[MLP_B][h] = -S * thr[kind]
        Wdown[MLP_B][out, h] = 1.0


# ---- unembedding --------------------------------------------------------------
# Priorita': la riga di decisione vota OK/NF (e V, milestone 2) con pesi
# alti; le righe di emissione decodificano il byte pescato come somma
# separabile hi+lo (ogni altro byte manca almeno un addendo).
P_OK, P_NF, P_V, P_EOS = 4.0, 1.5, 6.0, 8.0
Wout = np.zeros((VOCAB, N_EMBD), dtype=np.float32)
Wout[OK_T, OKD] = P_OK
Wout[NF_T, GCMD[CMD_IDX["G"]]] = P_NF  # voto grezzo: il default delle G
Wout[V_T, VD] = P_V
Wout[EOS, EOSD] = P_EOS
for b in range(256):
    Wout[byte_tok(b), FV[hi(b)]] = 1.0
    Wout[byte_tok(b), FV[16 + lo(b)]] = 1.0

gamma1 = [np.ones(N_EMBD, dtype=np.float32) for _ in range(N_BLOCK)]
beta1 = [np.zeros(N_EMBD, dtype=np.float32) for _ in range(N_BLOCK)]
gamma2 = [np.ones(N_EMBD, dtype=np.float32) for _ in range(N_BLOCK)]
beta2 = [np.zeros(N_EMBD, dtype=np.float32) for _ in range(N_BLOCK)]
gammaf = np.ones(N_EMBD, dtype=np.float32)
betaf = np.zeros(N_EMBD, dtype=np.float32)


def quantize_f16():
    for a in [wte, wpe, Wout] + Wqkv + Wattn + Wup + Wdown:
        a[:] = a.astype(np.float16).astype(np.float32)


# =================== FORWARD FEDELE (stessi tensori del GGUF) ================
def ln_rows(X, g, b):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + EPS) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def forward_all(ids):
    """Restituisce (snaps, M, logits, dbg): snaps[b] = residuo dopo
    l'attenzione del blocco b (i MLP dei blocchi 1-2 sono nulli), M =
    input dell'MLP dell'ultimo blocco (post ffn_norm), logits per riga.
    dbg = {"A2": input LN dell'attenzione del blocco 3 (le grandezze che
    la passata di misura campiona), "SC": punteggi mascherati della testa
    di match (query x chiavi, in logit)}."""
    T = len(ids)
    assert T <= MAXPOS
    X = wte[ids] + wpe[:T]
    mask = np.triu(np.ones((T, T)), 1).astype(bool)
    snaps = []
    M = None
    dbg = {}
    for b in range(N_BLOCK):
        A = ln_rows(X, gamma1[b], beta1[b])
        if b == 2:
            dbg["A2"] = A
        qkv = A @ Wqkv[b].T + bqkv[b]
        q, k, v = qkv[:, :N_EMBD], qkv[:, N_EMBD : 2 * N_EMBD], qkv[:, 2 * N_EMBD :]
        att = np.zeros((T, N_EMBD), dtype=np.float32)
        for h in range(N_HEAD):
            sl = slice(h * HEAD_D, (h + 1) * HEAD_D)
            sc = (q[:, sl] @ k[:, sl].T) / np.sqrt(HEAD_D)
            sc[mask] = -1e9
            if b == 2 and h == 0:
                dbg["SC"] = sc.copy()
            Aw = np.exp(sc - sc.max(1, keepdims=True))
            Aw /= Aw.sum(1, keepdims=True)
            att[:, sl] = Aw @ v[:, sl]
        X = X + att @ Wattn[b].T + battn[b]
        snaps.append(X.copy())
        Mrows = ln_rows(X, gamma2[b], beta2[b])
        if b == MLP_B:
            M = Mrows
        X = X + gelu(Mrows @ Wup[b].T + bup[b]) @ Wdown[b].T + bdown[b]
    logits = ln_rows(X, gammaf, betaf) @ Wout.T
    return snaps, M, logits, dbg


# ==================== GROUND TRUTH POSIZIONALE ===============================
def analyze(cmds):
    """Per ogni comando: la risposta attesa (autorita': replay) e, per le
    G, lo slot del turno S/D piu' recente con quella chiave (None se mai
    scritta) piu' gli slot STANTII (scritture precedenti della stessa
    chiave). Il bersaglio va vinto; gli stantii sono match pieni piu'
    vecchi: la rampa deve tenerli a contaminazione softmax trascurabile,
    NON sotto il sink. Le G senza bersaglio devono affondare."""
    replies = replay(cmds)
    hist = {}
    info = []
    for t, cmd in enumerate(cmds):
        op, key = cmd[0], cmd[1]
        slots = hist.get(key, []) if op == "G" else []
        info.append(
            {
                "reply": replies[t],
                "target_slot": slots[-1] if slots else None,
                "stale_slots": slots[:-1],
            }
        )
        if op in STORABLE:
            hist.setdefault(key, []).append(t + 1)
    return info


# ========================= STREAM DI CALIBRAZIONE ============================
# flag di milestone: tutto acceso (M4): sovrascritture E delete
OVERWRITES = True
DELETES = True
NIBK = [bytes([17 * n] * KEY_LEN) for n in range(16)]  # spazzata dei nibble


def near_set(key):
    """Le 12 esche canoniche di una chiave: per ogni slot, nibble alto
    cambiato (match 9/10), nibble basso cambiato (9/10), byte intero
    cambiato (8/10). Sono i concorrenti piu' vicini possibili."""
    out = []
    for i in range(KEY_LEN):
        b = key[i]
        for nb in ((((b >> 4) ^ 0x3) << 4) | (b & 15), (b & 0xF0) | ((b & 15) ^ 0x3), b ^ 0xA5):
            k = bytearray(key)
            k[i] = nb & 0xFF
            out.append(bytes(k))
    return out


def streams(seed, n_random, n_cmds=24):
    rng = np.random.default_rng(seed)
    k0, v0 = b"key0", b"deadbeef"
    filler = [("G", bytes([200 + (i % 8), i % 256, 7, 13])) for i in range(MAX_CMDS - 2)]
    out = [
        [("S", k0, v0)],  # un solo comando, primo dopo il prefisso
        [("G", k0)],
        [("D", k0)],
        [("S", k0, v0), ("G", k0)],  # il paio minimo, a distanza minima
        # doppia lettura mai scritta: i pad della prima G non devono fare
        # da archivio alla seconda (STORABLE esclude i turni G)
        [("G", k0), ("G", k0)],
        # spazzata dei nibble su chiavi E valori, con rilettura completa
        [("S", NIBK[n], bytes([17 * n] * VAL_LEN)) for n in range(16)]
        + [("G", NIBK[n]) for n in range(16)],
        # le esche canoniche contro una S vera, poi la lettura giusta
        [("S", b"ABCD", b"01234567")]
        + [("G", nk) for nk in near_set(b"ABCD")]
        + [("G", b"ABCD")],
        # distanza massima: S nel primo slot, G nell'ultimo
        [("S", k0, v0)] + filler + [("G", k0)],
        # --- recenza (M3) ---
        # sovrascrittura adiacente: slot consecutivi, il caso piu' stretto
        [("S", k0, b"AAAAAAAA"), ("S", k0, b"BBBBBBBB"), ("G", k0)],
        # catena di 8 sovrascritture: l'accumulo dei match stantii
        [("S", k0, bytes([65 + i] * VAL_LEN)) for i in range(8)] + [("G", k0)],
        # letture intercalate: ogni G legge il valore del momento
        [("S", k0, b"11111111"), ("G", k0), ("S", k0, b"22222222"), ("G", k0),
         ("S", k0, b"33333333"), ("G", k0)],
        # sovrascrittura a grande distanza
        [("S", k0, b"OLDVALUE")] + filler[:24] + [("S", k0, b"NEWVALUE")]
        + filler[24:48] + [("G", k0)],
        # il 10/10 vecchio contro i 9/10 recenti: la S vera al primo slot,
        # poi TUTTE le esche scritte dopo, e la lettura della chiave vera
        [("S", k0, b"EXACTVAL")]
        + [("S", nk, b"NEARNEAR") for nk in near_set(k0)]
        + [("G", k0)],
        # --- tombstone (M4): una via sola di scrittura ---
        [("S", k0, v0), ("D", k0), ("G", k0)],  # set-then-delete -> NF
        [("D", k0), ("G", k0)],  # delete di chiave mai scritta -> NF
        [("S", k0, v0), ("D", k0), ("S", k0, b"RISORTO!"), ("G", k0)],  # D poi S
        [("D", k0), ("S", k0, v0), ("G", k0)],  # delete-then-set -> V
        # delete a distanza: S presto, D tardi, G alla fine
        [("S", k0, v0)] + filler[:24] + [("D", k0)] + filler[24:48] + [("G", k0)],
        # catena S/D alternati sulla stessa chiave
        [c for i in range(4) for c in (("S", k0, bytes([48 + i] * VAL_LEN)), ("D", k0))]
        + [("G", k0)],
        gen_commands(rng, MAX_CMDS, overwrite=OVERWRITES, delete=DELETES),
    ]
    for _ in range(n_random):
        out.append(gen_commands(rng, n_cmds, overwrite=OVERWRITES, delete=DELETES))
    return out


def dec_row(t):
    """Posizione della riga di decisione del comando t (slot t+1)."""
    return W * (t + 1) + PH_DEC


def _sim_rows(cmds, t, reply):
    """Il nastro durante la generazione della risposta al comando t:
    log(0..t) + la risposta senza l'EOS finale. Le righe generate stanno
    a partire da dec_row(t)+1 = W*(t+2)."""
    return encode_log(cmds[: t + 1]) + reply_tokens(reply)[:-1]


# ========================== MISURA DEL MATCH =================================
def equalize_ballast():
    """Le zavorre dei fetch mancati devono pesare quanto un fetch vero
    (due nibble): stessa energia -> stessa varianza di riga -> stessa
    scala post-LN per righe altrimenti identiche. Senza questo, lo slot 1
    (che pesca la chiave-query dal prefisso, zavorra invece di byte) esce
    ~3% amplificato rispetto agli slot successivi: +200 logit di rumore
    sistematico, dove la rampa di recenza lavora a 8 logit per slot."""
    k0 = b"key0"
    cmds = [("S", k0, b"AAAAAAAA"), ("S", k0, b"BBBBBBBB"), ("G", k0), ("G", b"ZZZZ")]
    snaps, _, _, _ = forward_all(encode_log(cmds))
    X1, X2 = snaps[1], snaps[2]
    r_bal, r_real = W + 5, 2 * W + 5  # fase 5 dello slot 1 (prefisso) e 2 (reale)
    qk = np.mean(
        [X1[r_real, QK[i][hi(k0[i])]] for i in range(KEY_LEN)]
        + [X1[r_real, QK[i][16 + lo(k0[i])]] for i in range(KEY_LEN)]
    )
    qbal = np.mean(X1[r_bal, QBAL])
    s1 = float(qk * np.sqrt(2) / qbal)
    for i in range(KEY_LEN):  # blocco 2: 4 teste, zavorra alla coordinata 32
        Wattn[1][QBAL[i], i * HEAD_D + 32] *= s1
    fv = np.mean(
        [X2[dec_row(2), FV[hi(ord("B"))]], X2[dec_row(2), FV[16 + lo(ord("B"))]]]
    )
    fbal = float(X2[dec_row(3), FBAL])
    s2 = float(fv * np.sqrt(2) / fbal)
    Wattn[2][FBAL, 0 * HEAD_D + 33] *= s2
    print(f"  zavorre equalizzate: QBAL x{s1:.3f}  FBAL x{s2:.3f}")


def measure_components(seed=7):
    """Prima passata di misura: le grandezze post-LN reali che entrano
    nel punteggio del match (nibble raccolti lato chiave e lato query,
    one-hot di posizione, lettera comando, zavorra del <NOOP>), su nastri
    veri e su righe di risposta simulate. Da qui i pesi per componente:
    un addendo combaciante deve valere U_MATCH logit, chiunque sia."""
    rng = np.random.default_rng(seed)
    probes = [
        [("S", NIBK[n], bytes([17 * n] * VAL_LEN)) for n in range(16)]
        + [("G", NIBK[n]) for n in range(4)],
        [("S", b"MKEY", b"probeval"), ("G", b"MKEY"), ("D", b"DKEY")],
        gen_commands(rng, 30, overwrite=OVERWRITES, delete=DELETES),
    ]
    acc = {k: [] for k in ("gk", "kph", "kst", "kpad", "qnib", "qph", "qon", "kbal")}
    for cmds in probes:
        info = analyze(cmds)
        _, _, _, dbg = forward_all(encode_log(cmds))
        A2 = dbg["A2"]
        acc["kbal"].append(float(A2[0, BAL_SRC]))
        for t, cmd in enumerate(cmds):
            base, key = W * (t + 1), cmd[1]
            if cmd[0] in STORABLE:
                for ph in range(5, 13):
                    row = A2[base + ph]
                    acc["gk"] += [float(row[GK[i][hi(key[i])]]) for i in range(KEY_LEN)]
                    acc["gk"] += [float(row[GK[i][16 + lo(key[i])]]) for i in range(KEY_LEN)]
                    acc["kph"].append(float(row[KPH[ph - 5]]))
                    acc["kst"].append(float(row[GCMD[CMD_IDX[cmd[0]]]]))
            else:
                row = A2[base + PH_DEC]
                acc["qnib"] += [float(row[QK[i][hi(key[i])]]) for i in range(KEY_LEN)]
                acc["qph"].append(float(row[QPH[0]]))  # la decisione chiede fase 5
                acc["qon"].append(float(row[QON]))
                for ph in range(5, 13):  # i pad della G: la chiave del veto
                    acc["kpad"].append(float(A2[base + ph, PADBAL]))
        for t, cmd in enumerate(cmds):  # righe di risposta V + byte
            if info[t]["reply"][0] != "V":
                continue
            _, _, _, d2 = forward_all(_sim_rows(cmds, t, info[t]["reply"]))
            A2s, L = d2["A2"], W * (t + 2)
            for j in range(8):
                row = A2s[L + j]
                acc["qnib"] += [float(row[QK[i][hi(cmd[1][i])]]) for i in range(KEY_LEN)]
                acc["qph"].append(float(row[QPH[_match_target_phase(j) - 5]]))
                acc["qon"].append(float(row[QON]))
    med = {k: float(np.median(v)) for k, v in acc.items()}
    for k, v in acc.items():
        print(
            f"  misura {k:6s}: mediana {med[k]:7.3f}  "
            f"[{np.min(v):7.3f}, {np.max(v):7.3f}]  n={len(v)}"
        )
    return med


def _score_row(sc, r, trow, stale_rows):
    """(pieno, concorrente, contaminazione) alla riga di query r: il
    concorrente esclude bersaglio e stantii; la contaminazione e' la
    massa softmax degli stantii rispetto al bersaglio."""
    others = sc[r, 1 : r + 1].copy()
    others[trow - 1] = -np.inf
    for sr in stale_rows:
        others[sr - 1] = -np.inf
    stale = float(sum(np.exp(sc[r, sr] - sc[r, trow]) for sr in stale_rows))
    return float(sc[r, trow]), float(others.max()), stale


def measure_scores(seed=11, n_random=6):
    """Seconda passata: la distribuzione dei punteggi VERI della testa di
    match. Per ogni riga che deve pescare: punteggio del bersaglio, del
    miglior concorrente (esclusi gli stantii) e contaminazione stantia;
    per ogni riga che deve affondare (G senza scrittura): il punteggio
    migliore. Il sink va in mezzo al gap con almeno MIN_SAT logit per
    lato e la contaminazione resta sotto MAX_STALE, o il generatore
    rifiuta."""
    full, comp, sunk, stale = [], [], [], []
    for cmds in streams(seed, n_random):
        info = analyze(cmds)
        _, _, _, dbg = forward_all(encode_log(cmds))
        sc = dbg["SC"]
        for t, cmd in enumerate(cmds):
            if cmd[0] != "G":
                continue
            r, tgt = dec_row(t), info[t]["target_slot"]
            if tgt is None:
                sunk.append(float(sc[r, 1 : r + 1].max()))
            else:
                f, c, s = _score_row(sc, r, W * tgt + 5, [W * s + 5 for s in info[t]["stale_slots"]])
                full.append(f)
                comp.append(c)
                stale.append(s)
        for t, cmd in enumerate(cmds):  # righe di emissione delle G con valore
            if info[t]["reply"][0] != "V":
                continue
            _, _, _, d2 = forward_all(_sim_rows(cmds, t, info[t]["reply"]))
            scs, L, tgt = d2["SC"], W * (t + 2), info[t]["target_slot"]
            for j in range(8):
                f, c, s = _score_row(
                    scs, L + j, W * tgt + 5 + j,
                    [W * s + 5 + j for s in info[t]["stale_slots"]],
                )
                full.append(f)
                comp.append(c)
                stale.append(s)
    full, comp, sunk, stale = map(np.array, (full, comp, sunk, stale))
    # vincolo PER RIGA: il bersaglio batte il suo miglior concorrente
    # (stessa query, stessa scala); vincolo GLOBALE solo per il sink, che
    # e' una costante: sopra ogni riga che deve affondare, sotto ogni
    # match pieno. La contaminazione stantia e' il vincolo della rampa.
    row_gap = (full - comp).min()
    lo, hi = sunk.max(), full.min()
    print(
        f"  punteggi ({len(full)} fetch, {len(sunk)} sink): "
        f"gap_per_riga>={row_gap:7.1f}  pieno>={hi:8.1f}  "
        f"affondanti<={lo:8.1f}  gap_sink={hi - lo:7.1f} logit"
    )
    print(f"  contaminazione stantia massima: {stale.max():.4f} (tetto {MAX_STALE})")
    assert row_gap >= MIN_SAT, "un concorrente batte un bersaglio: non scrivo il GGUF"
    assert hi - lo >= 2 * MIN_SAT, "gap del sink sotto il margine: non scrivo il GGUF"
    assert stale.max() <= MAX_STALE, "rampa troppo piatta: non scrivo il GGUF"
    return float((hi + lo) / 2)


# ===================== PLUMBING: ASSERT SUI GATHER ===========================
def _assert_onehot(vec, dims, want, ctx):
    """dentro `dims` (16 dim) deve essere acceso want e solo want."""
    v = vec[dims]
    got = int(np.argmax(v))
    rest = np.delete(v, got)
    assert got == want and v[got] > 1.0 and np.abs(rest).max() < 0.2 * v[got], (
        f"gather rotto ({ctx}): atteso {want}, vinto {got} "
        f"(val {v[got]:.2f}, secondo {np.abs(rest).max():.2f})"
    )


def check_plumbing(cmds, snaps, coverage):
    X1, X2 = snaps[0], snaps[1]
    for t, cmd in enumerate(cmds):
        op, key = cmd[0], cmd[1]
        base = W * (t + 1)
        for ph in range(5, 13):
            row = X1[base + ph]
            for i in range(KEY_LEN):
                _assert_onehot(row, [GK[i][d] for d in range(16)], hi(key[i]), f"GK{i}hi f{ph}")
                _assert_onehot(row, [GK[i][d + 16] for d in range(16)], lo(key[i]), f"GK{i}lo f{ph}")
                coverage.add(("g", i, "h", hi(key[i])))
                coverage.add(("g", i, "l", lo(key[i])))
            _assert_onehot(row, GCMD, CMD_IDX[op], f"GCMD f{ph}")
        for ph in range(0, 5):  # le fasi 0..4 non raccolgono: sink pulito
            row = X1[base + ph]
            gk = np.concatenate([row[GK[i]] for i in range(KEY_LEN)])
            assert np.abs(gk).max() < 0.5, f"GK sporco a fase {ph}: {np.abs(gk).max():.2f}"
        # lato query: la riga di decisione porta la chiave del proprio turno
        row = X2[dec_row(t)]
        for i in range(KEY_LEN):
            _assert_onehot(row, [QK[i][d] for d in range(16)], hi(key[i]), f"QK{i}hi dec")
            _assert_onehot(row, [QK[i][d + 16] for d in range(16)], lo(key[i]), f"QK{i}lo dec")
        # le fasi 0..7 del turno portano la chiave del turno PRECEDENTE
        if t >= 1:
            pkey = cmds[t - 1][1]
            for ph in range(0, 8):
                row = X2[base + ph]
                for i in range(KEY_LEN):
                    _assert_onehot(row, [QK[i][d] for d in range(16)], hi(pkey[i]), f"QK{i}hi f{ph}")


# ======================= CALIBRAZIONE PER GRUPPO =============================
GROUPS = ["dec", "eos_ans", "vdec", "eos_v7"]


def neuron_sums(M, pos, neg):
    s = M[:, pos].sum(axis=1)
    if neg:
        s = s - NEG * M[:, neg].sum(axis=1)
    return s


def check_fv(cmds, t, inf, snaps, coverage):
    """Il fetch del match, verificato dim per dim: alla riga di decisione
    FV = v0 del bersaglio, alla riga di risposta a fase j FV = v_j."""
    X2 = snaps[2]
    val = inf["reply"][1]
    rows = [(dec_row(t), 0)] + [(W * (t + 2) + j, j) for j in range(8)]
    for r, j in rows:
        if r >= len(X2):
            continue
        _assert_onehot(X2[r], [FV[d] for d in range(16)], hi(val[j]), f"FVhi v{j}")
        _assert_onehot(X2[r], [FV[16 + d] for d in range(16)], lo(val[j]), f"FVlo v{j}")
        coverage.add(("fv", "h", hi(val[j])))
        coverage.add(("fv", "l", lo(val[j])))


def calibrate(seed=11, n_random=8):
    must = {g: [] for g in GROUPS}
    mustnot = {g: [] for g in GROUPS}
    coverage = set()
    n_dec = {"S": 0, "G": 0, "D": 0, "hit": 0, "miss": 0}
    n_sim = 0
    for si, cmds in enumerate(streams(seed, n_random)):
        info = analyze(cmds)
        ids = encode_log(cmds)
        snaps, M, _, _ = forward_all(ids)
        if si < 9:
            check_plumbing(cmds, snaps, coverage)
        # classificazione delle righe del nastro base
        T = len(ids)
        phases = np.arange(T) % W
        slot_cmd = np.full(T, -1)  # indice CMD_IDX del turno, -1 nel prefisso
        byte_row = np.array([BYTE0 <= tk < BYTE0 + 256 for tk in ids])
        ghit12 = np.zeros(T, bool)  # righe di decisione G col match pieno
        for t, cmd in enumerate(cmds):
            slot_cmd[W * (t + 1) : W * (t + 2)] = CMD_IDX[cmd[0]]
            n_dec[cmd[0]] += 1
            if cmd[0] == "G":
                hit = info[t]["target_slot"] is not None
                n_dec["hit" if hit else "miss"] += 1
                if info[t]["reply"][0] == "V":  # un match su tombstone NON accende V
                    ghit12[dec_row(t)] = True
                if not hit:  # sink pulito: una G senza scrittura pesca zeri
                    fv = snaps[2][dec_row(t)][FV]
                    assert np.abs(fv).max() < 0.5, (
                        f"sink sporco su G ignota (cmd {t}): |FV|={np.abs(fv).max():.2f}"
                    )
                elif info[t]["reply"][0] == "NF":  # match su tombstone: FV=TOMB
                    fv = snaps[2][dec_row(t)][FV]
                    assert fv[32] > 1.0, (
                        f"tombstone non pescato (cmd {t}): FV[TOMB]={fv[32]:.2f}"
                    )
        for pos, neg, _, kind in hidden:
            s = neuron_sums(M, pos, neg)
            care = np.ones(T, bool)
            if kind == "dec":
                fire = (phases >= 5) & (slot_cmd == GCMD.index(pos[0]))
            elif kind == "vdec":
                # vdec legge FV: sulle righe di prompt MAI campionate il
                # fetch del match e' spazzatura (query = chiave del turno
                # precedente) e puo' uscire come MISCELA di piu' nibble a
                # pari punteggio, che somma oltre ogni fetch legittimo.
                # Il greedy campiona solo le righe di decisione (fase 12)
                # e quelle generate: il gruppo si calibra solo li'.
                fire = ghit12
                care = (phases == PH_DEC) & (np.arange(T) >= W)
            elif kind == "eos_v7":
                fire = byte_row & (phases == 8)
            else:  # eos_ans: mai sul nastro base
                fire = np.zeros(T, bool)
            must[kind] += list(s[fire & care])
            mustnot[kind] += list(s[~fire & care])
        # risposte simulate: tutte le V (copertura dell'emissione), piu'
        # un campione di OK/NF per l'EOS
        sample = [t for t in range(len(cmds)) if info[t]["reply"][0] == "V"]
        sample += [0, len(cmds) - 1]
        sample = sorted(
            set(
                sample
                + list(np.random.default_rng(seed + si).integers(0, len(cmds), 4))
            )
        )
        for t in sample:
            rep = info[t]["reply"]
            rids = _sim_rows(cmds, t, rep)
            snaps_s, Ms, _, _ = forward_all(rids)
            n_sim += 1
            if rep[0] == "V":
                check_fv(cmds, t, info[t], snaps_s, coverage)
            L = W * (t + 2)
            for r in range(L, len(rids)):
                tok = rids[r]
                for pos, neg, _, kind in hidden:
                    sv = float(neuron_sums(Ms[r : r + 1], pos, neg)[0])
                    if kind == "eos_ans":
                        fire_r = (tok == OK_T and pos[0] == OKTOKD) or (
                            tok == NF_T and pos[0] == NFTOKD
                        )
                    elif kind == "eos_v7":
                        fire_r = BYTE0 <= tok < BYTE0 + 256 and r % W == 8
                    else:  # dec e vdec: mai sulle righe generate
                        fire_r = False
                    (must if fire_r else mustnot)[kind].append(sv)
    # copertura: tutti i nibble visti sia nei gather sia nei valori pescati
    need = {("g", i, hl, n) for i in range(KEY_LEN) for hl in "hl" for n in range(16)}
    need |= {("fv", hl, n) for hl in "hl" for n in range(16)}
    missing = need - coverage
    assert not missing, f"copertura incompleta: {sorted(missing)[:6]}"
    print(
        f"  eventi: comandi {dict(n_dec)}  risposte_simulate={n_sim}  "
        f"copertura_nibble={len(coverage & need)}/{len(need)}"
    )
    thr = {}
    ok_all = True
    for g in GROUPS:
        mn, nn = np.array(must[g]), np.array(mustnot[g])
        assert len(mn), f"gruppo {g}: nessun evento 'deve sparare'"
        gap = nn.max() < mn.min()
        ok_all &= gap
        thr[g] = float((nn.max() + mn.min()) / 2)
        print(
            f"  {g:8s}: deve>={mn.min():6.2f}  non_deve<={nn.max():6.2f}  "
            f"gap {'OK' if gap else 'SOVRAPPOSTO'}  thr={thr[g]:.2f}"
        )
    assert ok_all, "gap sovrapposto: rivedi ROUTE, i boost o i gruppi"
    return thr


# ============================ VERIFICA =======================================
def verify(seed=500, n_random=8):
    """Ogni comando di ogni stream: la risposta INTERA, token per token,
    contro il riferimento. Il nastro con la risposta appesa riproduce
    esattamente i contesti della generazione greedy (induzione: se i
    token precedenti combaciano, il contesto e' identico)."""
    ok = tot = 0
    bad = []
    n_rep = {"V": 0, "OK": 0, "NF": 0}
    for cmds in streams(seed, n_random):
        info = analyze(cmds)
        for t, cmd in enumerate(cmds):
            rep = info[t]["reply"]
            n_rep[rep[0]] += 1
            expected = reply_tokens(rep)
            _, _, logits, _ = forward_all(_sim_rows(cmds, t, rep))
            for j, want in enumerate(expected):
                pred = int(logits[dec_row(t) + j].argmax())
                tot += 1
                if pred == want:
                    ok += 1
                else:
                    bad.append((cmd[0], j, tokens[pred], tokens[want]))
    if bad:
        print(f"  primi errori: {bad[:8]}")
    print(f"  risposte verificate: {n_rep}")
    return ok, tot


# ============================== MAIN =========================================
if __name__ == "__main__":
    print(
        f"n_embd={N_EMBD} n_head={N_HEAD} head_dim={HEAD_D} n_block={N_BLOCK} "
        f"n_ff={H} vocab={VOCAB} ctx={MAXPOS} ({MAX_CMDS} comandi)"
    )
    print(f"MILESTONE 4: tombstone (rampa {RAMP_SLOT} logit/slot, fuzz completo)")
    quantize_f16()
    print("equalizzazione delle zavorre (forward reale, pesi f16):")
    equalize_ballast()
    quantize_f16()
    print("misura delle componenti del match (forward reale, pesi f16):")
    med = measure_components()
    w_nib = U_MATCH * np.sqrt(HEAD_D) / (med["qnib"] * med["gk"])
    w_ph = U_META * np.sqrt(HEAD_D) / (med["qph"] * med["kph"])
    w_st = U_META * np.sqrt(HEAD_D) / (med["qon"] * med["kst"])
    w_veto = VETO_U * U_MATCH * np.sqrt(HEAD_D) / (med["qon"] * med["kpad"])
    print(
        f"  pesi: nibble={w_nib:.1f} fase={w_ph:.1f} scrivibile={w_st:.1f} "
        f"veto={w_veto:.1f}"
    )
    build_match(w_nib, w_ph, w_st, w_veto, RAMP_SLOT / U_META)
    quantize_f16()
    print("misura dei punteggi del match:")
    sink = measure_scores()
    set_sink(sink, med["kbal"])
    print(f"  sink a {sink:.1f} logit")
    print("calibrazione soglie (forward reale, pesi f16):")
    build_mlp(calibrate())
    quantize_f16()
    ok, tot = verify()
    print(f"verifica: {ok}/{tot} token corretti ({100 * ok / max(tot, 1):.1f}%)")
    assert ok == tot, "verifica fallita: non scrivo il GGUF"

    w = gguf.GGUFWriter(OUT, "gpt2")
    w.add_name("kv-transformer")
    w.add_context_length(MAXPOS)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_BLOCK)
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
    w.add_bos_token_id(1)
    w.add_eos_token_id(EOS)
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    # i byte di chiavi/valori viaggiano come testo grezzo tra i token di
    # controllo: senza questo, l'SPM antepone "▁" a ogni frammento e
    # disallinea il frame (snake/life non lo vedono: solo token speciali)
    w.add_add_space_prefix(False)

    f16 = lambda a: a.astype(np.float16)
    w.add_tensor("token_embd.weight", f16(wte))
    w.add_tensor("position_embd.weight", f16(wpe))
    w.add_tensor("output_norm.weight", gammaf)
    w.add_tensor("output_norm.bias", betaf)
    w.add_tensor("output.weight", f16(Wout))
    for b in range(N_BLOCK):
        w.add_tensor(f"blk.{b}.attn_norm.weight", gamma1[b])
        w.add_tensor(f"blk.{b}.attn_norm.bias", beta1[b])
        w.add_tensor(f"blk.{b}.attn_qkv.weight", f16(Wqkv[b]))
        w.add_tensor(f"blk.{b}.attn_qkv.bias", bqkv[b])
        w.add_tensor(f"blk.{b}.attn_output.weight", f16(Wattn[b]))
        w.add_tensor(f"blk.{b}.attn_output.bias", battn[b])
        w.add_tensor(f"blk.{b}.ffn_norm.weight", gamma2[b])
        w.add_tensor(f"blk.{b}.ffn_norm.bias", beta2[b])
        w.add_tensor(f"blk.{b}.ffn_up.weight", f16(Wup[b]))
        w.add_tensor(f"blk.{b}.ffn_up.bias", bup[b])
        w.add_tensor(f"blk.{b}.ffn_down.weight", f16(Wdown[b]))
        w.add_tensor(f"blk.{b}.ffn_down.bias", bdown[b])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"scritto {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")
