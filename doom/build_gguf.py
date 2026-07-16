#!/usr/bin/env python3
"""
Builds `doom.gguf`: a DOOM-LIKE first-person game as a hand-programmed gpt2
transformer. Everything wolf has (fixed map in the weights, raycast walls,
movement/collisions) plus: ONE relentless enemy that chases you, hitscan
shooting, and hit points. The frontend still sends ONLY the input keys.

New ideas on top of wolf/build_gguf.py (which remains the reference for the
tape-is-not-a-history protocol, absolute selectors via bias, the ROM and the
phase bonus):

  - GENERATED TOKENS AS INTERMEDIATE REGISTERS. The joint state space
    (player x enemy) cannot be tabulated (~25M combinations). It CAN be
    factored into a pipeline of small tables, but one MLP layer cannot chain
    computations. The trick: every generated token is one full pass through
    the circuit, and it lands at a KNOWN absolute position — so later rows
    can fetch it. The tick emits a chain of intermediate tokens (relative
    position -> line of sight -> hit -> enemy move -> damage -> sprite),
    each one computed by a small table from previously generated tokens.

  - DERIVED VALUES RIDE IN THE EMBEDDINGS. A token's embedding can carry
    any function of its value, precomputed at build time: <X:x> carries both
    the sub-cell one-hot and the CELL one-hot (x//2); <T:t> carries the tick
    counter AND the hash respawn cell for tick t (the pseudo-random dice,
    as in the autonomous snake, but keyed to the tick counter because with
    a re-serialized context the positions never change).

  - MUTUALLY EXCLUSIVE CIRCUITS GATED BY GENERATED FLAGS. The <HIT:0|1>
    token gates the enemy update: move neurons AND on HIT=0, respawn
    neurons AND on HIT=1. The <ATK:0|1> token gates the HP update the same
    way (the fcopy/linc pattern from the autonomous snake).

  - SINK-BY-DEFAULT FOR STAGED FETCHES. 22 heads fetch 22 absolute
    positions, but early generation rows cannot yet see the later ones
    (causal mask): without a sink they would smear the whole prefix into
    their fetch blocks. Every head gets a constant sink bias toward <NOOP>
    (ballast as key) that any visible real target outranks: masked ->
    clean ballast, visible -> exact fetch. Per-row fetch profiles are then
    deterministic, so per-group thresholds hold.

Game rules (all deterministic, all in the weights):
  - player: half-cell grid 24x24 over a 12x12 map, 16 view angles,
    inputs L/R (turn), F/B (step), S (fire; no movement);
  - enemy: lives on whole cells, takes one king step per tick along a REAL
    BFS path toward the player (computed at build time, tabulated as one
    neuron per (enemy cell, player cell) pair), stops next to you and
    bites: Chebyshev distance <= 1 costs 1 HP per tick;
  - shooting: hits iff the enemy is within +-2 columns of the screen center
    AND there is line of sight (cell-to-cell, precomputed);
  - a hit kills: the enemy respawns at HASHCELL[t] (pseudo-random, baked in
    the <T:t> embedding); you start with 5 HP; at 0 the state is <DEAD>;
  - rendering: 32 raycast wall columns (like wolf) + a sprite token
    <SPR:column,size> for the enemy (visible iff line of sight and inside
    the view cone; binary occlusion).

Usage:
    pip install gguf numpy
    python build_gguf.py                # produces doom.gguf
    ollama create doom -f Modelfile
"""

import os

import gguf
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doom.gguf")

# ============================== WORLD =======================================
MAP = [
    "############",
    "#....#.....#",
    "#.##.#.###.#",
    "#.#........#",
    "#.#.##.###.#",
    "#....#...#.#",
    "#.##.###.#.#",
    "#.#....#...#",
    "#.#.##...#.#",
    "#.#..#.###.#",
    "#..........#",
    "############",
]
GC = 12
assert len(MAP) == GC and all(len(r) == GC for r in MAP)


def wall(cx, cy):
    return MAP[cy][cx] == "#"


SUB = 2
GS = GC * SUB  # 24: player sub-cell grid
NANG = 16
INPUTS = ["L", "R", "F", "B", "S"]  # turn, turn, step, step, shoot
NIN = len(INPUTS)
NCOLS = 32
NH_LEV = 14
NSHADE = 2
NCV = NH_LEV * NSHADE
HSC = 10.0
FOV_T = 0.66

HP0 = 5
TMOD = 32  # tick counter modulus (drives the respawn dice)
SPAWN = (3, 3, 0)  # player: sub-cell (3,3), facing east
ESPAWN = (10, 10)  # enemy: cell
NSIZE = 6  # sprite size buckets
RMAX = GC - 1  # relative cell offsets in [-11, 11]
NREL = 2 * RMAX + 1  # 23


def open_sub(sx, sy):
    return 0 <= sx < GS and 0 <= sy < GS and not wall(sx // SUB, sy // SUB)


def reachable_subs():
    seen = {(SPAWN[0], SPAWN[1])}
    todo = [(SPAWN[0], SPAWN[1])]
    while todo:
        x, y = todo.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (x + dx, y + dy)
                if open_sub(*n) and n not in seen:
                    seen.add(n)
                    todo.append(n)
    return seen


OPEN = sorted(reachable_subs())
assert len(OPEN) == sum(open_sub(x, y) for x in range(GS) for y in range(GS))
OPENC = sorted((x, y) for x in range(GC) for y in range(GC) if not wall(x, y))
assert not wall(*ESPAWN)

PSTATES = [(sx, sy, a) for (sx, sy) in OPEN for a in range(NANG)]


def theta(a):
    return a * 2 * np.pi / NANG


def p_transition(s, i):
    sx, sy, a = s
    if INPUTS[i] == "L":
        return (sx, sy, (a - 1) % NANG)
    if INPUTS[i] == "R":
        return (sx, sy, (a + 1) % NANG)
    if INPUTS[i] == "S":
        return s
    dx = int(np.rint(np.cos(theta(a))))
    dy = int(np.rint(np.sin(theta(a))))
    if INPUTS[i] == "B":
        dx, dy = -dx, -dy
    n = (sx + dx, sy + dy)
    return (n[0], n[1], a) if open_sub(*n) else s


def raycast(s):
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


def los(pc, ec):
    """Cell-to-cell line of sight (center to center, sampled)."""
    if pc == ec:
        return True
    x0, y0 = pc[0] + 0.5, pc[1] + 0.5
    x1, y1 = ec[0] + 0.5, ec[1] + 0.5
    n = int(np.hypot(x1 - x0, y1 - y0) / 0.02) + 1
    for k in range(1, n):
        x, y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
        if wall(int(x), int(y)):
            return False
    return True


def _neighbors(c):
    """King moves; diagonals cannot cut through two wall corners."""
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx, dy) == (0, 0):
                continue
            n = (c[0] + dx, c[1] + dy)
            if wall(*n):
                continue
            if dx and dy and wall(c[0] + dx, c[1]) and wall(c[0], c[1] + dy):
                continue
            out.append(n)
    return out


def bfs_dist(target):
    """Distance field toward `target` over open cells (king metric)."""
    dist = {target: 0}
    frontier = [target]
    while frontier:
        nxt = []
        for c in frontier:
            for n in _neighbors(c):
                if n not in dist:
                    dist[n] = dist[c] + 1
                    nxt.append(n)
        frontier = nxt
    return dist


def e_move(ec, pc, dist):
    """Real pathfinding: one step along the BFS field toward the player;
    already adjacent (or unreachable) -> stay and bite."""
    if max(abs(ec[0] - pc[0]), abs(ec[1] - pc[1])) <= 1 or ec not in dist:
        return ec
    return min(_neighbors(ec), key=lambda n: (dist.get(n, 1e9), n))


def sprite_geom(rdx, rdy, a):
    """(column, size) if the enemy cell offset (rdx,rdy) is inside the view
    cone at angle a, else None. (0,0) = you are inside it: full screen."""
    if rdx == 0 and rdy == 0:
        return (NCOLS // 2, NSIZE)
    diff = np.arctan2(rdy, rdx) - theta(a)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    if abs(diff) >= np.arctan(1.6):  # generous cone, then clamp by column
        return None
    col = (np.tan(diff) / FOV_T + 1) / 2 * NCOLS
    if not (0 <= col < NCOLS):
        return None
    dist = float(np.hypot(rdx, rdy))
    size = max(1, min(NSIZE, int(np.rint(7.0 / dist))))
    return (int(col), size)


AIM_COLS = {14, 15, 16, 17}  # +-2 columns around the screen center

print("precomputing world tables...")
PTRANS = {(s, i): p_transition(s, i) for s in PSTATES for i in range(NIN)}
FRAME = {s: raycast(s) for s in PSTATES}
LOS = {(pc, ec): los(pc, ec) for pc in OPENC for ec in OPENC}
_DIST = {pc: bfs_dist(pc) for pc in OPENC}
EMOVE = {(ec, pc): e_move(ec, pc, _DIST[pc]) for ec in OPENC for pc in OPENC}
SPRITE = {
    (rdx, rdy, a): sprite_geom(rdx, rdy, a)
    for rdx in range(-RMAX, RMAX + 1)
    for rdy in range(-RMAX, RMAX + 1)
    for a in range(NANG)
}
_hash_rng = np.random.default_rng(20260713)
HASHCELL = {t: OPENC[int(_hash_rng.integers(len(OPENC)))] for t in range(TMOD)}

# ============================ VOCABULARY =====================================
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
EX_T = {v: game(f"<EX:{v}>") for v in range(GC)}
EY_T = {v: game(f"<EY:{v}>") for v in range(GC)}
HP_T = {v: game(f"<HP:{v}>") for v in range(HP0 + 1)}
T_T = {v: game(f"<T:{v}>") for v in range(TMOD)}
I_T = {i: game(f"<I:{INPUTS[i]}>") for i in range(NIN)}
HIT_T = {b: game(f"<HIT:{b}>") for b in range(2)}
ATK_T = {b: game(f"<ATK:{b}>") for b in range(2)}
VISA_T = {b: game(f"<VISA:{b}>") for b in range(2)}
VISR_T = {b: game(f"<VISR:{b}>") for b in range(2)}
RDXA_T = {v: game(f"<RDXA:{v}>") for v in range(-RMAX, RMAX + 1)}
RDYA_T = {v: game(f"<RDYA:{v}>") for v in range(-RMAX, RMAX + 1)}
RDXR_T = {v: game(f"<RDXR:{v}>") for v in range(-RMAX, RMAX + 1)}
RDYR_T = {v: game(f"<RDYR:{v}>") for v in range(-RMAX, RMAX + 1)}
SPR_T = {
    (c, z): game(f"<SPR:{c},{z}>") for c in range(NCOLS) for z in range(1, NSIZE + 1)
}
SPRN_T = game("<SPR:none>")
OK_T, DEAD_T, KILL_T = game("<OK>"), game("<DEAD>"), game("<KILL>")
C_T = {
    (i, h, sd): game(f"<C{i}:{h},{sd}>")
    for i in range(NCOLS)
    for h in range(1, NH_LEV + 1)
    for sd in range(NSHADE)
}
V = len(tokens)

# ========================== HYPERPARAMETERS ==================================
HEAD_D = 64
N_HEAD = 32
N_EMBD = N_HEAD * HEAD_D  # 2048
MAXPOS = 60
SINKDIM = MAXPOS  # q/k coordinate for the default sink (outside POSB)
EPS = 1e-5
SEL_T = 60.0  # real-target bias strength
SEL_S = 2.0  # sink bias strength (any visible target outranks it)
S = 12.0
RAW = 9.0
DSCALE = 0.1  # circuit outputs stay small: the phase bonus is the referee

# --- protocol: one tick = 58 tokens at fixed absolute positions -------------
# input (frontend: echoed state + key):
P_X, P_Y, P_A, P_EX, P_EY, P_HP, P_T, P_I = 1, 2, 3, 4, 5, 6, 7, 8
# generated:
P_XN, P_YN, P_AN = 9, 10, 11  # new player state (ROM)
P_RDXA, P_RDYA = 12, 13  # enemy rel. to NEW player (cells)
P_VISA = 14  # line of sight for aiming
P_HIT = 15  # did the shot connect?
P_EXN, P_EYN = 16, 17  # enemy: chase step or respawn
P_TN = 18  # tick counter + 1
P_ATK = 19  # enemy adjacent -> bite
P_HPN = 20  # hit points after the bite
P_ST = 21  # <OK>/<DEAD>/<KILL>
P_RDXR, P_RDYR = 22, 23  # rel. for rendering (new enemy pos)
P_VISR = 24  # line of sight for the sprite
P_SPR = 25  # sprite column+size (or none)
P_C0 = 26  # 32 wall columns: 26..57
TICKLEN = P_C0 + NCOLS  # 58
GEN0 = P_I  # first generation row (row 8 emits position 9)
N_GEN = TICKLEN - GEN0 - 1  # 49 generated tokens


# --- residual layout ---------------------------------------------------------
def blk(n, c=[0]):
    s = c[0]
    c[0] += n
    return list(range(s, s + n))


_c = [0]
# token identity blocks (embeddings; PAD equalizes per-token variance)
XV = blk(GS, _c)
PCX = blk(GC, _c)  # player cell x = x//2, baked into <X:x>
YV = blk(GS, _c)
PCY = blk(GC, _c)
AV = blk(NANG, _c)
EXV = blk(GC, _c)
EYV = blk(GC, _c)
HPV = blk(HP0 + 1, _c)
TV = blk(TMOD, _c)
HXC = blk(GC, _c)  # respawn cell for tick t, baked into <T:t>
HYC = blk(GC, _c)
IV = blk(NIN, _c)
HITV = blk(2, _c)
ATKV = blk(2, _c)
VISV = blk(2, _c)  # shared by <VISA:.> and <VISR:.> (heads differ)
RDXV = blk(NREL, _c)  # shared by <RDXA:.> and <RDXR:.>
RDYV = blk(NREL, _c)
PAD = blk(1, _c)[0]  # second raw dim for single-one-hot tokens
BAL = blk(1, _c)[0]  # <NOOP> ballast: sink key + sink fetch value
# fetch destinations (one head per fetched position)
GX = blk(GS, _c)
GPCX = blk(GC, _c)
GY = blk(GS, _c)
GPCY = blk(GC, _c)
GA = blk(NANG, _c)
GEX = blk(GC, _c)
GEY = blk(GC, _c)
GHP = blk(HP0 + 1, _c)
GT = blk(TMOD, _c)
GHXC = blk(GC, _c)
GHYC = blk(GC, _c)
GI = blk(NIN, _c)
GXN = blk(GS, _c)
GPCXN = blk(GC, _c)
GYN = blk(GS, _c)
GPCYN = blk(GC, _c)
GAN = blk(NANG, _c)
GRDXA = blk(NREL, _c)
GRDYA = blk(NREL, _c)
GVISA = blk(2, _c)
GHIT = blk(2, _c)
GEXN = blk(GC, _c)
GEYN = blk(GC, _c)
GATK = blk(2, _c)
GHPN = blk(HP0 + 1, _c)
GRDXR = blk(NREL, _c)
GRDYR = blk(NREL, _c)
GVISR = blk(2, _c)
BALK = blk(22, _c)  # per-head ballast landing spots (sinked fetches)
# circuit outputs (MLP writes, unembedding reads)
NX = blk(GS, _c)
NY = blk(GS, _c)
NA = blk(NANG, _c)
FCOL = {i: blk(NCV, _c) for i in range(NCOLS)}
NRDXA = blk(NREL, _c)
NRDYA = blk(NREL, _c)
NRDXR = blk(NREL, _c)
NRDYR = blk(NREL, _c)
VISFA = blk(1, _c)[0]
VISFR = blk(1, _c)[0]
HITF = blk(1, _c)[0]
NEX = blk(GC, _c)
NEY = blk(GC, _c)
NT = blk(TMOD, _c)
ATKF = blk(1, _c)[0]
NHP = blk(HP0 + 1, _c)
SD = blk(1, _c)[0]
SKILL = blk(1, _c)[0]
NSPRC = blk(NCOLS, _c)
NSPRS = blk(NSIZE, _c)
POSB = blk(MAXPOS, _c)
assert _c[0] <= N_EMBD, f"layout {_c[0]} does not fit in {N_EMBD}"

# ============================== WEIGHTS ======================================
wte = np.zeros((V, N_EMBD), dtype=np.float32)


def emb(t, dims):
    for d in dims:
        wte[t, d] = RAW


for v, t in X_T.items():
    emb(t, [XV[v], PCX[v // SUB]])
for v, t in Y_T.items():
    emb(t, [YV[v], PCY[v // SUB]])
for v, t in A_T.items():
    emb(t, [AV[v], PAD])
for v, t in EX_T.items():
    emb(t, [EXV[v], PAD])
for v, t in EY_T.items():
    emb(t, [EYV[v], PAD])
for v, t in HP_T.items():
    emb(t, [HPV[v], PAD])
for v, t in T_T.items():
    emb(t, [TV[v], HXC[HASHCELL[v][0]], HYC[HASHCELL[v][1]]])
for i, t in I_T.items():
    emb(t, [IV[i], PAD])
for b, t in HIT_T.items():
    emb(t, [HITV[b], PAD])
for b, t in ATK_T.items():
    emb(t, [ATKV[b], PAD])
for b, t in VISA_T.items():
    emb(t, [VISV[b], PAD])
for b, t in VISR_T.items():
    emb(t, [VISV[b], PAD])
for v, t in RDXA_T.items():
    emb(t, [RDXV[v + RMAX], PAD])
for v, t in RDYA_T.items():
    emb(t, [RDYV[v + RMAX], PAD])
for v, t in RDXR_T.items():
    emb(t, [RDXV[v + RMAX], PAD])
for v, t in RDYR_T.items():
    emb(t, [RDYV[v + RMAX], PAD])
for t in list(SPR_T.values()) + [SPRN_T, OK_T, DEAD_T, KILL_T] + list(C_T.values()):
    emb(t, [PAD])  # never fetched: identity does not matter, variance does
emb(NOOP, [BAL, PAD])

wpe = np.zeros((MAXPOS, N_EMBD), dtype=np.float32)
for p in range(MAXPOS):
    wpe[p, POSB[p]] = 1.0

# ---- attention: 22 absolute selectors, sink-by-default ----------------------
# (target position, [(source block, destination block), ...])
HEADS = [
    (P_X, [(XV, GX), (PCX, GPCX)]),
    (P_Y, [(YV, GY), (PCY, GPCY)]),
    (P_A, [(AV, GA)]),
    (P_EX, [(EXV, GEX)]),
    (P_EY, [(EYV, GEY)]),
    (P_HP, [(HPV, GHP)]),
    (P_T, [(TV, GT), (HXC, GHXC), (HYC, GHYC)]),
    (P_I, [(IV, GI)]),
    (P_XN, [(XV, GXN), (PCX, GPCXN)]),
    (P_YN, [(YV, GYN), (PCY, GPCYN)]),
    (P_AN, [(AV, GAN)]),
    (P_RDXA, [(RDXV, GRDXA)]),
    (P_RDYA, [(RDYV, GRDYA)]),
    (P_VISA, [(VISV, GVISA)]),
    (P_HIT, [(HITV, GHIT)]),
    (P_EXN, [(EXV, GEXN)]),
    (P_EYN, [(EYV, GEYN)]),
    (P_ATK, [(ATKV, GATK)]),
    (P_HPN, [(HPV, GHPN)]),
    (P_RDXR, [(RDXV, GRDXR)]),
    (P_RDYR, [(RDYV, GRDYR)]),
    (P_VISR, [(VISV, GVISR)]),
]
assert len(HEADS) <= N_HEAD

Wqkv = np.zeros((3 * N_EMBD, N_EMBD), dtype=np.float32)
bqkv = np.zeros(3 * N_EMBD, dtype=np.float32)
vrow = 2 * N_EMBD
for h, (target, copies) in enumerate(HEADS):
    assert sum(len(src) for src, _ in copies) <= HEAD_D - 1, f"head {h} overflows"
    qrow, krow = h * HEAD_D, N_EMBD + h * HEAD_D
    for j in range(MAXPOS):
        Wqkv[krow + j, POSB[j]] = 1.0
    Wqkv[krow + SINKDIM, BAL] = 1.0  # only <NOOP> answers the sink
    bqkv[qrow + target] = SEL_T * np.sqrt(HEAD_D)
    bqkv[qrow + SINKDIM] = SEL_S * np.sqrt(HEAD_D)
    d = 0
    for src, dst in copies:
        for i in range(len(src)):
            Wqkv[vrow + h * HEAD_D + d + i, src[i]] = 1.0
        d += len(src)
    Wqkv[vrow + h * HEAD_D + HEAD_D - 1, BAL] = 1.0  # ballast rides along

Wattn = np.zeros((N_EMBD, N_EMBD), dtype=np.float32)
battn = np.zeros(N_EMBD, dtype=np.float32)
for h, (target, copies) in enumerate(HEADS):
    d = 0
    for src, dst in copies:
        for i in range(len(src)):
            Wattn[dst[i], h * HEAD_D + d + i] = 1.0
        d += len(src)
    Wattn[BALK[h], h * HEAD_D + HEAD_D - 1] = 1.0

# ---- MLP: the factored circuits ---------------------------------------------
# neuron = (positive input dims, [(output dim, weight)], group)
hidden = []

# player ROM: (x, y, a, input) -> new state + the whole wall frame
for s in PSTATES:
    for i in range(NIN):
        s2 = PTRANS[(s, i)]
        outs = [(NX[s2[0]], 1.0), (NY[s2[1]], 1.0), (NA[s2[2]], 1.0)]
        for ci, (hh, sd) in enumerate(FRAME[s2]):
            outs.append((FCOL[ci][(hh - 1) * NSHADE + sd], 1.0))
        hidden.append(([GX[s[0]], GY[s[1]], GA[s[2]], GI[i]], outs, "rom"))

# relative position, aim flavor: (player cell', enemy cell) -> offset
for i in range(GC):
    for j in range(GC):
        hidden.append(([GPCXN[i], GEX[j]], [(NRDXA[j - i + RMAX], 1.0)], "relxa"))
        hidden.append(([GPCYN[i], GEY[j]], [(NRDYA[j - i + RMAX], 1.0)], "relya"))
        hidden.append(([GPCXN[i], GEXN[j]], [(NRDXR[j - i + RMAX], 1.0)], "relxr"))
        hidden.append(([GPCYN[i], GEYN[j]], [(NRDYR[j - i + RMAX], 1.0)], "relyr"))

# line of sight (only TRUE pairs get a neuron; false -> baseline 0 wins)
for pc in OPENC:
    for ec in OPENC:
        if LOS[(pc, ec)]:
            hidden.append(
                ([GPCXN[pc[0]], GPCYN[pc[1]], GEX[ec[0]], GEY[ec[1]]],
                 [(VISFA, 1.0)], "losa"))
            hidden.append(
                ([GPCXN[pc[0]], GPCYN[pc[1]], GEXN[ec[0]], GEYN[ec[1]]],
                 [(VISFR, 1.0)], "losr"))

# aim: enemy near the crosshair + LOS + fire -> HIT
for (rdx, rdy, a), g in SPRITE.items():
    if g is not None and g[0] in AIM_COLS:
        hidden.append(
            ([GRDXA[rdx + RMAX], GRDYA[rdy + RMAX], GAN[a], GVISA[1],
              GI[INPUTS.index("S")]], [(HITF, 1.0)], "aim"))

# enemy update: real BFS pathfinding as a (enemy cell, player cell) table,
# gated by HIT=0; respawn gated by HIT=1
for ec in OPENC:
    for pc in OPENC:
        e2 = EMOVE[(ec, pc)]
        hidden.append(
            ([GEX[ec[0]], GEY[ec[1]], GPCXN[pc[0]], GPCYN[pc[1]], GHIT[0]],
             [(NEX[e2[0]], 1.0), (NEY[e2[1]], 1.0)], "emove"))
for cx in range(GC):
    hidden.append(([GHXC[cx], GHIT[1]], [(NEX[cx], 1.0)], "respawn"))
    hidden.append(([GHYC[cx], GHIT[1]], [(NEY[cx], 1.0)], "respawn"))

# tick counter
for t in range(TMOD):
    hidden.append(([GT[t]], [(NT[(t + 1) % TMOD], 1.0)], "tinc"))

# bite: enemy (new pos) within a king move of the player (new cell)
for pc in OPENC:
    for ec in OPENC:
        if max(abs(pc[0] - ec[0]), abs(pc[1] - ec[1])) <= 1:
            hidden.append(
                ([GPCXN[pc[0]], GPCYN[pc[1]], GEXN[ec[0]], GEYN[ec[1]]],
                 [(ATKF, 1.0)], "adj"))

# hit points: decrement on ATK=1 (floor 0), copy on ATK=0
for h in range(HP0 + 1):
    hidden.append(([GHP[h], GATK[1]], [(NHP[max(h - 1, 0)], 1.0)], "hp"))
    hidden.append(([GHP[h], GATK[0]], [(NHP[h], 1.0)], "hp"))

# state: DEAD if HP'==0, KILL if the shot connected (priorities in Wout)
hidden.append(([GHPN[0]], [(SD, 1.0)], "stdead"))
hidden.append(([GHIT[1]], [(SKILL, 1.0)], "stkill"))

# sprite: in-cone geometry gated by line of sight
for (rdx, rdy, a), g in SPRITE.items():
    if g is not None:
        hidden.append(
            ([GRDXR[rdx + RMAX], GRDYR[rdy + RMAX], GAN[a], GVISR[1]],
             [(NSPRC[g[0]], 1.0), (NSPRS[g[1] - 1], 1.0)], "sprite"))

H = len(hidden)
GROUPS = sorted({g for _, _, g in hidden})
# first row from which each group's inputs are all visible (max fetch pos)
GROUP_ROW = {
    "rom": P_I, "relxa": P_XN, "relya": P_YN, "relxr": P_EXN, "relyr": P_EYN,
    "losa": P_YN, "losr": P_EYN, "aim": P_VISA, "emove": P_HIT,
    "respawn": P_HIT, "tinc": P_I, "adj": P_EYN, "hp": P_ATK,
    "stdead": P_HPN, "stkill": P_HIT, "sprite": P_VISR,
}

Wup = np.zeros((H, N_EMBD), dtype=np.float32)
bup = np.zeros(H, dtype=np.float32)
Wdown = np.zeros((N_EMBD, H), dtype=np.float32)
bdown = np.zeros(N_EMBD, dtype=np.float32)


def build_mlp(thr):
    Wup[:] = 0
    bup[:] = 0
    Wdown[:] = 0
    for h, (pos, outs, g) in enumerate(hidden):
        for d in pos:
            Wup[h, d] = S
        bup[h] = -S * thr[g]
        for d, w in outs:
            Wdown[d, h] = DSCALE * w


# ---- unembedding: circuit value + phase bonus --------------------------------
# Every token type gets a bonus from the POSB column of its generation row;
# "zero/none" baseline tokens get bonus B+delta so they win when no circuit
# fires and lose (by D-delta) when one does.
Wout = np.zeros((V, N_EMBD), dtype=np.float32)
for v, t in X_T.items():
    Wout[t, NX[v]] = 1.0
for v, t in Y_T.items():
    Wout[t, NY[v]] = 1.0
for v, t in A_T.items():
    Wout[t, NA[v]] = 1.0
for v, t in RDXA_T.items():
    Wout[t, NRDXA[v + RMAX]] = 1.0
for v, t in RDYA_T.items():
    Wout[t, NRDYA[v + RMAX]] = 1.0
for v, t in RDXR_T.items():
    Wout[t, NRDXR[v + RMAX]] = 1.0
for v, t in RDYR_T.items():
    Wout[t, NRDYR[v + RMAX]] = 1.0
Wout[VISA_T[1], VISFA] = 1.0
Wout[VISR_T[1], VISFR] = 1.0
Wout[HIT_T[1], HITF] = 1.0
Wout[ATK_T[1], ATKF] = 1.0
for v, t in EX_T.items():
    Wout[t, NEX[v]] = 1.0
for v, t in EY_T.items():
    Wout[t, NEY[v]] = 1.0
for v, t in T_T.items():
    Wout[t, NT[v]] = 1.0
for v, t in HP_T.items():
    Wout[t, NHP[v]] = 1.0
Wout[DEAD_T, SD] = 6.0
Wout[KILL_T, SKILL] = 4.0
for (c, z), t in SPR_T.items():
    Wout[t, NSPRC[c]] = 1.0
    Wout[t, NSPRS[z - 1]] = 1.0
for (ci, hh, sd), t in C_T.items():
    Wout[t, FCOL[ci][(hh - 1) * NSHADE + sd]] = 1.0

# token type -> (generation row, is_baseline)
SLOTS = (
    [(t, P_XN - 1, False) for t in X_T.values()]
    + [(t, P_YN - 1, False) for t in Y_T.values()]
    + [(t, P_AN - 1, False) for t in A_T.values()]
    + [(t, P_RDXA - 1, False) for t in RDXA_T.values()]
    + [(t, P_RDYA - 1, False) for t in RDYA_T.values()]
    + [(VISA_T[1], P_VISA - 1, False), (VISA_T[0], P_VISA - 1, True)]
    + [(HIT_T[1], P_HIT - 1, False), (HIT_T[0], P_HIT - 1, True)]
    + [(t, P_EXN - 1, False) for t in EX_T.values()]
    + [(t, P_EYN - 1, False) for t in EY_T.values()]
    + [(t, P_TN - 1, False) for t in T_T.values()]
    + [(ATK_T[1], P_ATK - 1, False), (ATK_T[0], P_ATK - 1, True)]
    + [(t, P_HPN - 1, False) for t in HP_T.values()]
    + [(DEAD_T, P_ST - 1, False), (KILL_T, P_ST - 1, False), (OK_T, P_ST - 1, True)]
    + [(t, P_RDXR - 1, False) for t in RDXR_T.values()]
    + [(t, P_RDYR - 1, False) for t in RDYR_T.values()]
    + [(VISR_T[1], P_VISR - 1, False), (VISR_T[0], P_VISR - 1, True)]
    + [(t, P_SPR - 1, False) for t in SPR_T.values()]
    + [(SPRN_T, P_SPR - 1, True)]
    + [(t, P_C0 - 1 + ci, False) for (ci, _, _), t in C_T.items()]
)


def set_phase_bonus(B, delta):
    for t, row, base in SLOTS:
        Wout[t, POSB[row]] = B + (delta if base else 0.0)


gamma1 = np.ones(N_EMBD, dtype=np.float32)
beta1 = np.zeros(N_EMBD, dtype=np.float32)
gamma2 = gamma1.copy()
beta2 = beta1.copy()
gammaf = gamma1.copy()
betaf = beta1.copy()


def quantize_f16():
    for a in (wte, wpe, Wqkv, Wattn, Wup, Wdown, Wout):
        a[:] = a.astype(np.float16).astype(np.float32)


# =================== FAITHFUL FORWARD (same tensors as the GGUF) =============
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


# ==================== REFERENCE: one full tick ================================
def ref_tick(st, i):
    """st = (px, py, a, ex, ey, hp, t); returns (ids, st', info)."""
    px, py, a, ex, ey, hp, t = st
    p2 = PTRANS[((px, py, a), i)]
    pc = (p2[0] // SUB, p2[1] // SUB)
    rda = (ex - pc[0], ey - pc[1])
    visa = LOS[(pc, (ex, ey))]
    g = SPRITE.get((rda[0], rda[1], p2[2]))
    hit = INPUTS[i] == "S" and visa and g is not None and g[0] in AIM_COLS
    if hit:
        e2 = HASHCELL[t]
    else:
        e2 = EMOVE[((ex, ey), pc)]
    t2 = (t + 1) % TMOD
    atk = max(abs(pc[0] - e2[0]), abs(pc[1] - e2[1])) <= 1
    hp2 = max(hp - 1, 0) if atk else hp
    stt = DEAD_T if hp2 == 0 else KILL_T if hit else OK_T
    rdr = (e2[0] - pc[0], e2[1] - pc[1])
    visr = LOS[(pc, e2)]
    gr = SPRITE.get((rdr[0], rdr[1], p2[2]))
    spr = SPR_T[(gr[0], gr[1])] if (visr and gr is not None) else SPRN_T
    ids = [NOOP, X_T[px], Y_T[py], A_T[a], EX_T[ex], EY_T[ey], HP_T[hp], T_T[t],
           I_T[i],
           X_T[p2[0]], Y_T[p2[1]], A_T[p2[2]],
           RDXA_T[rda[0]], RDYA_T[rda[1]], VISA_T[int(visa)], HIT_T[int(hit)],
           EX_T[e2[0]], EY_T[e2[1]], T_T[t2], ATK_T[int(atk)], HP_T[hp2], stt,
           RDXR_T[rdr[0]], RDYR_T[rdr[1]], VISR_T[int(visr)], spr]
    ids += [C_T[(ci, hh, sd)] for ci, (hh, sd) in enumerate(FRAME[p2])]
    assert len(ids) == TICKLEN
    st2 = (p2[0], p2[1], p2[2], e2[0], e2[1], hp2, t2)
    info = dict(p2=p2, pc=pc, rda=rda, visa=visa, hit=hit, e2=e2, t=t,
                atk=atk, hp=hp, hp2=hp2, rdr=rdr, visr=visr, gr=gr)
    return ids, st2, info


def firing(info, i):
    """group -> list of input-dims tuples of the neurons that must fire."""
    f = {}
    p2, pc, e2 = info["p2"], info["pc"], info["e2"]
    f["relxa"] = [(GPCXN[pc[0]], GEX[pc[0] + info["rda"][0]])]
    f["relya"] = [(GPCYN[pc[1]], GEY[pc[1] + info["rda"][1]])]
    f["relxr"] = [(GPCXN[pc[0]], GEXN[e2[0]])]
    f["relyr"] = [(GPCYN[pc[1]], GEYN[e2[1]])]
    if info["visa"]:
        ex, ey = pc[0] + info["rda"][0], pc[1] + info["rda"][1]
        f["losa"] = [(GPCXN[pc[0]], GPCYN[pc[1]], GEX[ex], GEY[ey])]
    if info["visr"]:
        f["losr"] = [(GPCXN[pc[0]], GPCYN[pc[1]], GEXN[e2[0]], GEYN[e2[1]])]
    if info["hit"]:
        f["aim"] = [(GRDXA[info["rda"][0] + RMAX], GRDYA[info["rda"][1] + RMAX],
                     GAN[p2[2]], GVISA[1], GI[INPUTS.index("S")])]
        f["stkill"] = [(GHIT[1],)]
        hx, hy = HASHCELL[info["t"]]
        f["respawn"] = [(GHXC[hx], GHIT[1]), (GHYC[hy], GHIT[1])]
    else:
        ex, ey = pc[0] + info["rda"][0], pc[1] + info["rda"][1]
        f["emove"] = [(GEX[ex], GEY[ey], GPCXN[pc[0]], GPCYN[pc[1]], GHIT[0])]
    f["tinc"] = [(GT[info["t"]],)]
    if info["atk"]:
        f["adj"] = [(GPCXN[pc[0]], GPCYN[pc[1]], GEXN[e2[0]], GEYN[e2[1]])]
    f["hp"] = [(GHP[info["hp"]], GATK[int(info["atk"])])]
    if info["hp2"] == 0:
        f["stdead"] = [(GHPN[0],)]
    if info["visr"] and info["gr"] is not None:
        f["sprite"] = [(GRDXR[info["rdr"][0] + RMAX], GRDYR[info["rdr"][1] + RMAX],
                        GAN[p2[2]], GVISR[1])]
    return f


# ---- game policies (coverage) ------------------------------------------------
def start_state():
    return (SPAWN[0], SPAWN[1], SPAWN[2], ESPAWN[0], ESPAWN[1], HP0, 0)


def aim_error(st):
    """Signed column error of the enemy w.r.t. the crosshair (None if off-cone)."""
    px, py, a, ex, ey, _, _ = st
    pc = (px // SUB, py // SUB)
    g = SPRITE.get((ex - pc[0], ey - pc[1], a))
    return None if g is None else g[0] - (NCOLS - 1) / 2


def make_hunter(rng):
    """Turns toward the enemy and fires when it is centered and visible.
    The enemy chases the player, so standing ground works: it comes to us."""

    def choose(st):
        px, py, a, ex, ey, _, _ = st
        pc = (px // SUB, py // SUB)
        err = aim_error(st)
        seen = LOS[(pc, (ex, ey))]
        if err is not None and abs(err) <= 2 and seen:
            return INPUTS.index("S")
        if err is not None:  # in the cone: micro-adjust toward the crosshair
            return INPUTS.index("R" if err > 0 else "L")
        if not seen and rng.random() < 0.35:
            return INPUTS.index("F")
        diff = np.arctan2(ey - pc[1], ex - pc[0]) - theta(a)
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return INPUTS.index("R" if diff > 0 else "L")

    return choose


def make_bait(rng):
    """Spins in place without ever firing: the enemy walks up and bites it
    to death (adjacency attacks + the DEAD state for calibration)."""

    def choose(st):
        return INPUTS.index("L" if rng.random() < 0.8 else "F")

    return choose


def make_random(rng):
    def choose(st):
        return int(rng.integers(NIN))

    return choose


def games(seed, n_ticks=60):
    rng = np.random.default_rng(seed)
    out = []
    for mk in (make_hunter, make_hunter, make_hunter, make_bait, make_bait,
               make_random, make_random, make_random):
        policy = mk(rng)
        st = start_state()
        for _ in range(n_ticks):
            i = policy(st)
            ids, st2, info = ref_tick(st, i)
            out.append((st, i, ids, info))
            if st2[5] == 0:
                break
            st = st2
    return out


# ======================= PER-GROUP CALIBRATION ================================
def calibrate(seed=11):
    idx_of = {}
    for h, (pos, _, g) in enumerate(hidden):
        idx_of[(g, tuple(pos))] = h
    P = np.zeros((H, N_EMBD), dtype=np.float32)
    gidx = {g: [] for g in GROUPS}
    for h, (pos, _, g) in enumerate(hidden):
        P[h, pos] = 1.0
        gidx[g].append(h)
    gidx = {g: np.array(v) for g, v in gidx.items()}
    must = {g: np.inf for g in GROUPS}
    notm = {g: -np.inf for g in GROUPS}
    stats = dict(hits=0, deaths=0, atks=0, vis=0, blocked=0, respawn_seen=0)
    ticks = games(seed)
    for st, i, ids, info in ticks:
        M, _, _ = forward_all(ids)
        sums = M[GEN0:TICKLEN - 1] @ P.T  # [49, H] ; row r = position GEN0+r
        fire = firing(info, i)
        fire["rom"] = [tuple([GX[st[0]], GY[st[1]], GA[st[2]], GI[i]])]
        stats["hits"] += info["hit"]
        stats["deaths"] += info["hp2"] == 0
        stats["atks"] += info["atk"]
        stats["vis"] += ids[P_SPR] != SPRN_T
        stats["blocked"] += INPUTS[i] in "FB" and info["p2"][:2] == st[:2]
        for g in GROUPS:
            cols = gidx[g]
            sub = sums[:, cols]
            drop = []
            r0 = GROUP_ROW[g] - GEN0  # designated neurons fire from this row on
            for key in fire.get(g, []):
                h = idx_of[(g, key)]
                j = int(np.where(cols == h)[0][0])
                must[g] = min(must[g], sub[r0:, j].min())
                drop.append(j)
            if drop:
                sub = np.delete(sub, drop, axis=1)
            if sub.size:
                notm[g] = max(notm[g], sub.max())
    print(f"  coverage: {stats}")
    assert stats["hits"] >= 3 and stats["deaths"] >= 1 and stats["atks"] >= 5
    assert stats["vis"] >= 10 and stats["blocked"] >= 1
    thr = {}
    ok = True
    for g in GROUPS:
        assert np.isfinite(must[g]), f"group {g}: no must-fire event covered"
        gap = notm[g] < must[g]
        ok &= gap
        thr[g] = float((notm[g] + must[g]) / 2)
        print(f"  {g:8s}: must>={must[g]:7.2f}  mustnot<={notm[g]:7.2f}  "
              f"gap {'OK' if gap else 'OVERLAP'}  thr={thr[g]:.2f}")
    assert ok, "overlapping gap: rework boosts or groups"
    return thr


def calibrate_phase(seed=17):
    """B (type separation) and delta (baseline margin) from measurements."""
    data_dims = sorted(set(d for _, outs, _ in hidden for d, _ in outs))
    dmax, dmin, pmin = 0.0, np.inf, np.inf
    for st, i, ids, info in games(seed, n_ticks=12):
        _, XLn, _ = forward_all(ids)
        for r in range(GEN0, TICKLEN - 1):
            dmax = max(dmax, XLn[r, data_dims].max())
            pmin = min(pmin, XLn[r, POSB[r]])
            tok = ids[r + 1]
            nz = np.nonzero(Wout[tok, :POSB[0]])[0]
            if len(nz):  # value of the correct token's own data dims
                dmin = min(dmin, XLn[r, nz].min())
    assert pmin > 0 and dmin > 0
    B = 3.0 * dmax / pmin
    delta = 0.5 * dmin / pmin  # in bonus units: delta*P ~ half the weakest data
    print(f"  phase bonus: Dmax={dmax:.2f} Dmin={dmin:.2f} Pmin={pmin:.2f} "
          f"-> B={B:.1f} delta={delta:.1f}")
    return B, delta


# ============================ VERIFICATION ====================================
def verify(seed=500):
    ok = tot = 0
    bad = []
    for st, i, ids, info in games(seed, n_ticks=45):
        _, _, logits = forward_all(ids)
        for r in range(GEN0, TICKLEN - 1):
            pred = int(logits[r].argmax())
            tot += 1
            if pred == ids[r + 1]:
                ok += 1
            else:
                bad.append((r + 1, tokens[pred], tokens[ids[r + 1]]))
    if bad:
        print(f"  first errors: {bad[:8]}")
    return ok, tot


# ============================== MAIN =========================================
if __name__ == "__main__":
    print(f"map {GC}x{GC}, {len(OPEN)} player sub-cells, {len(OPENC)} enemy "
          f"cells, {len(PSTATES)} player states")
    print(f"n_embd={N_EMBD} n_head={N_HEAD} (live {len(HEADS)}) head_dim={HEAD_D} "
          f"n_ff={H} vocab={V} ctx={MAXPOS} ({N_GEN} tokens/tick)")
    quantize_f16()
    print("calibrating thresholds (real forward, f16 weights):")
    build_mlp(calibrate())
    quantize_f16()
    set_phase_bonus(*calibrate_phase())
    quantize_f16()
    ok, tot = verify()
    print(f"verification: {ok}/{tot} tokens correct ({100 * ok / tot:.1f}%)")
    assert ok == tot, "verification failed: not writing the GGUF"

    w = gguf.GGUFWriter(OUT, "gpt2")
    w.add_name("doom-like")
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
    print(f"written {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")
