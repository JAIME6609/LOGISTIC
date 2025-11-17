# -*- coding: utf-8 -*-
"""
Deep Topological Intelligence for Sustainable Cross-Border Logistics
====================================================================

Version with robust Wasserstein distances for H0, H1 and H2
and safe Excel export.

- Key features:
  * Custom StandardScaler and train_test_split (no scikit-learn)
  * Manual graph report (replaces nx.info, removed in NetworkX>=3)
  * Synthetic multimodal logistics network (port, airport, customs, DC, city)
  * Edge attributes (dist, time, mode, cap, emis, cost) and OD flows
  * Network metrics (diameter, average path length, clustering, assortativity,
    global efficiency)
  * Embeddings with Autoencoders (two latent sizes) and matrix reconstruction
  * TDA with ripser/persim (persistence diagrams, Wasserstein distances,
    Betti curves); Wasserstein distances computed for H0, H1, H2
  * Resilience tests (random and betweenness-based edge removal)
  * Visualizations and export to Excel (with fallback name if the file is busy)

Requirements (the script will try to install them if missing):
  tensorflow, numpy, pandas, matplotlib, networkx,
  openpyxl, xlsxwriter, ripser, persim
"""

# ===============================================================
# 0) Imports, utilities, and reproducible environment
# ===============================================================
import os, sys, subprocess, importlib, gc, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd

# Non-interactive backend for figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx

# --- Package installer (avoids scikit-learn) ---
def _ensure_package(pkg):
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        try:
            print(f"[Installing] {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            importlib.import_module(pkg)
            return True
        except Exception as e:
            print(f"[Warning] Could not install {pkg}: {e}")
            return False

PKGS = [
    "tensorflow","numpy","pandas","matplotlib","networkx",
    "openpyxl","xlsxwriter","ripser","persim"
]
for _p in PKGS:
    _ensure_package(_p)

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

# TDA
from ripser import ripser
from persim import wasserstein, plot_diagrams

# Seeds
NP_SEED = 123
PY_SEED = 123
TF_SEED = 123
np.random.seed(NP_SEED)
random.seed(PY_SEED)
tf.random.set_seed(TF_SEED)

# Info GPU
print("[TF] Built with CUDA:", tf.test.is_built_with_cuda())
print("[TF] Physical GPUs:", tf.config.list_physical_devices('GPU'))

# Output directories
OUT_DIR = Path("logistics_dti_outputs")
IMG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUT_DIR / "DTI_logistics_results.xlsx"

# Figure style
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.grid": True
})

# ===============================================================
# Custom utilities replacing scikit-learn
# ===============================================================
class SimpleStandardScaler:
    """Standard scaler: (X - mean) / std per column."""
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.with_mean:
            self.mean_ = X.mean(axis=0, keepdims=True)
        else:
            self.mean_ = np.zeros((1, X.shape[1]), dtype=np.float32)
        if self.with_std:
            std = X.std(axis=0, keepdims=True)
            std = np.where(std < 1e-12, 1.0, std)
            self.scale_ = std
        else:
            self.scale_ = np.ones((1, X.shape[1]), dtype=np.float32)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def simple_train_test_split(X, test_size=0.2, random_state=123):
    """Reproducible random partition without scikit-learn."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    te_idx = idx[:n_test]
    tr_idx = idx[n_test:]
    return X[tr_idx], X[te_idx]

def mse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean((y_true - y_pred) ** 2))

# ===============================================================
# 1) General experiment parameters
# ===============================================================
N_NODES      = 160
AREA_SIDE    = 1000.0
MODES        = ["sea","air","rail","road"]
P_CONN_BASE  = 0.045
EMISS_FACT   = {"sea": 10.0, "air": 200.0, "rail": 15.0, "road": 60.0}
V_CAP_MODE   = {"sea": 3000., "air": 120., "rail": 800., "road": 200.}
SPEED_MODE   = {"sea": 30., "air": 800., "rail": 70., "road": 65.}
ALPHA_COST   = 1.0
BETA_TIME    = 1.0
GAMMA_EMISS  = 0.25

# Autoencoder
EPOCHS       = 120
BATCH_SIZE   = 64
LATENT_1D    = 8
LATENT_2D    = 16
LR           = 1e-3

# TDA
TDA_MAXDIM   = 2
TDA_SUBS     = 150

# Resilience
ATTACK_FRAC  = 0.15
ATTACK_TYPES = ["random_edges", "betweenness_edges"]

# ===============================================================
# 2) Synthetic logistics network generation
# ===============================================================
rng = np.random.default_rng(NP_SEED)

def _sample_node_types(n):
    labels = []
    for _ in range(n):
        r = rng.random()
        if r < 0.50: labels.append("city")
        elif r < 0.70: labels.append("dc")
        elif r < 0.85: labels.append("customs")
        elif r < 0.95: labels.append("port")
        else: labels.append("airport")
    return labels

def _mode_from_pair(a, b):
    pair = {a, b}
    if "port" in pair and ("city" in pair or "dc" in pair or "customs" in pair):
        return "sea"
    if "airport" in pair and ("city" in pair or "dc" in pair or "customs" in pair):
        return "air"
    if "dc" in pair and "city" in pair:
        return "road"
    return rng.choice(MODES, p=[0.25, 0.10, 0.20, 0.45])

def _distance(p, q):
    return float(np.linalg.norm(p - q))

# Create graph
G = nx.Graph()
positions = {}
types = _sample_node_types(N_NODES)
for i in range(N_NODES):
    x = rng.uniform(0, AREA_SIDE)
    y = rng.uniform(0, AREA_SIDE)
    positions[i] = np.array([x, y], dtype=np.float32)
    G.add_node(i, pos=positions[i], ntype=types[i])

# Connect with distance-decaying probability
for i in range(N_NODES):
    for j in range(i+1, N_NODES):
        d = _distance(positions[i], positions[j])
        p = P_CONN_BASE * math.exp(-(d/250.0)**2)
        if rng.random() < p:
            m = _mode_from_pair(types[i], types[j])
            cap = V_CAP_MODE[m] * rng.uniform(0.7, 1.3)
            speed = SPEED_MODE[m] * rng.uniform(0.9, 1.1)
            ttime = d / max(speed, 1e-3)
            emis  = (EMISS_FACT[m] * d) * rng.uniform(0.9, 1.1)
            cost  = ALPHA_COST*d + BETA_TIME*ttime + GAMMA_EMISS*(emis/1000.0)
            G.add_edge(i, j,
                       dist=d, time=ttime, mode=m, cap=cap,
                       emis=emis, cost=cost)

# Connect components (ensure connectivity)
cc = list(nx.connected_components(G))
if len(cc) > 1:
    print(f"[Info] Initial graph had {len(cc)} components. Connecting…")
    reps = [list(c) for c in cc]
    while len(reps) > 1:
        A_comp = reps.pop()
        B_comp = reps[0]
        best = None
        best_d = 1e9
        for u in A_comp:
            for v in B_comp:
                d = _distance(positions[u], positions[v])
                if d < best_d:
                    best_d = d
                    best = (u, v, d)
        u, v, d = best
        m = _mode_from_pair(types[u], types[v])
        cap = V_CAP_MODE[m] * rng.uniform(0.8, 1.2)
        speed = SPEED_MODE[m] * rng.uniform(0.9, 1.1)
        ttime = d / max(speed, 1e-3)
        emis  = (EMISS_FACT[m] * d) * rng.uniform(0.95, 1.05)
        cost  = ALPHA_COST*d + BETA_TIME*ttime + GAMMA_EMISS*(emis/1000.0)
        G.add_edge(u, v, dist=d, time=ttime, mode=m, cap=cap, emis=emis, cost=cost)
        cc = list(nx.connected_components(G))
        reps = [list(c) for c in cc]

# Manual report (replaces nx.info)
def graph_report(G):
    comps = nx.number_connected_components(G)
    return f"[Graph] nodes={G.number_of_nodes()} | edges={G.number_of_edges()} | components={comps}"

print(graph_report(G))

# ===============================================================
# 3) OD demand and flow assignment (shortest path)
# ===============================================================
N_OD = max(30, N_NODES // 4)
nodes = list(G.nodes())
OD_pairs = []
for _ in range(N_OD):
    s, t = rng.choice(nodes, size=2, replace=False)
    OD_pairs.append((s, t, rng.uniform(5.0, 25.0)))  # ton demand

def _generalized_cost(u, v, data):
    return float(data["cost"])

total_cost, total_emis, total_dist, total_time = 0.0, 0.0, 0.0, 0.0

for s, t, demand in OD_pairs:
    try:
        path = nx.shortest_path(G, s, t, weight=lambda u,v,d: _generalized_cost(u,v,d))
        for u, v in zip(path[:-1], path[1:]):
            d = G[u][v]["dist"]
            tt = G[u][v]["time"]
            em = G[u][v]["emis"] * (demand/10.0)
            co = G[u][v]["cost"] * (1.0 + 0.05*(demand/10.0))
            G[u][v]["flow"] = G[u][v].get("flow", 0.0) + demand
            G[u][v]["emis_flow"] = G[u][v].get("emis_flow", 0.0) + em
            G[u][v]["cost_flow"] = G[u][v].get("cost_flow", 0.0) + co
            total_dist += d
            total_time += tt
            total_emis += em
            total_cost += co
    except Exception:
        continue

print(f"[Flows] Total distance (km): {total_dist:.2f} | Time (h): {total_time:.2f} | "
      f"Emissions (kgCO2~): {total_emis/1000.0:.2f} | Cost-index: {total_cost:.2f}")

# ===============================================================
# 4) Structural metrics and matrices
# ===============================================================
A = nx.to_numpy_array(G, weight=None, dtype=np.float32)
W_cost = nx.to_numpy_array(G, weight="cost", dtype=np.float32)
W_dist = nx.to_numpy_array(G, weight="dist", dtype=np.float32)
W_time = nx.to_numpy_array(G, weight="time", dtype=np.float32)

diameter = nx.diameter(G) if nx.is_connected(G) else np.nan
avg_path_len = nx.average_shortest_path_length(G)
clustering = nx.average_clustering(G)
assort = nx.degree_assortativity_coefficient(G)

def global_efficiency(G):
    sp = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    s = 0.0
    cnt = 0
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                d = sp[i].get(j, np.inf)
                if d > 0 and np.isfinite(d):
                    s += 1.0 / d
                    cnt += 1
    return s / max(cnt, 1)

glob_eff = global_efficiency(G)

print(f"[Structure] diameter={diameter} | avg_path_len={avg_path_len:.3f} | "
      f"clustering={clustering:.3f} | assort={assort:.3f} | global_eff={glob_eff:.3f}")

# Geodesic matrix (dijkstra on 'dist')
sp_len = dict(nx.all_pairs_dijkstra_path_length(G, weight="dist"))
D_geo = np.full((N_NODES, N_NODES), fill_value=np.inf, dtype=np.float32)
for i in range(N_NODES):
    for j, d in sp_len[i].items():
        D_geo[i, j] = d
D_geo = np.minimum(D_geo, D_geo.T)
max_finite = np.nanmax(D_geo[np.isfinite(D_geo)])
D_geo[~np.isfinite(D_geo)] = max_finite * 1.25
D_geo = (D_geo / (D_geo.max() + 1e-6)).astype(np.float32)

# ===============================================================
# 5) Node embeddings and Autoencoders
# ===============================================================
def _row_norm(M):
    M = M.copy().astype(np.float32)
    if M.size == 0:
        return M
    mn = M.min()
    mx = M.max()
    if mx > mn:
        M = (M - mn)/(mx - mn)
    else:
        M[:] = 0.0
    return M

A_bin   = A
C_norm  = _row_norm(W_cost)
D_norm  = _row_norm(W_dist)
T_norm  = _row_norm(W_time)

X_feat = np.hstack([A_bin, C_norm, D_norm, T_norm]).astype(np.float32)

scaler = SimpleStandardScaler(with_mean=True, with_std=True)
X_feat_sc = scaler.fit_transform(X_feat).astype(np.float32)

X_tr, X_te = simple_train_test_split(X_feat_sc, test_size=0.2, random_state=NP_SEED)

def build_ae(input_dim, latent_dim):
    inp = layers.Input(shape=(input_dim,), dtype="float32")
    h   = layers.Dense(256, activation="relu")(inp)
    h   = layers.Dense(128, activation="relu")(h)
    z   = layers.Dense(latent_dim, name=f"latent_{latent_dim}")(h)
    hz  = layers.Dense(128, activation="relu")(z)
    hz  = layers.Dense(256, activation="relu")(hz)
    out = layers.Dense(input_dim, name="recon")(hz)
    ae  = Model(inp, out, name=f"AE_{latent_dim}")
    ae.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    enc = Model(inp, z, name=f"ENC_{latent_dim}")
    return ae, enc

input_dim = X_feat_sc.shape[1]
AE_1, ENC_1 = build_ae(input_dim, LATENT_1D)
AE_2, ENC_2 = build_ae(input_dim, LATENT_2D)

h1 = AE_1.fit(X_tr, X_tr, validation_data=(X_te, X_te),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
h2 = AE_2.fit(X_tr, X_tr, validation_data=(X_te, X_te),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

print(f"[AE_{LATENT_1D}] final train={h1.history['loss'][-1]:.6f} | val={h1.history['val_loss'][-1]:.6f}")
print(f"[AE_{LATENT_2D}] final train={h2.history['loss'][-1]:.6f} | val={h2.history['val_loss'][-1]:.6f}")

Z1 = ENC_1.predict(X_feat_sc, verbose=0).astype(np.float32)
Z2 = ENC_2.predict(X_feat_sc, verbose=0).astype(np.float32)
Xr1 = AE_1.predict(X_feat_sc, verbose=0).astype(np.float32)
Xr2 = AE_2.predict(X_feat_sc, verbose=0).astype(np.float32)

mse_node_1 = np.mean((X_feat_sc - Xr1)**2, axis=1).astype(np.float32)
mse_node_2 = np.mean((X_feat_sc - Xr2)**2, axis=1).astype(np.float32)

print(f"[MSE/node] AE_{LATENT_1D}: mean={mse_node_1.mean():.6f} | median={np.median(mse_node_1):.6f}")
print(f"[MSE/node] AE_{LATENT_2D}: mean={mse_node_2.mean():.6f} | median={np.median(mse_node_2):.6f}")

# ===============================================================
# 6) Matrix reconstruction and comparison
# ===============================================================
def _split_blocks(Xv, N):
    Ahat = Xv[:, 0:N]
    Chat = Xv[:, N:2*N]
    Dhat = Xv[:, 2*N:3*N]
    That = Xv[:, 3*N:4*N]
    return Ahat, Chat, Dhat, That

A1, C1, D1, T1 = _split_blocks(Xr1, N_NODES)
A2, C2, D2, T2 = _split_blocks(Xr2, N_NODES)

def _symmetrize(M):
    Xs = (M + M.T)/2.0
    return 1.0/(1.0 + np.exp(-Xs))

A1_hat = _symmetrize(A1)
A2_hat = _symmetrize(A2)

# ===============================================================
# 7) TDA: distances, persistence, and Wasserstein for H0,H1,H2
# ===============================================================
def _latent_distance_matrix(Z):
    N = Z.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        diff = Z[i][None, :] - Z
        D[i] = np.sqrt((diff**2).sum(axis=1))
    D = (D / (D.max()+1e-6)).astype(np.float32)
    return D

D_lat_1 = _latent_distance_matrix(Z1)
D_lat_2 = _latent_distance_matrix(Z2)

def _tda_from_distance(D, maxdim=2, subsample=150, seed=7):
    """
    Compute persistence diagrams from a distance matrix, with optional
    random subsampling to control complexity.
    """
    N = D.shape[0]
    idx = np.arange(N)
    if N > subsample:
        rng2 = np.random.default_rng(seed)
        idx = rng2.choice(N, size=subsample, replace=False)
    D_sub = D[np.ix_(idx, idx)].astype(np.float32, copy=False)
    res = ripser(D_sub, maxdim=maxdim, distance_matrix=True)
    dgms = res["dgms"]
    return dgms, idx

dgms_geo, idx_geo  = _tda_from_distance(D_geo,   TDA_MAXDIM, TDA_SUBS)
dgms_lat1, idx_l1  = _tda_from_distance(D_lat_1, TDA_MAXDIM, TDA_SUBS)
dgms_lat2, idx_l2  = _tda_from_distance(D_lat_2, TDA_MAXDIM, TDA_SUBS)

def _clean_dgm_allow_empty(D):
    """
    Clean a single persistence diagram:
      - convert to float64,
      - remove points with non-finite coordinates,
      - KEEP the diagram even if it becomes empty (shape (0,2)),
        so that wasserstein can still handle empty vs non-empty.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        # Already empty diagram, leave it as is
        return D.reshape(0, 2)
    mask = np.isfinite(D).all(axis=1)
    D = D[mask]
    if D.size == 0:
        return D.reshape(0, 2)
    return D

def _wd_by_dim(dA, dB):
    """
    Wasserstein distance by homological dimension, including H2.

    - If both diagrams are empty in H_k: the distance is 0.0
    - If one is empty and the other is not: wasserstein returns a positive value
      (distance from non-empty diagram to the diagonal).
    - Errors are caught and reported; in that case a NaN is stored.
    """
    out = {}
    maxdim = min(len(dA), len(dB))
    for k in range(maxdim):
        DA = _clean_dgm_allow_empty(dA[k])
        DB = _clean_dgm_allow_empty(dB[k])

        # both empty → distance 0.0
        if DA.shape[0] == 0 and DB.shape[0] == 0:
            out[f"H{k}"] = 0.0
            continue

        try:
            dist_k = float(wasserstein(DA, DB, matching=False))
            out[f"H{k}"] = dist_k
        except Exception as e:
            print(f"[Warning] wasserstein failed at H{k}: {e}")
            out[f"H{k}"] = np.nan
    return out

WD_geo_lat1 = _wd_by_dim(dgms_geo,  dgms_lat1)
WD_geo_lat2 = _wd_by_dim(dgms_geo,  dgms_lat2)

print("[Wasserstein] geo vs lat1:", WD_geo_lat1)
print("[Wasserstein] geo vs lat2:", WD_geo_lat2)

# ---------------------------------------------------------------
# Betti curves
# ---------------------------------------------------------------
def _betti_curve(dgm, eps_grid):
    if dgm is None or len(dgm) == 0:
        return np.zeros_like(eps_grid, dtype=np.int32)
    D = dgm.copy()
    if D.size == 0:
        return np.zeros_like(eps_grid, dtype=np.int32)
    births = D[:, 0].astype(np.float32)
    deaths = D[:, 1].astype(np.float32)
    deaths = np.where(np.isfinite(deaths),
                      deaths,
                      eps_grid.max() + eps_grid[1] - eps_grid[0])
    B = np.zeros_like(eps_grid, dtype=np.int32)
    for i, e in enumerate(eps_grid):
        B[i] = int(np.sum((births <= e) & (e < deaths)))
    return B

eps = np.linspace(0.0, 1.0, 200).astype(np.float32)

def _betti_dict(dgms):
    out = {}
    for k in range(min(3, len(dgms))):
        out[f"H{k}"] = _betti_curve(dgms[k], eps)
    return out

BC_geo  = _betti_dict(dgms_geo)
BC_l1   = _betti_dict(dgms_lat1)
BC_l2   = _betti_dict(dgms_lat2)

# ===============================================================
# 8) Resilience: edge removal
# ===============================================================
def _resilience_eval(G0, remove_frac=0.15, mode="random_edges"):
    Gc = G0.copy()
    m = Gc.number_of_edges()
    k = max(1, int(remove_frac * m))

    edges_sorted = list(Gc.edges())
    if mode == "random_edges":
        rng.shuffle(edges_sorted)
        rem = edges_sorted[:k]
    elif mode == "betweenness_edges":
        bw = nx.edge_betweenness_centrality(Gc, weight="dist")
        rem = sorted(bw, key=bw.get, reverse=True)[:k]
    else:
        rem = edges_sorted[:k]

    Gc.remove_edges_from(rem)

    comps = list(nx.connected_components(Gc))
    giant = max(comps, key=len)
    H = Gc.subgraph(giant).copy()

    eff = global_efficiency(H)
    cl  = nx.average_clustering(H)
    apl = nx.average_shortest_path_length(H) if H.number_of_nodes() > 1 else np.nan
    diam = nx.diameter(H) if nx.is_connected(H) and H.number_of_nodes() > 1 else np.nan

    S = len(giant) / G0.number_of_nodes()
    return {
        "removed": k,
        "edges_after": H.number_of_edges(),
        "nodes_after": H.number_of_nodes(),
        "giant_fraction": S,
        "global_eff": eff,
        "clustering": cl,
        "avg_path_len": apl,
        "diameter": diam,
        "mode": mode
    }

resilience_rows = []
for at in ATTACK_TYPES:
    row = _resilience_eval(G, ATTACK_FRAC, mode=at)
    resilience_rows.append(row)
DF_RES = pd.DataFrame(resilience_rows)
print("[Resilience]\n", DF_RES)

# ===============================================================
# 9) Visualizations
# ===============================================================
def _close(fig):
    plt.close(fig); gc.collect()

# (A) Network map by transport mode
color_by_mode = {"sea":"tab:blue", "air":"tab:red", "rail":"tab:green", "road":"tab:orange"}
fig = plt.figure(figsize=(8,7))
ax = plt.gca()
for (u,v,data) in G.edges(data=True):
    x1,y1 = positions[u]
    x2,y2 = positions[v]
    ax.plot([x1,x2],[y1,y2], alpha=0.4, lw=1.2, color=color_by_mode.get(data["mode"], "gray"))
for i in G.nodes():
    x,y = positions[i]
    ax.scatter(x,y, s=12, c="k")
ax.set_title("Synthetic Logistics Network (modal edges)")
ax.set_xlabel("km (x)"); ax.set_ylabel("km (y)")
plt.tight_layout(); fig.savefig(IMG_DIR/"network_map.png"); _close(fig)

# (B) Heatmaps A vs A_hat
def _plot_heatmaps(Aorig, Ahat, title, fname):
    fig = plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(Aorig, aspect="auto"); plt.title("Adjacency (orig)"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(Ahat, aspect="auto");  plt.title("Adjacency (recon)"); plt.colorbar()
    plt.suptitle(title)
    plt.tight_layout(); fig.savefig(IMG_DIR/fname); _close(fig)

_plot_heatmaps(A, A1_hat, f"Adjacency – AE_{LATENT_1D}", "adjacency_recon_lat1.png")
_plot_heatmaps(A, A2_hat, f"Adjacency – AE_{LATENT_2D}", "adjacency_recon_lat2.png")

# (C) Training curves
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(h1.history["loss"], label="train"); plt.plot(h1.history["val_loss"], label="val")
plt.title(f"AE_{LATENT_1D} training"); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend()
plt.subplot(1,2,2)
plt.plot(h2.history["loss"], label="train"); plt.plot(h2.history["val_loss"], label="val")
plt.title(f"AE_{LATENT_2D} training"); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend()
plt.tight_layout(); fig.savefig(IMG_DIR/"training_curves.png"); _close(fig)

# (D) Histogram of per-node reconstruction error
fig = plt.figure(figsize=(8,4))
plt.hist(mse_node_1, bins=30, alpha=0.7, label=f"AE_{LATENT_1D}")
plt.hist(mse_node_2, bins=30, alpha=0.7, label=f"AE_{LATENT_2D}")
fig.legend()
plt.title("Per-node reconstruction error (MSE)")
plt.xlabel("MSE"); plt.ylabel("freq")
plt.tight_layout(); fig.savefig(IMG_DIR/"mse_hist_nodes.png"); _close(fig)

# (E) 2D PCA projection of latents (if dim > 2)
def _pca2(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    return Z.astype(np.float32)

Z1_2d = Z1 if Z1.shape[1]==2 else _pca2(Z1)
Z2_2d = Z2 if Z2.shape[1]==2 else _pca2(Z2)

fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.scatter(Z1_2d[:,0], Z1_2d[:,1], s=8); plt.title(f"Latent AE_{LATENT_1D} (PCA2)")
plt.subplot(1,2,2); plt.scatter(Z2_2d[:,0], Z2_2d[:,1], s=8); plt.title(f"Latent AE_{LATENT_2D} (PCA2)")
plt.tight_layout(); fig.savefig(IMG_DIR/"latents_scatter.png"); _close(fig)

# (F) Persistence diagrams
def _plot_diagrams_wrapped(dgms, title, fname):
    fig = plt.figure(figsize=(6.5,5.5))
    plot_diagrams(dgms, show=False)
    plt.title(title)
    plt.tight_layout(); fig.savefig(IMG_DIR/fname); _close(fig)

_plot_diagrams_wrapped(dgms_geo,  "Persistence (geo)",      "tda_diagrams_geo.png")
_plot_diagrams_wrapped(dgms_lat1, "Persistence (latent1)",  "tda_diagrams_lat1.png")
_plot_diagrams_wrapped(dgms_lat2, "Persistence (latent2)",  "tda_diagrams_lat2.png")

# (G) Betti curves comparison
def _plot_betti_compare(BCA, BCB, BCC, title, fname):
    fig = plt.figure(figsize=(8,4.5))
    for k, lab in [("H0","geo H0"), ("H1","geo H1"), ("H2","geo H2")]:
        if k in BCA:
            plt.plot(eps, BCA[k], label=lab)
    for k, lab in [("H0","lat1 H0"), ("H1","lat1 H1"), ("H2","lat1 H2")]:
        if k in BCB:
            plt.plot(eps, BCB[k], "--", label=lab)
    for k, lab in [("H0","lat2 H0"), ("H1","lat2 H1"), ("H2","lat2 H2")]:
        if k in BCC:
            plt.plot(eps, BCC[k], ":", label=lab)
    plt.title(title); plt.xlabel("ε"); plt.ylabel("β"); plt.legend()
    plt.tight_layout(); fig.savefig(IMG_DIR/fname); _close(fig)

_plot_betti_compare(BC_geo, BC_l1, BC_l2, "Betti-curves: geo vs latents", "betti_curves_compare.png")

# (H) Resilience metrics
fig = plt.figure(figsize=(8,4.5))
for col in ["giant_fraction","global_eff","clustering"]:
    plt.plot(DF_RES["mode"], DF_RES[col], marker="o", label=col)
plt.title("Resilience metrics after edge removals")
plt.ylabel("value"); plt.legend()
plt.tight_layout(); fig.savefig(IMG_DIR/"resilience_metrics.png"); _close(fig)

# ===============================================================
# 10) Tables and Excel export (with fallback name)
# ===============================================================
def _choose_engine():
    try:
        import openpyxl
        return "openpyxl"
    except Exception:
        try:
            import xlsxwriter
            return "xlsxwriter"
        except Exception:
            return None

engine = _choose_engine()
print(f"[Excel] Using engine: {engine}")

DF_STRUCT = pd.DataFrame({
    "n_nodes":[G.number_of_nodes()],
    "n_edges":[G.number_of_edges()],
    "diameter":[diameter],
    "avg_path_len":[avg_path_len],
    "clustering":[clustering],
    "assortativity":[assort],
    "global_eff":[glob_eff],
    "total_distance_km":[total_dist],
    "total_time_h":[total_time],
    "total_emissions_kgCO2~":[total_emis/1000.0],
    "total_cost_index":[total_cost]
})

DF_WD = pd.DataFrame({
    "metric":["Wasserstein"],
    "H0_lat1":[WD_geo_lat1.get("H0", np.nan)],
    "H1_lat1":[WD_geo_lat1.get("H1", np.nan)],
    "H2_lat1":[WD_geo_lat1.get("H2", np.nan)],
    "H0_lat2":[WD_geo_lat2.get("H0", np.nan)],
    "H1_lat2":[WD_geo_lat2.get("H1", np.nan)],
    "H2_lat2":[WD_geo_lat2.get("H2", np.nan)]
})

DF_MSE = pd.DataFrame({
    "node": np.arange(N_NODES),
    f"mse_AE_{LATENT_1D}": mse_node_1,
    f"mse_AE_{LATENT_2D}": mse_node_2
})

DF_META = pd.DataFrame({
    "param":["N_NODES","EPOCHS","BATCH_SIZE","LATENT_1D","LATENT_2D","ATTACK_FRAC",
             "TDA_MAXDIM","TDA_SUBS","LR","AREA_SIDE"],
    "value":[N_NODES,EPOCHS,BATCH_SIZE,LATENT_1D,LATENT_2D,ATTACK_FRAC,
             TDA_MAXDIM,TDA_SUBS,LR,AREA_SIDE]
})

DF_LOSS_1 = pd.DataFrame({
    "epoch":np.arange(1,len(h1.history["loss"])+1),
    "train":h1.history["loss"],
    "val":h1.history["val_loss"]
})
DF_LOSS_2 = pd.DataFrame({
    "epoch":np.arange(1,len(h2.history["loss"])+1),
    "train":h2.history["loss"],
    "val":h2.history["val_loss"]
})

try:
    with pd.ExcelWriter(EXCEL_PATH, engine=engine, mode="w") as writer:
        DF_STRUCT.to_excel(writer, sheet_name="structure_metrics", index=False)
        DF_WD.to_excel(writer, sheet_name="tda_wasserstein", index=False)
        DF_MSE.to_excel(writer, sheet_name="mse_per_node", index=False)
        DF_RES.to_excel(writer, sheet_name="resilience", index=False)
        DF_META.to_excel(writer, sheet_name="metadata", index=False)
        DF_LOSS_1.to_excel(writer, sheet_name=f"loss_AE_{LATENT_1D}", index=False)
        DF_LOSS_2.to_excel(writer, sheet_name=f"loss_AE_{LATENT_2D}", index=False)
    print(f"[Done] Results written to: {EXCEL_PATH}")
except PermissionError:
    alt_path = OUT_DIR / f"DTI_logistics_results_{int(time.time())}.xlsx"
    print(f"[Excel] Permission denied for {EXCEL_PATH}. "
          f"Writing instead to: {alt_path}")
    with pd.ExcelWriter(alt_path, engine=engine, mode="w") as writer:
        DF_STRUCT.to_excel(writer, sheet_name="structure_metrics", index=False)
        DF_WD.to_excel(writer, sheet_name="tda_wasserstein", index=False)
        DF_MSE.to_excel(writer, sheet_name="mse_per_node", index=False)
        DF_RES.to_excel(writer, sheet_name="resilience", index=False)
        DF_META.to_excel(writer, sheet_name="metadata", index=False)
        DF_LOSS_1.to_excel(writer, sheet_name=f"loss_AE_{LATENT_1D}", index=False)
        DF_LOSS_2.to_excel(writer, sheet_name=f"loss_AE_{LATENT_2D}", index=False)
    print(f"[Done] Results written to: {alt_path}")

print(f"[Done] Figures stored in: {IMG_DIR}")

# Cleanup
K.clear_session()
gc.collect()


