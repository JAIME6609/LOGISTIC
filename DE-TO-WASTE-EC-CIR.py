# -*- coding: utf-8 -*-
"""
Unified E-waste Circular-Economy Analytics with AI + Topology
=========================================================================
This script merges:
- The robust, eager-only, topology-aware design of the Keras-3-safe version.
- The memory-saving LOW-RAM logic of the light versions.
- The latent Ridge regression + TSNE/PCA comparison block from the rich Version 3.

Main features
-------------
1) Synthetic e-waste supply-chain simulation and directed graph.
2) Tabular per-device dataset and manual standardization.
3) Autoencoder (AE) and Variational Autoencoder (VAE) with:
   - Isometry preservation, kNN-preservation, Laplacian smoothness (TensorFlow).
4) Eager-only training loops with separate optimizers (Keras 3 safe).
5) Reconstruction metrics (MSE, MAE, R2) for AE and VAE.
6) Latent embeddings (PCA2D; optional TSNE for latents if enabled).
7) Latent Ridge regression (AE and VAE) with:
   - RidgeCV + TSNE if scikit-learn is available and FULL_PLOTS is True.
   - Closed-form Ridge + PCA2D otherwise.
8) TSNE/PCA embeddings of input vs reconstructions (X, Xrec_AE, Xrec_VAE).
9) TDA (ripser/persim) with Betti curves and topological distances.
10) Excel export of reconstruction metrics, training histories, latent regression,
    Betti curves, topological distances, and configuration metadata.

Configuration flags:
- LOW_RAM: controls dataset size, batch size, and epochs.
- FULL_PLOTS: controls the richness of plots (including TSNE embedding).
- DO_TDA: activates or deactivates TDA computations.
"""

import os
import sys
import subprocess
import importlib
import json
import gc
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# Global configuration
# ============================================================
LOW_RAM          = True           # If True, use conservative settings
FULL_PLOTS       = True           # If False, reduce number of plots and heavy embeddings
DO_TDA           = True           # If False, skip TDA completely
N_DEVICES        = 2500 if LOW_RAM else 5000
BATCH_SIZE       = 96 if LOW_RAM else 128
N_EPOCHS_AE      = 40 if LOW_RAM else 60
N_EPOCHS_VAE     = 40 if LOW_RAM else 60
EMBED_METHOD     = "PCA2D"        # For latent embeddings; TSNE is optional and heavy
TDA_MAX_POINTS   = 400            # Subsample points for TDA
SAVE_PLOTS_LIGHT = True if LOW_RAM else False
EXCEL_LIGHT      = True if LOW_RAM else False

# ============================================================
# Dependency bootstrap
# ============================================================
def _ensure(pkg):
    """Install a package if missing; return True on success, False otherwise."""
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

# Base dependencies
for _p in ["numpy", "pandas", "matplotlib", "networkx", "openpyxl", "xlsxwriter"]:
    _ensure(_p)

_ensure("tensorflow")
HAS_SKLEARN = _ensure("scikit-learn")
HAS_RIPSER  = _ensure("ripser") if DO_TDA else False
HAS_PERSIM  = _ensure("persim") if DO_TDA else False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers, Model

if HAS_SKLEARN:
    from sklearn.manifold import TSNE

if HAS_RIPSER:
    from ripser import ripser

if HAS_PERSIM:
    from persim import plot_diagrams, wasserstein, bottleneck

# ============================================================
# TensorFlow runtime configuration
# ============================================================
tf.config.run_functions_eagerly(True)
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
print("[Runtime] Eager:", tf.executing_eagerly(), "| Mixed precision disabled")

np.random.seed(7)
tf.random.set_seed(7)

# Output paths
OUT_DIR = Path("ewaste_topo_unified_v2")
FIG_DIR = OUT_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUT_DIR / "results_ewaste_topo_unified_v2.xlsx"

plt.rcParams.update({"figure.dpi": 115, "savefig.dpi": 135, "axes.grid": True})

# ============================================================
# 1) Synthetic e-waste chain + directed graph
# ============================================================
def simulate_ewaste_graph(n_devices=N_DEVICES, seed=7):
    """
    Simulate a directed e-waste supply chain and a per-device dataset.

    Nodes:
        - Manufacturers (M)
        - Users (U)
        - Reuse centers (R)
        - Recyclers (C)
        - Secondary material nodes (S)

    Each device has:
        - Material mix (5 components).
        - Efficiencies: reuse, collection, recovery.
        - Policy index, transport distance, energy, carbon, costs.
        - Composite circularity index in [0,1].
    """
    rng = np.random.default_rng(seed)

    # Node roles
    if LOW_RAM:
        n_manuf, n_users, n_reuse, n_recycl, n_sec = 5, 8, 4, 4, 3
    else:
        n_manuf, n_users, n_reuse, n_recycl, n_sec = 6, 10, 5, 5, 4

    roles = (
        ["manufacturer"] * n_manuf
        + ["user"] * n_users
        + ["reuse"] * n_reuse
        + ["recycler"] * n_recycl
        + ["secondary"] * n_sec
    )
    names = (
        [f"M{i}" for i in range(n_manuf)]
        + [f"U{i}" for i in range(n_users)]
        + [f"R{i}" for i in range(n_reuse)]
        + [f"C{i}" for i in range(n_recycl)]
        + [f"S{i}" for i in range(n_sec)]
    )

    G = nx.DiGraph()
    G.add_nodes_from([(n, {"role": r}) for n, r in zip(names, roles)])

    # Device-level features
    material_mix = rng.dirichlet([3, 1.5, 2.5, 1.2, 1.8], size=n_devices)
    eff_reuse    = rng.uniform(0.15, 0.55, size=n_devices)
    eff_collect  = rng.uniform(0.35, 0.85, size=n_devices)
    eff_recov    = rng.uniform(0.40, 0.95, size=n_devices)
    policy_idx   = rng.uniform(0.00, 1.00, size=n_devices)
    trans_km     = rng.gamma(6.0, 35.0 if LOW_RAM else 40.0, size=n_devices)
    energy_kwh   = rng.gamma(3.0, 4.5 if LOW_RAM else 5.0, size=n_devices)
    carbon_kg    = 0.45 * energy_kwh + 0.08 * trans_km + rng.normal(0, 1.0 if LOW_RAM else 1.2, size=n_devices)
    cost_log     = 0.02 * trans_km + rng.normal(0, 0.4 if LOW_RAM else 0.5, size=n_devices)
    cost_proc    = 0.15 * energy_kwh + rng.normal(0, 0.6 if LOW_RAM else 0.8, size=n_devices)

    # Circularity index
    return_rate = eff_collect * (0.6 * eff_reuse + 0.4)
    recovery    = eff_recov
    circ_raw    = 0.45 * return_rate + 0.45 * recovery + 0.10 * policy_idx
    circ_carbon = np.clip(
        1.0 - (carbon_kg - carbon_kg.min()) / (np.ptp(carbon_kg) + 1e-6),
        0.0,
        1.0,
    )
    circularity = np.clip(0.7 * circ_raw + 0.3 * circ_carbon, 0.0, 1.0)

    # Node selection helper
    def pick(role, size):
        cand = [n for n, d in G.nodes(data=True) if d["role"] == role]
        return rng.choice(cand, size=size, replace=True)

    srcM = pick("manufacturer", n_devices)
    dstU = pick("user", n_devices)
    dstR = pick("reuse", n_devices)
    dstC = pick("recycler", n_devices)
    dstS = pick("secondary", n_devices)

    # Edge aggregator
    def add_edge(u, v, w):
        if G.has_edge(u, v):
            for k, val in w.items():
                if isinstance(val, (int, float)):
                    G[u][v][k] += val
        else:
            G.add_edge(u, v, **w)

    base_tons = 0.01 + 0.19 * (return_rate * recovery)
    for i in range(n_devices):
        tons = base_tons[i]
        # M -> U
        add_edge(
            srcM[i],
            dstU[i],
            dict(tons=tons, km=trans_km[i], co2=0.05 * trans_km[i],
                 cost=cost_log[i], stage="distribution"),
        )
        # U -> R
        add_edge(
            dstU[i],
            dstR[i],
            dict(tons=tons * eff_reuse[i] * 0.6, km=0.2 * trans_km[i],
                 co2=0.02 * trans_km[i], cost=0.4 * cost_log[i], stage="reuse"),
        )
        # U -> C
        add_edge(
            dstU[i],
            dstC[i],
            dict(tons=tons * eff_collect[i] * (1.0 - 0.6 * eff_reuse[i]),
                 km=0.6 * trans_km[i], co2=0.04 * trans_km[i],
                 cost=0.8 * cost_log[i], stage="collection"),
        )
        # C -> S
        add_edge(
            dstC[i],
            dstS[i],
            dict(tons=tons * recovery[i] * eff_collect[i],
                 km=0.4 * trans_km[i], co2=0.03 * trans_km[i],
                 cost=cost_proc[i], stage="recovery"),
        )

    cols = [
        "plast", "glass", "base_metals", "critical_metals", "pcb",
        "eff_reuse", "eff_collect", "eff_recov", "policy_idx",
        "trans_km", "energy_kwh", "carbon_kg", "cost_log", "cost_proc",
        "return_rate", "recovery", "circularity",
    ]
    data = np.column_stack(
        [
            material_mix,
            eff_reuse, eff_collect, eff_recov, policy_idx,
            trans_km, energy_kwh, carbon_kg, cost_log, cost_proc,
            return_rate, recovery, circularity,
        ]
    )
    df = pd.DataFrame(data, columns=cols)
    df["device_id"] = np.arange(len(df))

    return G, df


def plot_supply_graph(G, savepath):
    """Draw the supply-chain graph with colors per role."""
    roles = nx.get_node_attributes(G, "role")
    pos = nx.spring_layout(G, seed=7, k=1.0, iterations=40)
    cmap = {
        "manufacturer": "tab:blue",
        "user": "tab:green",
        "reuse": "tab:orange",
        "recycler": "tab:red",
        "secondary": "tab:purple",
    }
    figsize = (8.5, 5.4) if LOW_RAM else (11.5, 7.5)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=300 if LOW_RAM else 420,
        node_color=[cmap[roles[n]] for n in G.nodes()],
        alpha=0.9,
        linewidths=0.4,
        edgecolors="black",
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrowsize=8, width=0.6)
    if FULL_PLOTS and not LOW_RAM:
        nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title("E-waste supply chain (aggregated flows)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


# Generate synthetic network and plot the graph
G, df = simulate_ewaste_graph()
plot_supply_graph(G, FIG_DIR / "supply_graph.png")
del G
gc.collect()

# ============================================================
# 2) Data preparation
# ============================================================
features = [c for c in df.columns if c not in ["device_id", "circularity"]]
X_all = df[features].values.astype(np.float32)
y_all = df["circularity"].values.astype(np.float32)
del df
gc.collect()

# Manual standardization
X_mean = X_all.mean(axis=0)
X_std = X_all.std(axis=0) + 1e-8
X = ((X_all - X_mean) / X_std).astype(np.float32)
del X_all
gc.collect()

n = X.shape[0]
rng = np.random.RandomState(7)
perm = rng.permutation(n)
test_size = max(400, int(0.2 * n))
te_idx = perm[:test_size]
tr_idx = perm[test_size:]
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y_all[tr_idx], y_all[te_idx]
del X, y_all, perm
gc.collect()

input_dim = X_tr.shape[1]

# ============================================================
# 3) Topology-aware regularizers and utilities
# ============================================================
LATENT_DIM = 2
K_NEIGH    = 8 if LOW_RAM else 10

def pairwise_dists(x):
    """Squared Euclidean pairwise distances (TensorFlow)."""
    x = tf.cast(x, tf.float32)
    xx = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    D = xx - 2.0 * tf.matmul(x, x, transpose_b=True) + tf.transpose(xx)
    return tf.maximum(D, 0.0)


def knn_adjacency_from_dist(D, k):
    """Symmetric kNN adjacency from a distance matrix D."""
    D = tf.cast(D, tf.float32)
    n = tf.shape(D)[0]
    k_eff = tf.minimum(tf.cast(k, tf.int32), n - 1)
    _, idx = tf.math.top_k(-D, k=k_eff + 1)  # includes self
    idx = idx[:, 1:]

    row_idx = tf.reshape(tf.range(n, dtype=tf.int32), [-1, 1])
    row_idx = tf.tile(row_idx, [1, k_eff])
    ii = tf.reshape(row_idx, [-1])
    jj = tf.reshape(tf.cast(idx, tf.int32), [-1])

    updates = tf.ones_like(ii, dtype=tf.float32)
    shape = tf.stack([n, n], axis=0)
    A = tf.scatter_nd(indices=tf.stack([ii, jj], axis=1),
                      updates=updates,
                      shape=shape)
    A = tf.maximum(A, tf.transpose(A))
    return A


def loss_isometry(x_in, z_lat):
    Dx = pairwise_dists(x_in)
    Dz = pairwise_dists(z_lat)
    Dx = Dx / (tf.reduce_max(Dx) + 1e-8)
    Dz = Dz / (tf.reduce_max(Dz) + 1e-8)
    return tf.reduce_mean(tf.square(Dx - Dz))


def loss_knn_preservation(x_in, z_lat, k=K_NEIGH):
    Dx = pairwise_dists(x_in)
    Dz = pairwise_dists(z_lat)
    A_in  = knn_adjacency_from_dist(Dx, k)
    A_lat = knn_adjacency_from_dist(Dz, k)
    return tf.reduce_mean(tf.square(A_in - A_lat))


def loss_laplacian_smooth(z_lat, x_in=None, k=K_NEIGH):
    Dx = pairwise_dists(x_in)
    A  = knn_adjacency_from_dist(Dx, k)
    z  = tf.cast(z_lat, tf.float32)
    z_i = tf.expand_dims(z, 1)
    z_j = tf.expand_dims(z, 0)
    diff2 = tf.reduce_sum(tf.square(z_i - z_j), axis=2)
    return 0.5 * tf.reduce_mean(A * diff2)


# ============================================================
# 4) AE and VAE models
# ============================================================
# Autoencoder
inp_ae = layers.Input(shape=(input_dim,), dtype="float32")
h = layers.Dense(64, activation="relu")(inp_ae)
h = layers.Dense(64, activation="relu")(h)
z_ae = layers.Dense(LATENT_DIM, name="latent_ae")(h)
u = layers.Dense(64, activation="relu")(z_ae)
u = layers.Dense(64, activation="relu")(u)
out_ae = layers.Dense(input_dim)(u)
AE = Model(inp_ae, out_ae, name="AE")
AE_lat_model = Model(AE.input, AE.get_layer("latent_ae").output)

# VAE
class Sampling(layers.Layer):
    """Reparameterization trick with KL divergence added as an internal loss."""
    def __init__(self, kl_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_mean, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean), dtype=tf.float32)
        z = z_mean + tf.exp(0.5 * z_logvar) * eps
        # KL divergence term added as a loss
        kl = -0.5 * tf.reduce_mean(
            1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
        )
        self.add_loss(self.kl_weight * kl)
        return z

inp_vae = layers.Input(shape=(input_dim,), dtype="float32")
h = layers.Dense(64, activation="relu")(inp_vae)
h = layers.Dense(64, activation="relu")(h)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(h)
z_logvar = layers.Dense(LATENT_DIM, name="z_logvar")(h)
z_vae = Sampling(kl_weight=0.01)([z_mean, z_logvar])
u = layers.Dense(64, activation="relu")(z_vae)
u = layers.Dense(64, activation="relu")(u)
out_vae = layers.Dense(input_dim)(u)

VAE = Model(inp_vae, out_vae, name="VAE")
VAE_lat_model = Model(VAE.input, z_vae)
ENC = Model(inp_vae, [z_mean, z_logvar, z_vae], name="ENC")

# ============================================================
# 5) Training loops with topology-aware losses
# ============================================================
mse_loss = tf.keras.losses.MeanSquaredError()

ae_optimizer  = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

lambda_iso  = 0.3
lambda_knn  = 0.3
lambda_lap  = 0.4

def train_ae(X_tr, epochs=N_EPOCHS_AE, batch_size=BATCH_SIZE):
    n = X_tr.shape[0]
    history = {"recon": [], "iso": [], "knn": [], "lap": [], "total": []}
    for ep in range(epochs):
        idx = np.random.permutation(n)
        X_tr_shuf = X_tr[idx]
        ep_recon = ep_iso = ep_knn = ep_lap = ep_total = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            xb = X_tr_shuf[i:i+batch_size]
            xb_tf = tf.convert_to_tensor(xb, dtype=tf.float32)
            with tf.GradientTape() as tape:
                zb = AE_lat_model(xb_tf, training=True)
                xb_rec = AE(xb_tf, training=True)
                recon = mse_loss(xb_tf, xb_rec)
                iso = loss_isometry(xb_tf, zb)
                knn = loss_knn_preservation(xb_tf, zb, k=K_NEIGH)
                lap = loss_laplacian_smooth(zb, xb_tf, k=K_NEIGH)
                total = recon + lambda_iso*iso + lambda_knn*knn + lambda_lap*lap
            grads = tape.gradient(total, AE.trainable_variables)
            ae_optimizer.apply_gradients(zip(grads, AE.trainable_variables))

            ep_recon += float(recon)
            ep_iso   += float(iso)
            ep_knn   += float(knn)
            ep_lap   += float(lap)
            ep_total += float(total)
            steps += 1
        history["recon"].append(ep_recon/steps)
        history["iso"].append(ep_iso/steps)
        history["knn"].append(ep_knn/steps)
        history["lap"].append(ep_lap/steps)
        history["total"].append(ep_total/steps)
        print(f"[AE] Epoch {ep+1}/{epochs} - total={history['total'][-1]:.4f}")
    return history


def train_vae(X_tr, epochs=N_EPOCHS_VAE, batch_size=BATCH_SIZE):
    n = X_tr.shape[0]
    history = {"recon": [], "iso": [], "knn": [], "lap": [], "total": []}
    for ep in range(epochs):
        idx = np.random.permutation(n)
        X_tr_shuf = X_tr[idx]
        ep_recon = ep_iso = ep_knn = ep_lap = ep_total = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            xb = X_tr_shuf[i:i+batch_size]
            xb_tf = tf.convert_to_tensor(xb, dtype=tf.float32)
            with tf.GradientTape() as tape:
                zb = VAE_lat_model(xb_tf, training=True)
                xb_rec = VAE(xb_tf, training=True)
                recon = mse_loss(xb_tf, xb_rec)
                iso = loss_isometry(xb_tf, zb)
                knn = loss_knn_preservation(xb_tf, zb, k=K_NEIGH)
                lap = loss_laplacian_smooth(zb, xb_tf, k=K_NEIGH)
                topo = lambda_iso*iso + lambda_knn*knn + lambda_lap*lap
                total = recon + topo
                # VAE has an internal KL loss attached to the Sampling layer
                for l in VAE.losses:
                    total += l
            grads = tape.gradient(total, VAE.trainable_variables)
            vae_optimizer.apply_gradients(zip(grads, VAE.trainable_variables))

            ep_recon += float(recon)
            ep_iso   += float(iso)
            ep_knn   += float(knn)
            ep_lap   += float(lap)
            ep_total += float(total)
            steps += 1
        history["recon"].append(ep_recon/steps)
        history["iso"].append(ep_iso/steps)
        history["knn"].append(ep_knn/steps)
        history["lap"].append(ep_lap/steps)
        history["total"].append(ep_total/steps)
        print(f"[VAE] Epoch {ep+1}/{epochs} - total={history['total'][-1]:.4f}")
    return history


print("=== Training AE with topological regularization ===")
hist_ae = train_ae(X_tr)

print("=== Training VAE with topological regularization ===")
hist_vae = train_vae(X_tr)

# ============================================================
# 6) Reconstruction metrics and basic plots
# ============================================================
def reconstruction_metrics(X_true, X_rec):
    """Return MSE, MAE, R2 for reconstruction."""
    residual = X_true - X_rec
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    var = float(np.var(X_true))
    r2 = 1.0 - mse / (var + 1e-8)
    return mse, mae, r2

# AE recon
X_tr_ae_rec = AE(X_tr, training=False).numpy()
X_te_ae_rec = AE(X_te, training=False).numpy()
mse_tr_ae, mae_tr_ae, r2_tr_ae = reconstruction_metrics(X_tr, X_tr_ae_rec)
mse_te_ae, mae_te_ae, r2_te_ae = reconstruction_metrics(X_te, X_te_ae_rec)

# VAE recon
X_tr_vae_rec = VAE(X_tr, training=False).numpy()
X_te_vae_rec = VAE(X_te, training=False).numpy()
mse_tr_vae, mae_tr_vae, r2_tr_vae = reconstruction_metrics(X_tr, X_tr_vae_rec)
mse_te_vae, mae_te_vae, r2_te_vae = reconstruction_metrics(X_te, X_te_vae_rec)

recon_summary = pd.DataFrame(
    {
        "model": ["AE", "AE", "VAE", "VAE"],
        "split": ["train", "test", "train", "test"],
        "MSE": [mse_tr_ae, mse_te_ae, mse_tr_vae, mse_te_vae],
        "MAE": [mae_tr_ae, mae_te_ae, mae_tr_vae, mae_te_vae],
        "R2":  [r2_tr_ae, r2_te_ae, r2_tr_vae, r2_te_vae],
    }
)

# Training curves plots
def plot_training_history(hist, title_prefix, save_prefix):
    epochs = range(1, len(hist["total"]) + 1)
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(epochs, hist["recon"], label="Reconstruction")
    plt.plot(epochs, hist["iso"],   label="Isometry")
    plt.plot(epochs, hist["knn"],   label="kNN")
    plt.plot(epochs, hist["lap"],   label="Laplacian")
    plt.plot(epochs, hist["total"], label="Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{save_prefix}_losses.png")
    plt.close()

if FULL_PLOTS or not SAVE_PLOTS_LIGHT:
    plot_training_history(hist_ae,  "AE",  "ae")
    plot_training_history(hist_vae, "VAE", "vae")
else:
    # At least save total loss for both
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(hist_ae["total"], label="AE total loss")
    plt.plot(hist_vae["total"], label="VAE total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("AE and VAE total losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ae_vae_total_losses.png")
    plt.close()

# Histograms of reconstruction error
if FULL_PLOTS:
    err_ae = np.mean((X_te - X_te_ae_rec) ** 2, axis=1)
    err_vae = np.mean((X_te - X_te_vae_rec) ** 2, axis=1)
    plt.figure(figsize=(7.0, 4.0))
    plt.hist(err_ae, bins=40, alpha=0.6, label="AE")
    plt.hist(err_vae, bins=40, alpha=0.6, label="VAE")
    plt.xlabel("Per-sample MSE (test)")
    plt.ylabel("Frequency")
    plt.title("Reconstruction error distribution (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "reconstruction_error_hist.png")
    plt.close()

# ============================================================
# 7) Latent embeddings (PCA and optional TSNE)
# ============================================================
def pca_2d(Xmat):
    """Simple PCA to 2D implemented with NumPy."""
    Xc = Xmat - Xmat.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    W = eigvecs[:, order[:2]]
    Z = Xc @ W
    return Z

# Latent representations for test set
Z_te_AE  = AE_lat_model(X_te, training=False).numpy()
Z_te_VAE = VAE_lat_model(X_te, training=False).numpy()

Z_te_AE_pca  = pca_2d(Z_te_AE)
Z_te_VAE_pca = pca_2d(Z_te_VAE)

plt.figure(figsize=(7.0, 4.0))
plt.scatter(Z_te_AE_pca[:, 0], Z_te_AE_pca[:, 1],
            c=y_te, cmap="viridis", s=15, alpha=0.8)
plt.colorbar(label="Circularity")
plt.title("AE latent (PCA 2D)")
plt.tight_layout()
plt.savefig(FIG_DIR / "ae_latent_pca2d.png")
plt.close()

plt.figure(figsize=(7.0, 4.0))
plt.scatter(Z_te_VAE_pca[:, 0], Z_te_VAE_pca[:, 1],
            c=y_te, cmap="viridis", s=15, alpha=0.8)
plt.colorbar(label="Circularity")
plt.title("VAE latent (PCA 2D)")
plt.tight_layout()
plt.savefig(FIG_DIR / "vae_latent_pca2d.png")
plt.close()

if FULL_PLOTS and HAS_SKLEARN and EMBED_METHOD.upper() == "TSNE":
    # Optional t-SNE on AE latent
    tsne = TSNE(n_components=2, random_state=7, perplexity=30, init="random", learning_rate="auto")
    Z_te_AE_tsne = tsne.fit_transform(Z_te_AE)
    plt.figure(figsize=(7.0, 4.0))
    plt.scatter(Z_te_AE_tsne[:, 0], Z_te_AE_tsne[:, 1],
                c=y_te, cmap="viridis", s=15, alpha=0.8)
    plt.colorbar(label="Circularity")
    plt.title("AE latent (t-SNE 2D)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ae_latent_tsne2d.png")
    plt.close()

# ============================================================
# 8) Latent Ridge regression + TSNE/PCA on X and reconstructions
# ============================================================
# Compute VAE mean latent as in Version 3
X_te_tf = tf.convert_to_tensor(X_te, tf.float32)
Zmu_te, Zlv_te, Z_te_V = ENC(X_te_tf, training=False)
Zmu_te = Zmu_te.numpy()
Z_te_V = Z_te_V.numpy()
Xrec_te_AE  = X_te_ae_rec
Xrec_te_VAE = X_te_vae_rec

def ridge_closed_form(Z, y, alpha=1.0):
    """Closed-form Ridge regression used when scikit-learn is not available."""
    Z1 = np.hstack([Z, np.ones((Z.shape[0], 1), dtype=Z.dtype)])
    I = np.eye(Z1.shape[1], dtype=Z.dtype)
    I[-1, -1] = 0.0  # do not regularize bias
    w = np.linalg.solve(Z1.T @ Z1 + alpha * I, Z1.T @ y)
    return w

def ridge_predict(Z, w):
    Z1 = np.hstack([Z, np.ones((Z.shape[0], 1), dtype=Z.dtype)])
    return Z1 @ w

reg_stats = {}

if HAS_SKLEARN:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, mean_absolute_error

    reg_AE  = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(Z_te_AE, y_te)
    reg_VAE = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(Zmu_te,  y_te)
    y_hat_AE  = reg_AE.predict(Z_te_AE)
    y_hat_VAE = reg_VAE.predict(Zmu_te)

    reg_stats = dict(
        AE  = dict(R2=float(r2_score(y_te, y_hat_AE)),
                   MAE=float(mean_absolute_error(y_te, y_hat_AE))),
        VAE = dict(R2=float(r2_score(y_te, y_hat_VAE)),
                   MAE=float(mean_absolute_error(y_te, y_hat_VAE))),
    )

    # Embeddings of X and reconstructions
    if FULL_PLOTS:
        tsne_X = TSNE(n_components=2, init="random", learning_rate="auto",
                      perplexity=40, random_state=7)
        X_te_emb     = tsne_X.fit_transform(X_te)
        Xrec_AE_emb  = tsne_X.fit_transform(Xrec_te_AE)
        Xrec_VAE_emb = tsne_X.fit_transform(Xrec_te_VAE)
    else:
        # Lightweight PCA fallback even if sklearn is installed
        def pca2(Xm):
            Xm = Xm - Xm.mean(0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
            return Xm @ Vt[:2].T
        X_te_emb     = pca2(X_te)
        Xrec_AE_emb  = pca2(Xrec_te_AE)
        Xrec_VAE_emb = pca2(Xrec_te_VAE)

else:
    # Closed-form Ridge + PCA embeddings if sklearn is not available
    w_AE  = ridge_closed_form(Z_te_AE, y_te, alpha=1.0)
    w_VAE = ridge_closed_form(Zmu_te,  y_te, alpha=1.0)
    y_hat_AE  = ridge_predict(Z_te_AE, w_AE)
    y_hat_VAE = ridge_predict(Zmu_te,  w_VAE)
    y_bar = np.mean(y_te)
    ss_tot = np.sum((y_te - y_bar)**2) + 1e-12
    reg_stats = dict(
        AE  = dict(R2=float(1 - np.sum((y_te - y_hat_AE)**2)/ss_tot),
                   MAE=float(np.mean(np.abs(y_te - y_hat_AE)))),
        VAE = dict(R2=float(1 - np.sum((y_te - y_hat_VAE)**2)/ss_tot),
                   MAE=float(np.mean(np.abs(y_te - y_hat_VAE))))
    )

    def pca2(Xm):
        Xm = Xm - Xm.mean(0, keepdims=True)  # CORREGIDO: keepdims en lugar de keepsdims
        U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
        return Xm @ Vt[:2].T
    X_te_emb     = pca2(X_te)
    Xrec_AE_emb  = pca2(Xrec_te_AE)
    Xrec_VAE_emb = pca2(Xrec_te_VAE)

# Plots for embeddings of X and reconstructions (three panels)
if FULL_PLOTS:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    for ax, emb, title in zip(
        axes,
        [X_te_emb, Xrec_AE_emb, Xrec_VAE_emb],
        ["X test", "AE reconstruction", "VAE reconstruction"],
    ):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_te, cmap="viridis", s=10, alpha=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Circularity")
    fig.suptitle("Embeddings of input and reconstructions", fontsize=11)
    plt.savefig(FIG_DIR / "embeddings_X_vs_recon.png")
    plt.close()

# ============================================================
# 9) TDA: persistence diagrams, Betti curves, topological distances
# ============================================================
tda_results = {}
betti_curves_summary = []
topo_distances = []

def compute_betti_curve(diag, t_grid):
    """
    Compute Betti curve for a single homology dimension.

    diag: array of shape (n_points, 2) with (birth, death).
    t_grid: 1D array of thresholds.
    """
    if diag.size == 0:
        return np.zeros_like(t_grid)
    b = diag[:, 0]
    d = diag[:, 1]
    counts = []
    for t in t_grid:
        counts.append(np.sum((b <= t) & (t < d)))
    return np.array(counts, dtype=float)

if DO_TDA and HAS_RIPSER:
    # Subsample test points for TDA
    idx_tda = np.random.choice(X_te.shape[0],
                               size=min(TDA_MAX_POINTS, X_te.shape[0]),
                               replace=False)
    X_tda = X_te[idx_tda]
    Z_ae_tda  = Z_te_AE[idx_tda]
    Z_vae_tda = Z_te_VAE[idx_tda]

    # Compute persistence diagrams
    print("=== Running ripser on test input (X) ===")
    dg_X = ripser(X_tda, maxdim=2)
    print("=== Running ripser on AE latent (Z_ae) ===")
    dg_AE = ripser(Z_ae_tda, maxdim=2)
    print("=== Running ripser on VAE latent (Z_vae) ===")
    dg_VAE = ripser(Z_vae_tda, maxdim=2)

    t_grid = np.linspace(0.0, 1.0, 200)

    # Collect Betti curves per space and dimension
    for name, dg in [("true", dg_X), ("AE", dg_AE), ("VAE", dg_VAE)]:
        diagrams = dg["dgms"]
        for dim, diag in enumerate(diagrams):
            diag = np.asarray(diag, dtype=float)
            if diag.size == 0:
                continue
            births = diag[:, 0]
            deaths = diag[:, 1]
            finite = np.isfinite(deaths)
            if np.any(finite):
                b_fin = births[finite]
                d_fin = deaths[finite]
                min_b = b_fin.min()
                max_d = d_fin.max()
                scale = max_d - min_b + 1e-8
                diag_norm = np.stack([(births - min_b) / scale,
                                      (deaths - min_b) / scale], axis=1)
            else:
                diag_norm = diag.copy()
            betti = compute_betti_curve(diag_norm, t_grid)
            betti_curves_summary.append(
                {
                    "space": name,
                    "dim": dim,
                    "t_grid": json.dumps(t_grid.tolist()),
                    "betti": json.dumps(betti.tolist()),
                }
            )

    # Build comparative Betti arrays as in Version 3
    # H0
    H0_true = compute_betti_curve(
        np.asarray(dg_X["dgms"][0], dtype=float), t_grid
    ) if len(dg_X["dgms"]) > 0 else np.zeros_like(t_grid)
    H0_AE = compute_betti_curve(
        np.asarray(dg_AE["dgms"][0], dtype=float), t_grid
    ) if len(dg_AE["dgms"]) > 0 else np.zeros_like(t_grid)
    H0_VAE = compute_betti_curve(
        np.asarray(dg_VAE["dgms"][0], dtype=float), t_grid
    ) if len(dg_VAE["dgms"]) > 0 else np.zeros_like(t_grid)

    # H1
    H1_true = compute_betti_curve(
        np.asarray(dg_X["dgms"][1], dtype=float), t_grid
    ) if len(dg_X["dgms"]) > 1 else np.zeros_like(t_grid)
    H1_AE = compute_betti_curve(
        np.asarray(dg_AE["dgms"][1], dtype=float), t_grid
    ) if len(dg_AE["dgms"]) > 1 else np.zeros_like(t_grid)
    H1_VAE = compute_betti_curve(
        np.asarray(dg_VAE["dgms"][1], dtype=float), t_grid
    ) if len(dg_VAE["dgms"]) > 1 else np.zeros_like(t_grid)

    # Optional H2
    HAS_H2 = (len(dg_X["dgms"]) > 2) and (len(dg_AE["dgms"]) > 2) and (len(dg_VAE["dgms"]) > 2)
    if HAS_H2:
        H2_true = compute_betti_curve(
            np.asarray(dg_X["dgms"][2], dtype=float), t_grid
        )
        H2_AE = compute_betti_curve(
            np.asarray(dg_AE["dgms"][2], dtype=float), t_grid
        )
        H2_VAE = compute_betti_curve(
            np.asarray(dg_VAE["dgms"][2], dtype=float), t_grid
        )
    else:
        H2_true = H2_AE = H2_VAE = np.zeros_like(t_grid)

    # DataFrame for Betti curves (Version 3 style)
    cols = {
        "eps": t_grid,
        "H0_true": H0_true, "H0_AE": H0_AE, "H0_VAE": H0_VAE,
        "H1_true": H1_true, "H1_AE": H1_AE, "H1_VAE": H1_VAE,
    }
    if HAS_H2:
        cols.update({
            "H2_true": H2_true, "H2_AE": H2_AE, "H2_VAE": H2_VAE
        })
    df_betti = pd.DataFrame(cols)

    # Summary statistics for Betti curves
    def _summ_stats(curve, eps_grid):
        return dict(
            max=int(curve.max()),
            auc=float(np.trapz(curve.astype(np.float32), eps_grid)),
        )

    summary_rows = []
    for dim, triplet in [("H0", (H0_true, H0_AE, H0_VAE)),
                         ("H1", (H1_true, H1_AE, H1_VAE))]:
        summary_rows.append(dict(dimension=dim, model="true", **_summ_stats(triplet[0], t_grid)))
        summary_rows.append(dict(dimension=dim, model="AE",   **_summ_stats(triplet[1], t_grid)))
        summary_rows.append(dict(dimension=dim, model="VAE",  **_summ_stats(triplet[2], t_grid)))
    if HAS_H2:
        for model_name, curve in [("true", H2_true), ("AE", H2_AE), ("VAE", H2_VAE)]:
            summary_rows.append(dict(dimension="H2", model=model_name, **_summ_stats(curve, t_grid)))

    df_betti_summary = pd.DataFrame(summary_rows)

    # Plot Betti curves as in Version 3 (comparative true vs AE vs VAE)
    if FULL_PLOTS or not SAVE_PLOTS_LIGHT:
        plt.figure(figsize=(9.5, 4.8))
        plt.plot(t_grid, H0_true, label="H0 true")
        plt.plot(t_grid, H0_AE,   label="H0 AE")
        plt.plot(t_grid, H0_VAE,  label="H0 VAE")
        plt.xlabel("ε")
        plt.ylabel("β0(ε)")
        plt.title("Betti curves H0")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "betti_H0.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(9.5, 4.8))
        plt.plot(t_grid, H1_true, label="H1 true")
        plt.plot(t_grid, H1_AE,   label="H1 AE")
        plt.plot(t_grid, H1_VAE,  label="H1 VAE")
        plt.xlabel("ε")
        plt.ylabel("β1(ε)")
        plt.title("Betti curves H1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "betti_H1.png", bbox_inches="tight")
        plt.close()

        if HAS_H2:
            plt.figure(figsize=(9.5, 4.8))
            plt.plot(t_grid, H2_true, label="H2 true")
            plt.plot(t_grid, H2_AE,   label="H2 AE")
            plt.plot(t_grid, H2_VAE,  label="H2 VAE")
            plt.xlabel("ε")
            plt.ylabel("β2(ε)")
            plt.title("Betti curves H2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / "betti_H2.png", bbox_inches="tight")
            plt.close()

    # Plot persistence diagrams if persim is available
    if HAS_PERSIM and (FULL_PLOTS or not SAVE_PLOTS_LIGHT):
        try:
            plt.figure(figsize=(7.0, 4.0))
            plot_diagrams(dg_X["dgms"], show=False)
            plt.title("Persistence diagrams - true")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "diagrams_true.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not plot diagrams for true: {e}")

        try:
            plt.figure(figsize=(7.0, 4.0))
            plot_diagrams(dg_AE["dgms"], show=False)
            plt.title("Persistence diagrams - AE")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "diagrams_AE.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not plot diagrams for AE: {e}")

        try:
            plt.figure(figsize=(7.0, 4.0))
            plot_diagrams(dg_VAE["dgms"], show=False)
            plt.title("Persistence diagrams - VAE")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "diagrams_VAE.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not plot diagrams for VAE: {e}")

    # Topological distances (AE vs true, VAE vs true) per dimension
    if HAS_PERSIM:
        for dim in [0, 1, 2]:
            dgm_X = np.asarray(dg_X["dgms"][dim], dtype=float) if dim < len(dg_X["dgms"]) else np.empty((0, 2))
            dgm_AE = np.asarray(dg_AE["dgms"][dim], dtype=float) if dim < len(dg_AE["dgms"]) else np.empty((0, 2))
            dgm_VAE = np.asarray(dg_VAE["dgms"][dim], dtype=float) if dim < len(dg_VAE["dgms"]) else np.empty((0, 2))

            if dgm_X.size == 0 or (dgm_AE.size == 0 and dgm_VAE.size == 0):
                continue

            try:
                if dgm_AE.size > 0:
                    w_AE = float(wasserstein(dgm_X, dgm_AE))
                    b_AE = float(bottleneck(dgm_X, dgm_AE))
                    topo_distances.append(
                        {"space_pair": "true_vs_AE", "dim": dim,
                         "wasserstein": w_AE, "bottleneck": b_AE}
                    )
                if dgm_VAE.size > 0:
                    w_VAE = float(wasserstein(dgm_X, dgm_VAE))
                    b_VAE = float(bottleneck(dgm_X, dgm_VAE))
                    topo_distances.append(
                        {"space_pair": "true_vs_VAE", "dim": dim,
                         "wasserstein": w_VAE, "bottleneck": b_VAE}
                    )
            except Exception as e:
                print(f"[Warning] Could not compute topological distances (dim={dim}): {e}")

else:
    df_betti = pd.DataFrame()
    df_betti_summary = pd.DataFrame()

# Convert Betti curves and distances to DataFrames (for Excel)
betti_df = pd.DataFrame(betti_curves_summary) if betti_curves_summary else pd.DataFrame()
topo_df  = pd.DataFrame(topo_distances) if topo_distances else pd.DataFrame()

# ============================================================
# 10) Excel export (reconstruction, training, regression, Betti, topology)
# ============================================================
# Regression statistics DataFrame
reg_rows = []
for model_name in reg_stats:
    reg_rows.append(
        {
            "model": model_name,
            "R2": reg_stats[model_name]["R2"],
            "MAE": reg_stats[model_name]["MAE"],
        }
    )
reg_df = pd.DataFrame(reg_rows) if reg_rows else pd.DataFrame()

with pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter") as writer:
    # Reconstruction metrics
    recon_summary.to_excel(writer, sheet_name="reconstruction", index=False)

    # Regression on latents
    if not reg_df.empty:
        reg_df.to_excel(writer, sheet_name="latent_regression", index=False)

    # Training histories (optional)
    if not EXCEL_LIGHT:
        hist_ae_df = pd.DataFrame(hist_ae)
        hist_ae_df.to_excel(writer, sheet_name="ae_training", index=False)
        hist_vae_df = pd.DataFrame(hist_vae)
        hist_vae_df.to_excel(writer, sheet_name="vae_training", index=False)

    # Betti curves (per space/dim) and Version-3-style comparative curves
    if DO_TDA and not betti_df.empty:
        betti_df.to_excel(writer, sheet_name="betti_curves_raw", index=False)
    if DO_TDA and not df_betti.empty:
        df_betti.to_excel(writer, sheet_name="betti_curves_H", index=False)
    if DO_TDA and not df_betti_summary.empty:
        df_betti_summary.to_excel(writer, sheet_name="betti_summary", index=False)

    # Topological distances
    if DO_TDA and not topo_df.empty:
        topo_df.to_excel(writer, sheet_name="topo_distances", index=False)

    # Meta sheet
    meta = {
        "LOW_RAM": [LOW_RAM],
        "FULL_PLOTS": [FULL_PLOTS],
        "DO_TDA": [DO_TDA],
        "N_DEVICES": [N_DEVICES],
        "BATCH_SIZE": [BATCH_SIZE],
        "N_EPOCHS_AE": [N_EPOCHS_AE],
        "N_EPOCHS_VAE": [N_EPOCHS_VAE],
    }
    meta_df = pd.DataFrame(meta)
    meta_df.to_excel(writer, sheet_name="meta", index=False)

print(f"All results saved in: {EXCEL_PATH}")
print(f"Figures saved in: {FIG_DIR}")

# ============================================================
# 11) Additional pedagogical figures (Option A)
#      - AE/VAE train vs val curves (MSE)
#      - t-SNE overlays: original vs AE rec, original vs VAE rec
#      This block DOES NOT modify any of the previous results.
#      It trains auxiliary models only for visualization.
# ============================================================

def build_plain_ae_model(input_dim, latent_dim):
    """
    Auxiliary AE (without topological regularization) trained with Keras.fit
    solely to produce classical train/val MSE curves.
    """
    inp = layers.Input(shape=(input_dim,), dtype="float32")
    h = layers.Dense(64, activation="relu")(inp)
    h = layers.Dense(64, activation="relu")(h)
    z = layers.Dense(latent_dim, name="latent_plain_ae")(h)
    u = layers.Dense(64, activation="relu")(z)
    u = layers.Dense(64, activation="relu")(u)
    out = layers.Dense(input_dim)(u)
    model = Model(inp, out, name="AE_plain_trainval")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return model


def build_plain_vae_model(input_dim, latent_dim):
    """
    Auxiliary VAE (with KL via Sampling layer) trained with Keras.fit
    to obtain train/val curves comparable to the AE.
    """
    inp = layers.Input(shape=(input_dim,), dtype="float32")
    h = layers.Dense(64, activation="relu")(inp)
    h = layers.Dense(64, activation="relu")(h)
    z_mean = layers.Dense(latent_dim, name="z_mean_tv")(h)
    z_logvar = layers.Dense(latent_dim, name="z_logvar_tv")(h)
    z = Sampling(kl_weight=0.01)([z_mean, z_logvar])
    u = layers.Dense(64, activation="relu")(z)
    u = layers.Dense(64, activation="relu")(u)
    out = layers.Dense(input_dim)(u)
    model = Model(inp, out, name="VAE_plain_trainval")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"  # reconstruction loss; KL is added internally as model.losses
    )
    return model


def generate_train_val_curves_and_tsne():
    """
    Extra pedagogical block:

    1) Creates a small validation subset from X_tr (without touching X_te).
    2) Trains auxiliary AE and VAE models with Keras.fit to obtain
       train/val loss curves (MSE + KL in the VAE case).
    3) Generates a figure with two panels:
          - AE training (MSE)      [train vs val]
          - VAE training (recon+KL) [train vs val]
       saved as 'training_curves.png'.

    4) Builds t-SNE (or PCA if sklearn is not available) overlays for:
          - original test data vs AE reconstructions
          - original test data vs VAE reconstructions
       saved jointly as 'tsne_overlays.png'.

    None of this alters the previous AE/VAE models or any exported result.
    """
    print("=== [Extra] Training auxiliary AE/VAE for train/val curves ===")

    # ------------------------------------------------------------
    # 1) Validation split derived only from X_tr
    # ------------------------------------------------------------
    n_total = X_tr.shape[0]
    n_val = max(int(0.2 * n_total), 64)  # at least 64 points in validation
    n_val = min(n_val, n_total // 2)     # keep majority as training
    X_train_tv = X_tr[:-n_val]
    X_val_tv   = X_tr[-n_val:]

    # ------------------------------------------------------------
    # 2) Auxiliary models and training with validation
    # ------------------------------------------------------------
    EPOCHS_TV = 120  # número de épocas para las curvas pedagógicas

    ae_tv  = build_plain_ae_model(input_dim, LATENT_DIM)
    vae_tv = build_plain_vae_model(input_dim, LATENT_DIM)

    hist_ae_tv = ae_tv.fit(
        X_train_tv, X_train_tv,
        validation_data=(X_val_tv, X_val_tv),
        epochs=EPOCHS_TV,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    hist_vae_tv = vae_tv.fit(
        X_train_tv, X_train_tv,
        validation_data=(X_val_tv, X_val_tv),
        epochs=EPOCHS_TV,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    # ------------------------------------------------------------
    # 3) Figura: curvas de entrenamiento (AE y VAE)
    # ------------------------------------------------------------
    epochs_range = np.arange(1, EPOCHS_TV + 1)

    plt.figure(figsize=(13.5, 4.5))

    # Panel izquierdo: AE
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist_ae_tv.history["loss"],     label="train")
    plt.plot(epochs_range, hist_ae_tv.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AE training (MSE)")
    plt.legend()

    # Panel derecho: VAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist_vae_tv.history["loss"],     label="train")
    plt.plot(epochs_range, hist_vae_tv.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE training (recon + KL via layer)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_curves.png")
    plt.close()

    # ------------------------------------------------------------
    # 4) t-SNE / PCA overlays: original vs AE rec, original vs VAE rec
    # ------------------------------------------------------------
    print("=== [Extra] Building t-SNE/PCA overlays for original vs reconstructions ===")

    n_te = X_te.shape[0]

    if HAS_SKLEARN:
        # Se aplican dos t-SNE independientes para que cada overlay
        # se adapte a su estructura, pero siempre en un espacio conjunto
        # (original + reconstrucciones).
        from sklearn.manifold import TSNE

        tsne_ae = TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            perplexity=40,
            random_state=7,
        )
        Z_comb_ae = tsne_ae.fit_transform(np.vstack([X_te, X_te_ae_rec]))
        Z_orig_ae = Z_comb_ae[:n_te]
        Z_rec_ae  = Z_comb_ae[n_te:]

        tsne_vae = TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            perplexity=40,
            random_state=17,
        )
        Z_comb_vae = tsne_vae.fit_transform(np.vstack([X_te, X_te_vae_rec]))
        Z_orig_vae = Z_comb_vae[:n_te]
        Z_rec_vae  = Z_comb_vae[n_te:]
    else:
        # Si no se dispone de sklearn, se usa un PCA 2D compartido
        # para cada par (original, reconstrucciones).
        def pca_2d_shared(X1, X2):
            Xcat = np.vstack([X1, X2])
            Xc = Xcat - Xcat.mean(axis=0, keepdims=True)
            cov = np.cov(Xc, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            W = eigvecs[:, order[:2]]
            Z = Xc @ W
            n1 = X1.shape[0]
            return Z[:n1], Z[n1:]

        Z_orig_ae, Z_rec_ae   = pca_2d_shared(X_te, X_te_ae_rec)
        Z_orig_vae, Z_rec_vae = pca_2d_shared(X_te, X_te_vae_rec)

    # Figura con dos paneles: overlays AE y VAE
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), sharex=True, sharey=True)

    ax0 = axes[0]
    ax0.scatter(Z_orig_ae[:, 0], Z_orig_ae[:, 1], s=12, alpha=0.85, label="original")
    ax0.scatter(Z_rec_ae[:, 0],  Z_rec_ae[:, 1],  s=12, alpha=0.75, label="AE rec")
    ax0.set_title("t-SNE overlay: original vs AE")
    ax0.legend()
    ax0.grid(True, linestyle="--", alpha=0.4)

    ax1 = axes[1]
    ax1.scatter(Z_orig_vae[:, 0], Z_orig_vae[:, 1], s=12, alpha=0.85, label="original")
    ax1.scatter(Z_rec_vae[:, 0],  Z_rec_vae[:, 1],  s=12, alpha=0.75, label="VAE rec")
    ax1.set_title("t-SNE overlay: original vs VAE")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(FIG_DIR / "tsne_overlays.png")
    plt.close()

    print("Additional pedagogical figures saved in:")
    print(" -", FIG_DIR / "training_curves.png")
    print(" -", FIG_DIR / "tsne_overlays.png")


# Ejecutar el bloque extra únicamente cuando el archivo se ejecuta como script
if __name__ == "__main__":
    try:
        generate_train_val_curves_and_tsne()
    except Exception as e:
        print("[Warning] Could not generate additional pedagogical figures:", e)


