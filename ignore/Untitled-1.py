#%%
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as _cos

#%%
file_name = r"F:\Stuff\HyperSonographSessions\s2.h5"
data = []
with h5py.File(file_name, "r") as f:
    data.append(f['features'][()])

    # data.append(f['audio_model_features'][''])
    for dataset_name in f.keys():
        print(dataset_name)
        
#%%
import h5py

file_name = r"F:\Stuff\HyperSonographSessions\s2.h5"

with h5py.File(file_name, "r") as f:
    amf = f["audio_model_features"]
    for model_name, g in amf.items():
        # pak de dataset
        paths = g["song_path"][()]       # of [:]
        # indien als bytes opgeslagen: decodeer
        if getattr(paths, "dtype", None) is not None and (paths.dtype.kind == "S" or paths.dtype == object):
            paths = [p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p) for p in paths]
        print(model_name, len(paths), "paths")

        # %%
#%%
            if getattr(self, "model_features", None):
                grp = hdf.create_group("audio_model_features")
                for name, mf in self.model_features.items():
                    g = grp.create_group(name)
                    segs = mf.segments
                    g.create_dataset("segment_embeddings", data=segs.embeddings, dtype="f4")
                    g.create_dataset("segment_song_id", data=segs.song_id, dtype="i4")
                    g.create_dataset("segment_start_s", data=segs.start_s, dtype="f4")
                    g.create_dataset("segment_end_s", data=segs.end_s, dtype="f4")
                    songs = mf.songs
                    g.create_dataset("song_centroid", data=songs.centroid, dtype="f4")
                    g.create_dataset("song_stats2D", data=songs.stats2D, dtype="f4")
                    g.create_dataset("song_song_id", data=songs.song_id, dtype="i4")
                    g.create_dataset(
                        "song_path", data=np.array(list(songs.path), dtype=object), dtype=dt
                    )


#%%
features = data[1]
# %%
plt.plot(features)
# %%
cors = _cos(features)
# %%

import h5py, json, numpy as np, pandas as pd
from pathlib import Path

h5_path = Path(r"F:\Stuff\HyperSonographSessions\s3.h5")  # <-- change me

def print_tree(g, prefix=""):
    """Recursively print the file structure."""
    for k, v in g.items():
        if isinstance(v, h5py.Dataset):
            print(f"{prefix}/{k}  [dataset]  shape={v.shape} dtype={v.dtype}")
        else:
            print(f"{prefix}/{k}  [group]")
            print_tree(v, prefix + "/" + k)

def decode_str_dataset(ds):
    """Decode UTF-8 string datasets robustly."""
    arr = ds[()]
    if arr.ndim == 0:  # scalar string
        return arr.decode("utf-8", errors="replace")
    return [x.decode("utf-8", errors="replace") for x in arr]
#%%
loaded = {}
with h5py.File(h5_path, "r") as f:
    # 1) Overview
    print("=== FILE TREE ===")
    print_tree(f)

    print("\n=== ATTRIBUTES ===")
    for ak, av in f.attrs.items():
        print(f"{ak}: {av}")

    # 2) Always-present
    loaded["file_list"] = decode_str_dataset(f["/file_list"])
    loaded["clustering_results"] = f["/clustering_results"][:]
    loaded["catList"] = decode_str_dataset(f["/catList"])
    loaded["edge_origins"] = decode_str_dataset(f["/edge_origins"])
    loaded["edge_seen_times"] = f["/edge_seen_times"][:]
    loaded["features"] = f["/features"][:]

    # 3) Optional features
    for name in ("openclip_features", "places365_features", "umap_embedding"):
        if name in f:
            loaded[name] = f[name][:]

    # 4) Image UMAP (edge -> DataFrame)
    if "image_umap" in f:
        imap = {}
        for edge in f["/image_umap"].keys():
            arr = f["/image_umap"][edge][:]
            df = pd.DataFrame(arr, columns=["idx", "x", "y"])
            df["idx"] = df["idx"].astype(int)
            imap[edge] = df
        loaded["image_umap"] = imap

    # 5) Audio models
    if "audio_model_names" in f:
        loaded["audio_model_names"] = decode_str_dataset(f["/audio_model_names"])
    if "audio_model_features" in f:
        amf = {}
        agrp = f["/audio_model_features"]
        for model_name in agrp.keys():
            g = agrp[model_name]
            amf[model_name] = {
                "segments": {
                    "embeddings": g["segment_embeddings"][:],
                    "song_id": g["segment_song_id"][:],
                    "start_s": g["segment_start_s"][:],
                    "end_s": g["segment_end_s"][:],
                },
                "songs": {
                    "centroid": g["song_centroid"][:],
                    "stats2D": g["song_stats2D"][:],
                    "song_id": g["song_song_id"][:],
                    "path": decode_str_dataset(g["song_path"]),
                },
            }
        loaded["audio_model_features"] = amf

    # 6) Metadata
    if "metadata" in f:
        raw = f["/metadata"][()]
        if isinstance(raw, (bytes, np.bytes_)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            loaded["metadata_json"] = json.loads(raw)
            loaded["metadata_df"] = pd.read_json(raw, orient="table")
        except Exception:
            loaded["metadata_raw"] = raw

    # 7) Thumbnails
    thumbs_embedded = bool(f.attrs.get("thumbnails_are_embedded", False))
    loaded["thumbnails_are_embedded"] = thumbs_embedded
    if thumbs_embedded and "thumbnail_data_embedded" in f:
        v = f["/thumbnail_data_embedded"]
        loaded["thumbnail_byte_lengths"] = [len(a) for a in v]
    elif not thumbs_embedded and "thumbnail_relative_paths" in f:
        loaded["thumbnail_relative_paths"] = decode_str_dataset(f["/thumbnail_relative_paths"])

print("\nLoaded keys:", list(loaded.keys()))
# %%
with h5py.File(h5_path, "r") as f:
    if "audio_model_features" in f:
        for i, (model_name, g) in enumerate(f["/audio_model_features"].items(), start=1):
            varname = f"features_m{i}"
            globals()[varname] = g["segment_embeddings"][:]
            print(f"{varname} <- embeddings from model '{model_name}', shape={globals()[varname].shape}")
    else:
        print("No audio_model_features group found in this file.")
# %%
import umap
import matplotlib.pyplot as plt
def umap_maker(features):
    cors = _cos(features)

    # Convert cosine similarity to distance
    dist_matrix = 1 - cors  

    # Run UMAP with precomputed distance matrix
    reducer = umap.UMAP(random_state=0)
    embedding = reducer.fit_transform(dist_matrix)

    # Plot the 2D embedding
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    plt.title("UMAP projection of segment similarities")
    plt.show()

# %%
umap_maker(features_m1)
umap_maker(features_m2)
umap_maker(features_m3)
# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import umap
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd  # only used for factorize; remove if you prefer numpy.unique
#%%
model_to_plot = 'MERT'             
model_to_plot = 'OpenL3'
model_to_plot = 'CLAP'             
#%%
def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    # Robust cosine similarity (handles non-unit vectors)
    return cosine_similarity(X)

def umap_maker(features: np.ndarray, song_ids=None, random_state=0):
    # Compute cosine distance from cosine similarity
    cors = _cosine_sim_matrix(features)
    dist_matrix = 1.0 - cors

    # UMAP on precomputed distances
    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(dist_matrix)

    # Colorize by song_id if provided
    plt.figure(figsize=(8, 6))
    if song_ids is not None:
        # Map song_ids (which may be arbitrary ints/strings) to contiguous integer labels
        # Using pandas.factorize preserves order of first appearance
        labels, uniques = pd.factorize(song_ids, sort=False)
        sc = plt.scatter(embedding[:, 0], embedding[:, 1], s=10, c=labels, alpha=0.9)
        plt.title("UMAP projection of segment similarities (colored by song_id)")

        # Colorbar can be huge; only add if not too many songs
        n_unique = len(uniques)
        if n_unique <= 25:
            cbar = plt.colorbar(sc, shrink=0.8)
            # Build a few ticks (if many, keep it light)
            tick_positions = np.linspace(0, n_unique - 1, min(n_unique, 10), dtype=int)
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels([str(uniques[i]) for i in tick_positions])
        else:
            plt.colorbar(sc, shrink=0.8).set_label("song_id (factorized)")
            # For many songs, showing every label is impractical.
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
        plt.title("UMAP projection of segment similarities")

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

# ---- Load embeddings + song_ids from the H5 and plot one model ----
with h5py.File(h5_path, "r") as f:
    if "audio_model_features" not in f:
        raise RuntimeError("No 'audio_model_features' group found in this file.")

    agrp = f["/audio_model_features"]
    # Choose model
    if model_to_plot is None:
        model_name = next(iter(agrp.keys()))
    else:
        if model_to_plot not in agrp:
            raise KeyError(f"Model '{model_to_plot}' not found. Available: {list(agrp.keys())}")
        model_name = model_to_plot

    g = agrp[model_name]
    features = g["segment_embeddings"][:]
    song_ids = g["segment_song_id"][:]

    print(f"Model: {model_name} | features shape: {features.shape} | segments: {len(song_ids)}")
    umap_maker(features, song_ids=song_ids, random_state=0)
# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
h5_path = Path(r"F:\Stuff\HyperSonographSessions\s3.h5")  # <-- change me


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def per_song_similarity_stats(embeddings: np.ndarray, song_ids: np.ndarray) -> pd.DataFrame:
    """
    Compute per-song within- and between-song cosine similarity means without forming the full similarity matrix.
    Assumes rows of `embeddings` correspond to segments; `song_ids` holds the song ID for each row.
    """
    X = l2_normalize_rows(embeddings)                 # N x D, unit rows
    N = X.shape[0]
    unique_songs, inverse = np.unique(song_ids, return_inverse=True)
    n_songs = unique_songs.size

    # Sum vector per song: S_k = sum_{i in song k} x_i
    # And count per song: n_k
    # We can accumulate with np.add.at
    D = X.shape[1]
    S = np.zeros((n_songs, D), dtype=X.dtype)
    np.add.at(S, inverse, X)                          # S[group] += X[i]
    counts = np.bincount(inverse, minlength=n_songs).astype(np.int64)

    # Total sum vector
    S_tot = S.sum(axis=0)                             # D,
    N_tot = N

    # WITHIN-song mean (exclude self pairs):
    # sum_{i!=j in k} x_i · x_j = ||S_k||^2 - n_k   (since ||x_i||^2 = 1)
    # number of ordered pairs i!=j is n_k*(n_k-1). For unordered pairs, multiply denominator by 2; we’ll use ordered to keep consistency.
    Sk_sqnorm = (S * S).sum(axis=1)                   # ||S_k||^2
    within_sum = Sk_sqnorm - counts                   # ∑_{i!=j} x_i·x_j
    within_denom = counts * (counts - 1)              # number of ordered pairs
    within_mean = np.divide(within_sum, within_denom, out=np.full_like(within_sum, np.nan, dtype=float), where=within_denom>0)

    # BETWEEN-song mean vs all others:
    # sum_{i in k} sum_{j not in k} x_i·x_j = S_k · (S_tot - S_k)
    # number of ordered pairs is n_k * (N_tot - n_k)
    between_sum = (S * (S_tot - S)).sum(axis=1)
    between_denom = counts * (N_tot - counts)
    between_mean = np.divide(between_sum, between_denom, out=np.full_like(between_sum, np.nan, dtype=float), where=between_denom>0)

    df = pd.DataFrame({
        "song_id": unique_songs,
        "n_segments": counts,
        "within_mean": within_mean,
        "between_mean": between_mean,
    })
    return df

# ---- Run for all models and plot ----
per_model_stats = {}  # model_name -> DataFrame

with h5py.File(h5_path, "r") as f:
    if "audio_model_features" not in f:
        raise RuntimeError("No 'audio_model_features' group found in this file.")

    agrp = f["/audio_model_features"]
    for model_name in agrp.keys():
        g = agrp[model_name]
        X = g["segment_embeddings"][:]                # (N, D)
        sid = g["segment_song_id"][:]                 # (N,)

        df = per_song_similarity_stats(X, sid)
        per_model_stats[model_name] = df

        # Quick visual: scatter within vs between per song (one figure per model)
        plt.figure(figsize=(6, 5))
        plt.scatter(df["between_mean"], df["within_mean"], s=20, alpha=0.8)
        plt.xlabel("Between-song mean similarity (vs all others)")
        plt.ylabel("Within-song mean similarity")
        plt.title(f"Within vs Between similarity per song — {model_name}")
        plt.tight_layout()
        plt.show()

# Example: peek at one model’s table
for name, df in per_model_stats.items():
    print(f"\nModel: {name}")
    display(df.sort_values("within_mean", ascending=False).head(10))
    break  # remove this break to print all

# %%
summary_rows = []
for model_name, df in per_model_stats.items():
    within_mean = df["within_mean"].mean(skipna=True)
    between_mean = df["between_mean"].mean(skipna=True)
    summary_rows.append({
        "model": model_name,
        "mean_within": within_mean,
        "mean_between": between_mean,
        "gap": within_mean - between_mean,
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df)









# %%
AUDIO_DIR = r"D:\Muziek\Youtube\tets"   # <- change me
RECURSIVE = True                            # include subfolders
SEGMENT_SECONDS = 30
HOP_SECONDS = 15                          # None -> default hop = SEGMENT_SECONDS/2
COMBINE = "separate"                        # "separate" (per-model rows) or "concat" (all models in one vector)
DEVICE = None                               # None -> auto; or "cpu" / "cuda"
SAVE_PREFIX = None                          # e.g., "/path/to/output/run1" to save .npy/.csv

# --- Imports & setup ---
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as _cos

# If audio_features.py is in the same folder as the notebook, this is enough.
# If it's elsewhere, add that folder to sys.path:
AF_PATH = Path("utils/feature_extraction.py").resolve()
if not AF_PATH.exists():
    raise FileNotFoundError("Could not find audio_features.py next to this notebook. "
                            "Set AF_PATH below to its location.")
sys.path.insert(0, str(AF_PATH.parent))

# Import your module
import utils.feature_extraction as af
#%%
# --- Collect audio files ---
exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aif", ".aiff"}
root = Path(AUDIO_DIR).expanduser().resolve()
if not root.exists():
    raise FileNotFoundError(f"Audio folder not found: {root}")

if RECURSIVE:
    files = [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]
else:
    files = [str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
if not files:
    raise RuntimeError(f"No audio files in {root} with extensions: {sorted(exts)}")

print(f"Found {len(files)} audio files.")
#%%
import torch
DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
print(DEVICE)
#%%
# --- Build models ---
warnings.filterwarnings("default")
extractors = af.build_windows_extractors(
    segment_seconds=SEGMENT_SECONDS,
    hop_seconds=HOP_SECONDS,
    target_sr=48_000,
    device=DEVICE,
)
print("Enabled models:")
for e in extractors:
    e.normalize = False
# extractors[3].target_sr = 16000
for e in extractors:
    print(f"  - {e.model_name} (target_sr={e.target_sr}, device={e.device})")

multi = af.MultiModelAudioExtractor(
    extractors=extractors,
    combine=COMBINE,
    segment_seconds=SEGMENT_SECONDS,
    hop_seconds=HOP_SECONDS,
    ref_sr=48_000,
    normalize=False,
)
#%%
Xseg, rows, songs = extractor.extract_segments_and_songs(files)
#%%
# --- Run extraction ---
if COMBINE == "concat":
    # Single matrix; rows contain concatenated vectors across models
    X, rows, songs = multi.extract_segments_and_songs(files)
    rows_df = pd.DataFrame(rows)
    songs_df = pd.DataFrame([{
        "file": s.file, "model": s.model,
        "centroid_dim": int(s.centroid_D.shape[0]),
        "stats_dim": int(s.stats_2D.shape[0]),
        "centroid_D": s.centroid_D.tolist(),
        "stats_2D": s.stats_2D.tolist(),
    } for s in songs])

    print(f"\nSegments: {X.shape[0]} rows, dim={X.shape[1] if X.size else 0}")

else:
    # "separate": keep each model’s array separately (no forced vstack)
    arrays = multi.extract_feature_arrays(files)  # tuple aligned with extractors
    # Build per-model DataFrames of segment metadata (file/start/end/model/dim)
    # by re-running lightweight metadata pass to match segment counts; reuse the same segmentation logic
    # We can get metadata via extract_features_with_metadata with combine="concat" per-model simulation
    # but simpler: rebuild rows by running once and discarding vectors.
    # Here, we reconstruct rows per model using the same internal path as MultiModelAudioExtractor:
    # Easiest: call extract_features_with_metadata with combine="concat" on a one-extractor runner per model.

    per_model = []
    for ext, X_i in zip(extractors, arrays):
        single = af.MultiModelAudioExtractor([ext], combine="concat",
                                             segment_seconds=SEGMENT_SECONDS,
                                             hop_seconds=HOP_SECONDS,
                                             ref_sr=48_000)
        _, rows_i = single.extract_features_with_metadata(files)
        rows_df_i = pd.DataFrame(rows_i)
        assert len(rows_df_i) == len(X_i), (
            f"Row/vector count mismatch for {ext.model_name}: "
            f"{len(rows_df_i)} rows vs {len(X_i)} vecs"
        )
        per_model.append((ext.model_name, X_i, rows_df_i))

    # Show a compact summary and first few rows for each model
    for name, X_i, rows_df_i in per_model:
        print(f"\nModel: {name} -> segments: {X_i.shape[0]}, dim: {X_i.shape[1] if X_i.size else 0}")


# %%
import matplotlib.pyplot as plt

model_id = 2
cors = _cos([per_model[model_id][1][2],per_model[model_id][1][28]])
print(cors)
cosses_pos = []
cosses_neg = []
unq = per_model[model_id][2]['file'].unique()

per_song_cors = []
for item in unq:
    per_song_cors.append([])

for ind, seg in enumerate(per_model[model_id][1]):
    for indx, segx in enumerate(per_model[model_id][1]):
        cors = _cos([seg,segx])
        
        per_song_cors[np.where(unq==per_model[model_id][2].iloc[ind]['file'])[0][0]].append(cors[1,0])
        if per_model[model_id][2].iloc[ind]['file'] == per_model[model_id][2].iloc[indx]['file']:
            if indx != ind:
                cosses_pos.append(cors[1,0])
        else:
            cosses_neg.append(cors[1,0])

#%%
plt.plot(cosses_neg)
plt.plot(cosses_pos)
print('pos:',np.mean(cosses_pos),'neg:',np.mean(cosses_neg))
# %%
# Build ROC/PR from your lists of cosine similarities

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Assuming you already have:
# cosses_pos: list[float]  # same-song pairs (excluding self-pairs)
# cosses_neg: list[float]  # different-song pairs

# 1) Prepare labels and scores
y_true = np.concatenate([np.ones(len(cosses_pos), dtype=int), np.zeros(len(cosses_neg), dtype=int)])
scores = np.concatenate([np.array(cosses_pos, dtype=float), np.array(cosses_neg, dtype=float)])

# 2) ROC
fpr, tpr, roc_thresholds = roc_curve(y_true, scores)      # scores: larger => positive
roc_auc = auc(fpr, tpr)

# Youden J to pick a threshold (maximizes TPR - FPR)
youden_idx = np.argmax(tpr - fpr)
thr_best = roc_thresholds[youden_idx]
tpr_best, fpr_best = tpr[youden_idx], fpr[youden_idx]

# 3) Precision–Recall (useful with class imbalance)
prec, rec, pr_thresholds = precision_recall_curve(y_true, scores)
ap = average_precision_score(y_true, scores)

# 4) Quick summaries
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR  AP : {ap:.4f}")
print(f"Best threshold (Youden J): {thr_best:.4f} -> TPR={tpr_best:.3f}, FPR={fpr_best:.3f}")
print(f"Means  pos={np.mean(cosses_pos):.4f}, neg={np.mean(cosses_neg):.4f}")

# 5) Plots (no seaborn, single-plot each)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.scatter([fpr_best], [tpr_best], label=f"Best thr={thr_best:.3f}", zorder=3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC – same-song vs different-song")
plt.legend()
plt.show()

plt.figure()
plt.plot(rec, prec, label=f"PR (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall – same-song vs different-song")
plt.legend()
plt.show()

# (Optional) quick overlap check
# Histograms to visualize similarity overlap
plt.figure()
plt.hist(cosses_neg, bins=50, alpha=0.6, label="neg (diff song)", density=True)
plt.hist(cosses_pos, bins=50, alpha=0.6, label="pos (same song)", density=True)
plt.axvline(thr_best, linestyle="--", label=f"Best thr={thr_best:.3f}")
plt.xlabel("Cosine similarity")
plt.ylabel("Density")
plt.title("Similarity distributions")
plt.legend()
plt.show()

# %%
