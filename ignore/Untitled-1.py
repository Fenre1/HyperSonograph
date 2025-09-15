#%%
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as _cos

#%%
file_name = r"F:\Stuff\HyperSonographSessions\s3.h5"
data = []
with h5py.File(file_name, "r") as f:
    data.append(f['features'][()])
    for dataset_name in f.keys():
        print(dataset_name)
        
        
        # %%
features = data[0]
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
