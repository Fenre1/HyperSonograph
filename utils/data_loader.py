import os
import h5py
import numpy as np
import pandas as pd

DATA_DIRECTORY = "F:/PhD/Projects/HyperGalleryDash/Sessions"

def _prepare_hypergraph_structures(df):
    """Return (hyperedges_dict, image_mapping_dict)."""
    if "image_path" in df.columns:
        df = df.drop(columns=["image_path"])

    hyperedges = {col: set(np.where(df[col] == 1)[0]) for col in df.columns}

    image_mapping = {}
    rows, cols = np.where(df.values == 1)
    col_names = df.columns.to_numpy()

    for r, c in zip(rows, cols):
        image_mapping.setdefault(r, set()).add(col_names[c])

    return hyperedges, image_mapping


def _calculate_hyperedge_avg_features(hyperedges, features_arr):
    """mean feature vector per hyperedge."""
    n_feat = features_arr.shape[1]
    return {
        name: np.mean(features_arr[list(idx_set)], axis=0) if idx_set else np.zeros(n_feat)
        for name, idx_set in hyperedges.items()
    }


def get_h5_files_in_directory():
    if not os.path.isdir(DATA_DIRECTORY):
        return []
    return [f for f in os.listdir(DATA_DIRECTORY) if f.lower().endswith(".h5")]


def load_session_data(h5_path):
    """Return dict with im_list, cat_list, hyperedges, …"""
    with h5py.File(h5_path, "r") as hdf:
        im_list = list(map(lambda x: x.decode() if isinstance(x, bytes) else x,
                           hdf["file_list"][()]))

        matrix = hdf["clustering_results"][()]
        cat_list = (
            list(map(lambda x: x.decode() if isinstance(x, bytes) else x,
                 hdf["catList"][()]))
            if "catList" in hdf
            else [f"edge_{i}" for i in range(matrix.shape[1])]
        )
        df_hyper = pd.DataFrame(matrix, columns=cat_list)

        features = hdf["features"][()]

    hyperedges, image_map = _prepare_hypergraph_structures(df_hyper)
    he_avg = _calculate_hyperedge_avg_features(hyperedges, features)

    return dict(
        im_list=im_list,
        df_hyperedges=df_hyper,
        cat_list=cat_list,
        hyperedges=hyperedges,
        image_mapping=image_map,
        hyperedge_avg_features=he_avg,
        h5_path=h5_path,
    )
