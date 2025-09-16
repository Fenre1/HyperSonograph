from __future__ import annotations  
import uuid
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Set, Iterable, Optional
from PIL import Image, ExifTags, ImageOps
import io
from PyQt5.QtCore import QObject, pyqtSignal as Signal
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import time
from .similarity import SIM_METRIC
from .audio_schema import SegmentLevel, SongLevel, ModelFeatures

def generate_n_colors(n: int, saturation: int = 150, value: int = 230) -> list[str]:
    colors: list[str] = []
    for i in range(max(n, 1)):
        hue = int(360 * i / n) if n else 0
        color = QColor()
        color.setHsv(hue, saturation, value)
        colors.append(color.name())
    return colors

def jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)



class SessionModel(QObject):
    edgeRenamed      = Signal(str, str)      # old, new
    layoutChanged    = Signal()              # big regroup or reload
    similarityDirty  = Signal()              # vectors changed; views may flush
    hyperedgeModified = Signal(str)

    @classmethod
    def load_h5(cls, path: Path) -> "SessionModel":
        with h5py.File(path, "r") as hdf:
            im_list = [
                x.decode() if isinstance(x, bytes) else x for x in hdf["file_list"][()]
            ]
            matrix = hdf["clustering_results"][()]
            cat_raw = (
                hdf["catList"][()]
                if "catList" in hdf
                else [f"edge_{i}" for i in range(matrix.shape[1])]
            )
            cat_list = [x.decode() if isinstance(x, bytes) else x for x in cat_raw]
            df_edges = pd.DataFrame(matrix, columns=cat_list)
            features = hdf["features"][()]
            umap_emb = hdf["umap_embedding"][()] if "umap_embedding" in hdf else None
            image_umap: Optional[Dict[str, Dict[int, np.ndarray]]] = None
            if "image_umap" in hdf:
                image_umap = {}
                grp = hdf["image_umap"]
                for edge in grp:
                    data = grp[edge][()]
                    if data.size > 0:
                        image_umap[edge] = {
                            int(i): data[idx, 1:]
                            for idx, i in enumerate(data[:, 0].astype(int))
                        }
                    else:
                        image_umap[edge] = {}

            openclip_feats = (
                hdf["openclip_features"][()] if "openclip_features" in hdf else None
            )
            places365_feats = (
                hdf["places365_features"][()] if "places365_features" in hdf else None
            )

            if "edge_origins" in hdf:
                origin_raw = hdf["edge_origins"][()]
                edge_orig = [
                    o.decode() if isinstance(o, bytes) else str(o) for o in origin_raw
                ]
            else:
                edge_orig = ["swinv2"] * len(cat_list)

            # Backwards compatibility: older sessions may use "Loaded" as origin
            edge_orig = ["swinv2" if o == "Loaded" else o for o in edge_orig]

            thumbnails_embedded = hdf.attrs.get("thumbnails_are_embedded", True)
            thumbnail_data: Optional[List[bytes] | List[str]] = None
            if thumbnails_embedded and "thumbnail_data_embedded" in hdf:
                thumbnail_data = [
                    arr.tobytes() for arr in hdf["thumbnail_data_embedded"][:]
                ]
            elif not thumbnails_embedded and "thumbnail_relative_paths" in hdf:
                thumbnail_data = [
                    p.decode("utf-8") if isinstance(p, bytes) else str(p)
                    for p in hdf["thumbnail_relative_paths"][:]
                ]

            metadata_df: pd.DataFrame | None = None
            if "metadata" in hdf:
                meta_json = hdf["metadata"][()]
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode("utf-8")
                metadata_df = pd.read_json(meta_json, orient="table")

            seen_raw = hdf["edge_seen_times"][()] if "edge_seen_times" in hdf else None

            model_names: List[str] = []
            model_feats: dict[str, ModelFeatures] = {}
            if "audio_model_names" in hdf:
                names_raw = hdf["audio_model_names"][()]
                model_names = [n.decode() if isinstance(n, bytes) else str(n) for n in names_raw]
            if "audio_model_features" in hdf:
                grp = hdf["audio_model_features"]
                for name in grp:
                    g = grp[name]
                    segs = SegmentLevel(
                        embeddings=g["segment_embeddings"][()],
                        song_id=g["segment_song_id"][()].astype(np.int32),
                        start_s=g["segment_start_s"][()].astype(np.float32),
                        end_s=g["segment_end_s"][()].astype(np.float32),
                    )
                    songs = SongLevel(
                        centroid=g["song_centroid"][()],
                        stats2D=g["song_stats2D"][()],
                        song_id=g["song_song_id"][()].astype(np.int32),
                        path=[p.decode() if isinstance(p, bytes) else str(p) for p in g["song_path"][()]],
                    )
                    model_feats[name] = ModelFeatures(name=name, segments=segs, songs=songs)
                if not model_names:
                    model_names = list(grp.keys())

        return cls(
            im_list,
            df_edges,
            features,
            path,
            openclip_features=openclip_feats,
            places365_features=places365_feats,
            umap_embedding=umap_emb,
            image_umap=image_umap,
            thumbnail_data=thumbnail_data,
            thumbnails_are_embedded=thumbnails_embedded,
            edge_origins=edge_orig,
            edge_last_seen=seen_raw,
            metadata=metadata_df,
            model_features=model_feats,
            model_names=model_names,
        )


    def save_h5(self, path: Path | None = None) -> None:
        target = Path(path) if path else self.h5_path
        if not target.suffix:
            target = target.with_suffix(".h5")

        with h5py.File(target, "w") as hdf:
            print('starting save')
            dt = h5py.string_dtype(encoding="utf-8")
            hdf.create_dataset(
                "file_list", data=np.array(self.im_list, dtype=object), dtype=dt
            )
            print('saved filelist')
            hdf.create_dataset(
                "clustering_results",
                data=self.df_edges.values.astype("i8"),
                dtype="i8",
            )
            print('saved clustering results')
            hdf.create_dataset(
                "catList", data=np.array(self.cat_list, dtype=object), dtype=dt
            )
            hdf.create_dataset(
                "edge_origins",
                data=np.array(
                    [self.edge_origins.get(n, "swinv2") for n in self.cat_list],
                    dtype=object,
                ),
                dtype=dt,
            )
            print('saved edge origins')

            hdf.create_dataset(
                "edge_seen_times",
                data=np.array([self.edge_seen_times.get(n, 0.0) for n in self.cat_list], dtype="f8"),
                dtype="f8",
            )
            print('saved edge seen times')

            hdf.create_dataset("features", data=self.features, dtype="f4")
            print('saved features')
            if self.openclip_features is not None:
                hdf.create_dataset(
                    "openclip_features", data=self.openclip_features, dtype="f4"
                )
            print('saved openclip features')
            if self.places365_features is not None:
                hdf.create_dataset(
                    "places365_features", data=self.places365_features, dtype="f4"
                )
            print('saved places365_features')

            if getattr(self, "model_names", None):
                hdf.create_dataset(
                    "audio_model_names",
                    data=np.array(self.model_names, dtype=object),
                    dtype=dt,
                )
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

            if self.umap_embedding is not None:
                hdf.create_dataset(
                    "umap_embedding", data=self.umap_embedding, dtype="f4"
                )
            print('saved umap_embedding')
                
            if getattr(self, "image_umap", None):
                grp = hdf.create_group("image_umap")
                for edge, mapping in self.image_umap.items():
                    arr = np.array(
                        [[idx, vec[0], vec[1]] for idx, vec in mapping.items()],
                        dtype="f4",
                    )
                    grp.create_dataset(edge, data=arr, dtype="f4")
            print('saved image_umap')
            
            hdf.attrs["thumbnails_are_embedded"] = self.thumbnails_are_embedded
            print('saved thumbs emb')
            try:
                meta_json = (
                    self._sanitize_metadata(self.metadata).to_json(orient="table")
                )
                print('meta json 1')
            except Exception:
                meta_json = self.metadata.astype(str).to_json(orient="table")
                print('meta json 2')
            hdf.create_dataset("metadata", data=np.string_(meta_json), dtype=dt)            
            
            print('meta json 3')
            if self.thumbnail_data:
                if self.thumbnails_are_embedded:
                    dt_vlen = h5py.vlen_dtype(np.uint8)
                    arrs = [
                        np.frombuffer(b, dtype=np.uint8) for b in self.thumbnail_data
                    ]
                    hdf.create_dataset(
                        "thumbnail_data_embedded", data=arrs, dtype=dt_vlen
                    )
                    print('thumbs2')
                else:
                    hdf.create_dataset(
                        "thumbnail_relative_paths",
                        data=np.array(self.thumbnail_data, dtype=object),
                        dtype=dt,
                    )
                    print('thumbs3')
        self.h5_path = target


    def __init__(self,
                 im_list: List[str],
                 df_edges: pd.DataFrame,
                 features: np.ndarray,
                 h5_path: Path,
                 *,
                 openclip_features: np.ndarray | None = None,
                 places365_features: np.ndarray | None = None,
                 umap_embedding: np.ndarray | None = None,
                 image_umap: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
                 thumbnail_data: Optional[List[bytes] | List[str]] = None,
                 thumbnails_are_embedded: bool = True,
                 edge_origins: Optional[List[str]] | None = None,
                 edge_last_seen: Optional[List[float]] | None = None,
                 metadata: pd.DataFrame | None = None,
                 model_features: dict[str, ModelFeatures] | None = None,
                 model_names: List[str] | None = None):
        super().__init__()
        self.im_list  = im_list                              # list[str]
        self.cat_list = list(df_edges.columns)               # list[str]
        self.df_edges = df_edges                             # DataFrame (images×edges)
        self.hyperedges, self.image_mapping = self._prepare_hypergraph_structures(df_edges)

        self.features = features                             # np.ndarray (N×D)
        self.openclip_features = openclip_features
        self.places365_features = places365_features        
        self.hyperedge_avg_features = self._calculate_hyperedge_avg_features(features)

        feats32 = self.features.astype(np.float32, copy=False)
        norms = np.linalg.norm(feats32, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._features_unit = feats32 / norms

        self.edge_origins = edge_origins or {name: "swin" for name in self.cat_list}

        self.metadata = metadata if metadata is not None else self._extract_image_metadata(im_list)

        self.status_map = {n: {"uuid": str(uuid.uuid4()), "status": "Original"}
                           for n in self.cat_list}
        self.edge_seen_times = {
            name: (edge_last_seen[i] if edge_last_seen is not None and i < len(edge_last_seen) else 0.0)
            for i, name in enumerate(self.cat_list)
        }

        self.status_map = {n: {"uuid": str(uuid.uuid4()), "status": "Original"}
                           for n in self.cat_list}

        colors = generate_n_colors(len(self.cat_list))
        self.edge_colors = {
            name: colors[i % len(colors)]
            for i, name in enumerate(self.cat_list)
        }
        self.edge_origins = {
            name: (edge_origins[i] if edge_origins and i < len(edge_origins) else "swinv2")
            for i, name in enumerate(self.cat_list)
        }
        self.umap_embedding = umap_embedding
        self.image_umap = image_umap
        self.h5_path = h5_path
        
        self.thumbnail_data: Optional[List[bytes] | List[str]] = thumbnail_data
        self.thumbnails_are_embedded: bool = thumbnails_are_embedded

        self.model_features: dict[str, ModelFeatures] = model_features or {}
        self.model_names: List[str] = model_names or list(self.model_features.keys())

        self.edge_to_song_index: Dict[str, int] = {}
        self.song_edge_names: Dict[int, str] = {}
        for name, members in self.hyperedges.items():
            if len(members) == 1:
                idx = int(next(iter(members)))
                self.edge_to_song_index[name] = idx
                self.song_edge_names[idx] = name

        # Determine the primary audio model (OpenL3 in current workflow)
        self.primary_model: str | None = None
        if self.model_names:
            for cand in self.model_names:
                if cand in self.model_features:
                    self.primary_model = cand
                    break
        elif self.model_features:
            self.primary_model = next(iter(self.model_features))

        self.segment_embeddings = np.zeros((0, 0), dtype=np.float32)
        self.segment_embeddings_unit = np.zeros((0, 0), dtype=np.float32)
        self.segment_song_id = np.zeros((0,), dtype=np.int32)
        self.segment_start_s = np.zeros((0,), dtype=np.float32)
        self.segment_end_s = np.zeros((0,), dtype=np.float32)
        self.song_segment_map: Dict[int, np.ndarray] = {}
        self.song_durations = np.zeros(len(self.im_list), dtype=np.float32)

        if self.primary_model:
            mf = self.model_features.get(self.primary_model)
            if mf:
                seg = mf.segments
                emb = np.asarray(seg.embeddings, dtype=np.float32)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                elif emb.ndim == 0:
                    emb = np.zeros((0, 0), dtype=np.float32)
                self.segment_embeddings = emb

                self.segment_song_id = np.asarray(seg.song_id, dtype=np.int32).reshape(-1)
                self.segment_start_s = np.asarray(seg.start_s, dtype=np.float32).reshape(-1)
                self.segment_end_s = np.asarray(seg.end_s, dtype=np.float32).reshape(-1)

                if self.segment_embeddings.size:
                    norms = np.linalg.norm(self.segment_embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    self.segment_embeddings_unit = self.segment_embeddings / norms
                else:
                    self.segment_embeddings_unit = np.zeros_like(self.segment_embeddings)

        emb_dim = self.segment_embeddings.shape[1] if self.segment_embeddings.ndim == 2 else 0
        if self.segment_embeddings_unit.shape[1] != emb_dim:
            self.segment_embeddings_unit = np.zeros((0, emb_dim), dtype=np.float32)

        total_songs = len(self.im_list)
        for idx in range(total_songs):
            if self.segment_song_id.size:
                indices = np.where(self.segment_song_id == idx)[0]
            else:
                indices = np.zeros((0,), dtype=np.int32)
            self.song_segment_map[idx] = indices
            if indices.size:
                self.song_durations[idx] = float(self.segment_end_s[indices].max())

        if not self.segment_embeddings.size:
            self.segment_embeddings = np.zeros((0, emb_dim), dtype=np.float32)
            self.segment_embeddings_unit = np.zeros((0, emb_dim), dtype=np.float32)

        self.overview_triplets: Dict[str, tuple[int | None, ...]] | None = None
        self.compute_overview_triplets()


    @property
    def features_unit(self) -> np.ndarray:
        return self._features_unit

    @staticmethod
    def _prepare_hypergraph_structures(df):
        hyperedges = {col: set(np.where(df[col] == 1)[0]) for col in df.columns}
        image_mapping: Dict[int, Set[str]] = {}
        rows, cols = np.where(df.values == 1)
        for r, c in zip(rows, cols):
            image_mapping.setdefault(r, set()).add(df.columns[c])
        return hyperedges, image_mapping

    def _calculate_hyperedge_avg_features(self, features):
        n_feat = features.shape[1]
        return {name: features[list(idx)].mean(axis=0) if idx else np.zeros(n_feat)
                for name, idx in self.hyperedges.items()}

    @staticmethod
    def _sanitize_metadata(df: pd.DataFrame) -> pd.DataFrame:
        return df.applymap(
            lambda x: x
            if isinstance(x, (int, float, str, bool))
            else str(x)
        )
    
    @staticmethod
    def _extract_image_metadata(im_list: List[str]) -> pd.DataFrame:
        meta_rows = []
        for path in im_list:
            entry = {"image_path": path}
            try:
                with Image.open(path) as img:
                    exif = img._getexif()
                    if exif:
                        for k, v in exif.items():
                            tag = ExifTags.TAGS.get(k, k)
                            entry[tag] = v
            except Exception:
                pass
            meta_rows.append(entry)

        return pd.DataFrame(meta_rows)


    def generate_thumbnails(self, size: tuple[int, int] = (100, 100)) -> None:
        thumbs: List[bytes] = []
        for p in self.im_list:
            try:
                img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                img.thumbnail(size, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", size, "black")
                off_x = (size[0] - img.width) // 2
                off_y = (size[1] - img.height) // 2
                canvas.paste(img, (off_x, off_y))
                buf = io.BytesIO()
                canvas.save(buf, format="JPEG", quality=90)
                thumbs.append(buf.getvalue())
            except Exception as e:
                print(f"Thumbnail generation failed for {p}: {e}")
                thumbs.append(b"")

        self.thumbnail_data = thumbs
        self.thumbnails_are_embedded = True


    # ------------------------------------------------------------------
    def _update_edit_status(self, name: str, *, renamed: bool = False, modified: bool = False) -> None:
        entry = self.status_map.get(name)
        if not entry:
            return
        status = entry.get("status", "Original")
        if status in {"New", "Orphaned", "Cluster"}:
            return
        has_renamed = "Renamed" in status
        has_modified = "modified" in status.lower()
        if renamed:
            has_renamed = True
        if modified:
            has_modified = True
        if has_renamed and has_modified:
            entry["status"] = "Renamed and modified"
        elif has_renamed:
            entry["status"] = "Renamed"
        elif has_modified:
            entry["status"] = "Modified"
        else:
            entry["status"] = "Original"


    def rename_edge(self, old: str, new: str) -> bool:
        new = new.strip()
        if (not new) or (new in self.hyperedges):
            return False

        old_indices = self.hyperedges.get(old, set()).copy()

        self.hyperedges[new] = self.hyperedges.pop(old)
        self.df_edges.rename(columns={old: new}, inplace=True)
        self.cat_list[self.cat_list.index(old)] = new
        self.hyperedge_avg_features[new] = self.hyperedge_avg_features.pop(old)
        self.status_map[new] = self.status_map.pop(old)
        self._update_edit_status(new, renamed=True)
        if old in self.edge_origins:
            self.edge_origins[new] = self.edge_origins.pop(old)
        if old in self.edge_colors:
            self.edge_colors[new] = self.edge_colors.pop(old)
        if old in self.edge_seen_times:
            self.edge_seen_times[new] = self.edge_seen_times.pop(old)
        for idx in old_indices:
            imgs = self.image_mapping.get(idx)
            if imgs is not None and old in imgs:
                imgs.remove(old)
                imgs.add(new)

        if self.overview_triplets is not None and old in self.overview_triplets:
            self.overview_triplets[new] = self.overview_triplets.pop(old)
        self.edgeRenamed.emit(old, new)
        
        return True

    def add_empty_hyperedge(self, name: str) -> None:
        start12 = time.perf_counter()
        """Adds a new, empty hyperedge to the model."""
        self.hyperedges[name] = set()
        self.df_edges[name] = 0
        self.cat_list.append(name)
        n_features = self.features.shape[1]
        self.hyperedge_avg_features[name] = np.zeros(n_features)

        self.status_map[name] = {"uuid": str(uuid.uuid4()), "status": "New"}
        self.edge_origins[name] = "New"
        self.edge_colors[name] = generate_n_colors(len(self.edge_colors) + 1)[-1]

        idx = len(self.edge_colors)
        cmap_hues = max(idx + 1, 16)
        self.edge_colors[name] = pg.mkColor(pg.intColor(idx, hues=cmap_hues)).name()
        self.edge_seen_times[name] = 0.0
        self.overview_triplets = None
        print('12',time.perf_counter() - start12)
        self.layoutChanged.emit()
        print('13',time.perf_counter() - start12)
        self.hyperedgeModified.emit(name)
        print('14',time.perf_counter() - start12)

    def add_images_to_hyperedge(self, name: str, idxs: Iterable[int]) -> None:
        """Add selected images to an existing hyperedge."""
        if name not in self.hyperedges:
            return

        changed = False
        for idx in idxs:
            if idx not in self.hyperedges[name]:
                self.hyperedges[name].add(idx)
                self.df_edges.at[idx, name] = 1
                self.image_mapping.setdefault(idx, set()).add(name)
                changed = True

        if changed:
            indices = list(self.hyperedges[name])
            if indices:
                self.hyperedge_avg_features[name] = self.features[indices].mean(axis=0)
            else:
                self.hyperedge_avg_features[name] = np.zeros(self.features.shape[1])
            self._update_edit_status(name, modified=True)
            if self.overview_triplets is not None:
                self.overview_triplets.pop(name, None)
            self.layoutChanged.emit()
            self.similarityDirty.emit()
            self.hyperedgeModified.emit(name)

    def remove_images_from_edges(self, img_idxs: List[int], edges: List[str]) -> None:
        """Remove selected images from the specified hyperedges."""
        changed_edges: set[str] = set()
        for edge in edges:
            if edge not in self.hyperedges:
                continue
            members = self.hyperedges[edge]
            removed = [i for i in img_idxs if i in members]
            if not removed:
                continue
            changed_edges.add(edge)
            for idx in removed:
                members.remove(idx)
                if idx in self.image_mapping:
                    self.image_mapping[idx].discard(edge)
                    if not self.image_mapping[idx]:
                        del self.image_mapping[idx]
                if idx < len(self.df_edges.index):
                    self.df_edges.at[idx, edge] = 0

            if members:
                self.hyperedge_avg_features[edge] = self.features[list(members)].mean(axis=0)
            else:
                self.hyperedge_avg_features[edge] = np.zeros(self.features.shape[1])
            self._update_edit_status(edge, modified=True)

        if changed_edges:
            if self.overview_triplets is not None:
                for edge in changed_edges:
                    self.overview_triplets.pop(edge, None)
            self.layoutChanged.emit()
            self.similarityDirty.emit()
            for edge in changed_edges:
                self.hyperedgeModified.emit(edge)

    def delete_hyperedge(self, name: str, orphan_name: str = "orphaned images") -> None:
        if name not in self.hyperedges:
            return

        members = self.hyperedges.pop(name)

        if name in self.df_edges.columns:
            self.df_edges.drop(columns=[name], inplace=True)
        if name in self.cat_list:
            self.cat_list.remove(name)
        self.hyperedge_avg_features.pop(name, None)
        self.status_map.pop(name, None)
        self.edge_origins.pop(name, None)
        self.edge_colors.pop(name, None)
        self.edge_seen_times.pop(name, None)        
        if getattr(self, "image_umap", None):
            self.image_umap.pop(name, None)

        if orphan_name not in self.hyperedges:
            self.hyperedges[orphan_name] = set()
            self.df_edges[orphan_name] = 0
            self.cat_list.append(orphan_name)
            n_features = self.features.shape[1]
            self.hyperedge_avg_features[orphan_name] = np.zeros(n_features)
            self.status_map[orphan_name] = {"uuid": str(uuid.uuid4()), "status": "Orphaned"}
            self.edge_colors[orphan_name] = generate_n_colors(len(self.edge_colors) + 1)[-1]
            self.edge_origins[orphan_name] = "system"
            self.edge_seen_times[orphan_name] = 0.0

        for idx in members:
            if idx in self.image_mapping:
                self.image_mapping[idx].discard(name)
                if not self.image_mapping[idx]:
                    self.image_mapping[idx].add(orphan_name)
                    self.hyperedges[orphan_name].add(idx)
                    self.df_edges.at[idx, orphan_name] = 1
            if idx < len(self.df_edges.index):
                pass

        orphans = self.hyperedges[orphan_name]
        if orphans:
            self.hyperedge_avg_features[orphan_name] = self.features[list(orphans)].mean(axis=0)
        else:
            self.hyperedge_avg_features[orphan_name] = np.zeros(self.features.shape[1])

        if self.overview_triplets is not None:
            self.overview_triplets.pop(name, None)
            self.overview_triplets.pop(orphan_name, None)
        self.layoutChanged.emit()
        self.similarityDirty.emit()
        self.hyperedgeModified.emit(orphan_name)
        # self.hyperedgeModified.emit(name)

    def prune_similar_edges(self, threshold: float) -> None:
        precedence = ["swinv2", "places365", "openclip"]
        ordered_names = self.cat_list[:]
        kept: list[str] = []
        remove: list[str] = []
        for origin in precedence:
            for name in ordered_names:
                if self.edge_origins.get(name) != origin or name in remove:
                    continue
                if origin == "swinv2":
                    kept.append(name)
                    continue
                is_dup = False
                for k in kept:
                    if jaccard_similarity(self.hyperedges[name], self.hyperedges[k]) > threshold:
                        is_dup = True
                        break
                if is_dup:
                    remove.append(name)
                else:
                    kept.append(name)

        for name in remove:
            self.delete_hyperedge(name)


    def vector_for(self, name: str) -> np.ndarray | None:
        return self.hyperedge_avg_features.get(name)

    def similarity_map(self, ref_name: str) -> Dict[str, float]:
        ref = self.vector_for(ref_name)
        if ref is None:
            return {}
        names = list(self.hyperedge_avg_features)
        mat = np.stack([self.hyperedge_avg_features[n] for n in names])
        sims = SIM_METRIC(ref.reshape(1, -1), mat)[0]
        return dict(zip(names, sims))
    
    def similarity_std(self, name: str) -> float | None:
        idxs = list(self.hyperedges.get(name, []))
        if not idxs:
            return None
        feats = self.features[idxs]
        avg = self.hyperedge_avg_features[name][None, :]
        sims = SIM_METRIC(avg, feats)[0]
        return float(np.std(sims))

    def overview_triplet_for(self, name: str) -> tuple[int | None, ...]:
        # not actually triplets anymore, but sextets
        if self.overview_triplets is None:
            self.overview_triplets = {}
        trip = self.overview_triplets.get(name)
        if trip is not None:
            return trip

        idxs = self.hyperedges.get(name)
        if not idxs:
            trip = ()
        else:
            idx_list = list(idxs)
            feats = self.features[idx_list]
            avg = self.hyperedge_avg_features[name].reshape(1, -1)
            sims = SIM_METRIC(avg, feats)[0]
            top_order = np.argsort(sims)[::-1]
            top = [idx_list[i] for i in top_order[:3]]

            extremes: list[int] = []
            if len(idx_list) >= 2:
                sim_mat = SIM_METRIC(feats, feats)
                np.fill_diagonal(sim_mat, 1.0)
                i, j = divmod(np.argmin(sim_mat), sim_mat.shape[1])
                extremes = [idx_list[i], idx_list[j]]

            far_order = np.argsort(sims)
            farthest: int | None = None
            for i in far_order:
                cand = idx_list[i]
                if cand not in top and cand not in extremes:
                    farthest = cand
                    break

            final: list[int | None] = []
            for idx in top + extremes:
                if idx not in final:
                    final.append(idx)
            if farthest is not None and farthest not in final:
                final.append(farthest)
            while len(final) < 6:
                final.append(None)
            trip = tuple(final[:6])

        self.overview_triplets[name] = trip
        return trip

    def compute_overview_triplets(self) -> Dict[str, tuple[int | None, ...]]:
        """Return and cache up to six representative image indices per edge."""
        if self.overview_triplets is None:
            self.overview_triplets = {}

        for name in self.hyperedges:
            if name not in self.overview_triplets:
                self.overview_triplets[name] = self.overview_triplet_for(name)

        return self.overview_triplets

    def apply_clustering_matrix(
        self, matrix: np.ndarray, *, prefix: str = "edge", origin: str = "swinv2"
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError("clustering matrix must be 2D")

        df_edges = pd.DataFrame(
            matrix.astype(int),
            columns=[f"{prefix}_{i}" for i in range(matrix.shape[1])],
        )
        self.df_edges = df_edges
        self.cat_list = list(df_edges.columns)
        self.edge_origins = {name: origin for name in self.cat_list}

        self.hyperedges, self.image_mapping = self._prepare_hypergraph_structures(
            df_edges
        )
        self.hyperedge_avg_features = self._calculate_hyperedge_avg_features(self.features)

        status = "Original" if origin == "places365" else "Origin"
        self.status_map = {
            name: {"uuid": str(uuid.uuid4()), "status": status}
            for name in self.cat_list
        }
        colors = generate_n_colors(len(self.cat_list))
        self.edge_colors = {name: colors[i % len(colors)] for i, name in enumerate(self.cat_list)}
        self.edge_seen_times = {name: 0.0 for name in self.cat_list}

        self.layoutChanged.emit()
        self.similarityDirty.emit()

    def append_clustering_matrix(
        self, matrix: np.ndarray, *, prefix: str = "edge", origin: str = "swinv2"
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError("clustering matrix must be 2D")
        if matrix.shape[0] != len(self.im_list):
            raise ValueError("matrix row count must match number of images")

        start_idx = len(self.cat_list)
        n_new = matrix.shape[1]
        color_list = generate_n_colors(start_idx + n_new)
        for i in range(n_new):
            name = f"{prefix}_{start_idx + i}"
            col = matrix[:, i].astype(int)
            self.df_edges[name] = col
            self.cat_list.append(name)

            idxs = set(np.where(col == 1)[0])
            self.hyperedges[name] = idxs
            for idx in idxs:
                self.image_mapping.setdefault(idx, set()).add(name)
            self.hyperedge_avg_features[name] = (
                self.features[list(idxs)].mean(axis=0)
                if idxs
                else np.zeros(self.features.shape[1])
            )
            status = "Original" if origin == "places365" else "Origin"
            self.status_map[name] = {"uuid": str(uuid.uuid4()), "status": status}
            # ).name()
            self.edge_colors[name] = color_list[start_idx + i]
            self.edge_origins[name] = origin
            self.edge_seen_times[name] = 0.0

        self.layoutChanged.emit()
        self.similarityDirty.emit()