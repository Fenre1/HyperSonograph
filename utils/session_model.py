from __future__ import annotations  
import uuid
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Set, Iterable, Optional, Sequence, Mapping
from PIL import Image, ExifTags, ImageOps
import io
from PyQt5.QtCore import QObject, pyqtSignal as Signal
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import time
from dataclasses import dataclass
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

@dataclass(frozen=True)
class SegmentMatchResult:
    """Best matching segment pair between two songs."""

    song_a: int
    song_b: int
    segment_a: int
    segment_b: int
    start_a: float
    start_b: float
    end_a: float
    end_b: float
    score: float

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

        self._update_feature_norms()


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

        self._rebuild_song_edge_maps()

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

        self._refresh_segment_cache()

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
    def _update_feature_norms(self) -> None:
        feats32 = np.asarray(self.features, dtype=np.float32)
        if feats32.ndim == 1:
            feats32 = feats32.reshape(1, -1)
        elif feats32.ndim == 0:
            feats32 = np.zeros((0, 0), dtype=np.float32)

        if feats32.size == 0:
            self._features_unit = feats32
            return

        norms = np.linalg.norm(feats32, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._features_unit = feats32 / norms

    def _rebuild_song_edge_maps(self) -> None:
        self.edge_to_song_index: Dict[str, int] = {}
        self.song_edge_names: Dict[int, str] = {}
        for name, members in self.hyperedges.items():
            if len(members) == 1:
                idx = int(next(iter(members)))
                self.edge_to_song_index[name] = idx
                self.song_edge_names[idx] = name

    def _refresh_segment_cache(self) -> None:
        emb_dim = 0
        if self.primary_model:
            mf = self.model_features.get(self.primary_model)
        else:
            mf = None

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
        else:
            self.segment_embeddings = np.zeros((0, 0), dtype=np.float32)
            self.segment_embeddings_unit = np.zeros((0, 0), dtype=np.float32)
            self.segment_song_id = np.zeros((0,), dtype=np.int32)
            self.segment_start_s = np.zeros((0,), dtype=np.float32)
            self.segment_end_s = np.zeros((0,), dtype=np.float32)

        if self.segment_embeddings_unit.shape[1] != emb_dim:
            self.segment_embeddings_unit = np.zeros((0, emb_dim), dtype=np.float32)

        total_songs = len(self.im_list)
        self.song_segment_map = {}
        self.song_durations = np.zeros(total_songs, dtype=np.float32)
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

    def add_songs(
        self,
        files: Sequence[str],
        features: np.ndarray,
        edge_names: Sequence[str],
        model_features: Mapping[str, ModelFeatures],
        *,
        origin: str = "song",
    ) -> None:
        """Append new songs with their audio features to the session."""

        n_new = len(files)
        if n_new == 0:
            return
        if len(edge_names) != n_new:
            raise ValueError("edge_names length must match files length")

        new_features = np.asarray(features, dtype=np.float32)
        if new_features.ndim == 1:
            new_features = new_features.reshape(1, -1)
        elif new_features.ndim == 0:
            new_features = np.zeros((0, 0), dtype=np.float32)
        if new_features.shape[0] != n_new:
            raise ValueError("features row count must match number of files")

        if self.features.size:
            if self.features.ndim != 2:
                raise ValueError("session features array must be 2D")
            if new_features.size and self.features.shape[1] != new_features.shape[1]:
                raise ValueError("feature dimension mismatch")
        else:
            # ensure features array has the correct shape for stacking later
            self.features = np.zeros((0, new_features.shape[1]), dtype=np.float32)

        start_idx = len(self.im_list)
        self.im_list.extend(str(p) for p in files)

        # Maintain thumbnail list length if thumbnails are stored
        if self.thumbnail_data is not None:
            if not isinstance(self.thumbnail_data, list):
                self.thumbnail_data = list(self.thumbnail_data)
            placeholder = b"" if self.thumbnails_are_embedded else ""
            self.thumbnail_data.extend([placeholder] * n_new)

        if isinstance(self.metadata, pd.DataFrame):
            new_meta = pd.DataFrame([{"image_path": path} for path in files])
            self.metadata = pd.concat([self.metadata, new_meta], ignore_index=True)

        # Extend dataframe with empty rows for the new songs
        base_cols = list(self.df_edges.columns)
        if base_cols:
            new_rows = pd.DataFrame(0, index=range(n_new), columns=base_cols, dtype=int)
        else:
            new_rows = pd.DataFrame(index=range(n_new))
        self.df_edges = pd.concat([self.df_edges, new_rows], ignore_index=True)

        for name in edge_names:
            if name in self.df_edges.columns:
                raise ValueError(f"Hyperedge '{name}' already exists")
            self.df_edges[name] = 0

        self.cat_list.extend(edge_names)

        # Prepare new feature vectors for hyperedges
        for offset, name in enumerate(edge_names):
            row_idx = start_idx + offset
            self.df_edges.at[row_idx, name] = 1
            self.hyperedges[name] = {row_idx}
            self.image_mapping[row_idx] = {name}
            vec = (
                new_features[offset].astype(np.float32, copy=False)
                if new_features.size
                else np.zeros(self.features.shape[1], dtype=np.float32)
            )
            self.hyperedge_avg_features[name] = vec
            self.status_map[name] = {"uuid": str(uuid.uuid4()), "status": "New"}
            color_idx = len(self.edge_colors)
            cmap_hues = max(color_idx + 1, 16)
            self.edge_colors[name] = pg.mkColor(pg.intColor(color_idx, hues=cmap_hues)).name()
            self.edge_origins[name] = origin
            self.edge_seen_times[name] = 0.0

        if new_features.size:
            self.features = np.vstack([self.features, new_features])
        else:
            self.features = np.vstack([self.features, np.zeros((n_new, self.features.shape[1]), dtype=np.float32)])

        self._update_feature_norms()

        # Merge model features
        for model_name, mf_new in model_features.items():
            seg_new = mf_new.segments
            songs_new = mf_new.songs

            emb_new = np.asarray(seg_new.embeddings, dtype=np.float32)
            if emb_new.ndim == 1:
                emb_new = emb_new.reshape(1, -1)
            elif emb_new.ndim == 0:
                emb_new = np.zeros((0, 0), dtype=np.float32)

            song_id_new = np.asarray(seg_new.song_id, dtype=np.int32).reshape(-1)
            start_new = np.asarray(seg_new.start_s, dtype=np.float32).reshape(-1)
            end_new = np.asarray(seg_new.end_s, dtype=np.float32).reshape(-1)

            centroid_new = np.asarray(songs_new.centroid, dtype=np.float32)
            if centroid_new.ndim == 1:
                centroid_new = centroid_new.reshape(1, -1)
            elif centroid_new.ndim == 0:
                centroid_new = np.zeros((0, 0), dtype=np.float32)

            stats_new = np.asarray(songs_new.stats2D, dtype=np.float32)
            if stats_new.ndim == 1:
                stats_new = stats_new.reshape(1, -1)
            elif stats_new.ndim == 0:
                stats_new = np.zeros((0, 0), dtype=np.float32)

            song_ids_new = np.asarray(songs_new.song_id, dtype=np.int32).reshape(-1)
            paths_new = list(songs_new.path)

            existing = self.model_features.get(model_name)
            if existing:
                seg_old = existing.segments
                songs_old = existing.songs

                emb_old = np.asarray(seg_old.embeddings, dtype=np.float32)
                if emb_old.ndim == 1:
                    emb_old = emb_old.reshape(1, -1)
                elif emb_old.ndim == 0:
                    emb_old = np.zeros((0, 0), dtype=np.float32)
                if emb_old.size == 0 and emb_new.size:
                    emb_old = np.zeros((0, emb_new.shape[1]), dtype=np.float32)
                if emb_old.size and emb_new.size and emb_old.shape[1] != emb_new.shape[1]:
                    raise ValueError(f"Segment embedding dimension mismatch for model '{model_name}'")

                seg_embeddings = (
                    np.vstack([emb_old, emb_new]) if emb_new.size else emb_old
                )

                song_id_old = np.asarray(seg_old.song_id, dtype=np.int32).reshape(-1)
                start_old = np.asarray(seg_old.start_s, dtype=np.float32).reshape(-1)
                end_old = np.asarray(seg_old.end_s, dtype=np.float32).reshape(-1)

                seg_song_id = np.concatenate([song_id_old, song_id_new]) if song_id_new.size else song_id_old
                seg_start = np.concatenate([start_old, start_new]) if start_new.size else start_old
                seg_end = np.concatenate([end_old, end_new]) if end_new.size else end_old

                cen_old = np.asarray(songs_old.centroid, dtype=np.float32)
                if cen_old.ndim == 1:
                    cen_old = cen_old.reshape(1, -1)
                elif cen_old.ndim == 0:
                    cen_old = np.zeros((0, 0), dtype=np.float32)
                if cen_old.size == 0 and centroid_new.size:
                    cen_old = np.zeros((0, centroid_new.shape[1]), dtype=np.float32)
                if cen_old.size and centroid_new.size and cen_old.shape[1] != centroid_new.shape[1]:
                    raise ValueError(f"Centroid dimension mismatch for model '{model_name}'")

                centroid = (
                    np.vstack([cen_old, centroid_new]) if centroid_new.size else cen_old
                )

                stats_old = np.asarray(songs_old.stats2D, dtype=np.float32)
                if stats_old.ndim == 1:
                    stats_old = stats_old.reshape(1, -1)
                elif stats_old.ndim == 0:
                    stats_old = np.zeros((0, 0), dtype=np.float32)
                if stats_old.size == 0 and stats_new.size:
                    stats_old = np.zeros((0, stats_new.shape[1]), dtype=np.float32)
                if stats_old.size and stats_new.size and stats_old.shape[1] != stats_new.shape[1]:
                    raise ValueError(f"Song feature dimension mismatch for model '{model_name}'")

                stats = (
                    np.vstack([stats_old, stats_new]) if stats_new.size else stats_old
                )

                song_ids_old = np.asarray(songs_old.song_id, dtype=np.int32).reshape(-1)
                song_ids = np.concatenate([song_ids_old, song_ids_new]) if song_ids_new.size else song_ids_old
                paths = list(songs_old.path) + paths_new

                combined_segments = SegmentLevel(
                    embeddings=seg_embeddings,
                    song_id=seg_song_id,
                    start_s=seg_start,
                    end_s=seg_end,
                )
                combined_songs = SongLevel(
                    centroid=centroid,
                    stats2D=stats,
                    song_id=song_ids,
                    path=paths,
                )
                self.model_features[model_name] = ModelFeatures(
                    name=model_name,
                    segments=combined_segments,
                    songs=combined_songs,
                )
            else:
                self.model_features[model_name] = ModelFeatures(
                    name=model_name,
                    segments=SegmentLevel(
                        embeddings=emb_new,
                        song_id=song_id_new,
                        start_s=start_new,
                        end_s=end_new,
                    ),
                    songs=SongLevel(
                        centroid=centroid_new,
                        stats2D=stats_new,
                        song_id=song_ids_new,
                        path=paths_new,
                    ),
                )
                if model_name not in self.model_names:
                    self.model_names.append(model_name)

        if not self.primary_model:
            if self.model_names:
                for cand in self.model_names:
                    if cand in self.model_features:
                        self.primary_model = cand
                        break
            elif self.model_features:
                self.primary_model = next(iter(self.model_features))

        self._rebuild_song_edge_maps()
        self._refresh_segment_cache()

        self.umap_embedding = None
        self.image_umap = {}

        self.overview_triplets = None
        self.similarityDirty.emit()
        self.layoutChanged.emit()

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

    def _segment_similarity(self, song_idx: int, *, average: bool) -> Dict[int, float]:
        if not (0 <= song_idx < len(self.im_list)):
            return {}
        if self.segment_embeddings_unit.size == 0:
            return {}

        base_indices = self.song_segment_map.get(song_idx)
        if base_indices is None or base_indices.size == 0:
            return {}

        base_vecs = self.segment_embeddings_unit[base_indices]
        if base_vecs.size == 0:
            return {}

        results: Dict[int, float] = {}
        for other_idx, seg_ids in self.song_segment_map.items():
            if seg_ids.size == 0:
                continue
            other_vecs = self.segment_embeddings_unit[seg_ids]
            if other_vecs.size == 0:
                continue
            sims = np.matmul(base_vecs, other_vecs.T)
            if sims.size == 0:
                continue
            value = float(np.mean(sims)) if average else float(np.max(sims))
            results[other_idx] = value

        return results

    def segment_similarity_single(self, song_idx: int) -> Dict[int, float]:
        """Return similarity scores using the best matching segment for each song."""

        return self._segment_similarity(song_idx, average=False)

    def segment_similarity_average(self, song_idx: int) -> Dict[int, float]:
        """Return similarity scores averaged over all segment comparisons."""

        return self._segment_similarity(song_idx, average=True)

    def song_level_similarity(self, song_idx: int) -> Dict[int, float]:
        """Return cosine similarity between song-level feature vectors."""

        if not (0 <= song_idx < len(self.im_list)):
            return {}
        if not self.primary_model:
            return {}

        mf = self.model_features.get(self.primary_model)
        if not mf:
            return {}

        stats = np.asarray(mf.songs.stats2D, dtype=np.float32)
        if stats.ndim == 1:
            stats = stats.reshape(1, -1)
        elif stats.ndim == 0 or stats.size == 0:
            return {}

        song_ids = np.asarray(mf.songs.song_id, dtype=np.int32).reshape(-1)
        if song_ids.size != stats.shape[0]:
            song_ids = np.arange(stats.shape[0], dtype=np.int32)

        id_to_idx = {int(sid): i for i, sid in enumerate(song_ids)}
        ref_idx = id_to_idx.get(int(song_idx))
        if ref_idx is None:
            return {}

        norms = np.linalg.norm(stats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        stats_unit = stats / norms

        ref_vec = stats_unit[ref_idx]
        sims = stats_unit @ ref_vec
        return {int(song_ids[i]): float(sims[i]) for i in range(len(song_ids))}

    def has_segment_features(self) -> bool:
        return bool(self.segment_embeddings_unit.size)

    def best_matching_segment_pair(
        self, song_a: int, song_b: int
    ) -> SegmentMatchResult | None:
        """Return the most similar segment pair between two songs.

        Parameters
        ----------
        song_a, song_b:
            Indices of the songs whose segments should be compared.

        Returns
        -------
        SegmentMatchResult | None
            Details about the best-matching segments, or ``None`` if no
            comparison could be made.
        """

        if not self.has_segment_features():
            return None

        indices_a = self.song_segment_map.get(song_a)
        indices_b = self.song_segment_map.get(song_b)
        if indices_a is None or indices_b is None:
            return None
        if indices_a.size == 0 or indices_b.size == 0:
            return None

        vecs_a = self.segment_embeddings_unit[indices_a]
        vecs_b = self.segment_embeddings_unit[indices_b]
        if vecs_a.size == 0 or vecs_b.size == 0:
            return None

        sims = np.matmul(vecs_a, vecs_b.T)
        if sims.size == 0:
            return None

        best_flat = int(np.argmax(sims))
        score = float(sims.flat[best_flat])
        cols = sims.shape[1]
        pos_a = best_flat // cols
        pos_b = best_flat % cols
        seg_idx_a = int(indices_a[pos_a])
        seg_idx_b = int(indices_b[pos_b])

        start_a = (
            float(self.segment_start_s[seg_idx_a])
            if self.segment_start_s.size > seg_idx_a
            else 0.0
        )
        end_a = (
            float(self.segment_end_s[seg_idx_a])
            if self.segment_end_s.size > seg_idx_a
            else start_a
        )
        start_b = (
            float(self.segment_start_s[seg_idx_b])
            if self.segment_start_s.size > seg_idx_b
            else 0.0
        )
        end_b = (
            float(self.segment_end_s[seg_idx_b])
            if self.segment_end_s.size > seg_idx_b
            else start_b
        )

        return SegmentMatchResult(
            song_a=song_a,
            song_b=song_b,
            segment_a=seg_idx_a,
            segment_b=seg_idx_b,
            start_a=start_a,
            start_b=start_b,
            end_a=end_a,
            end_b=end_b,
            score=score,
        )

    def has_song_level_features(self) -> bool:
        if not self.primary_model:
            return False
        mf = self.model_features.get(self.primary_model)
        if not mf:
            return False
        stats = np.asarray(mf.songs.stats2D, dtype=np.float32)
        if stats.ndim == 1:
            stats = stats.reshape(1, -1)
        return stats.ndim == 2 and stats.size > 0

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