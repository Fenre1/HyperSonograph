# audio_features.py
# %%
from __future__ import annotations
import logging
import os
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Callable
from collections import defaultdict
import numpy as np
import torch
import librosa
from contextlib import suppress
import soundfile as sf
import warnings
import tensorflow as tf
from tensorflow.keras import Model
import openl3
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf

# ----------------------------
# Model cache dir
# ----------------------------
try:
    _here = Path(__file__).resolve()
    _root = _here.parent.parent
except NameError:
    # __file__ can be undefined in notebooks; fall back to CWD
    _root = Path.cwd()

MODEL_DIR = _root / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(MODEL_DIR))

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
# ----------------------------
# Helpers: loading & segmentation
# ----------------------------

def load_audio_mono(path: str | Path, target_sr: int = 48000,
                    offset: float = 0.0, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    # first attempt
    y, sr = librosa.load(str(path), sr=target_sr, mono=True,
                         offset=float(offset), duration=duration)
    # tail rounding can yield empty; retry with a tiny pad
    if y.size == 0 and duration is not None and duration > 0:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True,
                             offset=float(offset), duration=float(duration) + 1e-2)
    return np.asarray(y, dtype=np.float32, order="C"), target_sr


def segment_bounds(
    total_seconds: float,
    segment_seconds: float = 30.0,
    hop_seconds: Optional[float] = None,
) -> List[Tuple[float, float]]:
    hop = float(hop_seconds) if hop_seconds is not None else float(segment_seconds) / 2.0
    if total_seconds <= 0 or segment_seconds <= 0 or hop <= 0:
        return []
    bounds: List[Tuple[float, float]] = []
    s = 0.0
    while s < total_seconds:
        e = min(s + segment_seconds, total_seconds)
        # avoid very tiny tail segments
        if e - s < 0.5:
            break
        bounds.append((s, e))
        s += hop
    return bounds


# ----------------------------
# Base API
# ----------------------------

@dataclass
class SegmentMeta:
    file: str
    start_s: float
    end_s: float
    model: str
    dim: int

@dataclass
class SongEmbedding:
    """Song-level representation for a single (file, model) pair."""
    file: str
    model: str
    centroid_D: np.ndarray
    stats_2D: np.ndarray


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of ``v`` (float32)."""
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.astype(np.float32)


def _compute_song_embeddings(X: np.ndarray, rows: List[Dict[str, Any]], normalize: bool = True) -> List[SongEmbedding]:
    """Aggregate segment vectors into song-level embeddings."""
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        grouped[(r["file"], r["model"])].append(idx)

    songs: List[SongEmbedding] = []
    for (file, model), idxs in grouped.items():
        V = X[idxs]
        weights = np.array([
            rows[i]["end_s"] - rows[i]["start_s"] for i in idxs
        ], dtype=np.float32).reshape(-1, 1)
        w_sum = float(weights.sum())
        if w_sum <= 0:
            D = V.shape[1]
            centroid = np.zeros(D, dtype=np.float32)
            stats = np.zeros(2 * D, dtype=np.float32)
        else:
            mean = (V * weights).sum(axis=0) / w_sum
            if normalize:
                centroid = _l2_normalize(mean)
                var = ((V - mean) ** 2 * weights).sum(axis=0) / w_sum
                std = np.sqrt(np.maximum(var, 1e-8))
                stats_vec = np.concatenate([mean, std], axis=0)
                stats = _l2_normalize(stats_vec)
            else:
                centroid = mean.astype(np.float32)
                stats = stats_vec.astype(np.float32)
        songs.append(
            SongEmbedding(
                file=file,
                model=model,
                centroid_D=centroid.astype(np.float32),
                stats_2D=stats.astype(np.float32),
            )
        )

    return songs




class AudioFeatureExtractorBase:
    model_name: str

    def __init__(
        self,
        segment_seconds: float = 30.0,
        hop_seconds: Optional[float] = None,
        target_sr: int = 48000,
        device: Optional[str] = None,
        normalize: bool = True,   # <-- NEW
    ):
        self.segment_seconds = float(segment_seconds)
        self.hop_seconds = hop_seconds
        self.target_sr = int(target_sr)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.normalize = normalize
        self._init_model()

    # to be implemented by subclasses
    def _init_model(self) -> None: ...
    
    def _embed_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Generic HF path: accepts mono float32 y, sampling rate sr.
        Handles both standard and custom processor APIs.
        If a subclass defines self.layer, we respect it; otherwise use last layer.
        """
        # prepare inputs (support both common processor signatures)
        try:
            inputs = self.processor([y], sampling_rate=sr, return_tensors="pt")  # type: ignore[attr-defined]
        except TypeError:
            inputs = self.processor(audios=[y], sampling_rate=sr, return_tensors="pt")  # type: ignore[attr-defined]

        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)  # type: ignore[attr-defined]

        # choose layer
        layer_idx = getattr(self, "layer", None)
        if hasattr(out, "hidden_states") and out.hidden_states is not None and layer_idx is not None:
            hs = out.hidden_states[layer_idx]
        else:
            hs = out.last_hidden_state

        # pool over time if needed
        if hs.dim() == 3:
            pooled = hs.mean(dim=1)
        elif hs.dim() == 2:
            pooled = hs
        else:
            raise RuntimeError(f"Unexpected tensor rank for hidden states: {hs.shape}")

        return pooled.squeeze(0).float().cpu().numpy()

    def close(self) -> None:
        """Release references to heavyweight models and clear GPU memory."""

        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            hook = getattr(model, "_hk", None)
            if hook is not None and hasattr(hook, "remove"):
                with suppress(Exception):
                    hook.remove()
            with suppress(Exception):
                model.to("cpu")
        if hasattr(self, "model"):
            self.model = None

        processor = getattr(self, "processor", None)
        if processor is not None and hasattr(processor, "to"):
            with suppress(Exception):
                processor.to("cpu")
        if hasattr(self, "processor"):
            self.processor = None

        if hasattr(self, "_tol3"):
            self._tol3 = None

        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.empty_cache()

        gc.collect()

    def output_dim(self) -> int: ...

    def extract_features_with_metadata(
        self,
        file_list: List[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        vecs: List[np.ndarray] = []

        total = len(file_list)

        for idx, f in enumerate(file_list, start=1):
            if progress_callback:
                try:
                    progress_callback(idx, total, f)
                except Exception:
                    logger.exception("Progress callback failed for %s", f)
            # get duration
            try:
                info = sf.info(f)
                total_seconds = float(info.frames) / float(info.samplerate)
            except Exception:
                y_tmp, sr_tmp = load_audio_mono(f, target_sr=self.target_sr)
                total_seconds = len(y_tmp) / float(sr_tmp)

            # compute segment bounds
            bounds = segment_bounds(total_seconds, self.segment_seconds, self.hop_seconds)
            if not bounds:
                continue

            # decode once at extractor SR
            y, sr = load_audio_mono(f, target_sr=self.target_sr)
            if y.size == 0:
                continue

            # single OpenL3 pass over the whole file
            frame_emb, ts = self._embed_file_frames(y, sr)  # (T, D), (T,)


            ts = np.asarray(ts, dtype=np.float32).reshape(-1)

            # make frame_emb (T, D)
            fe = np.asarray(frame_emb)
            if fe.ndim == 3 and fe.shape[0] == 1:
                fe = fe[0]                     # (D, T) or (T, D)
            if fe.ndim == 2 and fe.shape[0] != ts.shape[0] and fe.shape[1] == ts.shape[0]:
                fe = fe.T
            frame_emb = fe.astype(np.float32, copy=False)


            # for each segment, average the frames that fall inside it
            # note: OpenL3 frames are 1.0 s long; ts are either start times (center=False) or centers (center=True). 
            frame_len = 1.0
            for (s, e) in bounds:
                if self.center:
                    # frame covers [c-0.5, c+0.5]; include frames fully inside [s, e]
                    mask = (ts >= (s + 0.5)) & (ts <= (e - 0.5))
                else:
                    # frame covers [t, t+1.0]; include frames fully inside [s, e]
                    mask = (ts >= s) & (ts <= (e - frame_len))
                if not np.any(mask):
                    continue
                v = frame_emb[mask].mean(axis=0).astype(np.float32)
                if self.normalize:
                    v = _l2_normalize(v)
                vecs.append(v)
                rows.append({"file": f, "start_s": s, "end_s": e,
                            "model": self.model_name, "dim": int(v.shape[-1])})

        if not vecs:
            return np.zeros((0, self.output_dim()), dtype=np.float32), []

        X = np.vstack(vecs).astype(np.float32, copy=False)
        return X, rows


    def extract_segments_and_songs(
        self,
        file_list: List[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[SongEmbedding]]:
        """Return segment embeddings with metadata and pooled song-level vectors."""
        X, rows = self.extract_features_with_metadata(
            file_list, progress_callback=progress_callback
        )
        songs = _compute_song_embeddings(X, rows, normalize=self.normalize)
        return X, rows, songs

# ----------------------------
# OpenL3
# ----------------------------
class TorchOpenL3FeatureExtractor(AudioFeatureExtractorBase):
    """
    PyTorch OpenL3 via the official 'torchopenl3' package.
    - Single pass per file to get all 1 s frame embeddings + timestamps.
    - Pools frames inside each [start, end] segment.
    - Supports intermediate layers via forward hooks:
        layer=None            -> final OpenL3 embedding
        layer='penultimate'   -> input to the last nn.Linear (prelogits)
        layer=<int>           -> output of the leaf module at that index (supports negatives like -4)
    """
    def __init__(
        self,
        input_repr: str = "mel256",
        content_type: str = "music",
        embedding_size: int = 512,
        hop_size: float = 0.5,
        center: bool = False,
        batch_size: int = 64,
        layer: str | int | None = None,  # None | 'penultimate' | int index (e.g., -4)
        **kwargs,
    ):
        self.input_repr = str(input_repr)
        self.content_type = str(content_type)
        self.embedding_size = int(embedding_size)
        self.hop_size = float(hop_size)
        self.center = bool(center)
        self.batch_size = int(batch_size)
        self.layer_spec = layer
        super().__init__(**kwargs)

    # ------------- model init & wrapping -------------
    def _init_model(self) -> None:
        try:
            import torchopenl3 as tol3
        except Exception as e:
            raise ImportError(
                "torchopenl3 not available. Install with `pip install torchopenl3`."
            ) from e

        self._tol3 = tol3
        base = tol3.models.load_audio_embedding_model(
            input_repr=self.input_repr,
            embedding_size=self.embedding_size,
            content_type=self.content_type,
        )  # returns nn.Module
        base.to(self.device).eval()

        # Wrap for intermediate layers if requested
        self.model, self._output_dim, self._layer_name = self._make_layer_wrapper(base, self.layer_spec)

        self.model_name = f"torchopenl3_{self.content_type}_{self.input_repr}_{self.embedding_size}" \
                          + (f"@{self._layer_name}" if self.layer_spec is not None else "")

    def output_dim(self) -> int:
        return int(self._output_dim)

    # ------------- wrappers for intermediate outputs -------------
    @staticmethod
    def _flatten_leaf_modules(m: nn.Module):
        """Return list of (qualified_name, module) for leaf modules in forward order."""
        leaves = []
        for name, mod in m.named_modules():
            if len(list(mod.children())) == 0:
                leaves.append((name, mod))
        return leaves

    def _wrap_return_prelogits(self, base: nn.Module) -> tuple[nn.Module, int, str]:
        last_linear = None
        for _, mod in base.named_modules():
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is None:
            raise RuntimeError("Could not find final nn.Linear in OpenL3 model (needed for 'penultimate').")

        class _PreLogitsWrapper(nn.Module):
            def __init__(self, inner: nn.Module, target: nn.Module):
                super().__init__()
                self.inner = inner
                self.target = target
                self._buf = None
                def hook(mod, inp, out):
                    # capture the *input* to the last linear
                    self._buf = inp[0]
                self._hk = target.register_forward_hook(hook)

            def forward(self, x):
                _ = self.inner(x)
                y = self._buf
                # y expected to be (B, Din)
                return y

        wrapped = _PreLogitsWrapper(base, last_linear)
        # infer output dim with a tiny dummy pass through torchopenl3 frontend: handled later after first call if needed
        # here we pessimistically return embedding_size as placeholder; will update on first real run.
        return wrapped, -1, "prelogits"

    def _wrap_return_module(self, base: nn.Module, module: nn.Module, name: str) -> tuple[nn.Module, int, str]:
        class _ReturnModuleWrapper(nn.Module):
            def __init__(self, inner: nn.Module, target: nn.Module):
                super().__init__()
                self.inner = inner
                self.target = target
                self._buf = None
                def hook(mod, inp, out):
                    self._buf = out
                self._hk = target.register_forward_hook(hook)

            def forward(self, x):
                _ = self.inner(x)
                y = self._buf
                # If feature map (B,C,T) or (B,C,H,W): global-average pool spatial/time dims.
                if y.dim() > 2:
                    for _ in range(y.dim() - 2):
                        y = y.mean(dim=-1)
                return y

        return _ReturnModuleWrapper(base, module), -1, name

    def _make_layer_wrapper(self, base: nn.Module, layer_spec: str | int | None):
        if layer_spec is None:
            # final embedding
            return base, int(self.embedding_size), "embedding"

        if isinstance(layer_spec, str):
            if layer_spec.lower() in {"penultimate", "prelogits", "before_dense"}:
                return self._wrap_return_prelogits(base)
            raise ValueError("Unsupported string for layer: use 'penultimate'/'prelogits' or an integer index (e.g. -4).")

        # integer index: pick leaf module [-1 is last]
        leaves = self._flatten_leaf_modules(base)
        if not leaves:
            raise RuntimeError("OpenL3 model has no leaf modules?")

        idx = layer_spec if layer_spec >= 0 else len(leaves) + layer_spec
        if idx < 0 or idx >= len(leaves):
            raise IndexError(f"layer index {layer_spec} out of range for {len(leaves)} leaf modules.")

        name, mod = leaves[idx]
        return self._wrap_return_module(base, mod, name)

    # ------------- embedding of a whole file into 1 s frames -------------
    def _embed_file_frames(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          frame_emb: (T, D) float32
          ts:        (T,)  float32 timestamps (seconds). These correspond to 1 s windows at the configured hop.
        """
        # get_audio_embedding returns (embedding, timestamps)
        # embedding shape is commonly (B, D, T) for torch tensors; also supports numpy input.
        emb, ts = self._tol3.get_audio_embedding(
            audio=y,
            sr=sr,
            model=self.model,
            center=self.center,
            hop_size=self.hop_size,
            batch_size=self.batch_size,
            # input_repr/content_type are already baked into `model`, but passing them is harmless
            input_repr=self.input_repr,
            content_type=self.content_type,
        )

        # Convert to numpy float32 and squeeze batch dimension if present
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        emb = np.asarray(emb)
        # Expected shapes from torchopenl3: (T, D) or (1, D, T) or (1, T, D)
        if emb.ndim == 3:
            if emb.shape[0] == 1 and emb.shape[1] == self.embedding_size:
                # (1, D, T) -> (T, D)
                emb = np.transpose(emb[0], (1, 0))
            elif emb.shape[0] == 1:
                # (1, T, D) -> (T, D)
                emb = emb[0]
            else:
                # (B, T, D) -> concatenate along batch
                emb = emb.reshape(-1, emb.shape[-1])
        elif emb.ndim == 2:
            # (T, D) ok
            pass
        else:
            raise RuntimeError(f"Unexpected embedding shape from torchopenl3: {emb.shape}")

        if isinstance(ts, torch.Tensor):
            ts = ts.detach().cpu().numpy()
        ts = np.asarray(ts, dtype=np.float32)

        # If this is the *first* run with a wrapper, discover dimension
        if getattr(self, "_output_dim", -1) == -1:
            self._output_dim = int(emb.shape[-1])

        return emb.astype(np.float32, copy=False), ts.astype(np.float32, copy=False)

    # ------------- main extraction: single pass per file + segment pooling -------------
    def extract_features_with_metadata(
        self,
        file_list: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        vecs: list[np.ndarray] = []

        total = len(file_list)

        for idx, f in enumerate(file_list, start=1):
            if progress_callback:
                try:
                    progress_callback(idx, total, f)
                except Exception:
                    logger.exception("Progress callback failed for %s", f)
            # duration
            try:
                info = sf.info(f)
                total_seconds = float(info.frames) / float(info.samplerate)
            except Exception:
                y_tmp, sr_tmp = load_audio_mono(f, target_sr=self.target_sr)
                total_seconds = len(y_tmp) / float(sr_tmp)

            bounds = segment_bounds(total_seconds, self.segment_seconds, self.hop_seconds)
            if not bounds:
                continue

            # decode whole file once at target_sr (48 kHz by default)
            y, sr = load_audio_mono(f, target_sr=self.target_sr)
            if y.size == 0:
                continue

            # single torchopenl3 pass over the whole file to get frame-wise embeddings
            frame_emb, ts = self._embed_file_frames(y, sr)  # (T, D), (T,)
            ts = np.asarray(ts, dtype=np.float32).reshape(-1)    # (T,)
            fe = np.asarray(frame_emb)
            if fe.ndim == 3 and fe.shape[0] == 1:                # (1,*,*) -> (*,*)
                fe = fe[0]
            if fe.ndim == 2 and fe.shape[0] != ts.shape[0] and fe.shape[1] == ts.shape[0]:
                fe = fe.T                                        # (T,D)
            frame_emb = fe.astype(np.float32, copy=False)        # ensure (T,D)

            frame_len = 1.0  # seconds; OpenL3 uses 1 s windows
            eps = 1e-7       # guard for float rounding

            for (s, e) in bounds:
                # frames fully inside [s, e]
                if self.center:
                    start_cut = s + 0.5
                    end_cut   = e - 0.5
                else:
                    start_cut = s
                    end_cut   = e - frame_len
                if end_cut <= start_cut + eps:
                    continue

                # ts is sorted ascending
                i0 = int(np.searchsorted(ts, start_cut - eps, side="left"))
                i1 = int(np.searchsorted(ts, end_cut   + eps, side="right"))  # exclusive
                if i1 <= i0:
                    continue

                v = frame_emb[i0:i1].mean(axis=0).astype(np.float32)
                if self.normalize:
                    v = _l2_normalize(v)
                vecs.append(v)
                rows.append({"file": f, "start_s": s, "end_s": e,
                            "model": self.model_name, "dim": int(v.shape[-1])})

        if not vecs:
            return np.zeros((0, self.output_dim()), dtype=np.float32), []

        X = np.vstack(vecs).astype(np.float32, copy=False)
        return X, rows



# class OpenL3FeatureExtractor(AudioFeatureExtractorBase):
#     """
#     Optional: requires openl3 (works well on Python 3.11).
#     pip install openl3
#     """

#     def __init__(
#         self,
#         input_repr: str = "mel256",
#         content_type: str = "music",
#         embedding_size: int = 512,
#         center: bool = False,
#         hop_size: float = 0.5,
#         **kwargs,
#     ):
#         self.input_repr = input_repr
#         self.content_type = content_type
#         self.embedding_size = int(embedding_size)
#         self.center = center
#         self.hop_size = float(hop_size)
#         super().__init__(**kwargs)

#     def _init_model(self) -> None:
#         try:
#             import openl3  # lazy import
#             self._openl3 = openl3
#             self.model_name = f"openl3_{self.content_type}_{self.input_repr}_{self.embedding_size}"
#         except Exception as e:
#             raise ImportError(
#                 "OpenL3 not available. Install with `pip install openl3` (Python 3.11 recommended)."
#             ) from e

#     def _embed_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
#         emb, _ = self._openl3.get_audio_embedding(
#             y,
#             sr,
#             input_repr=self.input_repr,
#             content_type=self.content_type,
#             embedding_size=self.embedding_size,
#             center=self.center,
#             hop_size=self.hop_size,
#         )
#         return np.asarray(emb.mean(axis=0), dtype=np.float32, order="C")

#     def output_dim(self) -> int:
#         return int(self.embedding_size)

class OpenL3FeatureExtractor(AudioFeatureExtractorBase):
    """
    Optional: requires openl3 (works well on Python 3.11).
    pip install openl3 tensorflow
    """

    def __init__(
        self,
        input_repr: str = "mel256",
        content_type: str = "music",
        embedding_size: int = 512,
        center: bool = False,
        hop_size: float = 0.5,
        layer: str | int | None = None,
        frontend: str = "kapre",
        batch_size: int = 128,           
        **kwargs,
    ):
        self.input_repr = input_repr
        self.content_type = content_type
        self.embedding_size = int(embedding_size)
        self.center = center
        self.hop_size = float(hop_size)
        self.layer_spec = layer
        self.frontend = str(frontend)
        self.batch_size = int(batch_size)
        super().__init__(**kwargs)

    def _init_model(self) -> None:
        
        self._openl3 = openl3
        base = openl3.models.load_audio_embedding_model(
            input_repr=self.input_repr,
            content_type=self.content_type,
            embedding_size=self.embedding_size,
            frontend=self.frontend,  # allow 'kapre' (GPU) or 'librosa' (CPU)
        )


        # Pick layer
        out_tensor = base.output
        layer_name = "embedding"  # nice default label

        if self.layer_spec is not None:
            if isinstance(self.layer_spec, int):
                chosen = base.layers[self.layer_spec]
                out_tensor = chosen.output
                layer_name = chosen.name
            elif isinstance(self.layer_spec, str):
                # convenience aliases
                if self.layer_spec.lower() in {"embedding", "final", "last"}:
                    out_tensor = base.output
                    layer_name = "embedding"
                elif self.layer_spec.lower() in {"penultimate", "prelogits", "before_dense"}:
                    # common case: second-to-last layer
                    chosen = base.layers[-2]
                    out_tensor = chosen.output
                    layer_name = chosen.name
                else:
                    chosen = base.get_layer(self.layer_spec)
                    out_tensor = chosen.output
                    layer_name = chosen.name
            else:
                raise ValueError("layer must be None, int (index), or str (name)")

            # If this is a feature map, pool spatial/time dims so we return a vector per frame
            rank = len(out_tensor.shape)
            if rank == 4:
                out_tensor = tf.keras.layers.GlobalAveragePooling2D(name="l3_gap2d")(out_tensor)
            elif rank == 3:
                out_tensor = tf.keras.layers.GlobalAveragePooling1D(name="l3_gap1d")(out_tensor)
            elif rank == 2:
                pass  # already (batch, D)
            else:
                raise RuntimeError(f"Unexpected tensor rank {rank} for layer '{layer_name}'")

        # Build the submodel we’ll hand to openl3.get_audio_embedding
        
        self.submodel = Model(inputs=base.input, outputs=out_tensor,
                            name=f"openl3_sub_{layer_name}")
        self._output_dim = int(self.submodel.output_shape[-1])
        self.model_name = f"openl3_{self.content_type}_{self.input_repr}_{self.embedding_size}" \
                        + (f"@{layer_name}" if self.layer_spec is not None else "")


    def _embed_file_frames(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        # returns (frames_TxD, ts_T)
        emb, ts = self._openl3.get_audio_embedding(
            audio=y, sr=sr, model=self.submodel,
            center=self.center, hop_size=self.hop_size,
            batch_size=self.batch_size,
            frontend=self.frontend,   # honored when model has no kapre frontend
            verbose=False
        )
        # emb: (T, D), ts: (T,)
        return np.asarray(emb, np.float32, order="C"), np.asarray(ts, np.float32)


    def _embed_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Use openl3’s framing, but with our submodel to fetch the desired layer
        emb, _ts = self._openl3.get_audio_embedding(
            y,
            sr,
            model=self.submodel,  # <-- custom layer output
            center=self.center,
            hop_size=self.hop_size,
        )
        # mean over frames -> one vector for the segment
        return np.asarray(emb.mean(axis=0), dtype=np.float32, order="C")

    def output_dim(self) -> int:
        return self._output_dim


# Multi-model wrapper
# ----------------------------

class MultiModelAudioExtractor:
    """
    Runs multiple models per segment.
    combine="concat" -> one row per segment with concatenated vectors.
    combine="separate" -> one row per model per segment.
    """

    def __init__(
        self,
        extractors: List[AudioFeatureExtractorBase],
        combine: str = "concat",
        segment_seconds: float = 30.0,
        hop_seconds: Optional[float] = None,
        ref_sr: int = 48000,
        normalize: bool = True,   
    ):
        assert combine in ("concat", "separate")
        if not extractors:
            raise ValueError("At least one extractor required")
        self.extractors = extractors
        self.combine = combine
        self.segment_seconds = float(segment_seconds)
        self.hop_seconds = hop_seconds if hop_seconds is not None else float(segment_seconds) / 2.0
        self.ref_sr = int(ref_sr)
        self.normalize = normalize

    def extract_features_with_metadata(
        self,
        file_list: List[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        vecs: List[np.ndarray] = []

        total = len(file_list)

        for idx, f in enumerate(file_list, start=1):
            if progress_callback:
                try:
                    progress_callback(idx, total, f)
                except Exception:
                    logger.exception("Progress callback failed for %s", f)
            # get duration
            try:
                info = sf.info(f)
                total_seconds = float(info.frames) / float(info.samplerate)
            except Exception:
                y_tmp, sr_tmp = load_audio_mono(f, target_sr=self.target_sr)
                total_seconds = len(y_tmp) / float(sr_tmp)

            # compute segment bounds
            bounds = segment_bounds(total_seconds, self.segment_seconds, self.hop_seconds)
            if not bounds:
                continue

            # decode once at extractor SR
            y, sr = load_audio_mono(f, target_sr=self.target_sr)
            if y.size == 0:
                continue

            # single OpenL3 pass over the whole file
            frame_emb, ts = self._embed_file_frames(y, sr)  # (T, D), (T,)

            # for each segment, average the frames that fall inside it
            # note: OpenL3 frames are 1.0 s long; ts are either start times (center=False) or centers (center=True). 
            frame_len = 1.0
            for (s, e) in bounds:
                if self.center:
                    # frame covers [c-0.5, c+0.5]; include frames fully inside [s, e]
                    mask = (ts >= (s + 0.5)) & (ts <= (e - 0.5))
                else:
                    # frame covers [t, t+1.0]; include frames fully inside [s, e]
                    mask = (ts >= s) & (ts <= (e - frame_len))
                if not np.any(mask):
                    continue
                print('1',frame_emb.shape)
                v = frame_emb[mask].mean(axis=0).astype(np.float32)
                if self.normalize:
                    v = _l2_normalize(v)
                vecs.append(v)
                rows.append({"file": f, "start_s": s, "end_s": e,
                            "model": self.model_name, "dim": int(v.shape[-1])})

        if not vecs:
            return np.zeros((0, self.output_dim()), dtype=np.float32), []

        X = np.vstack(vecs).astype(np.float32, copy=False)
        return X, rows


    def extract_segments_and_songs(
        self,
        file_list: List[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[SongEmbedding]]:
        """Return segment embeddings with metadata and pooled song-level vectors."""
        X, rows = self.extract_features_with_metadata(
            file_list, progress_callback=progress_callback
        )
        songs = _compute_song_embeddings(X, rows, normalize=self.normalize)
        return X, rows, songs


    def extract_feature_arrays(self, file_list: List[str]) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of (X_model0, X_model1, ...), in the same order as self.extractors.
        Each X_modeli is float32 with shape (N_segments_total, D_i).
        No metadata, no padding, no mixing.
        """
        per_vecs: List[List[np.ndarray]] = [[] for _ in self.extractors]

        for f in file_list:
            # duration
            try:
                info = sf.info(f)
                total_seconds = float(info.frames) / float(info.samplerate)
            except Exception:
                y_tmp, sr_tmp = load_audio_mono(f, target_sr=self.ref_sr)
                total_seconds = len(y_tmp) / float(sr_tmp)

            bounds = segment_bounds(total_seconds, self.segment_seconds, self.hop_seconds)
            if not bounds:
                continue

            # decode once per segment at reference SR and filter empties/very short
            seg_cache: List[Tuple[np.ndarray, int]] = []
            for (s, e) in bounds:
                y_ref, sr_ref = load_audio_mono(f, target_sr=self.ref_sr, offset=s, duration=(e - s))
                if y_ref.size == 0 or len(y_ref) < int(0.05 * self.ref_sr):
                    continue
                seg_cache.append((y_ref, sr_ref))

            # run all extractors for each kept segment
            for y_ref, sr_ref in seg_cache:
                for j, ext in enumerate(self.extractors):
                    if ext.target_sr != sr_ref:
                        y = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=ext.target_sr)
                        y = np.asarray(y, dtype=np.float32, order="C")
                        sr = ext.target_sr
                    else:
                        y, sr = y_ref, sr_ref
                    if y.size == 0:
                        continue
                    v = ext._embed_waveform(y, sr)
                    v = np.asarray(v, dtype=np.float32)
                    if v.ndim != 1:
                        v = v.reshape(-1)
                    per_vecs[j].append(v)

        # stack per model; if none for a model, return (0, D_i)
        arrays: List[np.ndarray] = []
        for j, ext in enumerate(self.extractors):
            if per_vecs[j]:
                X = np.vstack(per_vecs[j]).astype(np.float32, copy=False)
            else:
                X = np.zeros((0, ext.output_dim()), dtype=np.float32)
            arrays.append(X)

        return tuple(arrays)


def create_default_openl3_extractor(
    *,
    segment_seconds: float = 30.0,
    hop_seconds: float | None = None,
    target_sr: int = 48_000,
    device: str | None = None,
) -> TorchOpenL3FeatureExtractor:
    return TorchOpenL3FeatureExtractor(
        input_repr="mel256",
        content_type="music",
        embedding_size=512,
        # same framing as before
        hop_size=0.5,
        center=False,
        batch_size=256,
        # --- this is your TF choice; negative leaf index in PyTorch variant ---
        layer=-4,  # or try 'penultimate' for a robust prelogits feature
        segment_seconds=segment_seconds,
        hop_seconds=hop_seconds,
        target_sr=target_sr,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

# ----------------------------
# Factory: build available models (OpenL3 optional)
# ----------------------------
def create_default_openl3_extractor_torch(
    *,
    segment_seconds: float = 30.0,
    hop_seconds: float | None = None,
    target_sr: int = 48_000,
    device: str | None = None,
) -> TorchOpenL3FeatureExtractor:
    return TorchOpenL3FeatureExtractor(
        input_repr="mel256",
        content_type="music",
        embedding_size=512,
        hop_size=0.5,
        segment_seconds=segment_seconds,
        hop_seconds=hop_seconds,
        target_sr=target_sr,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )


def build_windows_extractors(
    segment_seconds: float = 30.0,
    hop_seconds: float | None = None,
    target_sr: int = 48_000,
    device: str | None = None,
    prefer: str = "torch",   # "torch" | "tf"
    **_: Any,
):
    exts: List[AudioFeatureExtractorBase] = []
    if prefer == "torch":
        try:
            exts.append(
                create_default_openl3_extractor_torch(
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    target_sr=target_sr,
                    device=device,
                )
            )
            return exts
        except ImportError as e:
            warnings.warn(f"{e} Falling back to TensorFlow openl3.")
    # TF fallback (your existing function)
    try:
        exts.append(
            create_default_openl3_extractor(
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                target_sr=target_sr,
                device=device or "cpu",
            )
        )
    except ImportError as e:
        warnings.warn(str(e))
    return exts


