# audio_features.py
# %%
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import numpy as np
import torch
import librosa
import soundfile as sf
import warnings
import tensorflow
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
    
    def output_dim(self) -> int: ...

    def extract_features_with_metadata(
        self, file_list: List[str]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        vecs: List[np.ndarray] = []

        for f in file_list:
            try:
                info = sf.info(f)
                total_seconds = float(info.frames) / float(info.samplerate)
            except Exception:
                y_tmp, sr_tmp = load_audio_mono(f, target_sr=self.target_sr)
                total_seconds = len(y_tmp) / float(sr_tmp)

            for (s, e) in segment_bounds(total_seconds, self.segment_seconds, self.hop_seconds):
                y, sr = load_audio_mono(f, target_sr=self.target_sr, offset=s, duration=(e - s))
                if y.size == 0:
                    continue  # skip empty tails safely
                v = self._embed_waveform(y, sr)
                v = np.asarray(v, dtype=np.float32)
                if v.ndim != 1:
                    v = v.reshape(-1)  # force 1D
                if self.normalize:
                    v = _l2_normalize(v)        
                vecs.append(v)
                rows.append({"file": f, "start_s": s, "end_s": e,
                            "model": self.model_name, "dim": int(v.shape[-1])})

        if not vecs:
            return np.zeros((0, self.output_dim()), dtype=np.float32), []

        X = np.vstack(vecs).astype(np.float32, copy=False)  # (N, D)
        return X, rows

    def extract_segments_and_songs(
        self, file_list: List[str]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[SongEmbedding]]:
        """Return segment embeddings with metadata and pooled song-level vectors."""
        X, rows = self.extract_features_with_metadata(file_list)
        songs = _compute_song_embeddings(X, rows, normalize=self.normalize)
        return X, rows, songs

# ----------------------------
# OpenL3
# ----------------------------

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
        # NEW:
        layer: str | int | None = None,   # None -> default embedding output
        **kwargs,
    ):
        self.input_repr = input_repr
        self.content_type = content_type
        self.embedding_size = int(embedding_size)
        self.center = center
        self.hop_size = float(hop_size)
        self.layer_spec = layer
        super().__init__(**kwargs)

    def _init_model(self) -> None:
        import tensorflow as tf
        import openl3

        self._openl3 = openl3
        # Load the standard OpenL3 embedding model
        base = openl3.models.load_audio_embedding_model(
            input_repr=self.input_repr,
            content_type=self.content_type,
            embedding_size=self.embedding_size,
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
        from tensorflow.keras import Model
        self.submodel = Model(inputs=base.input, outputs=out_tensor, name=f"openl3_sub_{layer_name}")
        self._output_dim = int(self.submodel.output_shape[-1])

        self.model_name = f"openl3_{self.content_type}_{self.input_repr}_{self.embedding_size}" \
                          + (f"@{layer_name}" if self.layer_spec is not None else "")

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


# ----------------------------
# CLAP (LAION-CLAP)
# ----------------------------

class CLAPFeatureExtractor(AudioFeatureExtractorBase):
    """
    pip install laion-clap
    """

    def __init__(self, clap_model: str = "HTSAT-large", **kwargs):
        self.clap_model = clap_model
        super().__init__(**kwargs)

    def _init_model(self) -> None:
        import laion_clap
        self._clap = laion_clap
        self.model = self._clap.CLAP_Module(enable_fusion=False, amodel=self.clap_model, device=str(self.device))
        self.model.eval()
        self.model_name = f"clap_{self.clap_model}"
        self.target_sr = 48000  

    def _embed_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Ensure float32, contiguous
        y = np.asarray(y, dtype=np.float32, order="C")

        with torch.no_grad():
            try:
                # Newer laion-clap versions
                emb_list = self.model.get_audio_embedding_from_data(x=[y], sr=sr, use_tensor=False)
            except TypeError:
                # Older laion-clap: no `sr` kw. It expects 48kHz audio.
                if sr != 48000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=48000)
                    y = np.asarray(y, dtype=np.float32, order="C")
                emb_list = self.model.get_audio_embedding_from_data(x=[y], use_tensor=False)

        return np.asarray(emb_list[0], dtype=np.float32, order="C")

    def output_dim(self) -> int:
        # CLAP projection dim is typically 512 for HTSAT variants
        return 512


# ----------------------------
# MERT (HF Transformers)
# ----------------------------

class MERTFeatureExtractor(AudioFeatureExtractorBase):
    """
    MERT from Hugging Face. Needs nnAudio for CQT features.
    pip install transformers torchaudio nnAudio
    """

    def __init__(self, hf_model: str = "m-a-p/MERT-v1-95M", layer: int = -1, **kwargs):
        self.hf_model = hf_model
        self.layer = layer  # respect this in base _embed_waveform
        super().__init__(**kwargs)

    def _init_model(self) -> None:
        from transformers import AutoProcessor, AutoModel
        self.model_name = f"mert_{Path(self.hf_model).name}"
        try:
            self.processor = AutoProcessor.from_pretrained(self.hf_model, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.hf_model, trust_remote_code=True, use_safetensors=True
            ).to(self.device).eval()
            # Align to the processor's expected sampling rate (e.g., 24000)
            sr = getattr(self.processor, "sampling_rate", None)
            if sr is None:
                fe = getattr(self.processor, "feature_extractor", None)
                sr = getattr(fe, "sampling_rate", 24000)
            self.target_sr = int(sr)  # <-- add these lines
        except Exception as e:
            raise RuntimeError(
                "MERT load failed. Tips: `pip install nnAudio`; ensure internet "
                f"access for first run; consider device='cpu'. Original error: {e}"
            ) from e

    def output_dim(self) -> int:
        try:
            return int(getattr(self.model.config, "hidden_size"))  # type: ignore[attr-defined]
        except Exception:
            # Fallback for known variant
            return 768  # 95M variant

###############################################################
########## HFAudioClassifierFeatureExtractor ###########
############################################################
class HFAudioClassifierFeatureExtractor(AudioFeatureExtractorBase):
    """
    Wrap an HF audio classification checkpoint (e.g., Wav2Vec2/AST) and expose
    hidden-state embeddings with simple time pooling.
    Examples:
      - "dima806/music_genres_classification" (Wav2Vec2 genre; 16 kHz)
      - "MIT/ast-finetuned-audioset-10-10-0.4593" (AST; feature extractor handles spectrogram)
    """

    def __init__(
        self,
        hf_model: str,
        layer: int = -1,                 # which hidden layer to use; -1 = last
        target_sr: Optional[int] = None, # if None, inferred from feature extractor (for Wav2Vec2 often 16000)
        **kwargs,
    ):
        self.hf_model = hf_model
        self.layer = layer
        self._explicit_sr = target_sr
        super().__init__(**kwargs)

    def _init_model(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.model_name = f"hf_audio_cls_{Path(self.hf_model).name}"
        # Feature extractor prepares raw waveforms / spectrograms per checkpoint
        self.processor = AutoFeatureExtractor.from_pretrained(self.hf_model)
        self.model = AutoModelForAudioClassification.from_pretrained(self.hf_model).to(self.device).eval()

        # Decide sampling rate
        sr = self._explicit_sr
        if sr is None:
            # Most HF audio feature extractors expose a sampling_rate field
            sr = getattr(self.processor, "sampling_rate", None)
        if sr is None:
            # Fallback defaults: Wav2Vec2 commonly 16k; AST uses its own featurization but 16k works
            sr = 48_000
        self.target_sr = int(sr)

    def _embed_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Standard HF audio path: processor returns inputs for model
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Pick a hidden layer (shape: [B, T, D] or sometimes [B, D])
        hs = out.hidden_states[self.layer] if hasattr(out, "hidden_states") and out.hidden_states is not None else out.logits
        if hs.dim() == 3:
            pooled = hs.mean(dim=1)     # [B, D]
        elif hs.dim() == 2:
            pooled = hs                  # already [B, D]
        else:
            raise RuntimeError(f"Unexpected hidden_states shape: {tuple(hs.shape)}")
        return pooled.squeeze(0).float().cpu().numpy()

    def output_dim(self) -> int:
        try:
            return int(getattr(self.model.config, "hidden_size"))
        except Exception:
            # Fall back to classifier head width if needed
            return int(self.model.classifier.in_features)  # type: ignore[attr-defined]


# ----------------------------
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

    def extract_features_with_metadata(self, file_list: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        all_rows: List[Dict[str, Any]] = []
        all_vecs: List[np.ndarray] = []

        for f in file_list:
            try:
                info = sf.info(f)
                total_seconds = float(info.frames) / float(info.samplerate)
            except Exception:
                y_tmp, sr_tmp = load_audio_mono(f, target_sr=self.ref_sr)
                total_seconds = len(y_tmp) / float(sr_tmp)

            bounds = segment_bounds(total_seconds, self.segment_seconds, self.hop_seconds)
            if not bounds:
                continue

            # Decode once per bound at reference SR
            seg_cache: List[Tuple[np.ndarray, int]] = []
            filtered_bounds: List[Tuple[float, float]] = []
            for (s, e) in bounds:
                y_ref, sr_ref = load_audio_mono(f, target_sr=self.ref_sr, offset=s, duration=(e - s))
                if y_ref.size == 0 or len(y_ref) < int(0.05 * self.ref_sr):  # skip empty/ultra-short
                    continue
                seg_cache.append((y_ref, sr_ref))
                filtered_bounds.append((s, e))

            for i, (s, e) in enumerate(filtered_bounds):
                y_ref, sr_ref = seg_cache[i]
                if self.combine == "concat":
                    parts: List[np.ndarray] = []

                for ext in self.extractors:
                    # resample per extractor SR
                    if ext.target_sr != sr_ref:
                        y = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=ext.target_sr)
                        y = np.asarray(y, dtype=np.float32, order="C")
                        sr = ext.target_sr
                    else:
                        y, sr = y_ref, sr_ref
                    if y.size == 0:
                        # skip this segment entirely
                        parts = None  # type: ignore
                        break

                    v = ext._embed_waveform(y, sr)
                    v = np.asarray(v, dtype=np.float32)
                    if v.ndim != 1:
                        v = v.reshape(-1)
                    if self.normalize:
                        v = _l2_normalize(v)
                    if self.combine == "concat":
                        parts.append(v)  # type: ignore
                    else:
                        all_vecs.append(v)
                        all_rows.append({"file": f, "start_s": s, "end_s": e,
                                        "model": ext.model_name, "dim": int(v.shape[-1])})

                if self.combine == "concat":
                    if parts is None or len(parts) != len(self.extractors):
                        continue  # a sub-embed failed/was empty
                    vcat = np.concatenate(parts, axis=0)
                    if self.normalize:
                        vcat = _l2_normalize(vcat)
                    all_vecs.append(vcat)
                    model_tag = "+".join(ext.model_name for ext in self.extractors)
                    all_rows.append({"file": f, "start_s": s, "end_s": e,
                                    "model": model_tag, "dim": int(vcat.shape[-1])})

        # Always return a tuple
        if not all_vecs:
            if self.combine == "concat":
                total_dim = int(sum(ext.output_dim() for ext in self.extractors))
            else:
                # can't know per-model dim here reliably; return 0-width matrix
                total_dim = 0
            return np.zeros((0, total_dim), dtype=np.float32), []

        X = np.vstack(all_vecs).astype(np.float32, copy=False)  # (N, D)
        return X, all_rows

    def extract_segments_and_songs(self, file_list: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]], List[SongEmbedding]]:
        """Return segment embeddings with metadata and pooled song-level vectors."""
        X, rows = self.extract_features_with_metadata(file_list)
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

# ----------------------------
# Factory: build available models (OpenL3 optional)
# ----------------------------

def build_windows_extractors(
    segment_seconds: float = 30.0,
    hop_seconds: float | None = None,
    target_sr: int = 48_000,
    device: str | None = None,
    hf_genre_ckpt: str = "dima806/music_genres_classification",  # unused here; keep/remove as you like
    hf_genre_layer: int = -1,                                     # unused here
):
    exts: List[AudioFeatureExtractorBase] = []

    # OpenL3 (final embedding layer)
    try:
        exts.append(
            OpenL3FeatureExtractor(
                input_repr="mel256",
                content_type="music",
                embedding_size=512,
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                target_sr=target_sr,
                device=device or "cpu",
            )
        )
    except ImportError as e:
        warnings.warn(str(e))

    # OpenL3 (penultimate layer)
    try:
        exts.append(
            OpenL3FeatureExtractor(
                input_repr="mel256",
                content_type="music",
                embedding_size=512,
                layer=-3,               # <-- key difference
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                target_sr=target_sr,
                device=device or "cpu",
            )
        )
    except ImportError as e:
        warnings.warn(str(e))

    try:
        exts.append(
            OpenL3FeatureExtractor(
                input_repr="mel256",
                content_type="music",
                embedding_size=512,
                layer=-4,               # <-- key difference
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                target_sr=target_sr,
                device=device or "cpu",
            )
        )
    except ImportError as e:
        warnings.warn(str(e))

    return exts

# Usage example:
# files = ["song1.mp3", "song2.flac"]
# extractors = build_windows_extractors(segment_seconds=30.0, hop_seconds=15.0)
# multi = MultiModelAudioExtractor(extractors, combine="separate", segment_seconds=30.0, hop_seconds=15.0)
# vecs, meta = multi.extract_features_with_metadata(files)
# %%
