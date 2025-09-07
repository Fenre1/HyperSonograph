from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class SegmentLevel:
    """Per-song segment features.

    Attributes
    ----------
    embeddings:
        Array of shape ``(N_segments, D)`` containing segment embedding vectors
        for a single model. The dtype should be ``float32`` or ``float16``.
    song_id:
        Integer array of shape ``(N_segments,)`` mapping each segment to its
        parent song.
    start_s:
        Float array of shape ``(N_segments,)`` with segment start times in
        seconds.
    end_s:
        Float array of shape ``(N_segments,)`` with segment end times in
        seconds.
    """

    embeddings: np.ndarray
    song_id: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray


@dataclass
class SongLevel:
    """Song level aggregated features."""

    centroid: np.ndarray
    stats2D: np.ndarray
    song_id: np.ndarray
    path: Sequence[str]


@dataclass
class ModelFeatures:
    """Container storing audio features for a single model.

    The schema groups segment-level arrays with their corresponding song-level
    aggregations. The arrays are structured so that future utilities can:

    * retrieve all segments for a song via ``segments.song_id``,
    * compare a segment vector to the song centroid via ``songs.centroid``, and
    * compare songs via ``songs.stats2D``.
    """

    name: str
    segments: SegmentLevel
    songs: SongLevel

    def get_song_segments(self, song: int) -> np.ndarray:
        """Return embeddings for all segments belonging to ``song``."""
        mask = self.segments.song_id == song
        return self.segments.embeddings[mask]

    def compare_segment_to_song(self, seg_vec: np.ndarray, song: int) -> float:
        """Compare ``seg_vec`` to the centroid of ``song``.

        Parameters
        ----------
        seg_vec:
            The segment embedding vector to compare.
        song:
            Integer identifier of the target song.
        Returns
        -------
        float
            Cosine similarity between the segment vector and the song's centroid.
        """
        idx = np.where(self.songs.song_id == song)[0]
        if idx.size == 0:
            raise KeyError(f"Song id {song} not found")
        centroid = self.songs.centroid[idx[0]]
        seg = np.asarray(seg_vec, dtype=np.float32)
        seg /= np.linalg.norm(seg) + 1e-8
        cen = centroid.astype(np.float32)
        cen /= np.linalg.norm(cen) + 1e-8
        return float(np.dot(seg, cen))

    def compare_songs(self, song_a: int, song_b: int) -> float:
        """Compare two songs using their ``stats2D`` representations."""
        idx_a = np.where(self.songs.song_id == song_a)[0]
        idx_b = np.where(self.songs.song_id == song_b)[0]
        if idx_a.size == 0 or idx_b.size == 0:
            raise KeyError("Song id not found")
        vec_a = self.songs.stats2D[idx_a[0]].astype(np.float32)
        vec_b = self.songs.stats2D[idx_b[0]].astype(np.float32)
        vec_a /= np.linalg.norm(vec_a) + 1e-8
        vec_b /= np.linalg.norm(vec_b) + 1e-8
        return float(np.dot(vec_a, vec_b))
        