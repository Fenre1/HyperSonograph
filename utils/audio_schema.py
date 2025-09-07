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
        raise NotImplementedError

    def compare_segment_to_song(self, seg_vec: np.ndarray, song: int) -> float:
        """Compare ``seg_vec`` to the centroid of ``song``."""
        raise NotImplementedError

    def compare_songs(self, song_a: int, song_b: int) -> float:
        """Compare two songs using their ``stats2D`` representations."""
        raise NotImplementedError