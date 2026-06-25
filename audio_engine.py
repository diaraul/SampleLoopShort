"""
audio_engine.py
================
Core backend module for the Automated Audio-Chopper & Loop-Rearranger.

Responsibilities:
    - Load audio and build a precise rhythmic "beat grid" using librosa
      (BPM detection, beat timestamps in milliseconds, per-beat RMS energy).
    - Provide sample-accurate slicing of audio via pydub, mapped strictly
      to the millisecond beat grid (no rhythmic drift).
    - Apply short linear crossfades at every slice boundary -- including
      the loop's own wrap-point -- to eliminate clicks/pops.
    - Implement the VariationEngine: five distinct algorithmic strategies
      for generating catchy, seamless 4-beat loop configurations.

This module has no I/O-path or CLI concerns -- it is a pure processing
library. Entry-point behavior (where files come from, where they go)
lives in app.py.

Dependencies: librosa, pydub, numpy
"""

import random
import logging
from typing import List, Tuple, Optional

import numpy as np
import librosa
from pydub import AudioSegment

log = logging.getLogger("audio_engine")

# --------------------------------------------------------------------------
# Module-level configuration (tunable constants)
# --------------------------------------------------------------------------
CROSSFADE_MS = 4            # anti-pop crossfade length (spec range: 2-5ms)
BEATS_PER_LOOP = 4
SAMPLE_RATE = 44100          # librosa analysis sample rate


# --------------------------------------------------------------------------
# Custom Exceptions
# --------------------------------------------------------------------------
class AudioLoadError(Exception):
    """Raised when the source audio file cannot be loaded or analyzed."""
    pass


class BeatGridError(Exception):
    """Raised when the beat grid is unusable (too few beats, bad tempo, etc.)."""
    pass


class LoopExportError(Exception):
    """Raised when a generated loop cannot be assembled or exported."""
    pass


# --------------------------------------------------------------------------
# Core Data Structure: the Beat Grid
# --------------------------------------------------------------------------
class BeatGrid:
    """
    Analyzes a single audio file and exposes:
        - bpm                 : detected tempo
        - beat_times_ms        : list[int] beat onset timestamps in ms
        - beat_energy          : list[float] per-beat RMS energy
        - segment               : pydub.AudioSegment for sample-accurate cuts
        - n_beats               : number of beats detected

    All slicing is mapped strictly to beat_times_ms so cuts never drift
    from the actual rhythmic grid of the source material.
    """

    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.bpm: float = 0.0
        self.beat_times_ms: List[int] = []
        self.beat_energy: List[float] = []
        self.segment: Optional[AudioSegment] = None
        self.n_beats: int = 0

        self._analyze()

    def _analyze(self):
        # ---- 1. Load for librosa analysis ----
        try:
            y, sr = librosa.load(self.audio_path, sr=SAMPLE_RATE, mono=True)
        except FileNotFoundError:
            raise AudioLoadError(f"File not found: {self.audio_path}")
        except Exception as e:
            raise AudioLoadError(f"Librosa failed to load '{self.audio_path}': {e}")

        if y is None or len(y) == 0:
            raise AudioLoadError(f"Loaded audio is empty: {self.audio_path}")

        duration_sec = librosa.get_duration(y=y, sr=sr)
        if duration_sec < 2.0:
            raise BeatGridError(
                f"Audio too short ({duration_sec:.2f}s) for reliable beat tracking."
            )

        # ---- 2. Beat tracking ----
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        except Exception as e:
            raise BeatGridError(f"Beat tracking failed: {e}")

        # librosa can return tempo as a 0-d numpy array depending on version
        self.bpm = float(np.atleast_1d(tempo)[0])

        if self.bpm <= 0 or len(beat_frames) < BEATS_PER_LOOP + 1:
            raise BeatGridError(
                f"Insufficient beat grid (BPM={self.bpm:.1f}, "
                f"beats found={len(beat_frames)}). Need at least "
                f"{BEATS_PER_LOOP + 1} beats to build 4-beat loops safely."
            )

        beat_times_sec = librosa.frames_to_time(beat_frames, sr=sr)
        self.beat_times_ms = [int(round(t * 1000)) for t in beat_times_sec]
        self.n_beats = len(self.beat_times_ms)

        # ---- 3. Per-beat RMS energy (drives "high-energy slice" selection) ----
        end_ms = int(round(duration_sec * 1000))
        boundaries = self.beat_times_ms + [end_ms]

        hop_length = 512
        rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_times_ms = librosa.frames_to_time(
            np.arange(len(rms_frames)), sr=sr, hop_length=hop_length
        ) * 1000

        for i in range(self.n_beats):
            start, end = boundaries[i], boundaries[i + 1]
            mask = (rms_times_ms >= start) & (rms_times_ms < end)
            seg_rms = rms_frames[mask]
            energy = float(np.mean(seg_rms)) if len(seg_rms) > 0 else 0.0
            self.beat_energy.append(energy)

        # ---- 4. Load the actual audio into pydub for sample-accurate cutting ----
        try:
            self.segment = AudioSegment.from_file(self.audio_path)
        except Exception as e:
            raise AudioLoadError(f"pydub failed to load '{self.audio_path}': {e}")

        log.info(
            f"Beat grid built: {self.bpm:.1f} BPM, {self.n_beats} beats, "
            f"duration={duration_sec:.2f}s"
        )

    def beat_duration_ms(self, beat_index: int) -> int:
        """Duration of a given beat slice in ms (clamped to track length)."""
        start = self.beat_times_ms[beat_index]
        if beat_index + 1 < self.n_beats:
            end = self.beat_times_ms[beat_index + 1]
        else:
            end = len(self.segment)  # last beat runs to track end
        return max(end - start, 1)

    def slice_beat(self, beat_index: int) -> AudioSegment:
        """Extract one full beat as an AudioSegment, by ms boundary."""
        start = self.beat_times_ms[beat_index]
        end = start + self.beat_duration_ms(beat_index)
        end = min(end, len(self.segment))
        return self.segment[start:end]

    def slice_ms(self, start_ms: int, dur_ms: int) -> AudioSegment:
        """Extract an arbitrary ms-accurate slice (used for stutters/triplets)."""
        start_ms = max(0, min(start_ms, len(self.segment)))
        end_ms = max(start_ms, min(start_ms + dur_ms, len(self.segment)))
        return self.segment[start_ms:end_ms]

    def high_energy_indices(self, exclude: Optional[set] = None, top_n: int = 5) -> List[int]:
        """Returns beat indices sorted by descending RMS energy."""
        exclude = exclude or set()
        ranked = sorted(
            (i for i in range(self.n_beats) if i not in exclude),
            key=lambda i: self.beat_energy[i],
            reverse=True,
        )
        return ranked[:top_n]


# --------------------------------------------------------------------------
# Anti-pop stitching helpers
# --------------------------------------------------------------------------
def stitch_with_crossfade(slices: List[AudioSegment], crossfade_ms: int = CROSSFADE_MS) -> AudioSegment:
    """
    Concatenates a list of AudioSegments using pydub's built-in crossfade,
    and ALSO crossfades the loop's tail back into its own head, so the loop
    point itself (the most common source of clicks on repeat) is seamless.
    """
    if not slices:
        raise LoopExportError("No slices provided to stitch.")

    def safe_append(base: AudioSegment, nxt: AudioSegment) -> AudioSegment:
        # Crossfade can't exceed the shorter of two adjacent clips, or
        # pydub raises -- clamp per-junction.
        cf = min(crossfade_ms, len(base), len(nxt))
        cf = max(cf, 0)
        if cf < 2:
            return base + nxt
        return base.append(nxt, crossfade=cf)

    stitched = slices[0]
    for s in slices[1:]:
        stitched = safe_append(stitched, s)

    # ---- Loop-point seam fix ----
    # Crossfade the very end of the loop into the very start, so the wrap
    # point doesn't click when the loop repeats back-to-back.
    seam_cf = min(crossfade_ms, len(stitched) // 4) if len(stitched) > 8 else 0
    if seam_cf >= 2:
        head = stitched[:seam_cf]
        tail = stitched[:-seam_cf]
        stitched = tail.append(head, crossfade=seam_cf)

    return stitched


def fade_edges(segment: AudioSegment, ms: int = CROSSFADE_MS) -> AudioSegment:
    """Apply a short linear fade-in/out to a segment's outer boundaries."""
    ms = min(ms, len(segment) // 2) if len(segment) > 4 else 0
    if ms < 1:
        return segment
    return segment.fade_in(ms).fade_out(ms)


# --------------------------------------------------------------------------
# Variation Engine
# --------------------------------------------------------------------------
class VariationEngine:
    """
    Generates distinct 4-beat loop configurations from a BeatGrid using
    five different compositional strategies:

        1. Swapper            - keeps beats 1 & 3, swaps 2 & 4 with
                                  high-energy donor slices.
        2. Stutter/Syncopator  - subdivides one transient beat into a
                                  triplet bounce, capped by a clean beat.
        3. Textural Flip       - maps later-song energy onto the intro's
                                  rhythmic grid.
        4 & 5. Algorithmic Chaos - randomized, rhythmically-stable
                                  permutations of clean transient slices.
    """

    def __init__(self, grid: BeatGrid):
        self.grid = grid
        if grid.n_beats < BEATS_PER_LOOP + 1:
            raise BeatGridError(
                "Not enough beats in track to safely generate 4-beat loops "
                "with donor material outside the base grid."
            )
        # Reserve beats [0..3] as "the standard intro grid" baseline,
        # and treat everything else as donor material for swaps/textures.
        self.base_indices = list(range(min(BEATS_PER_LOOP, grid.n_beats)))
        self.donor_pool = [i for i in range(grid.n_beats) if i not in self.base_indices]

    # ---- Variation 1: The Swapper ----
    def variation_swapper(self) -> AudioSegment:
        """
        Keeps beats 1 & 3 (indices 0, 2) standard; swaps beats 2 & 4
        (indices 1, 3) with high-energy donor slices from elsewhere.
        """
        donors = self.grid.high_energy_indices(exclude=set(self.base_indices), top_n=10)
        if len(donors) < 2:
            donors = self.donor_pool[:2] if len(self.donor_pool) >= 2 else self.base_indices[:2]

        random.shuffle(donors)
        swap_beat2 = donors[0] if len(donors) > 0 else self.base_indices[1]
        swap_beat4 = donors[1] if len(donors) > 1 else self.base_indices[3]

        chosen = [
            self.base_indices[0],   # beat 1 - standard
            swap_beat2,               # beat 2 - high-energy swap
            self.base_indices[2],   # beat 3 - standard
            swap_beat4,               # beat 4 - high-energy swap
        ]
        slices = [fade_edges(self.grid.slice_beat(i)) for i in chosen]
        return stitch_with_crossfade(slices)

    # ---- Variation 2: The Stutter / Syncopator ----
    def variation_stutter(self) -> AudioSegment:
        """
        Takes one strong transient beat and subdivides it into a
        triplet-feel stutter, repeated to fill most of the 4-beat loop,
        then capped off with a standard beat for grid resolution.
        """
        donors = self.grid.high_energy_indices(exclude=set(), top_n=5)
        stutter_idx = donors[0] if donors else self.base_indices[0]

        full_dur = self.grid.beat_duration_ms(stutter_idx)
        start_ms = self.grid.beat_times_ms[stutter_idx]

        # Split the beat into a triplet (3 equal subdivisions) for a
        # syncopated bounce feel.
        third = max(full_dur // 3, 10)  # guard against degenerate tiny beats
        stutter_unit = fade_edges(self.grid.slice_ms(start_ms, third))

        # Build: stutter, stutter, stutter, [standard closing beat]
        closing_idx = self.base_indices[-1]
        closing_beat = fade_edges(self.grid.slice_beat(closing_idx))

        slices = [stutter_unit, stutter_unit, stutter_unit, closing_beat]
        return stitch_with_crossfade(slices)

    # ---- Variation 3: The Textural Flip ----
    def variation_textural_flip(self) -> AudioSegment:
        """
        Maps structural chops from the LATER half of the song onto the
        intro's rhythmic grid -- same beat-count/timing, different
        textural content (e.g. drop/chorus energy over verse rhythm).
        """
        half_point = self.grid.n_beats // 2
        later_half = list(range(half_point, self.grid.n_beats))

        if len(later_half) < BEATS_PER_LOOP:
            # Track too short to have a distinct "later half" -- fall back
            # to the highest-energy beats available anywhere.
            chosen = self.grid.high_energy_indices(top_n=BEATS_PER_LOOP)
            while len(chosen) < BEATS_PER_LOOP:
                chosen.append(chosen[-1] if chosen else 0)
        else:
            # Rank later-half beats by energy, take the top 4, but keep
            # them in chronological order so the grid still "flows".
            ranked_later = sorted(
                later_half, key=lambda i: self.grid.beat_energy[i], reverse=True
            )[:BEATS_PER_LOOP]
            chosen = sorted(ranked_later)

        slices = [fade_edges(self.grid.slice_beat(i)) for i in chosen]
        return stitch_with_crossfade(slices)

    # ---- Variation 4 & 5: Algorithmic Chaos ----
    def variation_chaos(self, energy_threshold_percentile: float = 40.0) -> AudioSegment:
        """
        Randomized but rhythmically-stable permutation: selects 4 beats
        from the pool of "clean transient" beats (above an energy
        percentile threshold, to avoid silence/noise-floor slices) in
        random order.
        """
        energies = np.array(self.grid.beat_energy)
        threshold = np.percentile(energies, energy_threshold_percentile)
        clean_pool = [i for i in range(self.grid.n_beats) if self.grid.beat_energy[i] >= threshold]

        if len(clean_pool) < BEATS_PER_LOOP:
            clean_pool = list(range(self.grid.n_beats))  # fallback: use everything

        chosen = random.sample(clean_pool, k=min(BEATS_PER_LOOP, len(clean_pool)))
        while len(chosen) < BEATS_PER_LOOP:
            chosen.append(random.choice(clean_pool))

        slices = [fade_edges(self.grid.slice_beat(i)) for i in chosen]
        return stitch_with_crossfade(slices)

    def generate_all(self) -> List[Tuple[str, AudioSegment]]:
        """
        Runs all five strategies and returns a list of (name, AudioSegment)
        pairs. Individual strategy failures are logged and skipped rather
        than aborting the whole batch.
        """
        results = []

        strategies = [
            ("variation_1_swapper", self.variation_swapper),
            ("variation_2_stutter_syncopator", self.variation_stutter),
            ("variation_3_textural_flip", self.variation_textural_flip),
            ("variation_4_chaos_a", self.variation_chaos),
            ("variation_5_chaos_b", self.variation_chaos),  # independent random draw
        ]

        for name, fn in strategies:
            try:
                loop = fn()
                results.append((name, loop))
                log.info(f"Generated {name} ({len(loop)} ms)")
            except Exception as e:
                log.error(f"Failed to generate {name}: {e}")
                continue

        if not results:
            raise LoopExportError("All variation strategies failed; no loops generated.")

        return results


# --------------------------------------------------------------------------
# Convenience top-level function (used by app.py)
# --------------------------------------------------------------------------
def process_file(audio_path: str) -> List[Tuple[str, AudioSegment]]:
    """
    End-to-end convenience function: builds a BeatGrid from audio_path,
    runs the full VariationEngine, and returns the resulting loops.

    Raises AudioLoadError, BeatGridError, or LoopExportError on failure --
    callers (e.g. app.py) are expected to catch and report these.
    """
    grid = BeatGrid(audio_path)
    engine = VariationEngine(grid)
    return engine.generate_all()
