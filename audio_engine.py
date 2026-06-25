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
    - Implement the VariationEngine: algorithmic strategies tuned for
      mellow, Alchemist-inspired hip-hop 32-beat loop configurations.

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
BEATS_PER_LOOP = 32         # Expanded to 32 beats for 10-20 second loop generation
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

        if self.bpm <= 0 or len(beat_frames) < 4:
            raise BeatGridError(
                f"Insufficient beat grid (BPM={self.bpm:.1f}, "
                f"beats found={len(beat_frames)})."
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
    Generates distinct 32-beat loop configurations from a BeatGrid using
    compositional layouts structured to match a mellow, Alchemist hip-hop pocket:

        1. Swapper            - keeps structural foundations standard; swaps backbeats
                                  with smooth, high-energy soul elements.
        2. Syncopator A       - rolling natural flow with a clean MPC-style double-trigger
                                  pad hit right before the phrase turnaround point.
        3. Syncopator B       - classic vinyl-feel triplet swing turnaround at
                                  phrase boundaries while keeping the core groove static.
        4. Textural Flip A    - alternates 4-beat structural blocks from the song's later
                                  half to form an evolving melodic bridge structure.
        5. Textural Flip B    - maps chunks in broad 8-beat soul blocks to keep vintage
                                  vocal lines or jazz loops completely intact and un-chopped.
        6. Algorithmic Chaos  - randomized layout pulling from steady transient pockets.
    """

    def __init__(self, grid: BeatGrid):
        self.grid = grid

    def add_background_drums(self, foreground_loop: AudioSegment) -> AudioSegment:
        """Helper to overlay an optional instrumental backing track loop if needed."""
        return foreground_loop

    # ---- Variation 1: The Swapper ----
    def variation_swapper(self) -> AudioSegment:
        """
        Keeps beats 1 & 3 standard over an extended timeline; swaps beats 2 & 4
        with high-energy donor slices dynamically chosen from elsewhere in the track.
        """
        chosen_indices = []
        for b in range(BEATS_PER_LOOP):
            if b % 4 == 1 or b % 4 == 3:
                # Target a smooth, high-energy pool to fill the backbeat points for an Alchemist style pocket
                high_energy_pool = sorted(range(self.grid.n_beats), key=lambda x: self.grid.beat_energy[x],
                                          reverse=True)[:10]
                chosen_indices.append(random.choice(high_energy_pool))
            else:
                chosen_indices.append(b % self.grid.n_beats)

        slices = [fade_edges(self.grid.slice_beat(i)) for i in chosen_indices]
        return self.add_background_drums(stitch_with_crossfade(slices))

    # ---- Variation 2: The Syncopator A (MPC Style Double-Trigger) ----
    def variation_syncopator_a(self) -> AudioSegment:
        """
        Mellow Alchemist vibe: plays naturally, then creates a clean MPC-style
        double-trigger chop (divided by 2) at the end of the phrase turnaround.
        """
        slices = []
        for b in range(BEATS_PER_LOOP):
            if b % 16 == 14:  # Clean half-beat pad re-trigger right before the phrase loops
                dur = self.grid.beat_duration_ms(b % self.grid.n_beats)
                start = self.grid.beat_times_ms[b % self.grid.n_beats]
                subdivision = max(dur // 2, 10)
                slices.append(fade_edges(self.grid.slice_ms(start, subdivision)))
                slices.append(fade_edges(self.grid.slice_ms(start, subdivision)))
            else:
                slices.append(fade_edges(self.grid.slice_beat(b % self.grid.n_beats)))
        return self.add_background_drums(stitch_with_crossfade(slices))

    # ---- Variation 3: The Syncopator B (Triplet Bounce Turnaround) ----
    def variation_syncopator_b(self) -> AudioSegment:
        """
        Plays naturally, then delivers a smooth vinyl-style triplet turnaround
        at the phrase boundaries for a classic laid-back boom-bap bounce.
        """
        slices = []
        for b in range(BEATS_PER_LOOP):
            if b % 16 >= 12 and b % 16 < 15:
                dur = self.grid.beat_duration_ms(b % self.grid.n_beats)
                start = self.grid.beat_times_ms[b % self.grid.n_beats]
                subdivision = max(dur // 3, 10)
                slices.append(fade_edges(self.grid.slice_ms(start, subdivision)))
                slices.append(fade_edges(self.grid.slice_ms(start, subdivision)))
                slices.append(fade_edges(self.grid.slice_ms(start, subdivision)))
            else:
                slices.append(fade_edges(self.grid.slice_beat(b % self.grid.n_beats)))
        return self.add_background_drums(stitch_with_crossfade(slices))

    # ---- Variation 4: The Textural Flip A (4-Beat Block Swap) ----
    def variation_textural_flip_a(self) -> AudioSegment:
        """
        Maps structural chops from the later half of the song onto the
        rhythmic grid in alternating 4-beat blocks, preserving longer melodic phrasing.
        """
        slices = []
        midpoint = self.grid.n_beats // 2 if self.grid.n_beats > 1 else 0
        for b in range(BEATS_PER_LOOP):
            if b % 8 >= 4 and midpoint > 0:
                idx = (midpoint + b) % self.grid.n_beats
            else:
                idx = b % self.grid.n_beats
            slices.append(fade_edges(self.grid.slice_beat(idx)))
        return self.add_background_drums(stitch_with_crossfade(slices))

    # ---- Variation 5: The Textural Flip B (8-Beat Soul Block Swap) ----
    def variation_textural_flip_b(self) -> AudioSegment:
        """
        Alchemist soul-sample style: maps chops in larger 8-beat structural blocks
        to keep vintage vocal or jazz expressions completely intact and easy to loop over.
        """
        slices = []
        midpoint = self.grid.n_beats // 2 if self.grid.n_beats > 1 else 0
        for b in range(BEATS_PER_LOOP):
            if b % 16 >= 8 and midpoint > 0:
                idx = (midpoint + b) % self.grid.n_beats
            else:
                idx = b % self.grid.n_beats
            slices.append(fade_edges(self.grid.slice_beat(idx)))
        return self.add_background_drums(stitch_with_crossfade(slices))

    # ---- Variation 6: Algorithmic Chaos ----
    def variation_chaos(self, energy_threshold_percentile: float = 40.0) -> AudioSegment:
        """
        Randomized but rhythmically-stable permutation: selects 32 beats
        from the pool of "clean transient" beats (above an energy
        percentile threshold, to avoid silence/noise-floor slices) in
        random order.
        """
        energies = np.array(self.grid.beat_energy)
        threshold = np.percentile(energies, energy_threshold_percentile)
        clean_pool = [i for i in range(self.grid.n_beats) if self.grid.beat_energy[i] >= threshold]

        if not clean_pool:
            clean_pool = list(range(self.grid.n_beats))  # fallback: use everything

        chosen = [random.choice(clean_pool) for _ in range(BEATS_PER_LOOP)]
        slices = [fade_edges(self.grid.slice_beat(i)) for i in chosen]
        return self.add_background_drums(stitch_with_crossfade(slices))

    def generate_all(self) -> List[Tuple[str, AudioSegment]]:
        """
        Runs all six strategies and returns a list of (name, AudioSegment)
        pairs. Individual strategy failures are logged and skipped rather
        than aborting the whole batch.
        """
        results = []

        strategies = [
            ("variation_1_swapper", self.variation_swapper),
            ("variation_2_syncopator_a", self.variation_syncopator_a),
            ("variation_3_syncopator_b", self.variation_syncopator_b),
            ("variation_4_textural_flip_a", self.variation_textural_flip_a),
            ("variation_5_textural_flip_b", self.variation_textural_flip_b),
            ("variation_6_chaos", self.variation_chaos),
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