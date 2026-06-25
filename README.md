# Automated Audio-Chopper & Loop-Rearranger

A Python toolkit that maps the rhythmic grid of an audio file and
algorithmically generates a batch of distinct, seamless **4-beat loop
variations** — useful for sample-flipping, beat-making, or as a
downstream processing stage after stem separation (vocals / drums /
bass / melody).

Given one input track, the engine outputs **5 unique loop variations**,
each built with a different compositional strategy, all rhythmically
grid-aligned and click-free at every cut and loop point.

---

## How It Works

The pipeline has two stages, split across two modules:

1. **`audio_engine.py`** — the processing core.
   - Loads the audio and runs beat detection via `librosa` to get exact
     BPM and a list of beat timestamps **in milliseconds**.
   - Computes per-beat RMS energy, used to identify "high-energy" /
     transient-rich beats for the variation algorithms.
   - Uses `pydub` to slice the audio at those exact millisecond
     boundaries — no drift from the real rhythmic grid.
   - Applies short linear crossfades (2–5 ms) at every slice boundary,
     **including the loop's own wrap-point**, so loops play back
     click-free even when repeated back-to-back.
   - Implements `VariationEngine`, which runs five generation
     strategies (see below).

2. **`app.py`** — the entry point.
   - Takes a file path from the command line.
   - Calls into `audio_engine.py` to run the full analysis +
     generation pipeline.
   - Exports the resulting batch into an organized output directory.

---

## The Five Variations

| # | Name | Strategy |
|---|------|----------|
| 1 | **The Swapper** | Keeps beats 1 & 3 from the intro grid standard; swaps beats 2 & 4 with high-energy donor slices pulled from elsewhere in the track. |
| 2 | **The Stutter / Syncopator** | Takes the single most transient-rich beat in the track, subdivides it into a triplet, and repeats it three times before capping the loop with one clean closing beat — producing a catchy, syncopated bounce. |
| 3 | **The Textural Flip** | Identifies the highest-energy beats from the *later half* of the song (e.g. drop/chorus material) and maps them onto a 4-beat grid, in chronological order, replacing the intro's original texture while preserving rhythmic flow. |
| 4 | **Algorithmic Chaos A** | Randomly samples 4 beats from a pool of "clean transient" beats (filtered above the 40th energy percentile to avoid silence/noise-floor slices) and arranges them in random order. |
| 5 | **Algorithmic Chaos B** | Same strategy as #4, run with an independent random draw — guaranteed to differ from Chaos A as long as the donor pool has more than 4 eligible beats. |

All five variations are 4-beat loops, individually exported, and safe
to drop into a DAW for instant audition.

---

## Repository Structure

```
.
├── audio_engine.py      # Core: beat-grid analysis, slicing, crossfades, variation algorithms
├── app.py                # CLI entry point: drives the engine, organizes output
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── output/                # Generated on first run -- organized by track name
    └── <track_name>/
        ├── loop_variation_1_swapper.wav
        ├── loop_variation_2_stutter_syncopator.wav
        ├── loop_variation_3_textural_flip.wav
        ├── loop_variation_4_chaos_a.wav
        └── loop_variation_5_chaos_b.wav
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/audio-loop-rearranger.git
cd audio-loop-rearranger
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

`pydub` relies on `ffmpeg` for loading/exporting non-WAV formats (MP3,
OGG, FLAC) and for decoding compressed source files. It is **not**
installable via pip.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows
# Download a build from https://ffmpeg.org/download.html
# and add the bin/ folder to your system PATH.
```

---

## Usage

Basic run, default settings (WAV output, `output/` directory):

```bash
python app.py path/to/your_track.wav
```

Custom output directory:

```bash
python app.py path/to/your_track.wav --output-dir my_loops
```

Export in a different format (requires ffmpeg):

```bash
python app.py path/to/your_track.wav --format mp3
```

### Example

```bash
python app.py samples/drum_stem.wav
```

```
[INFO] Beat grid built: 128.0 BPM, 64 beats, duration=30.21s
[INFO] Generated variation_1_swapper (1875 ms)
[INFO] Generated variation_2_stutter_syncopator (1875 ms)
[INFO] Generated variation_3_textural_flip (1875 ms)
[INFO] Generated variation_4_chaos_a (1875 ms)
[INFO] Generated variation_5_chaos_b (1875 ms)
[INFO] Exported -> output/drum_stem/loop_variation_1_swapper.wav
[INFO] Exported -> output/drum_stem/loop_variation_2_stutter_syncopator.wav
[INFO] Exported -> output/drum_stem/loop_variation_3_textural_flip.wav
[INFO] Exported -> output/drum_stem/loop_variation_4_chaos_a.wav
[INFO] Exported -> output/drum_stem/loop_variation_5_chaos_b.wav
[INFO] Done. 5/5 loop variation(s) written to 'output/drum_stem/'.
```

Listen through the batch and pick your favorite — each file is named
clearly by variation strategy so you know exactly what algorithm
produced it.

---

## Use With Stem-Separated Audio

This script is designed to sit downstream of a neural stem-separation
stage (e.g. a U-Net-style source separator that splits a track into
Vocals / Drums / Bass / Melody). Point `app.py` at any isolated stem
file the same way you would a full mix:

```bash
python app.py stems/drums.wav --output-dir output/drums
python app.py stems/melody.wav --output-dir output/melody
```

Drum stems tend to produce the cleanest results for the Stutter and
Chaos variations, since transients are already isolated from melodic
or vocal content.

---

## Error Handling Notes

- **File won't load**: raises a clear error if the file is missing,
  unreadable, or not a valid audio format.
- **Beat detection too sparse**: if fewer than 5 beats are detected
  (the minimum needed for 4-beat loop variations with donor material),
  the script exits with a descriptive error rather than producing
  broken output.
- **Individual variation failure**: if one strategy fails (e.g. due to
  unusual track structure), it is logged and skipped — the rest of the
  batch is still generated and exported.
- **Export failure**: a failed export for one variation does not abort
  the others; you'll get a partial batch with a clear log of what
  succeeded.

---

## Tuning

A few constants at the top of `audio_engine.py` are safe to adjust:

```python
CROSSFADE_MS = 4        # anti-pop crossfade length, keep within 2-5ms
BEATS_PER_LOOP = 4       # change to build loops of a different length
SAMPLE_RATE = 44100       # librosa analysis sample rate
```

---

## License

MIT — use freely, attribution appreciated.
