"""
app.py
=======
Entry-point for the Automated Audio-Chopper & Loop-Rearranger.

This is a CLI script (not a web server) -- it takes an input audio file,
hands it to audio_engine.py for analysis and loop generation, and writes
the resulting batch of 4-beat loop variations to an organized output
directory.

Usage:
    python app.py path/to/track.wav
    python app.py path/to/track.wav --output-dir my_loops
    python app.py path/to/track.wav --format mp3

Output structure:
    output/
      <track_name>/
        loop_variation_1_swapper.wav
        loop_variation_2_stutter_syncopator.wav
        loop_variation_3_textural_flip.wav
        loop_variation_4_chaos_a.wav
        loop_variation_5_chaos_b.wav
"""

import os
import sys
import argparse
import logging

from audio_engine import (
    process_file,
    AudioLoadError,
    BeatGridError,
    LoopExportError,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("app")

DEFAULT_OUTPUT_ROOT = "output"
DEFAULT_FORMAT = "wav"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a batch of seamless 4-beat loop variations from an audio file."
    )
    parser.add_argument(
        "input_path",
        help="Path to the source audio file (e.g. a full track or an isolated stem).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for generated loops (default: '{DEFAULT_OUTPUT_ROOT}').",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=["wav", "mp3", "flac", "ogg"],
        help=f"Export format for the generated loops (default: '{DEFAULT_FORMAT}'). "
             f"Note: mp3/ogg export requires ffmpeg on PATH.",
    )
    return parser


def make_track_output_dir(output_root: str, input_path: str) -> str:
    """
    Builds an organized per-track output directory:
        output/<track_name_without_extension>/
    """
    track_name = os.path.splitext(os.path.basename(input_path))[0]
    track_dir = os.path.join(output_root, track_name)
    os.makedirs(track_dir, exist_ok=True)
    return track_dir


def export_loops(results, track_dir: str, export_format: str):
    """
    Writes each generated (name, AudioSegment) pair to disk.
    A failure exporting one loop is logged and does not abort the
    rest of the batch.
    """
    exported_paths = []

    for name, loop in results:
        filename = f"loop_{name}.{export_format}"
        out_path = os.path.join(track_dir, filename)
        try:
            loop.export(out_path, format=export_format)
            log.info(f"Exported -> {out_path}")
            exported_paths.append(out_path)
        except Exception as e:
            log.error(f"Export failed for '{name}': {e}")

    return exported_paths


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        log.error(f"Input file does not exist: {args.input_path}")
        sys.exit(1)

    # ---- Run the analysis + generation engine ----
    try:
        results = process_file(args.input_path)
    except AudioLoadError as e:
        log.error(f"Audio could not be loaded: {e}")
        sys.exit(1)
    except BeatGridError as e:
        log.error(f"Beat grid could not be built: {e}")
        sys.exit(1)
    except LoopExportError as e:
        log.error(f"Loop generation failed entirely: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch-all so unexpected library errors don't print a raw
        # traceback to an end user running this from the CLI.
        log.error(f"Unexpected error during processing: {e}")
        sys.exit(1)

    # ---- Organize and export output ----
    track_dir = make_track_output_dir(args.output_dir, args.input_path)
    exported_paths = export_loops(results, track_dir, args.format)

    if not exported_paths:
        log.error("No loops were successfully exported.")
        sys.exit(1)

    log.info(
        f"Done. {len(exported_paths)}/{len(results)} loop variation(s) "
        f"written to '{track_dir}/'."
    )


if __name__ == "__main__":
    main()
