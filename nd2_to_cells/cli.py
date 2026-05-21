"""Command-line interface for nd2_to_cells."""

import click

from .export import run_export
from .align import run_align
from .track import run_track


@click.group()
def cli():
    """nd2_to_cells: ND2 microscopy files → per-cell HDF5 files.

    Three sequential subcommands:

    \b
      export  — split ND2 into per-position TIFFs
      align   — drift-correct phase and fluor images
      track   — link cells across frames and write cell*.h5 files

    Run each subcommand with --help for details.
    """


@cli.command("export")
@click.option(
    "--input", "nd2_path", required=True, type=click.Path(exists=True),
    help="Path to the input ND2 file.",
)
@click.option(
    "--output", "output_dir", required=True, type=click.Path(),
    help="Output directory. Created if it does not exist.",
)
@click.option(
    "--basename", required=True, type=str,
    help="Filename prefix for all output TIFFs (e.g. '260430').",
)
@click.option(
    "--phase-channel", "phase_channel", default=0, show_default=True, type=int,
    help="0-based index of the phase-contrast channel in the ND2 file. "
         "All other channels become fluor1, fluor2, ... in order.",
)
@click.option(
    "--z-project", "z_project", default="mean",
    type=click.Choice(["mean", "max"], case_sensitive=False),
    show_default=True,
    help="Z-projection method for Z-stacks.",
)
def export_cmd(nd2_path, output_dir, basename, phase_channel, z_project):
    """Export an ND2 file to per-position TIFFs.

    Creates one xy{N}/ subdirectory per microscope position, each containing:

    \b
      phase/    — phase-contrast frames
      fluor1/   — first fluorescence channel frames
      fluor2/   — second fluorescence channel frames (if present)
      ...
      masks/    — empty directory (populated by Omnipose later)
      cell/     — empty directory (populated by nd2_to_cells track later)

    Also writes raw_im/ containing pre-alignment copies of all TIFFs.
    """
    run_export(
        nd2_path=nd2_path,
        output_dir=output_dir,
        basename=basename,
        phase_channel=phase_channel,
        z_project=z_project,
    )


@cli.command("align")
@click.option(
    "--data", "data_dir", required=True, type=click.Path(exists=True),
    help="Directory containing xy*/ subdirectories (output of nd2_to_cells export).",
)
@click.option(
    "--align-channel", "align_channel", default="phase", show_default=True,
    help="Channel subdirectory name to use for computing shifts (default: phase).",
)
@click.option(
    "--workers", default=1, show_default=True, type=int,
    help="Number of parallel worker processes (one per xy position).",
)
def align_cmd(data_dir, align_channel, workers):
    """Correct stage drift across frames for all xy positions.

    Computes frame-to-frame shifts from the align channel using
    phase cross-correlation, then applies the same shifts to all
    channel images. Images are padded to a common canvas size
    (no cropping). Originals are preserved in raw_im/.
    """
    run_align(
        data_dir=data_dir,
        align_channel=align_channel,
        workers=workers,
    )


@cli.command("track")
@click.option(
    "--data", "data_dir", required=True, type=click.Path(exists=True),
    help="Directory containing xy*/ subdirectories.",
)
@click.option(
    "--preset", default="100XEc", show_default=True,
    help="Preset name (e.g. '100XEc') or path to a custom .toml file.",
)
@click.option(
    "--workers", default=1, show_default=True, type=int,
    help="Number of parallel worker processes (one per xy position).",
)
@click.option(
    "--pad", default=5, show_default=True, type=int,
    help="Padding in pixels added around each cell's bounding box.",
)
def track_cmd(data_dir, preset, workers, pad):
    """Link cells across frames and write per-cell HDF5 files.

    Reads Omnipose PNG masks from xy{N}/masks/, links regions across
    frames using IoU-based assignment with a centroid-distance fallback,
    detects division events, and writes one HDF5 file per tracked cell
    to xy{N}/cell/.

    Cell files are named cell{ID:07d}.h5 (lowercase) or Cell{ID:07d}.h5
    (uppercase) for cells with a complete observed cell cycle (both birth
    and division observed, length >= min_cell_age).
    """
    run_track(
        data_dir=data_dir,
        preset=preset,
        workers=workers,
        pad=pad,
    )
