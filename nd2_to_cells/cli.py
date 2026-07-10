"""Command-line interface for nd2_to_cells."""

import click

from .align import run_align
from .export import run_export
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
    "--input",
    "nd2_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the input ND2 file.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory. Created if it does not exist.",
)
@click.option(
    "--basename",
    required=True,
    type=str,
    help="Filename prefix for all output TIFFs (e.g. '260430').",
)
@click.option(
    "--phase-channel",
    "phase_channel",
    default=0,
    show_default=True,
    type=int,
    help="0-based index of the phase-contrast channel in the ND2 file. "
    "All other channels become fluor1, fluor2, ... in order.",
)
@click.option(
    "--z-project",
    "z_project",
    default="mean",
    type=click.Choice(["mean", "max"], case_sensitive=False),
    show_default=True,
    help="Z-projection method for Z-stacks.",
)
def export_cmd(nd2_path, output_dir, basename, phase_channel, z_project):
    """Export an ND2 file to per-position TIFFs.

    Creates SuperSegger-like folder and exports frames to raw_im/ directory.

    Files will be names {basename}_t{???}_xy{??}_c{?}.tif.
    Also creates one xy{N}/ subdirectory per microscope position, each containing:

    \b
      cell/     — empty directory (populated by nd2_to_cells track later)
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
    "--data",
    "data_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing xy*/ subdirectories (output of nd2_to_cells export).",
)
@click.option(
    "--align-channel",
    "align_channel",
    default="phase",
    show_default=True,
    help="Channel subdirectory name to use for computing shifts (default: phase).",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of parallel worker processes (one per xy position).",
)
@click.option(
    "--max-shift-px",
    "max_shift_px",
    default=50.0,
    show_default=True,
    type=float,
    help="Shifts larger than this (pixels) are clamped to 0. "
    "Guards against spurious large shifts from blurry or artifact frames.",
)
@click.option(
    "--align-to-first",
    "align_to_first",
    is_flag=True,
    default=False,
    help="Register all frames against the first frame, instead of the default"
    "sequential mode.",
)
def align_cmd(data_dir, align_channel, workers, max_shift_px, align_to_first):
    """Correct stage drift across frames for all xy positions.

    Output frames will be saved to channel-specific subfolders in the xy{N}/ directory.

    By default uses sequential mode: every frame is registered against the previous
    frame. This works well for slow drift or fast-changing contents.

    Use align-to-first flag to align frames directly against frame 0. This avoids
    compounding of subpixel errors over long movies.

    """
    run_align(
        data_dir=data_dir,
        align_channel=align_channel,
        workers=workers,
        max_shift_px=max_shift_px,
        align_to_first=align_to_first,
    )


@cli.command("track")
@click.option(
    "--data",
    "data_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing xy*/ subdirectories.",
)
@click.option(
    "--preset",
    default="100XEc",
    show_default=True,
    help="Preset name (e.g. '100XEc') or path to a custom .toml file.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of parallel worker processes (one per xy position).",
)
@click.option(
    "--pad",
    default=5,
    show_default=True,
    type=int,
    help="Padding in pixels added around each cell's bounding box.",
)
@click.option(
    "--consolidated",
    is_flag=True,
    default=False,
    help="Write one cells.h5 per position (one group per cell) instead of one file per cell.",
)
def track_cmd(data_dir, preset, workers, pad, consolidated):
    """Link cells across frames and write HDF5 output.

    Reads Omnipose PNG masks from xy{N}/masks/, links regions across
    frames using IoU-based assignment with a centroid-distance fallback,
    detects division events, and writes HDF5 output to xy{N}/cell/.

    By default, writes one file per tracked cell: cell{ID:07d}.h5
    (lowercase) or Cell{ID:07d}.h5 (uppercase, capital C = complete cell
    cycle: both birth and division observed, length >= min_cell_age).

    With --consolidated, writes a single cells.h5 per position where each
    cell is stored as a group (e.g. cells.h5/Cell0000002/mask).
    """
    run_track(
        data_dir=data_dir,
        preset=preset,
        workers=workers,
        pad=pad,
        consolidated=consolidated,
    )
