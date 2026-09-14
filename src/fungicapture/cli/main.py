"""
Command-line runner.

The graphical application is for setting layouts and checking results by eye.
This is for everything else: running a large batch on a compute cluster,
scripting a re-analysis, or reproducing a published result from a project
folder.

Both front ends call the same functions in ``core.pipeline``, so they cannot
drift apart in behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _progress_printer(quiet: bool):
    """A progress callback that rewrites one line rather than scrolling."""
    state = {"last": -1}

    def progress(done: int, total: int, message: str = "") -> bool:
        if quiet or not total:
            return True
        percent = int(100 * done / total)
        if percent != state["last"]:
            state["last"] = percent
            sys.stderr.write(f"\r  {percent:3d}%  {message[:60]:<60s}")
            sys.stderr.flush()
        return True

    return progress


def command_new(args) -> int:
    from ..core.project import ImagingMode, PlateLayout, Project

    project = Project.create(
        args.root,
        name=args.name or Path(args.root).name,
        mode=ImagingMode(args.mode),
        image_dir=args.images,
        layout=PlateLayout(rows=args.rows, cols=args.cols),
        exist_ok=True,
    )
    if args.media:
        project.media_label = args.media
        project.save()
    print(f"Created project '{project.name}' at {project.root}")
    print(f"  mode: {project.mode.label} ({project.mode.description})")
    print(f"  params hash: {project.params_hash()}")
    return 0


def command_run(args) -> int:
    from ..core.pipeline import ALL_STEPS, run_pipeline
    from ..core.platemap import PlateMap
    from ..core.project import Project

    project = Project.load(args.project)
    steps = args.steps.split(",") if args.steps else list(ALL_STEPS)

    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        print(f"Unknown step(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Valid steps: {', '.join(ALL_STEPS)}", file=sys.stderr)
        return 2

    plate_map = None
    if args.plate_map:
        plate_map = PlateMap.load_csv(
            args.plate_map, "well" if project.mode.needs_grid else "file"
        )
    elif project.plate_map_path and Path(project.plate_map_path).exists():
        plate_map = PlateMap.load_csv(
            project.plate_map_path, "well" if project.mode.needs_grid else "file"
        )

    print(f"Project: {project.name}  ({project.mode.label})")
    print(f"Steps:   {', '.join(steps)}")
    print(f"Params:  {project.params_hash()}")
    print()

    results = run_pipeline(
        project,
        steps,
        progress=_progress_printer(args.quiet),
        plate_map=plate_map,
        backend=args.backend,
        auto_centre=not args.no_auto_centre,
        skip_edge_wells=args.skip_edge_wells,
    )

    if not args.quiet:
        sys.stderr.write("\r" + " " * 70 + "\r")

    exit_code = 0
    for result in results:
        print(result.summary())
        for warning in result.warnings:
            print(f"  note: {warning}")
        for name, error in result.errors[:10]:
            print(f"  error: {Path(name).name}: {error}", file=sys.stderr)
        if result.errors:
            exit_code = 1

    print()
    print(f"Outputs are in {project.root}")
    return exit_code


def command_report(args) -> int:
    from ..core.pipeline import pair_crops_and_masks
    from ..core.project import Project
    from ..core.report import build_context, write_report
    from ..core.validate import VerdictStore, build_panel_data

    project = Project.load(args.project)
    pairs = pair_crops_and_masks(project)
    if not pairs:
        print("No colonies found. Run the pipeline first.", file=sys.stderr)
        return 1

    store = VerdictStore(project.qc_dir / VerdictStore.FILENAME)
    progress = _progress_printer(args.quiet)

    panels = []
    for index, (crop_path, mask_path) in enumerate(pairs, start=1):
        progress(index, len(pairs), crop_path.name)
        try:
            panels.append(build_panel_data(crop_path, mask_path, {}, project.params))
        except Exception as error:
            print(f"  skipped {crop_path.name}: {error}", file=sys.stderr)

    output = Path(args.output or project.qc_dir / f"qc_report_{args.mode}.pdf")
    context = build_context(project, args.mode, len(panels), store)
    write_report(output, panels, project.params, context, mode=args.mode, verdicts=store)

    if not args.quiet:
        sys.stderr.write("\r" + " " * 70 + "\r")
    print(f"Report written to {output}")
    return 0


def command_info(args) -> int:
    from ..core.project import Project

    project = Project.load(args.project)
    print(f"Name:        {project.name}")
    print(f"Mode:        {project.mode.label} — {project.mode.description}")
    print(f"Root:        {project.root}")
    print(f"Images:      {project.image_dir}")
    print(f"Medium:      {project.media_label}")
    print(f"Plate:       {project.layout.rows} x {project.layout.cols}")
    print(f"Params hash: {project.params_hash()}")
    print()
    print("Settings that change the numbers:")
    print(f"  ring fraction:    {project.params.ring.fraction}")
    print(f"  perimeter method: {project.params.morphology.perimeter_method}")
    print(f"  GLCM levels:      {project.params.glcm.levels}")
    print(f"  LBP:              P={project.params.lbp.n_points} "
          f"R={project.params.lbp.radius}")
    print(f"  Lab white point:  {project.params.colour.lab_illuminant}")
    print(f"  colour card:      {project.params.calibration.use_colour_card}")
    print()
    if project.runs:
        print("Run history:")
        for record in project.runs[-10:]:
            print(
                f"  {record.finished}  {record.step:<12s} "
                f"{record.n_outputs}/{record.n_inputs} ok, "
                f"{record.n_errors} errors  [{record.params_hash}]"
            )
    else:
        print("No runs recorded yet.")
    return 0


def command_gui(args) -> int:
    from ..gui.app import main as gui_main

    return gui_main([sys.argv[0]] + ([args.project] if args.project else []))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fungicapture",
        description=(
            "Image-based phenotyping of fungal colonies, from an agar plate "
            "photograph to GWAS-ready scores."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    new = subparsers.add_parser("new", help="create a project")
    new.add_argument("root", help="folder for the new project")
    new.add_argument("--name")
    new.add_argument(
        "--mode",
        default="plate_batch",
        choices=["single_dish", "dish_batch", "gridded_plate", "plate_batch"],
        help="what one photograph contains",
    )
    new.add_argument("--images", help="folder holding the photographs")
    new.add_argument("--rows", type=int, default=8)
    new.add_argument("--cols", type=int, default=12)
    new.add_argument("--media", help="growth medium label, e.g. SDA")
    new.set_defaults(function=command_new)

    run = subparsers.add_parser("run", help="run the pipeline")
    run.add_argument("project", help="project folder")
    run.add_argument(
        "--steps",
        help="comma-separated: crop,segment,features,phenotypes (default: all)",
    )
    run.add_argument("--plate-map", dest="plate_map", help="plate map CSV")
    run.add_argument(
        "--backend", default="auto", choices=["auto", "sam3", "classical"]
    )
    run.add_argument("--no-auto-centre", action="store_true")
    run.add_argument("--skip-edge-wells", action="store_true")
    run.add_argument("-q", "--quiet", action="store_true")
    run.set_defaults(function=command_run)

    report = subparsers.add_parser("report", help="write a QC PDF")
    report.add_argument("project")
    report.add_argument(
        "--mode", default="full", choices=["full", "failures", "contact_sheet"]
    )
    report.add_argument("-o", "--output")
    report.add_argument("-q", "--quiet", action="store_true")
    report.set_defaults(function=command_report)

    info = subparsers.add_parser("info", help="show a project's settings and history")
    info.add_argument("project")
    info.set_defaults(function=command_info)

    gui = subparsers.add_parser("gui", help="launch the graphical application")
    gui.add_argument("project", nargs="?")
    gui.set_defaults(function=command_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
