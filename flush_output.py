"""
flush_output.py
Wipes all generated artifacts from previous runs so you can start fresh.
Keeps the folder structure intact — just deletes the files inside.

Usage:
    python flush_output.py
    python flush_output.py --dry-run     (preview what would be deleted)
    python flush_output.py --yes         (skip confirmation prompt)
"""
import os
import sys
import shutil
import argparse

SUBDIRS = [
    "scenes",
    "audio",
    "images",
    "videos",
    "manim",
    "checkpoints",
]

OUTPUT_ROOT = "output"


def count_files(root: str) -> tuple[int, float]:
    """Return (file_count, total_size_mb)."""
    total_files = 0
    total_bytes = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_bytes += os.path.getsize(fp)
                total_files += 1
            except OSError:
                pass
    return total_files, total_bytes / 1_048_576


def flush(root: str, dry_run: bool = False) -> None:
    if not os.path.exists(root):
        print(f"Output folder '{root}' does not exist — nothing to flush.")
        return

    deleted_files  = 0
    deleted_bytes  = 0
    skipped        = 0

    for subdir in SUBDIRS:
        subpath = os.path.join(root, subdir)
        if not os.path.exists(subpath):
            continue

        for entry in os.scandir(subpath):
            try:
                size = entry.stat().st_size
                if dry_run:
                    print(f"  [dry-run] would delete: {entry.path}  ({size/1024:.1f} KB)")
                else:
                    if entry.is_dir():
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
                deleted_files += 1
                deleted_bytes += size
            except Exception as e:
                print(f"  WARNING: could not delete {entry.path}: {e}")
                skipped += 1

    size_mb = deleted_bytes / 1_048_576
    verb    = "Would delete" if dry_run else "Deleted"
    print(f"\n{verb} {deleted_files} file(s) — {size_mb:.1f} MB freed.")
    if skipped:
        print(f"Skipped {skipped} file(s) (check warnings above).")
    if dry_run:
        print("\nRun without --dry-run to actually delete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flush ProcEx output artifacts")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be deleted without deleting anything"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_ROOT,
        help=f"Output root directory (default: {OUTPUT_ROOT})"
    )
    args = parser.parse_args()

    root = args.output_dir

    if not os.path.exists(root):
        print(f"Nothing to flush — '{root}' folder doesn't exist yet.")
        return

    n_files, size_mb = count_files(root)

    if n_files == 0:
        print("Output folder is already empty.")
        return

    print(f"\nProcEx Output Flush")
    print(f"{'='*40}")
    print(f"  Target : {os.path.abspath(root)}")
    print(f"  Found  : {n_files} file(s) — {size_mb:.1f} MB")
    print(f"  Dirs   : {', '.join(SUBDIRS)}")
    print()

    if args.dry_run:
        flush(root, dry_run=True)
        return

    if not args.yes:
        answer = input("Delete all output files? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted — nothing deleted.")
            return

    flush(root, dry_run=False)
    print("\nReady for a fresh run:")
    print(f"  python main.py --input papers\\your_paper.pdf --topic \"Your Topic\" --minutes 5")


if __name__ == "__main__":
    main()