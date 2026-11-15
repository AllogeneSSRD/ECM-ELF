"""Restore the UTF-8 BOM state recorded in Git HEAD."""

import argparse
import codecs
import io
import os
from pathlib import Path
import subprocess


SOURCE_EXTENSIONS = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc"}
BOM = codecs.BOM_UTF8


def git(repo, *args, input_data=None):
    return subprocess.run(
        ["git", *args], cwd=repo, input=input_data, stdout=subprocess.PIPE, check=True
    ).stdout


def head_boms(repo, blob_ids):
    ids = list(dict.fromkeys(blob_ids))
    if not ids:
        return {}
    objects = io.BytesIO(git(repo, "cat-file", "--batch", input_data=b"\n".join(ids) + b"\n"))
    states = {}
    for oid in ids:
        header = objects.readline().split()
        if len(header) != 3 or header[:2] != [oid, b"blob"]:
            raise RuntimeError(f"Unexpected Git batch response for {oid.decode('ascii')}")
        size = int(header[2])
        prefix = objects.read(min(size, len(BOM)))
        objects.seek(size - len(prefix), io.SEEK_CUR)
        if objects.read(1) != b"\n":
            raise RuntimeError(f"Incomplete Git blob {oid.decode('ascii')}")
        states[oid] = None if size < len(BOM) else prefix == BOM
    return states


def guard(repo, paths, no_fix):
    # Source archives have no Git HEAD to compare with. Worktrees also use .git.
    if not (repo / ".git").exists():
        print("BOM guard: skipped (no Git checkout)")
        return

    roots = [(repo / path).resolve() for path in paths]
    files = {}
    for entry in git(repo, "ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, _, oid = metadata.split()
        path = repo / os.fsdecode(name)
        if (mode in (b"100644", b"100755") and path.suffix.lower() in SOURCE_EXTENSIONS
                and any(path.is_relative_to(root) for root in roots)):
            if not path.is_symlink() and path.is_file():
                files[path] = oid

    states = head_boms(repo, files.values())
    restored = cleared = 0
    for path, oid in files.items():
        expected = states[oid]
        with path.open("rb", buffering=0) as source:
            prefix = source.read(len(BOM))
            if expected is None or len(prefix) < len(BOM) or (prefix == BOM) == expected:
                continue
            if not no_fix:
                remainder = source.read()
        name = path.relative_to(repo).as_posix()
        if no_fix:
            print(f"  [BOM] {name} differs from HEAD (HEAD bom={expected}, working bom={prefix == BOM})")
        elif expected:
            path.write_bytes(BOM + prefix + remainder)
            print(f"  [BOM] restored UTF-8 BOM in {name}")
            restored += 1
        else:
            path.write_bytes(remainder)
            print(f"  [BOM] removed BOM from {name}")
            cleared += 1

    print(f"BOM guard: {len(files)} file(s) scanned, {restored} restored, {cleared} cleared")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fix", action="store_true")
    parser.add_argument("paths", nargs="*", default=["kernels", "src"])
    args = parser.parse_args()
    guard(Path(__file__).resolve().parents[3], args.paths, args.no_fix)


if __name__ == "__main__":
    main()
