"""Sort final NTT table rows without changing measured values or other sections."""
import argparse
import pathlib
import re
import tomllib


def format_rows(source: str) -> tuple[str, int]:
    document = tomllib.loads(source)
    changed = 0
    section = re.compile(r"(?m)^\[ntt\.(\d+)\.table\.(\d+)\][^\r\n]*\r?\n(?:(?!^\[).)*", re.S | re.M)

    def rewrite(match):
        nonlocal changed
        table = document["ntt"][match[1]]["table"][match[2]]
        block = match[0]
        rows = re.search(r"(?m)^rows\s*=\s*\[\s*\r?\n(.*?)^\][ \t]*", block, re.S)
        if rows is None:
            raise ValueError("Expected generated one-row-per-line NTT table")
        lines = rows[1].splitlines(keepends=True)
        if len(lines) != len(table["rows"]):
            raise ValueError("Cannot reorder a table with multiline/commented rows")
        records = [dict(table, **dict(zip(table["columns"], row))) for row in table["rows"]]
        order = sorted(range(len(records)), key=lambda i: tuple(
            int(records[i].get(key, 0)) for key in ("log2_length", "length", "batch", "tile")))
        if order != list(range(len(records))):
            changed += 1
        return block[:rows.start(1)] + "".join(lines[i] for i in order) + block[rows.end(1):]

    result = section.sub(rewrite, source)
    return result, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=pathlib.Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--in-place", action="store_true")
    destination.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    raw = args.profile.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    result, changed = format_rows(raw.decode("utf-8-sig"))
    target = args.profile if args.in_place else args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".format.partial")
    with partial.open("xb") as stream:
        stream.write((b"\xef\xbb\xbf" if bom else b"") + result.encode("utf-8"))
    partial.replace(target)
    print(f"NTT tables reordered: {changed}; output: {target}")


if __name__ == "__main__":
    main()
