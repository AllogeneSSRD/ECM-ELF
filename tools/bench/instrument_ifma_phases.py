"""Generate the marked IFMA kernels for the standalone phase probe."""

import argparse
from pathlib import Path


def function(source, name):
    signature = f"static void {name}("
    start = source.index(signature)
    brace = source.index("\n{", start)
    while ";" in source[start:brace]:
        start = source.index(signature, start + len(signature))
        brace = source.index("\n{", start)
    end = source.index("\n}", brace)
    return source[start:end + 2]


def instrument(source, name, markers, replacements=()):
    body = function(source, name).replace(f"void {name}(", f"void profile_{name}(", 1)
    for original, replacement in replacements:
        if body.count(original) != 1:
            raise ValueError(f"{name}: expected exactly one {original!r}")
        body = body.replace(original, replacement)
    for anchor, phase in markers:
        if body.count(anchor) != 1:
            raise ValueError(f"{name}: expected exactly one {anchor!r}")
        body = body.replace(anchor, f"    mark(Phase::{phase});\n{anchor}", 1)
    return body


def generate(source):
    fold_body = function(source, "ifma_mersenne_fold_top")
    if "/* ---- 4. drain" in fold_body:
        fold_markers = [("    /* ---- 4. drain", "fold_high"),
                        ("    /* ---- 5. canonical", "fold_carry")]
        fold_phases = '"fold_high_ns", "fold_carry_ns"'
    else:
        fold_markers = [("    // Wrap", "fold_high"), ("    if (is_modulus)", "fold_carry")]
        fold_phases = '"fold_add_carry_ns", "fold_wrap_ns"'
    fold = instrument(source, "ifma_mersenne_fold_top",
                      [*fold_markers, ("\n}", "canonical")])
    finish = instrument(source, "ifma_mersenne_finish", [
        ("    profile_ifma_mersenne_fold_top", "carry"),
    ], [("    ifma_mersenne_fold_top(out, t, c, top);",
         "    profile_ifma_mersenne_fold_top(out, t, c, top);")])
    mul = instrument(source, "ifma_mersenne_mul", [
        ("    /* ---- 1. product", "clear"),
        ("    profile_ifma_mersenne_finish", "product"),
    ], [("    ifma_mersenne_finish(out, t, c, zero);",
         "    profile_ifma_mersenne_finish(out, t, c, zero);")])
    sqr = instrument(source, "ifma_mersenne_sqr", [
        ("    /* ---- off-diagonal", "clear"),
        ("    /* ---- double the off-diagonal", "product"),
        ("    profile_ifma_mersenne_fold_top", "carry"),
    ], [("    ifma_mersenne_fold_top(out, t, c, cy);",
         "    profile_ifma_mersenne_fold_top(out, t, c, cy);")])
    cios = instrument(source, "ifma_cios", [
        ("    __m512i cy = zero;", "clear"),
        ("    /* --- result = columns", "product"),
        ("    /* --- conditional subtraction", "carry"),
        ("\n}", "canonical"),
    ])
    return ("// Generated from modular_ifma.cpp by instrument_ifma_phases.py.\n\n"
            f"static constexpr const char* fold_columns[] = {{{fold_phases}}};\n\n"
            + "\n\n".join((fold, finish, mul, sqr, cios)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(encoding="utf-8-sig")),
                           encoding="utf-8", newline="\n")
