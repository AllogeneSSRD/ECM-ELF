"""ecm_prob.py — 用户 CLI：概率预测 / 四元组反解 / GMP 推荐表。

整合原 predict.py 与 params.py 的用户功能。模型逻辑在 model.py。

用法：
  python ecm_prob.py predict --all <bit> <B1> [B2]     # 全部曲线表
  python ecm_prob.py predict <curve> <bit> <B1> [B2]   # 单条曲线
  python ecm_prob.py solve --bit X --B1 Y [--B2 Z] [--curves N|--prob P] [--curve C]
  python ecm_prob.py gmp-table [bits...]
"""

from __future__ import annotations

import argparse
import sys

import model
import curves


def _parse_int(s):
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def predict_table(bit: int, B1: int, B2: float | None = None) -> str:
    if B2 is None:
        B2 = 100.0 * B1
    d_eff = model.load_representative_d_eff()
    t_by_name = {c["name"]: c.get("T") for c in curves.ROSTER}
    lines = [f"ECM prediction  (bit={bit}, B1={B1}, B2={B2:g})", ""]
    lines.append(f"{'curve':16s} {'T':>3s} {'D_emp':>8s} "
                 f"{'f_s1':>8s} {'N_s1':>7s} {'f_s12':>8s} {'N_s12':>7s}")
    lines.append("-" * 66)
    for c in curves.ROSTER:
        name = c["name"]
        d = d_eff.get(name)
        if d is None:
            continue
        f1 = model.predict_fraction(name, bit, B1, d)
        f12 = model.predict_fraction(name, bit, B1, d, B2)
        lines.append(f"{name:16s} {t_by_name[name]:>3d} {d:8.2f} "
                     f"{100*f1:8.3f}% {1.0/f1:7.1f} "
                     f"{100*f12:8.3f}% {1.0/f12:7.1f}")
    lines.append("")
    lines.append("f_s1 = stage-1 only;  N_s1 = 1/f_s1")
    lines.append(f"f_s12 = stage-1+stage-2 (B2={B2:g});  N_s12 = 1/f_s12")
    return "\n".join(lines)


def predict_one(name: str, bit: int, B1: int, B2: float | None) -> str:
    d_eff = model.load_representative_d_eff()
    dt = model.THEORETICAL_D.get(name)
    f1 = model.predict_fraction(name, bit, B1, d_eff[name])
    lines = [f"{name}: bit={bit}, B1={B1}" + (f", B2={B2:g}" if B2 else "")]
    lines.append(f"  D_eff = {d_eff[name]:.2f}" +
                 (f",  D_theoretical = {dt:.2f}" if dt else ""))
    lines.append(f"  stage-1 fraction        = {100*f1:.4f}%   (curves {1.0/f1:.1f})")
    if B2 and B2 > B1:
        f12 = model.predict_fraction(name, bit, B1, d_eff[name], B2)
        lines.append(f"  stage-1+stage-2 fraction = {100*f12:.4f}%   (curves {1.0/f12:.1f})")
    return "\n".join(lines)


def cmd_predict(a: argparse.Namespace) -> None:
    B2 = a.B2 if hasattr(a, "B2") else None
    if a.all:
        print(predict_table(a.bit, a.B1, B2))
    else:
        print(predict_one(a.curve, a.bit, a.B1, B2))


def cmd_solve(a: argparse.Namespace) -> None:
    D = model.resolve_D(a.curve, a.D)
    print(model.solve(a.bit, a.B1, a.curves, a.prob, D, a.B2))


def cmd_gmp_table(a: argparse.Namespace) -> None:
    print(model.gmp_table(a.bits))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ecm_prob")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="predict success fraction / expected curves")
    p.add_argument("--all", action="store_true", help="table over all curves")
    p.add_argument("--curve", default="suyama_s10", help="single curve name")
    p.add_argument("--bit", type=_parse_int, default=25)
    p.add_argument("--B1", type=_parse_int, default=256)
    p.add_argument("--B2", type=float, default=None)
    p.set_defaults(func=cmd_predict)

    s = sub.add_parser("solve", help="solve {bit,B1,N,p} 4-tuple")
    s.add_argument("--bit", type=float)
    s.add_argument("--B1", type=float)
    s.add_argument("--B2", type=float, help="default 100*B1")
    s.add_argument("--curves", type=float)
    s.add_argument("--prob", type=float)
    s.add_argument("--curve", default="suyama_s10")
    s.add_argument("--D", type=float)
    s.set_defaults(func=cmd_solve)

    g = sub.add_parser("gmp-table", help="GMP-ECM recommended bit->B1->curves table")
    g.add_argument("bits", type=int, nargs="*", default=list(range(30, 226, 5)))
    g.set_defaults(func=cmd_gmp_table)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
