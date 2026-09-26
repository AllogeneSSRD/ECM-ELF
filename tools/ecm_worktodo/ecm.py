"""worktodopipeline

将 ECM2 任务拆分为两阶段：
1) Linux 阶段一：生成 `worktodo.sh`
2) Windows 阶段二：生成 `worktodo.csv`

支持：
- 读取并解析 ECM2 行
- 条件筛选（常用范围筛选）
- 重写 B1/B2
- 设置 has_na（缺失 AID 时补为 N/A）
- 按字段排序
- Linux/Windows 输出文件可选追加写入
"""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ECMTask:
	aid: Optional[str]
	has_na: bool
	k: str
	b: str
	n: int
	c: str
	b1: str
	b2: str
	curves_to_run: int
	known_factors: List[str]


def _split_csv_line(text: str) -> List[str]:
	row = next(csv.reader(io.StringIO(text), skipinitialspace=True), None)
	if row is None:
		return []
	return [item.strip() for item in row]


def _is_int_like(value: str) -> bool:
	try:
		int(value)
		return True
	except ValueError:
		return False


def _to_sort_number(value: str) -> float:
	try:
		return float(value)
	except ValueError:
		return float("inf")


def parse_ecm2_line(line: str) -> ECMTask:
	s = line.strip()
	if not s.startswith("ECM2="):
		raise ValueError(f"不是 ECM2 行: {line}")

	payload = s[len("ECM2=") :]
	cols = _split_csv_line(payload)
	if len(cols) < 8:
		raise ValueError(f"字段不足(至少 8 列): {line}")

	idx = 0
	aid: Optional[str] = None
	if not _is_int_like(cols[0]):
		aid_raw = cols[0]
		aid = aid_raw if aid_raw else None
		idx = 1

	needed = idx + 7
	if len(cols) < needed:
		raise ValueError(f"核心字段不足: {line}")

	k = cols[idx]
	b = cols[idx + 1]
	n = int(cols[idx + 2])
	c = cols[idx + 3]
	b1 = cols[idx + 4]
	b2 = cols[idx + 5]
	curves_to_run = int(cols[idx + 6])

	known_factors: List[str] = []

	optional_cols = cols[idx + 7 :]
	if len(optional_cols) >= 1:
		factors_joined = optional_cols[0]
		if factors_joined:
			known_factors = [x.strip() for x in factors_joined.split(",") if x.strip()]

	has_na = (aid is None) or (aid.upper() == "N/A")
	return ECMTask(
		aid=aid,
		has_na=has_na,
		k=k,
		b=b,
		n=n,
		c=c,
		b1=b1,
		b2=b2,
		curves_to_run=curves_to_run,
		known_factors=known_factors,
	)


def build_expression(task: ECMTask) -> str:
	c_int = int(task.c)
	sign = "+" if c_int >= 0 else "-"
	c_abs = abs(c_int)
	body = f"({task.k}*{task.b}^{task.n}{sign}{c_abs})"
	if task.known_factors:
		return f"{body}/({'*'.join(task.known_factors)})"
	return body


def _csv_quote(value: str) -> str:
	return f'"{value.replace("\"", "\"\"")}"'


def to_linux_line(task: ECMTask, save_name: str, gpu_curves: int, gmpecm_b2: str) -> str:
	expr = build_expression(task)
	return (
		f"echo '{expr}' | ./ecm -v -savea {save_name} "
		f"-gpu -gpuckpt 300 -gpucurves {gpu_curves} {task.b1} {gmpecm_b2}"
	)


def to_stage2_line(task: ECMTask, save_name: str, p95_b2: str, skip_curves: int, num_curves: int) -> str:
	if task.aid:
		parts = [
			f"ECMSTAGE2={task.aid}",
			task.k,
			task.b,
			str(task.n),
			task.c,
			_csv_quote(save_name),
			p95_b2,
			str(skip_curves),
			str(num_curves),
		]
	else:
		parts = [
			f"ECMSTAGE2={task.k}",
			task.b,
			str(task.n),
			task.c,
			_csv_quote(save_name),
			p95_b2,
			str(skip_curves),
			str(num_curves),
		]
	if task.known_factors:
		parts.append(_csv_quote(",".join(task.known_factors)))
	return ",".join(parts)


def prmers_to_p95_stage2_line(task: ECMTask, num_curves: int) -> str:
	if task.aid:
		parts = [
			f"ECM2={task.aid}",
			task.k,
			task.b,
			str(task.n),
			task.c,
			task.b1,
			task.b2,
			str(num_curves),
		]
	else:
		parts = [
			f"ECM2={task.k}",
			task.b,
			str(task.n),
			task.c,
			task.b1,
			task.b2,
			str(num_curves),
		]
	if task.known_factors:
		parts.append(_csv_quote(",".join(task.known_factors)))
	return ",".join(parts)


def read_tasks(input_path: Path) -> List[ECMTask]:
	tasks: List[ECMTask] = []
	with input_path.open("r", encoding="utf-8") as f:
		for raw in f:
			line = raw.strip().lstrip("\ufeff")
			if not line or line.startswith("#"):
				continue
			if not line.startswith("ECM2="):
				continue
			tasks.append(parse_ecm2_line(line))
	return tasks


def apply_processing(
	tasks: Iterable[ECMTask],
	set_b1: Optional[str],
	set_b2: Optional[str],
	force_has_na: bool,
	min_n: Optional[int],
	max_n: Optional[int],
	min_curves: Optional[int],
	max_curves: Optional[int],
) -> List[ECMTask]:
	out: List[ECMTask] = []
	for task in tasks:
		if min_n is not None and task.n < min_n:
			continue
		if max_n is not None and task.n > max_n:
			continue
		if min_curves is not None and task.curves_to_run < min_curves:
			continue
		if max_curves is not None and task.curves_to_run > max_curves:
			continue

		if set_b1 is not None:
			task.b1 = set_b1
		if set_b2 is not None:
			task.b2 = set_b2
		if force_has_na and (task.aid is None or task.aid == ""):
			task.aid = "N/A"
			task.has_na = True

		out.append(task)
	return out


def sort_tasks(tasks: List[ECMTask], sort_by: str, desc: bool) -> List[ECMTask]:
	if not sort_by:
		return tasks

	key_map = {
		"n": lambda t: t.n,
		"k": lambda t: _to_sort_number(t.k),
		"b": lambda t: _to_sort_number(t.b),
		"c": lambda t: _to_sort_number(t.c),
		"b1": lambda t: _to_sort_number(t.b1),
		"b2": lambda t: _to_sort_number(t.b2),
		"curves": lambda t: t.curves_to_run,
		"aid": lambda t: t.aid or "",
	}
	if sort_by not in key_map:
		raise ValueError(f"不支持排序字段: {sort_by}")

	return sorted(tasks, key=key_map[sort_by], reverse=desc)


def save_outputs(
	tasks: List[ECMTask],
	out_linux: Path,
	out_windows: Path,
	out_prmers: Path,
	save_pattern: str,
	gpu_curves: int,
	gmpecm_b2: str,
	p95_b2: str,
	skip_curves: int,
	append_linux: bool,
	append_windows: bool,
	append_prmers: bool,
) -> None:
	linux_lines: List[str] = []
	windows_lines: List[str] = []
	p95_lines: List[str] = []

	for t in tasks:
		save_name = save_pattern.format(k=t.k, b=t.b, n=t.n, c=t.c, b1=t.b1, b2=t.b2)
		linux_lines.append(to_linux_line(t, save_name, gpu_curves, gmpecm_b2))
		windows_lines.append(to_stage2_line(t, save_name, p95_b2, skip_curves, gpu_curves))
		p95_lines.append(prmers_to_p95_stage2_line(t, gpu_curves))

	out_linux.parent.mkdir(parents=True, exist_ok=True)
	out_windows.parent.mkdir(parents=True, exist_ok=True)
	out_prmers.parent.mkdir(parents=True, exist_ok=True)

	linux_payload = "\n".join(linux_lines) + ("\n" if linux_lines else "")
	linux_mode = "a" if append_linux else "w"
	if linux_payload:
		prefix = ""
		if append_linux and out_linux.exists() and out_linux.stat().st_size > 0:
			with out_linux.open("rb") as existing:
				existing.seek(-1, 2)
				if existing.read(1) != b"\n":
					prefix = "\n"
		with out_linux.open(linux_mode, encoding="utf-8", newline="") as f:
			f.write(prefix + linux_payload)

	windows_payload = "\n".join(windows_lines) + ("\n" if windows_lines else "")
	windows_mode = "a" if append_windows else "w"
	if windows_payload:
		prefix = ""
		if append_windows and out_windows.exists() and out_windows.stat().st_size > 0:
			with out_windows.open("rb") as existing:
				existing.seek(-1, 2)
				if existing.read(1) != b"\n":
					prefix = "\n"
		with out_windows.open(windows_mode, encoding="utf-8", newline="") as f:
			f.write(prefix + windows_payload)

	p95_payload = "\n".join(p95_lines) + ("\n" if p95_lines else "")
	p95_mode = "a" if append_prmers else "w"
	if p95_payload:
		prefix = ""
		if append_prmers and out_prmers.exists() and out_prmers.stat().st_size > 0:
			with out_prmers.open("rb") as existing:
				existing.seek(-1, 2)
				if existing.read(1) != b"\n":
					prefix = "\n"
		with out_prmers.open(p95_mode, encoding="utf-8", newline="") as f:
			f.write(prefix + p95_payload)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="ECM worktodo 两阶段管线生成器")
	parser.add_argument("--input", required=True, help="输入 assignment.csv / 文本文件")
	parser.add_argument("--out-linux", default="worktodo.sh", help="Linux 阶段输出脚本")
	parser.add_argument("--out-windows", default="worktodo.csv", help="Windows 阶段输出 csv")
	parser.add_argument("--out-prmers", default="worktodo_p95_stage2.txt", help="P95 阶段输出文本")

	parser.add_argument("--set-b1", help="重写 B1")
	parser.add_argument("--set-b2", help="重写 B2")
	parser.add_argument("--set-has-na", action="store_true", help="缺失 AID 时设为 N/A")

	parser.add_argument("--min-n", type=int, help="仅保留 n >= min_n")
	parser.add_argument("--max-n", type=int, help="仅保留 n <= max_n")
	parser.add_argument("--min-curves", type=int, help="仅保留 curves_to_run >= min_curves")
	parser.add_argument("--max-curves", type=int, help="仅保留 curves_to_run <= max_curves")

	parser.add_argument(
		"--sort-by",
		default="",
		choices=["", "aid", "k", "b", "n", "c", "b1", "b2", "curves"],
		help="排序字段",
	)
	parser.add_argument("--desc", action="store_true", help="降序")

	parser.add_argument("--save-pattern", default="m{n}_{b1}.save", help="保存文件名模板")
	parser.add_argument("--gpu-curves", type=int, default=192, help="Linux -gpucurves 参数")
	parser.add_argument("--gmpecm-b2", default="0", help="Linux 阶段 ECM 命令中的 B2 参数")
	parser.add_argument("--p95-b2", default="0", help="Windows ECMSTAGE2 的 B2-or-zero")
	parser.add_argument("--skip-curves", type=int, default=0, help="Windows ECMSTAGE2 skip_curves")
	parser.add_argument("--append-linux", action="store_true", help="Linux 输出文件追加写入")
	parser.add_argument("--append-windows", action="store_true", help="Windows 输出文件追加写入")
	parser.add_argument("--append-prmers", action="store_true", help="PrMers 输出文件追加写入")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	tasks = read_tasks(Path(args.input))
	tasks = apply_processing(
		tasks,
		set_b1=args.set_b1,
		set_b2=args.set_b2,
		force_has_na=args.set_has_na,
		min_n=args.min_n,
		max_n=args.max_n,
		min_curves=args.min_curves,
		max_curves=args.max_curves,
	)
	tasks = sort_tasks(tasks, args.sort_by, args.desc)

	save_outputs(
		tasks,
		out_linux=Path(args.out_linux),
		out_windows=Path(args.out_windows),
		out_prmers=Path(args.out_prmers),
		save_pattern=args.save_pattern,
		gpu_curves=args.gpu_curves,
		gmpecm_b2=args.gmpecm_b2,
		p95_b2=args.p95_b2,
		skip_curves=args.skip_curves,
		append_linux=args.append_linux,
		append_windows=args.append_windows,
		append_prmers=args.append_prmers,
	)

	print(f"输入任务: {len(tasks)}")
	print(f"Linux 输出: {args.out_linux}")
	print(f"Windows 输出: {args.out_windows}")
	print(f"PrMers 输出: {args.out_prmers}")


if __name__ == "__main__":
	main()

