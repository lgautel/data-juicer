#!/usr/bin/env python3
"""解析 LeRobot 数据集 parquet 文件的元信息与指定行内容.

数据目录示例:
  /mnt/r/DATA/pre_train_v1/post_train/stack_bowls_three/data/chunk-000/

用法:
  # 只看元信息(schema / 行数 / row group / 文件级 KV 元数据)
  /mnt/r/VENV/dj/bin/python inspect_parquet.py --file episode_000000.parquet

  # 用纯数字索引指定 episode
  /mnt/r/VENV/dj/bin/python inspect_parquet.py --file 0

  # 查看第 10 行
  /mnt/r/VENV/dj/bin/python inspect_parquet.py --file 0 --row 10

  # 查看多行/区间(0,5,10 或 0-9)
  /mnt/r/VENV/dj/bin/python inspect_parquet.py --file 0 --rows 0-4

  # 列出列的每元素长度(对 list 列很有用), 并完整打印数组
  /mnt/r/VENV/dj/bin/python inspect_parquet.py --file 0 --row 0 --full

  # 读取 task_index 并去 meta/tasks.jsonl 查出该 episode 的 task prompt
  /mnt/r/VENV/dj/bin/python inspect_parquet.py --file 0 --show-task
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_DATA_ROOT = Path(
    "/mnt/r/DATA/pre_train_v1/post_train/stack_bowls_three/data/chunk-000"
)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="解析 parquet 文件的元信息与指定行内容"
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="chunk-000 目录(默认 stack_bowls_three/data/chunk-000)",
    )
    p.add_argument(
        "--file",
        default="episode_000000.parquet",
        help="parquet 文件名, 或纯数字索引如 0 / 3",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--row", type=int, default=None, help="查看单行行号(从 0 开始)")
    g.add_argument(
        "--rows",
        default=None,
        help="查看多行: 逗号列表(0,5,10) 或区间(0-9)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="完整打印数组内容(默认对长 list 做截断预览)",
    )
    p.add_argument(
        "--max-preview",
        type=int,
        default=8,
        help="非 --full 时, list 列预览的元素个数(默认 8)",
    )
    p.add_argument(
        "--show-task",
        action="store_true",
        help="读取 parquet 的 task_index 列, 去 meta/tasks.jsonl 查出对应 task prompt",
    )
    p.add_argument(
        "--tasks-file",
        type=Path,
        default=None,
        help="tasks.jsonl 路径(默认 <data-root>/../../meta/tasks.jsonl)",
    )
    return p.parse_args()


def resolve_path(data_root: Path, file_arg: str) -> Path:
    if file_arg.isdigit():
        name = f"episode_{int(file_arg):06d}.parquet"
    elif file_arg.endswith(".parquet"):
        name = file_arg
    else:
        name = f"{file_arg}.parquet"
    path = data_root / name
    if not path.exists():
        raise FileNotFoundError(f"找不到 parquet 文件: {path}")
    return path


def parse_rows_arg(rows: str, num_rows: int) -> list[int]:
    result: list[int] = []
    if "-" in rows and "," not in rows:
        a, b = rows.split("-", 1)
        result = list(range(int(a), int(b) + 1))
    else:
        result = [int(x) for x in rows.split(",") if x.strip() != ""]
    valid = [r for r in result if 0 <= r < num_rows]
    dropped = [r for r in result if r not in valid]
    if dropped:
        print(f"[警告] 越界行已忽略: {dropped} (有效范围 0-{num_rows - 1})")
    return valid


def resolve_tasks_file(data_root: Path, tasks_file: Path | None) -> Path:
    if tasks_file is not None:
        return tasks_file
    # data_root 形如 <dataset>/data/chunk-000, meta 与 data 同级
    return data_root.parent.parent / "meta" / "tasks.jsonl"


def load_tasks(tasks_path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks[int(obj["task_index"])] = obj["task"]
    return tasks


def print_task(path: Path, pf: pq.ParquetFile, tasks_path: Path) -> None:
    print("=" * 60)
    print("Task prompt:")
    if "task_index" not in pf.schema_arrow.names:
        print("  [警告] 该 parquet 没有 task_index 列, 无法查询 task")
        return
    if not tasks_path.exists():
        print(f"  [警告] 找不到 tasks 文件: {tasks_path}")
        return
    tasks = load_tasks(tasks_path)
    print(f"  tasks 文件: {tasks_path} (共 {len(tasks)} 条)")
    task_indices = pf.read(columns=["task_index"]).column("task_index").to_pylist()
    unique_ti = sorted(set(task_indices))
    if len(unique_ti) == 1:
        ti = unique_ti[0]
        print(f"  task_index={ti}  ->  {tasks.get(ti, '[未在 tasks.jsonl 中找到]')}")
    else:
        print(f"  该 episode 含多个 task_index: {unique_ti}")
        for ti in unique_ti:
            print(f"    task_index={ti}  ->  {tasks.get(ti, '[未在 tasks.jsonl 中找到]')}")


def print_metadata(path: Path, pf: pq.ParquetFile) -> None:
    md = pf.metadata
    size = path.stat().st_size
    print("=" * 60)
    print(f"文件: {path}")
    print(f"大小: {size} bytes ({size / 1024:.1f} KiB)")
    print(f"行数(num_rows): {md.num_rows}")
    print(f"列数(num_columns): {md.num_columns}")
    print(f"row group 数: {md.num_row_groups}")
    print(f"创建者(created_by): {md.created_by}")
    print(f"format_version: {md.format_version}")

    print("-" * 60)
    print("Schema (列名: 类型):")
    schema = pf.schema_arrow
    for field in schema:
        print(f"  - {field.name}: {field.type}")

    # 文件级 key-value 元数据(LeRobot 有时会写入)
    kv = schema.metadata
    if kv:
        print("-" * 60)
        print("文件级 KV 元数据:")
        for k, v in kv.items():
            key = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
            val = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
            try:
                val_obj = json.loads(val)
                val = json.dumps(val_obj, ensure_ascii=False)
            except (ValueError, TypeError):
                pass
            if len(val) > 500:
                val = val[:500] + f"...(共 {len(val)} 字符)"
            print(f"  [{key}] {val}")

    print("-" * 60)
    print("各 row group 行数:")
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        print(f"  row group {i}: rows={rg.num_rows} bytes={rg.total_byte_size}")


def format_value(value, full: bool, max_preview: int):
    if isinstance(value, list):
        n = len(value)
        if full or n <= max_preview:
            return f"list(len={n}) {value}"
        head = value[:max_preview]
        return f"list(len={n}) {head} ...(截断, 用 --full 查看全部)"
    return value


def print_rows(pf: pq.ParquetFile, row_indices: list[int], full: bool, max_preview: int) -> None:
    table = pf.read()
    columns = table.column_names
    py = {c: table.column(c).to_pylist() for c in columns}
    for r in row_indices:
        print("=" * 60)
        print(f"第 {r} 行:")
        for c in columns:
            val = format_value(py[c][r], full, max_preview)
            print(f"  {c}: {val}")


def main() -> None:
    args = build_args()
    path = resolve_path(args.data_root, args.file)
    pf = pq.ParquetFile(path)

    print_metadata(path, pf)

    if args.show_task:
        tasks_path = resolve_tasks_file(args.data_root, args.tasks_file)
        print_task(path, pf, tasks_path)

    num_rows = pf.metadata.num_rows
    row_indices: list[int] = []
    if args.row is not None:
        if 0 <= args.row < num_rows:
            row_indices = [args.row]
        else:
            print(f"[警告] 行号越界: {args.row} (有效范围 0-{num_rows - 1})")
    elif args.rows is not None:
        row_indices = parse_rows_arg(args.rows, num_rows)

    if row_indices:
        print_rows(pf, row_indices, args.full, args.max_preview)


if __name__ == "__main__":
    main()
