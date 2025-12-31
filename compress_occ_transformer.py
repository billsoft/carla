#!/usr/bin/env python3
"""
代码压缩脚本 - 专用于压缩 occ_transformer 目录供 AI 分析

用法:
    python compress_occ_transformer.py
"""

import os
import argparse
from pathlib import Path
import re


def remove_comments_and_empty_lines(content, file_ext):
    """移除注释和多余空行"""
    lines = content.split('\n')
    cleaned = []
    in_multiline_comment = False

    for line in lines:
        stripped = line.strip()

        # Python 文件处理
        if file_ext == '.py':
            # 跳过多行注释
            if '"""' in stripped or "'''" in stripped:
                if in_multiline_comment:
                    in_multiline_comment = False
                    continue
                else:
                    in_multiline_comment = True
                    continue

            if in_multiline_comment:
                continue

            # 跳过单行注释（但保留 shebang）
            if stripped.startswith('#') and not stripped.startswith('#!'):
                continue

            # 移除行尾注释
            if '#' in line and not line.strip().startswith('#!'):
                code_part = line.split('#')[0].rstrip()
                if code_part:
                    cleaned.append(code_part)
            elif stripped:  # 保留非空行
                cleaned.append(line.rstrip())

        # YAML/配置文件处理
        elif file_ext in ['.yaml', '.yml', '.json']:
            if stripped.startswith('#'):
                continue
            if stripped:
                cleaned.append(line.rstrip())

        # 其他文件保持原样
        else:
            if stripped:
                cleaned.append(line.rstrip())

    # 移除连续空行
    result = []
    prev_empty = False
    for line in cleaned:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return '\n'.join(result)


def get_file_tree(root_dir, exclude_patterns=None):
    """生成文件树结构"""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.git', '.pytest_cache', '*.pyc', '*.pyo',
                           '*.egg-info', 'outputs', 'checkpoints', '*.pth', '*.ckpt', 'wandb']

    tree_lines = []
    root_path = Path(root_dir)

    def should_exclude(path):
        path_str = str(path)
        for pattern in exclude_patterns:
            if pattern in path_str or path.name.startswith('.'):
                return True
        return False

    def walk_dir(directory, prefix=''):
        try:
            entries = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return

        filtered_entries = [e for e in entries if not should_exclude(e)]
        
        for i, entry in enumerate(filtered_entries):
            is_last = i == len(filtered_entries) - 1
            current_prefix = '└── ' if is_last else '├── '
            tree_lines.append(f"{prefix}{current_prefix}{entry.name}")

            if entry.is_dir():
                extension = '    ' if is_last else '│   '
                walk_dir(entry, prefix + extension)

    tree_lines.append(root_path.name + '/')
    walk_dir(root_path)

    return '\n'.join(tree_lines)


def collect_code_files(root_dir, include_exts=None):
    """收集代码文件"""
    if include_exts is None:
        include_exts = ['.py', '.yaml', '.yml', '.json', '.md']

    exclude_patterns = ['__pycache__', '.git', '.pytest_cache', 'outputs',
                       'checkpoints', '.egg-info', 'wandb']
    exclude_files = [] # 可以添加要排除的具体文件名

    files_content = []
    root_path = Path(root_dir)

    for file_path in sorted(root_path.rglob('*')):
        # 跳过目录
        if file_path.is_dir():
            continue

        # 跳过排除的路径
        if any(pattern in str(file_path) for pattern in exclude_patterns):
            continue

        # 跳过排除的文件
        if file_path.name in exclude_files:
            continue

        # 只处理指定扩展名
        if file_path.suffix not in include_exts:
            continue

        # 跳过大文件 (>200KB)
        if file_path.stat().st_size > 200 * 1024:
            continue

        # 读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue

        # 压缩内容
        compressed = remove_comments_and_empty_lines(content, file_path.suffix)

        # 计算相对路径
        rel_path = file_path.relative_to(root_path)

        files_content.append({
            'path': str(rel_path),
            'content': compressed,
            'lines': len(compressed.split('\n'))
        })

    return files_content


def generate_summary(root_dir, output_file):
    """生成代码摘要文件"""
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"错误: 目录不存在 - {root_dir}")
        return

    print(f"正在分析目录: {root_dir}")

    # 1. 生成文件树
    print("  [1/3] 生成文件树...")
    file_tree = get_file_tree(root_dir)

    # 2. 收集代码文件
    print("  [2/3] 收集代码文件...")
    files = collect_code_files(root_dir)

    # 3. 生成输出
    print("  [3/3] 生成摘要文件...")

    output = []
    output.append("=" * 80)
    output.append(f"代码摘要 - {root_path.name}")
    output.append("=" * 80)
    output.append("")

    # 统计信息
    total_lines = sum(f['lines'] for f in files)
    output.append(f"统计信息:")
    output.append(f"  文件数: {len(files)}")
    output.append(f"  总行数: {total_lines}")
    output.append("")

    # 文件树
    output.append("=" * 80)
    output.append("文件结构")
    output.append("=" * 80)
    output.append("")
    output.append(file_tree)
    output.append("")

    # 文件内容
    output.append("=" * 80)
    output.append("文件内容")
    output.append("=" * 80)
    output.append("")

    for file_info in files:
        output.append("-" * 80)
        output.append(f"文件: {file_info['path']} ({file_info['lines']} 行)")
        output.append("-" * 80)
        output.append(file_info['content'])
        output.append("")

    # 写入文件
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"\n✅ 摘要文件已生成: {output_path}")
    print(f"   文件数: {len(files)}")
    print(f"   总行数: {total_lines}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='压缩 occ_transformer 代码用于 AI 分析')
    parser.add_argument('--dir', default='occ_transformer',
                       help='要分析的目录 (默认: occ_transformer)')
    parser.add_argument('--output', default='occ_transformer_code_summary.txt',
                       help='输出文件名 (默认: occ_transformer_code_summary.txt)')

    args = parser.parse_args()

    # 构建完整路径
    # 假设脚本在 d:\code\carla\ 下运行
    script_dir = Path(__file__).parent
    target_dir = script_dir / args.dir
    output_file = script_dir / args.output

    print("\n" + "=" * 80)
    print("代码压缩脚本 - 用于 AI 分析".center(80))
    print("=" * 80)
    print(f"\n目标目录: {target_dir}")
    print(f"输出文件: {output_file}\n")

    generate_summary(target_dir, output_file)

    print("\n" + "=" * 80)
    print("完成！".center(80))
    print("=" * 80)
    print(f"\n可以复制 {args.output} 的内容给其他 AI 分析\n")


if __name__ == '__main__':
    main()
