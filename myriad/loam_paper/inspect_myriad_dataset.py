#!/usr/bin/env python3
"""
inspect_myriad_dataset.py

Usage:
  python inspect_myriad_dataset.py "C:\\path\\to\\code\\myriad\\loam_paper\\dataset"

This script walks the dataset directory, examines JSON files, and reports
whether they contain likely commit-level file modification information
(e.g., fields named "files", "filename", or string values that look like
file paths with extensions such as .js, .py, .cpp, .md, or 'test' directories).
"""

import sys
import os
import json
import re
from pathlib import Path

FILE_EXT_RE = re.compile(r'\b[\w\-/\\]+\.('
                         r'js|py|cpp|c|h|java|ts|rb|go|rs|php|scss|css|html|md|rst|txt'
                         r')\b', re.IGNORECASE)
COMMIT_KEYS = {'files', 'file', 'filename', 'filenames', 'files_changed', 'changed_files', 'patch', 'paths', 'modified_files', 'files_modified', 'files_removed'}

def inspect_json_file(path: Path, max_bytes=10_000_000):
    report = {
        'path': str(path),
        'size_bytes': path.stat().st_size,
        'loaded': False,
        'top_type': None,
        'sample_item_keys': set(),
        'has_commit_keys': False,
        'has_file_like_strings': False,
        'file_like_examples': []
    }

    try:
        # try to load (may be large)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read(max_bytes)
            # If file is larger than max_bytes, we'll still analyze the truncated text
            try:
                data = json.loads(raw)
                report['loaded'] = True
            except Exception:
                # fallback: not valid JSON in first chunk, but still scan raw text
                data = None

    except Exception as e:
        report['error'] = str(e)
        return report

    # If we have parsed JSON structure, inspect first few items
    def analyze_obj(obj):
        if isinstance(obj, dict):
            report['top_type'] = 'object'
            report['sample_item_keys'].update(obj.keys())
            # check keys for commit-file indicators
            if any(k.lower() in COMMIT_KEYS for k in obj.keys()):
                report['has_commit_keys'] = True
            # scan values for file-like strings
            for v in obj.values():
                if isinstance(v, str):
                    if FILE_EXT_RE.search(v):
                        report['has_file_like_strings'] = True
                        report['file_like_examples'].append(v)
                elif isinstance(v, (list, tuple)):
                    for el in v[:10]:
                        if isinstance(el, str) and FILE_EXT_RE.search(el):
                            report['has_file_like_strings'] = True
                            report['file_like_examples'].append(el)
        elif isinstance(obj, list):
            report['top_type'] = 'array'
            # sample up to 5 items
            for item in obj[:5]:
                if isinstance(item, dict):
                    report['sample_item_keys'].update(item.keys())
                    if any(k.lower() in COMMIT_KEYS for k in item.keys()):
                        report['has_commit_keys'] = True
                    for v in item.values():
                        if isinstance(v, str) and FILE_EXT_RE.search(v):
                            report['has_file_like_strings'] = True
                            report['file_like_examples'].append(v)
                        elif isinstance(v, (list, tuple)):
                            for el in v[:10]:
                                if isinstance(el, str) and FILE_EXT_RE.search(el):
                                    report['has_file_like_strings'] = True
                                    report['file_like_examples'].append(el)
                elif isinstance(item, str):
                    if FILE_EXT_RE.search(item):
                        report['has_file_like_strings'] = True
                        report['file_like_examples'].append(item)
        else:
            report['top_type'] = type(obj).__name__

    if report['loaded'] and data is not None:
        analyze_obj(data)
    else:
        # parse the truncated raw text for likely keys / file-like strings
        raw_lower = raw.lower()
        if any(k in raw_lower for k in ('"files"', '"filename"', '"files_changed"', '"patch"', '"modified_files"')):
            report['has_commit_keys'] = True
        m = FILE_EXT_RE.search(raw)
        if m:
            report['has_file_like_strings'] = True
            report['file_like_examples'].append(m.group(0))

    # limit examples
    report['file_like_examples'] = report['file_like_examples'][:10]
    return report

def main(dataset_dir):
    p = Path(dataset_dir)
    if not p.exists():
        print(f"Path not found: {p}")
        return 2

    json_files = list(p.rglob('*.json'))
    if not json_files:
        print("No JSON files found under the dataset path.")
        return 1

    print(f"Found {len(json_files)} JSON files. Scanning for commit-level file info...\n")

    positive = []
    for jf in json_files:
        r = inspect_json_file(jf)
        # print a compact row if something looks promising
        if r.get('has_commit_keys') or r.get('has_file_like_strings'):
            positive.append(r)
        # compact debug: print the files that clearly match the dataset filenames you're likely to check
        print(f"- {jf} | size={r['size_bytes']:,} bytes | type={r['top_type']} | commit_keys={r['has_commit_keys']} | file_strings={r['has_file_like_strings']}")

    print("\nSummary: files that appear to contain file modification info or file-like strings:")
    for r in positive:
        print(f"\nFile: {r['path']}")
        print(f"  size: {r['size_bytes']:,} bytes")
        print(f"  loaded: {r['loaded']}")
        print(f"  top_type: {r['top_type']}")
        if r['sample_item_keys']:
            print(f"  sample keys: {sorted(list(r['sample_item_keys']))[:10]}")
        print(f"  has_commit_keys: {r['has_commit_keys']}")
        print(f"  has_file_like_strings: {r['has_file_like_strings']}")
        if r['file_like_examples']:
            print(f"  file-like examples: {r['file_like_examples'][:5]}")

    if not positive:
        print("\nNo JSON files clearly contain commit-level file lists or file-like strings.")
        print("If this is the case, you likely only have aggregate contribution counts (per repo),")
        print("which are insufficient to infer task-based roles from file modification patterns.")

    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_myriad_dataset.py <dataset_dir>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    sys.exit(main(dataset_dir))