#!/usr/bin/env python3
"""
Flask Endpoint Conflict Auto-Fixer

Fixes the "View function mapping is overwriting an existing endpoint function" error
by finding duplicate function names across Flask route files and renaming them.

Usage:
    python flask_endpoint_fixer.py /path/to/your/app/

Or import and use:
    from flask_endpoint_fixer import fix_duplicate_endpoint
    fix_duplicate_endpoint("list_datasets", "/path/to/app/")
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# Pre-compiled patterns
_ROUTE_DECORATOR = re.compile(r'@\w+\.route\s*\([^)]+\)')
_FUNCTION_DEF = re.compile(r'^(\s*)def\s+(\w+)\s*\(')
_ENDPOINT_ERROR = re.compile(r'endpoint function:\s*(\w+)')


def find_duplicate_endpoints(directory: str, endpoint_name: str) -> List[Tuple[str, int, str]]:
    """
    Find all occurrences of a function name that's registered as a Flask route.

    Returns list of (filepath, line_number, full_line) tuples.
    """
    matches = []

    for root, dirs, files in os.walk(directory):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'venv', '.venv']]

        for filename in files:
            if not filename.endswith('.py'):
                continue

            filepath = os.path.join(root, filename)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except Exception:
                continue

            # Look for the function definition
            in_route_context = False

            for i, line in enumerate(lines):
                # Check if previous line(s) have a route decorator
                if i > 0:
                    # Look back up to 5 lines for a route decorator
                    for j in range(max(0, i-5), i):
                        if _ROUTE_DECORATOR.search(lines[j]):
                            in_route_context = True
                            break
                    else:
                        in_route_context = False

                # Check for function definition matching our endpoint name
                match = _FUNCTION_DEF.match(line)
                if match and match.group(2) == endpoint_name:
                    if in_route_context:
                        matches.append((filepath, i + 1, line.rstrip()))

    return matches


def generate_unique_name(original_name: str, filepath: str) -> str:
    """Generate a unique function name based on the file it's in."""

    # Extract meaningful prefix from filepath
    basename = os.path.basename(filepath).replace('.py', '')

    # Common prefixes to try
    prefixes = [
        basename.replace('_', ''),  # pi_dataset_registry -> pidatasetregistry
        basename.split('_')[0],     # pi_dataset_registry -> pi
        basename,                    # pi_dataset_registry
    ]

    # Create candidates
    for prefix in prefixes:
        # Clean up prefix
        prefix = re.sub(r'[^a-zA-Z0-9]', '_', prefix)
        if prefix and not prefix[0].isdigit():
            candidate = f"{prefix}_{original_name}"
            return candidate

    # Fallback: just append a number
    return f"{original_name}_alt"


def fix_duplicate_endpoint(endpoint_name: str, directory: str, dry_run: bool = False) -> bool:
    """
    Fix a duplicate Flask endpoint by renaming one of the functions.

    Args:
        endpoint_name: The duplicate endpoint function name
        directory: Directory to search in
        dry_run: If True, just print what would be done

    Returns:
        True if fix was applied, False otherwise
    """
    print(f"[FLASK-FIXER] Looking for duplicate endpoint: {endpoint_name}")

    matches = find_duplicate_endpoints(directory, endpoint_name)

    if len(matches) < 2:
        print(f"[FLASK-FIXER] Found {len(matches)} occurrence(s) - no duplicate to fix")
        return False

    print(f"[FLASK-FIXER] Found {len(matches)} occurrences:")
    for filepath, line_num, line in matches:
        print(f"  {filepath}:{line_num}")
        print(f"    {line.strip()}")

    # Decide which one to rename (not the one in app.py typically)
    # Prefer renaming files that aren't the main app
    to_rename = None
    for filepath, line_num, line in matches:
        basename = os.path.basename(filepath)
        if basename != 'app.py':
            to_rename = (filepath, line_num, line)
            break

    # Fallback to last one if all are app.py (unlikely)
    if not to_rename:
        to_rename = matches[-1]

    filepath, line_num, line = to_rename
    new_name = generate_unique_name(endpoint_name, filepath)

    print(f"[FLASK-FIXER] Will rename in {os.path.basename(filepath)}:")
    print(f"  {endpoint_name} -> {new_name}")

    if dry_run:
        print("[FLASK-FIXER] Dry run - no changes made")
        return True

    # Apply the fix
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace the function definition
        # Be careful to only replace the def line, not calls to it
        old_def = f"def {endpoint_name}("
        new_def = f"def {new_name}("

        # Only replace if it's a function definition (has 'def ' before it)
        pattern = re.compile(r'^(\s*)def\s+' + re.escape(endpoint_name) + r'\s*\(', re.MULTILINE)

        new_content = pattern.sub(rf'\1def {new_name}(', content, count=1)

        if new_content == content:
            print(f"[FLASK-FIXER] ERROR: Could not find function to rename")
            return False

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"[FLASK-FIXER] ✓ Fixed! Renamed {endpoint_name} to {new_name} in {filepath}")
        return True

    except Exception as e:
        print(f"[FLASK-FIXER] ERROR: {e}")
        return False


def fix_from_error_message(error_message: str, directory: str, dry_run: bool = False) -> bool:
    """
    Parse a Flask AssertionError message and fix the duplicate endpoint.

    Args:
        error_message: The full error message/traceback
        directory: Directory to search in
        dry_run: If True, just print what would be done
    """
    # Extract endpoint name from error
    match = _ENDPOINT_ERROR.search(error_message)
    if not match:
        print("[FLASK-FIXER] Could not parse endpoint name from error message")
        return False

    endpoint_name = match.group(1)
    return fix_duplicate_endpoint(endpoint_name, directory, dry_run)


def scan_and_fix_all(directory: str, dry_run: bool = False) -> int:
    """
    Scan directory for potential duplicate endpoints and fix them all.

    Returns number of fixes applied.
    """
    print(f"[FLASK-FIXER] Scanning {directory} for potential duplicates...")

    # Find all route-decorated function names
    endpoint_counts = {}

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'venv', '.venv']]

        for filename in files:
            if not filename.endswith('.py'):
                continue

            filepath = os.path.join(root, filename)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except Exception:
                continue

            in_route_context = False

            for i, line in enumerate(lines):
                if _ROUTE_DECORATOR.search(line):
                    in_route_context = True
                    continue

                if in_route_context:
                    match = _FUNCTION_DEF.match(line)
                    if match:
                        func_name = match.group(2)
                        if func_name not in endpoint_counts:
                            endpoint_counts[func_name] = []
                        endpoint_counts[func_name].append((filepath, i + 1))
                        in_route_context = False

    # Find duplicates
    duplicates = {name: locs for name, locs in endpoint_counts.items() if len(locs) > 1}

    if not duplicates:
        print("[FLASK-FIXER] No duplicate endpoints found!")
        return 0

    print(f"[FLASK-FIXER] Found {len(duplicates)} duplicate endpoint(s):")
    for name, locs in duplicates.items():
        print(f"  {name}: {len(locs)} occurrences")

    fixes = 0
    for endpoint_name in duplicates:
        if fix_duplicate_endpoint(endpoint_name, directory, dry_run):
            fixes += 1

    return fixes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python flask_endpoint_fixer.py ./App/")
        print("  python flask_endpoint_fixer.py ./App/ --scan")
        print("  python flask_endpoint_fixer.py ./App/ --fix list_datasets")
        print("  python flask_endpoint_fixer.py ./App/ --dry-run")
        sys.exit(1)

    directory = sys.argv[1]
    dry_run = '--dry-run' in sys.argv

    if not os.path.isdir(directory):
        print(f"[ERROR] Not a directory: {directory}")
        sys.exit(1)

    if '--scan' in sys.argv:
        fixes = scan_and_fix_all(directory, dry_run)
        print(f"\n[COMPLETE] Applied {fixes} fix(es)")

    elif '--fix' in sys.argv:
        try:
            idx = sys.argv.index('--fix')
            endpoint_name = sys.argv[idx + 1]
            fix_duplicate_endpoint(endpoint_name, directory, dry_run)
        except (IndexError, ValueError):
            print("[ERROR] --fix requires an endpoint name")
            sys.exit(1)

    else:
        # Default: scan and fix all
        fixes = scan_and_fix_all(directory, dry_run)
        print(f"\n[COMPLETE] Applied {fixes} fix(es)")


if __name__ == "__main__":
    main()
