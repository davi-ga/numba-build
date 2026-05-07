#!/usr/bin/env python3
"""
Convenience wrapper — delegates to the installed build_pipeline module.

For local development (without installing the package):
    python refactor_script.py --source-dir my_project/ --output-dir dist/

When the package is installed, prefer the `numba-build` CLI command instead.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from build_pipeline import main

if __name__ == "__main__":
    main()
