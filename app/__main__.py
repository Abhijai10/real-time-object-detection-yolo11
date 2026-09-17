"""Entry point that makes the project runnable with ``python -m app``.

The module simply delegates to :func:`app.cli.main` and forwards its exit code
to the operating system, so shell scripts can detect failures.
"""

from __future__ import annotations

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
