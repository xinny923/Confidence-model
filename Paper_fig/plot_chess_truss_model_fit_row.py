from __future__ import annotations

import sys

from manu_figure import main


if __name__ == "__main__":
    if "--include-truss-row" not in sys.argv:
        sys.argv.append("--include-truss-row")
    main()
