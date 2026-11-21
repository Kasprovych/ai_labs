"""Renders a fragment of a real run log as a terminal-window PNG (for the "screenshots" report section).

The text is taken verbatim from artifacts/labN/run_output.txt, which main_labN.py writes during an
actual run; nothing is typed by hand. Rendering: HTML -> headless Google Chrome screenshot.

Usage:
    python scripts/render_terminal.py LOG PNG --start "text of the first line" [--lines 40]
                                      [--command "python main_lab3.py"] [--max-width 150]
"""

import argparse
import html
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT_SIZE_PX = 13
LINE_HEIGHT_PX = 18
CHAR_WIDTH_PX = 7.83  # Menlo at 13 px
PADDING_PX = 14
TITLE_BAR_PX = 30

PAGE_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; background: #1e1e1e; }}
.bar {{ height: {bar}px; background: #3a3a3a; color: #d0d0d0; font: 13px -apple-system, Helvetica, sans-serif;
        display: flex; align-items: center; padding-left: 12px; }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; display: inline-block; }}
pre {{ margin: 0; padding: {pad}px; color: #f2f2f2; font: {font}px/{line}px Menlo, Monaco, monospace;
       white-space: pre; }}
</style></head><body>
<div class="bar"><span class="dot" style="background:#ff5f56"></span><span class="dot" style="background:#ffbd2e">
</span><span class="dot" style="background:#27c93f"></span><span style="margin-left:10px">{title}</span></div>
<pre>{body}</pre></body></html>"""


def select_lines(log_path: Path, start: str, count: int, max_width: int) -> list[str]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    first = next((index for index, line in enumerate(lines) if start in line), None)
    if first is None:
        raise SystemExit(f"Start text not found in {log_path}: {start!r}")
    # Over-long lines are cut with an ellipsis so that the image keeps a readable width
    return [line if len(line) <= max_width else line[:max_width - 1] + "…" for line in lines[first:first + count]]


def render(lines: list[str], png_path: Path, title: str) -> None:
    width = int(max(len(line) for line in lines) * CHAR_WIDTH_PX) + 2 * PADDING_PX + 8
    height = len(lines) * LINE_HEIGHT_PX + 2 * PADDING_PX + TITLE_BAR_PX
    page = PAGE_TEMPLATE.format(bar=TITLE_BAR_PX, pad=PADDING_PX, font=FONT_SIZE_PX, line=LINE_HEIGHT_PX,
                                title=html.escape(title), body=html.escape("\n".join(lines)))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as page_file:
        page_file.write(page)
    try:
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={width},{height}", f"--screenshot={png_path}",
                        Path(page_file.name).as_uri()], check=True, capture_output=True)
    finally:
        Path(page_file.name).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("--start", required=True, help="text contained in the first line to show")
    parser.add_argument("--lines", type=int, default=40)
    parser.add_argument("--command", default="", help="command shown in the window title")
    parser.add_argument("--max-width", type=int, default=150, help="maximum characters per line")
    args = parser.parse_args()
    lines = select_lines(args.log, args.start, args.lines, args.max_width)
    render(lines, args.png, f"ai_labs — {args.command}" if args.command else "ai_labs")
    print(f"{args.png}: {len(lines)} lines")


if __name__ == "__main__":
    main()
