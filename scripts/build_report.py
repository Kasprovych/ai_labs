"""Builds printable versions of REPORT_LAB1.md: DOCX (pandoc) and PDF (pandoc HTML -> headless Chrome).

The Markdown title page uses HTML tags for GitHub rendering. DOCX ignores raw HTML, so for DOCX
the title block is rewritten into pandoc custom-style divs (centered Word styles) plus a page break.

Usage: python scripts/build_report.py
Requires: pandoc; Google Chrome (for PDF).
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "REPORT_LAB1.md"
OUT_DIR = ROOT / "report"
CSS = Path(__file__).resolve().parent / "report.css"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PAGE_BREAK_DOCX = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'


def split_title_block(markdown: str) -> tuple[str, str]:
    """Returns (title_html, body): the title page is everything before the first '---' rule."""
    title, _, body = markdown.partition("\n---\n")
    return title, body


def title_lines(title_html: str) -> list[str]:
    """Extracts visible text lines from the HTML title block, dropping tags and markdown markers."""
    lines = []
    for raw in re.split(r"<br\s*/?>|\n", title_html):
        text = re.sub(r"<[^>]+>", "", raw).strip()
        text = re.sub(r"^#+\s*", "", text)
        if text:
            lines.append(text)
    return lines


def docx_title_page(lines: list[str]) -> str:
    """Maps title lines onto centered Word styles (Subtitle / Title / Author / Date)."""
    blocks = []
    for line in lines:
        plain = line.replace("**", "")
        if plain == "ЗВІТ":
            style = "Title"
        elif plain.startswith(("Виконав", "Перевірив", "Каспрович")):
            style = "Author"
        elif "— 20" in plain:
            style = "Date"
        else:
            style = "Subtitle"
        blocks.append(f'::: {{custom-style="{style}"}}\n{line}\n:::')
    return "\n\n".join(blocks) + "\n\n" + PAGE_BREAK_DOCX + "\n"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_docx(markdown: str) -> Path:
    title_html, body = split_title_block(markdown)
    docx_md = docx_title_page(title_lines(title_html)) + body
    out = OUT_DIR / "REPORT_LAB1.docx"
    with tempfile.NamedTemporaryFile("w", suffix=".md", dir=ROOT, delete=False, encoding="utf-8") as tmp:
        tmp.write(docx_md)
    try:
        run(["pandoc", tmp.name, "-f", "markdown", "-o", str(out),
             "--resource-path", str(ROOT), "--metadata", "lang=uk-UA"])
    finally:
        Path(tmp.name).unlink()
    return out


def build_pdf() -> Path | None:
    if not Path(CHROME).exists():
        print("Google Chrome not found, skipping PDF", file=sys.stderr)
        return None
    html = OUT_DIR / "REPORT_LAB1.html"
    pdf = OUT_DIR / "REPORT_LAB1.pdf"
    run(["pandoc", str(SOURCE), "-f", "markdown", "-s", "--embed-resources",
         "--css", str(CSS), "--resource-path", str(ROOT),
         "--metadata", "pagetitle=Звіт ЛР1", "--metadata", "lang=uk-UA", "-o", str(html)])
    run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf}", html.as_uri()])
    html.unlink()
    return pdf


def main() -> None:
    if not shutil.which("pandoc"):
        sys.exit("pandoc is required: brew install pandoc")
    OUT_DIR.mkdir(exist_ok=True)
    markdown = SOURCE.read_text(encoding="utf-8")
    print("DOCX:", build_docx(markdown))
    print("PDF: ", build_pdf())


if __name__ == "__main__":
    main()
