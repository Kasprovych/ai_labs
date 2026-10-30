"""Builds printable versions of the lab reports: DOCX (pandoc) and PDF (pandoc HTML -> headless Chrome).

lab1: REPORT_LAB1.md. The Markdown title page uses HTML tags for GitHub rendering. DOCX ignores raw
      HTML, so for DOCX the title block is rewritten into pandoc custom-style divs plus a page break.
lab2: REPORT_LAB2.md, formatted like the submitted Lab 1 report (Lviv Polytechnic style). The source
      uses pandoc custom-style divs directly; DOCX takes its formatting from scripts/reference_lpnu.docx
      (see make_reference_docx.py), PDF from scripts/report_lpnu.css. `<!-- pagebreak -->` marks a page break.

Usage: python scripts/build_report.py [lab1|lab2]   (default: lab1)
Requires: pandoc; Google Chrome (for PDF); python-docx (lab2 DOCX post-processing).
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "report"
SCRIPTS = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

REPORTS = {
    "lab1": {"source": ROOT / "REPORT_LAB1.md", "css": SCRIPTS / "report.css", "title": "Звіт ЛР1"},
    "lab2": {"source": ROOT / "REPORT_LAB2.md", "css": SCRIPTS / "report_lpnu.css", "title": "Звіт ЛР2",
             "reference_docx": SCRIPTS / "reference_lpnu.docx", "pandoc_args": ["--syntax-highlighting=none"]},
}

PAGE_BREAK_DOCX = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'
PAGE_BREAK_HTML = '```{=html}\n<div class="page-break"></div>\n```'
PAGE_BREAK_MARK = "<!-- pagebreak -->"


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


def pandoc_from_text(markdown: str, args: list[str]) -> None:
    """Runs pandoc on an in-memory Markdown string (temp file in ROOT keeps relative image paths)."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", dir=ROOT, delete=False, encoding="utf-8") as tmp:
        tmp.write(markdown)
    try:
        run(["pandoc", tmp.name, "-f", "markdown", "--resource-path", str(ROOT),
             "--metadata", "lang=uk-UA", *args])
    finally:
        Path(tmp.name).unlink()


def polish_lpnu_docx(path: Path) -> None:
    """Table cells in 'Table Text' style with a bold header row; tables span the text width."""
    from docx import Document
    from docx.shared import Pt

    doc = Document(str(path))
    for table in doc.tables:
        table.autofit = True
        for row_idx, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.style = doc.styles["Table Text"]
                    for run_ in paragraph.runs:
                        run_.font.size = Pt(11)
                        if row_idx == 0:
                            run_.bold = True
        tbl_pr = table._tbl.tblPr
        width = tbl_pr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblW")
        if width is not None:
            width.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type", "pct")
            width.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w", "5000")
    doc.save(str(path))


def build_docx(lab: str, markdown: str) -> Path:
    spec = REPORTS[lab]
    out = OUT_DIR / f"REPORT_{lab.upper()}.docx"
    if lab == "lab1":
        title_html, body = split_title_block(markdown)
        pandoc_from_text(docx_title_page(title_lines(title_html)) + body, ["-o", str(out)])
        return out
    pandoc_from_text(markdown.replace(PAGE_BREAK_MARK, PAGE_BREAK_DOCX),
                     ["-o", str(out), "--reference-doc", str(spec["reference_docx"]), "--syntax-highlighting=none"])
    polish_lpnu_docx(out)
    return out


def build_pdf(lab: str, markdown: str) -> Path | None:
    if not Path(CHROME).exists():
        print("Google Chrome not found, skipping PDF", file=sys.stderr)
        return None
    spec = REPORTS[lab]
    html = OUT_DIR / f"REPORT_{lab.upper()}.html"
    pdf = OUT_DIR / f"REPORT_{lab.upper()}.pdf"
    pandoc_from_text(markdown.replace(PAGE_BREAK_MARK, PAGE_BREAK_HTML),
                     ["-s", "--embed-resources", "--css", str(spec["css"]), *spec.get("pandoc_args", []),
                      "--metadata", f"pagetitle={spec['title']}", "-o", str(html)])
    run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf}", html.as_uri()])
    html.unlink()
    return pdf


def main() -> None:
    lab = sys.argv[1] if len(sys.argv) > 1 else "lab1"
    if lab not in REPORTS:
        sys.exit(f"Unknown report '{lab}', expected one of: {', '.join(REPORTS)}")
    if not shutil.which("pandoc"):
        sys.exit("pandoc is required: brew install pandoc")
    OUT_DIR.mkdir(exist_ok=True)
    markdown = REPORTS[lab]["source"].read_text(encoding="utf-8")
    print("DOCX:", build_docx(lab, markdown))
    print("PDF: ", build_pdf(lab, markdown))


if __name__ == "__main__":
    main()
