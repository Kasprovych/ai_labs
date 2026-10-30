"""Generates scripts/reference_lpnu.docx: the pandoc reference document for the Lab 2 report.

Reproduces the formatting of the submitted Lab 1 report (Lviv Polytechnic style): A4, Times New Roman
14 pt, 1.5 line spacing, justified text with 1.25 cm first-line indent, centered upper-case section
headings, right-aligned table captions above tables, centered figure captions, bordered tables,
shaded Courier New code blocks and page numbers bottom-right (not on the title page).

Usage: python scripts/make_reference_docx.py   (requires pandoc and python-docx)
"""

import subprocess
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(__file__).resolve().parent / "reference_lpnu.docx"
FONT = "Times New Roman"
INDENT = Cm(1.25)


def set_font(style, size: float = 14, bold: bool | None = None, italic: bool | None = None,
             name: str = FONT) -> None:
    font = style.font
    font.name = name
    font.size = Pt(size)
    font.color.rgb = RGBColor(0, 0, 0)
    font.bold = bold
    font.italic = italic
    # East-Asian / complex-script slots too, otherwise Word may fall back to the theme font
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(attr), None)


def set_paragraph(style, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line=INDENT, spacing: float = 1.5,
                  before: float = 0, after: float = 0, left=None, keep_next: bool = False) -> None:
    fmt = style.paragraph_format
    fmt.alignment = align
    fmt.first_line_indent = first_line
    fmt.left_indent = left
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = spacing
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.keep_with_next = keep_next


def get_style(doc, name: str, kind=WD_STYLE_TYPE.PARAGRAPH):
    try:
        return doc.styles[name]
    except KeyError:
        style = doc.styles.add_style(name, kind)
        style.base_style = doc.styles["Normal"] if kind == WD_STYLE_TYPE.PARAGRAPH else None
        style.quick_style = True
        return style


def shade(style, fill: str) -> None:
    ppr = style.element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def table_borders(style) -> None:
    """Single black borders on every cell edge of pandoc's 'Table' style."""
    tbl_pr = style.element.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        style.element.append(tbl_pr)
    old = tbl_pr.find(qn("w:tblBorders"))
    if old is not None:
        tbl_pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tbl_pr.append(borders)


def page_number_footer(section) -> None:
    """Right-aligned PAGE field in the footer; the title page (first page) has none."""
    section.different_first_page_header_footer = True
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.paragraph_format.first_line_indent = Cm(0)
    run = paragraph.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    run.font.name = FONT
    run.font.size = Pt(12)


def main() -> None:
    subprocess.run(["pandoc", "-o", str(OUT), "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(str(OUT))

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(2), Cm(2)
    section.left_margin, section.right_margin = Cm(2.5), Cm(1.5)
    page_number_footer(section)

    for name in ("Normal", "Body Text", "First Paragraph"):
        style = get_style(doc, name)
        set_font(style)
        set_paragraph(style)
    compact = get_style(doc, "Compact")
    set_font(compact)
    set_paragraph(compact, first_line=Cm(0))

    for name in ("Title", "Subtitle", "Author", "Date", "Abstract", "Block Text"):
        set_font(get_style(doc, name))

    h1 = get_style(doc, "Heading 1")
    set_font(h1, bold=True)
    h1.font.all_caps = True
    set_paragraph(h1, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=Cm(0), before=12, after=6, keep_next=True)
    h2 = get_style(doc, "Heading 2")
    set_font(h2, bold=True)
    set_paragraph(h2, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=6, keep_next=True)
    h3 = get_style(doc, "Heading 3")
    set_font(h3, bold=True, italic=False)
    set_paragraph(h3, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=0, keep_next=True)

    caption = get_style(doc, "Caption")
    set_font(caption, italic=False)
    set_paragraph(caption, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=Cm(0), spacing=1.15, after=6)
    table_caption = get_style(doc, "Table Caption")
    set_font(table_caption, italic=False)
    set_paragraph(table_caption, align=WD_ALIGN_PARAGRAPH.RIGHT, first_line=Cm(0), spacing=1.15,
                  before=6, keep_next=True)
    image_caption = get_style(doc, "Image Caption")
    set_font(image_caption, italic=False)
    set_paragraph(image_caption, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=Cm(0), spacing=1.15, after=8)
    for name in ("Figure", "Captioned Figure"):
        style = get_style(doc, name)
        set_paragraph(style, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=Cm(0), spacing=1.0, keep_next=True)

    source = get_style(doc, "Source Code")
    set_font(source, size=9, name="Courier New")
    set_paragraph(source, align=WD_ALIGN_PARAGRAPH.LEFT, first_line=Cm(0), spacing=1.0, before=3, after=3)
    shade(source, "F2F2F2")
    verbatim = get_style(doc, "Verbatim Char", WD_STYLE_TYPE.CHARACTER)
    set_font(verbatim, size=11, name="Courier New")

    table_borders(get_style(doc, "Table", WD_STYLE_TYPE.TABLE))

    # Title page styles used via ::: {custom-style="..."} in the Markdown source
    for name, align, left in [("Title Header", WD_ALIGN_PARAGRAPH.CENTER, None),
                              ("Title Right", WD_ALIGN_PARAGRAPH.RIGHT, None),
                              ("Title Center", WD_ALIGN_PARAGRAPH.CENTER, None),
                              ("Title Signature", WD_ALIGN_PARAGRAPH.LEFT, Cm(10))]:
        style = get_style(doc, name)
        set_font(style)
        set_paragraph(style, align=align, first_line=Cm(0), spacing=1.5, left=left)
    # Table cell text: smaller, no indent (pandoc puts cell paragraphs in 'Compact')
    table_text = get_style(doc, "Table Text")
    set_font(table_text, size=11)
    set_paragraph(table_text, align=WD_ALIGN_PARAGRAPH.LEFT, first_line=Cm(0), spacing=1.0)

    doc.save(str(OUT))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
