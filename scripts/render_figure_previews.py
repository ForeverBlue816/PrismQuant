"""Render the checked-in author PDFs without modifying data, scales or layout.

Install: pip install -e '.[figures]'
Run: python scripts/render_figure_previews.py
"""
from pathlib import Path
import pymupdf

root = Path(__file__).resolve().parents[1] / 'Figures'
for path in sorted(root.glob('*.pdf')):
    if path.stem == 'offset_precision':
        continue  # This existing native figure has its own original exports.
    doc = pymupdf.open(path)
    if len(doc) != 1:
        raise ValueError(f'Expected a one-page figure: {path}')
    page = doc[0]
    scale = 2000 / page.rect.width
    page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(path.with_suffix('.png'))
    path.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True))
