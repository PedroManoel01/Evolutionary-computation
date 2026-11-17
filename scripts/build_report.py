"""Converte docs/relatorio.md em docs/relatorio.html e docs/relatorio.pdf (via Chrome headless).

Uso:
    python scripts/build_report.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

CSS = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 980px; margin: 24px auto;
       color: #1d1d1f; line-height: 1.45; font-size: 13px; padding: 0 16px; }
h1 { color: #1f4e79; font-size: 24px; border-bottom: 3px solid #1f4e79; padding-bottom: 6px; }
h2 { color: #1f4e79; font-size: 19px; border-bottom: 1px solid #c8d3e0; margin-top: 28px; }
h3 { color: #1f4e79; font-size: 15px; margin-top: 20px; }
table { border-collapse: collapse; margin: 10px 0 16px; font-size: 11px; width: 100%; }
th { background: #1f4e79; color: white; padding: 4px 6px; text-align: left; }
td { border-bottom: 1px solid #dde3ea; padding: 3px 6px; }
tr:nth-child(even) td { background: #f5f8fb; }
code, pre { background: #eef2f6; border-radius: 3px; font-size: 11px; }
pre { padding: 8px; overflow-x: auto; }
img { max-width: 100%; display: block; margin: 8px auto; }
blockquote { border-left: 4px solid #1f4e79; background: #eef3f8; margin: 10px 0; padding: 6px 12px; }
@media print { h2 { page-break-after: avoid; } table, img { page-break-inside: avoid; } }
"""


def main() -> None:
    text = (DOCS / "relatorio.md").read_text()
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Relatório</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = DOCS / "relatorio.html"
    out_html.write_text(html)
    chrome = shutil.which("google-chrome") or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if Path(chrome).exists():
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={DOCS / 'relatorio.pdf'}", out_html.as_uri()],
                       check=True, capture_output=True)
        print("PDF:", DOCS / "relatorio.pdf")
    print("HTML:", out_html)


if __name__ == "__main__":
    main()
