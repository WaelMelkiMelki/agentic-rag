import pymupdf4llm
from pathlib import Path

def parse_pdf(pdf_path: str) -> str:
    """Convert a PDF to Markdown text."""
    return pymupdf4llm.to_markdown(
        pdf_path,
        ignore_graphics=True,
        show_progress=False,
    )

def parse_all(raw_dir: str, output_dir: str):
    raw = Path(raw_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for pdf in sorted(raw.glob("*.pdf")):
        print(f"📄 Parsing {pdf.name}...")
        text = parse_pdf(str(pdf))
        out_path = out / f"{pdf.stem}.md"
        out_path.write_text(text, encoding="utf-8")
        print(f"   → {len(text)} chars saved to {out_path.name}")

if __name__ == "__main__":
    parse_all("data/raw", "data/processed")