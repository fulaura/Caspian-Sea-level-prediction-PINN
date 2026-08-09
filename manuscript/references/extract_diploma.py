#!/usr/bin/env python3
"""Extract all text from the diploma PDF using pymupdf."""
import fitz  # pymupdf

pdf_path = '/opt/paper_v2/manuscript/references/memoirthesis (2).pdf'
output_path = '/opt/paper_v2/manuscript/references/diploma_extracted.txt'

doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f"PDF has {total_pages} pages")

with open(output_path, 'w', encoding='utf-8') as f:
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()
        f.write(f"\n{'='*80}\n")
        f.write(f"PAGE {page_num + 1} of {total_pages}\n")
        f.write(f"{'='*80}\n\n")
        f.write(text)
        f.write("\n")

doc.close()
print(f"Extraction complete. Saved to {output_path}")
