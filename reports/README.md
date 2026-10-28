# Reports

| File | What it is |
|---|---|
| `ParkSmart_Report.docx` | Final report for submission (generated from the Markdown below with pandoc; add your name and PRN on the title page) |
| `ParkSmart_Report.md` | Same report in Markdown (renders on GitHub) |
| `presentation_script.md` | 10-slide outline with timings, speaker script, ParkSmart 2.0 Pune slides and demo, backup plan |
| `viva_preparation.md` | 50 viva questions with answers (including ParkSmart 2.0 / Pune), plus hard follow-ups |
| `screenshots/` | Real screenshots of all seven pages in light and dark mode (`python scripts/capture_screenshots.py --plain-map` with the app running) |

Regenerate the Word file after editing the Markdown:

```bash
cd reports && pandoc ParkSmart_Report.md -o ParkSmart_Report.docx --resource-path=.:.. && python3 -c "from docx import Document; from docx.shared import Mm; d=Document('ParkSmart_Report.docx'); [setattr(s,'page_width',Mm(210)) or setattr(s,'page_height',Mm(297)) for s in d.sections]; d.save('ParkSmart_Report.docx')"
```
