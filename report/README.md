# IEEE Conference Report: Compilation Guide

This directory contains the IEEE-style research report for the EE 305 Electromagnetic Waves Project (Theme 4): *Machine-Learning-Based Transmission Line Fault and Load Identification via Time-Domain Reflectometry*.

## Overleaf Compilation Instructions
1. Compress or upload the entire `report/` folder to Overleaf:
   - `report/main.tex` (Main entry document)
   - `report/sections/` (All manuscript sections)
   - `report/figures/` (Vector PDF figures)
   - `report/tables/` (Auto-generated LaTeX tables)
   - `report/refs.bib` (BibTeX bibliography)
2. Ensure the compiler is set to **pdfLaTeX** (default) with TeX Live 2022 or newer.
3. Set `main.tex` as the root document.
4. Click **Recompile**. All tables and vector PDF figures are pre-synchronized and will compile with zero missing inputs or undefined references.

## Local LaTeX Compilation
If you have a local TeX distribution (`pdflatex` + `bibtex`):
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
The output PDF will be generated at `report/main.pdf`.
