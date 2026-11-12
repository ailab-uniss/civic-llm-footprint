#!/bin/bash
# Regenerate numbers, tables and figures from the local results, refresh the generated blocks of paper/main.tex
# (between "% >>> name" and "% <<< name") and compile. The rest of main.tex is edited by hand and is never overwritten.
set -e
cd "$(dirname "$0")/.."
python scripts/make_numbers.py > /dev/null
python scripts/make_tables.py | tail -1
python scripts/make_cls.py 2>&1 | grep "macros written"
python src/civic/figures_cls.py --out paper/figures 2>&1 | tail -1
python scripts/tex_single.py sync
cd paper
pdflatex -interaction=nonstopmode main.tex > /dev/null; bibtex main > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null
grep -n "Undefined control\|Reference.*undefined\|Citation.*undefined\|^! " main.log | sort | uniq -c | head -20
grep -o "Output written on main.pdf ([0-9]* pages" main.log
