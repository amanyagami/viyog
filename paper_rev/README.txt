Viyog — CODES+ISSS 2026 (Paper #215) — Overleaf submission package
===================================================================

CONTENTS
--------
  final.tex      Revised manuscript (14 pages, IEEEtran journal, double column).
  response.tex   Response-to-reviewers letter (4 pages).
  refs.bib       Bibliography database.
  final.bbl      Pre-compiled bibliography (so bibtex is optional).
  IEEEtran.cls   Document class (bundled for a self-contained build).
  IEEEtran.bst   Bibliography style.
  figs/rebuttal/*.pdf   All 18 figures used by final.tex (PDF/Type-42 fonts,
                        PDF-eXpress compatible).

HOW TO BUILD ON OVERLEAF
------------------------
  1. Create a new project -> "Upload Project" -> select this .zip.
  2. Set the main document to final.tex (Menu -> Main document).
  3. Compiler: pdfLaTeX. The bundled final.bbl means a single pdfLaTeX pass
     resolves all citations; for a clean rebuild use pdfLaTeX -> BibTeX ->
     pdfLaTeX -> pdfLaTeX.
  4. To build the response letter, set the main document to response.tex
     (it has no bibliography; two pdfLaTeX passes suffice).

HOW TO BUILD LOCALLY
--------------------
  pdflatex final && bibtex final && pdflatex final && pdflatex final
  pdflatex response && pdflatex response

NOTES
-----
  - final.tex uses the `lineno` package (visible margin line numbers) for
    reviewer navigation; the response letter cites those line numbers.
  - Revision text is marked in blue via the \rev{} macro.
  - Expected output: final.pdf = 14 pages, response.pdf = 4 pages.
