Viyog — CODES+ISSS 2026 (Paper #215) — camera-ready manuscript source
=====================================================================

This directory builds the accepted camera-ready paper, final.pdf (14 pages).

CONTENTS
--------
  final.tex      Camera-ready manuscript (14 pages, IEEEtran, double column).
  final.pdf      The accepted paper. This is the PDF submitted for artifact
                 evaluation, and the one this artifact's claims refer to.
  refs.bib       Bibliography database.
  final.bbl      Pre-compiled bibliography (so bibtex is optional).
  IEEEtran.cls   Document class (bundled for a self-contained build).
  IEEEtran.bst   Bibliography style.
  figs/rebuttal/*.pdf   The 18 figures used by final.tex. These are generated
                 outputs and are gitignored -- nothing precomputed is shipped
                 in this artifact. Regenerate them with `make figures` from
                 the repository root.

HOW TO BUILD
------------
  From the repository root:
      make figures      # only if paper_rev/figs/rebuttal/ is empty
      make paper

  Or directly, in this directory:
      pdflatex final && bibtex final && pdflatex final && pdflatex final

  Expected output: final.pdf, 14 pages.

NOTES
-----
  - The \rev{} macro, which rendered revised passages in blue during the
    review round, is defined as the identity in the camera-ready: revision
    highlighting is off and the manuscript is entirely black. The
    review-time `lineno` margin line numbers are likewise disabled.
  - The response-to-reviewers letter and the combined manuscript+response
    PDF were removed after acceptance. Both were anonymized
    ("Anonymous Author(s)") artifacts of the rebuttal cycle, carried blue
    revision highlighting and margin line numbers, and do not belong in the
    camera-ready record.
