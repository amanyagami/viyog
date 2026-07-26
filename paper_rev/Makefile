# LaTeX build for the Viyog paper (final.tex) and rebuttal (response.tex).
# Uses the local TinyTeX toolchain (IEEEtran.cls is local; IEEEtran.bst from TeXLive).

export PATH := $(HOME)/.TinyTeX/bin/x86_64-linux:$(PATH)

PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX   := bibtex

.PHONY: all paper response combined clean

all: paper response combined

# final.tex is an IEEEtran manuscript with \bibliography{refs}: full bibtex cycle.
paper:
	$(PDFLATEX) final
	$(BIBTEX) final
	$(PDFLATEX) final
	$(PDFLATEX) final

# response.tex has no bibliography: two pdflatex passes.
response:
	$(PDFLATEX) response
	$(PDFLATEX) response

# Single submission PDF: revised manuscript (<=14 pp, revisions in blue)
# followed by the response (<=4 pp), losslessly merged with navigation
# bookmarks. Depends on final.pdf + response.pdf existing.
combined: paper response
	$(PDFLATEX) combine
	$(PDFLATEX) combine
	mv -f combine.pdf viyog_215_combined.pdf

# Remove LaTeX aux artifacts only. Keeps PDFs and .bbl.
clean:
	rm -f *.aux *.log *.out *.blg *.synctex.gz
