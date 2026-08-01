# LaTeX build for the Viyog camera-ready paper (final.tex).
# Uses the local TinyTeX toolchain (IEEEtran.cls is local; IEEEtran.bst from TeXLive).

export PATH := $(HOME)/.TinyTeX/bin/x86_64-linux:$(PATH)

PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX   := bibtex

.PHONY: all paper clean

all: paper

# final.tex is an IEEEtran manuscript with \bibliography{refs}: full bibtex cycle.
paper:
	$(PDFLATEX) final
	$(BIBTEX) final
	$(PDFLATEX) final
	$(PDFLATEX) final


# Remove LaTeX aux artifacts only. Keeps PDFs and .bbl.
clean:
	rm -f *.aux *.log *.out *.blg *.synctex.gz
