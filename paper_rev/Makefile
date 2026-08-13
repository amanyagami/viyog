# LaTeX build for the Viyog camera-ready paper (final.tex).
# Prefer a system TeX installation; fall back to Tectonic when available.

PDFLATEX := $(shell command -v pdflatex 2>/dev/null)
BIBTEX   := $(shell command -v bibtex 2>/dev/null)
TECTONIC := $(shell command -v tectonic 2>/dev/null)

.PHONY: all paper clean

all: paper

# final.tex is an IEEEtran manuscript with \bibliography{refs}.
paper:
ifneq ($(and $(PDFLATEX),$(BIBTEX)),)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error final
	$(BIBTEX) final
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error final
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error final
else ifneq ($(TECTONIC),)
	$(TECTONIC) -k --keep-logs final.tex
else
	@echo "error: install pdflatex+bibtex or tectonic to build final.pdf"
	@false
endif


# Remove LaTeX aux artifacts only. Keeps PDFs and .bbl.
clean:
	rm -f *.aux *.log *.out *.blg *.synctex.gz
