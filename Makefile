# Top-level convenience Makefile for the Viyog repo. Run `make help` for a
# summary. Delegates LaTeX work to paper_rev/Makefile.

UV       := uv run --frozen
FIGS_OUT := paper_rev/figs/rebuttal

.PHONY: help env figures paper response all clean demo-data demo

help:
	@echo "Viyog make targets:"
	@echo "  env       - uv sync the Python project"
	@echo "  figures   - regenerate paper figures from results/analysis CSVs"
	@echo "  paper     - build paper_rev/final.pdf (pdflatex+bibtex)"
	@echo "  response  - build paper_rev/response.pdf (pdflatex x2)"
	@echo "  all       - figures + paper + response"
	@echo "  demo-data - rebuild the ~38 KB CSVs the leaderboard app serves"
	@echo "  demo      - run the Viyog-vs-pytorch-ood leaderboard locally (:7860)"
	@echo "  clean     - remove LaTeX aux files in paper_rev/ (keeps PDFs, .bbl, data)"

# --- Python environment ----------------------------------------------------
env:
	uv sync

# --- Figures ---------------------------------------------------------------
# Config-pathed scripts write straight into paper_rev/figs/...; --out scripts
# are pointed at paper_rev/figs/rebuttal/ so the PDFs land where final.tex
# expects them. See README "Reproducing the paper" for the full pipeline.
figures:
	$(UV) python experiments/plot_rebuttal_figs.py
	$(UV) python experiments/make_final_figs.py
	$(UV) python experiments/plot_adaptive_ladder.py --out $(FIGS_OUT)/fig_adaptive_ladder.pdf
	$(UV) python experiments/plot_mechanism.py       --out $(FIGS_OUT)/fig_mechanism.pdf
	$(UV) python experiments/plot_seed_forest.py     --out $(FIGS_OUT)/fig_seed_forest_new.pdf
	$(UV) python experiments/plot_t3_breakdown.py    --out $(FIGS_OUT)/fig_t3_breakdown.pdf
	$(UV) python experiments/plot_problem_concept.py --out $(FIGS_OUT)/fig_problem_concept.pdf
	$(UV) python experiments/plot_pipeline.py        --out $(FIGS_OUT)/fig_pipeline.pdf

# --- Paper / rebuttal (delegate to paper_rev/Makefile) ---------------------
paper:
	$(MAKE) -C paper_rev paper

response:
	$(MAKE) -C paper_rev response

all: figures paper response

# --- Demo / leaderboard ----------------------------------------------------
# Distills results/analysis/*.csv into demo/data/*.csv (a few tens of KB) and
# runs the interactive Viyog-vs-pytorch-ood leaderboard. See demo/HOSTING_PLAN.md
# for the (free) HuggingFace Spaces deployment steps.
demo-data:
	$(UV) python demo/export_leaderboard.py

demo: demo-data
	cd demo && uv run --no-project --python 3.11 --with-requirements requirements.txt python app.py

# --- Cleaning --------------------------------------------------------------
clean:
	$(MAKE) -C paper_rev clean
