# FLIP-IT CKD baseline — convenience targets.
# Each target just calls the matching wrapper in scripts/ (the single home for run commands).
# First time only:  make setup
.DEFAULT_GOAL := help

.PHONY: help setup clinics baseline simulate simulate-mlp iid notebook lab lint clean paper paper-data paper-zip

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:          ## Create/repair the venv and install everything (dev + notebook extras)
	./scripts/setup.sh

clinics:        ## Generate synthetic per-clinic datasets into data/clinics/
	./scripts/clinics.sh

baseline:       ## Centralized pooled-data ceiling (logreg + mlp + xgboost)
	./scripts/baseline.sh

simulate:       ## Federated simulation: logistic regression, 12 practices, non-IID
	./scripts/simulate.sh

simulate-mlp:   ## Federated simulation with the MLP architecture
	./scripts/simulate.sh --model mlp

iid:            ## Federated simulation on an IID split (comparison)
	./scripts/simulate.sh --iid

notebook:       ## Re-run the exploration notebook headless (verifies it still works)
	./scripts/notebook.sh

lab:            ## Open the notebook interactively in your browser
	./scripts/lab.sh

paper:          ## Rebuild the expert report's figures/tables/macros from the cached run
	./scripts/paper.sh

paper-data:     ## Re-run every experiment, then rebuild the report's figures/tables/macros
	./scripts/paper.sh --recompute

paper-zip:      ## Package the expert report for upload to Overleaf
	./scripts/paper-zip.sh

lint:           ## Lint with ruff
	./scripts/lint.sh

clean:          ## Remove caches and the macOS .DS_Store junk
	./scripts/clean.sh
