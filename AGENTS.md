# Development Guidelines

This document contains critical information about working with this codebase.
Follow these guidelines precisely.

## Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Follow existing patterns exactly
   - Use Google style for docstring

3. Testing Requirements
   - Framework: `uv run --frozen pytest`
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Git
   - Follow the Conventional Commits style on commit messages.

## Code Formatting and Linting

Scope is `src/ tests/ examples/` (the installable package) -- matches CI.
`experiments/` is the research/reproduction pipeline and isn't held to the
same style bar.

1. Ruff
   - Format: `uv run --frozen ruff format src/ tests/ examples/`
   - Check: `uv run --frozen ruff check src/ tests/ examples/`
   - Fix: `uv run --frozen ruff check src/ tests/ examples/ --fix`
2. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Ruff (Python)
