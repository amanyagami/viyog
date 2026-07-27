# Contributing to Viyog

Thank you for your interest in contributing to **Viyog**. Contributions of all kinds are welcome, including:

* New scoring variants or robustness improvements
* Additional evaluation metrics
* Bug fixes
* Performance optimizations
* Documentation enhancements
* Unit tests and CI improvements

For significant architectural or API changes, please open an issue first to discuss your proposal before submitting a pull request.

---

## Development Setup

Viyog uses **uv** for dependency management and environment reproducibility.

### 1. Install `uv`

If not already installed:

```bash
pip install uv
```

Recommended installation:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

---

### 2. Clone the Repository

```bash
git clone <your-fork-url>
cd viyog_repo
```

---

### 3. Create and Sync the Environment

```bash
uv sync
```

This will:

* Create a virtual environment
* Install all project dependencies
* Respect the `uv.lock` file for reproducible installs

To activate the environment:

```bash
source .venv/bin/activate
```

---

## Development Workflow

1. Fork the repository

2. Sync your fork with the latest `main` branch

3. Create a feature branch:

   ```bash
   git checkout -b feature/my-change
   ```

4. Make your changes

5. Commit with clear, descriptive messages

6. Push to your fork

7. Open a Pull Request (PR)

**Do not open pull requests directly from `main`.**

---

## Code Style & Linting

This project uses:

* **Ruff** for linting
* **pre-commit** for automated formatting and checks

Configuration is stored in:

```
.pre-commit-config.yaml
```

### Install Pre-commit Hooks

```bash
uv run pre-commit install
```

Hooks will run automatically on every commit.

To run manually:

```bash
uv run pre-commit run --all-files
```

All linting and formatting checks must pass before merging.

---

## Testing

Run tests with:

```bash
uv run pytest
```

If you add or modify functionality:

* Include unit tests
* Ensure all tests pass
* Verify tensor shapes and device handling
* Avoid introducing gradients (Viyog is inference-only)
* Ensure CPU and CUDA behavior remain consistent (if applicable)

Pull requests without appropriate test coverage may be requested to add tests before review.

---

## Pull Request Guidelines

Before submitting a PR, ensure:

* Code passes linting and formatting checks
* All tests pass
* New functionality includes tests
* Public API changes are documented
* The README is updated if behavior changes

PR descriptions should clearly explain:

* What was changed
* Why the change was made
* Any trade-offs or limitations

---

## Reporting Issues

When opening an issue, please include:

* Python version
* PyTorch version
* Operating system
* Minimal reproducible example
* Expected vs. actual behavior
* Full error traceback (if applicable)

Incomplete reports may delay resolution.

---

## Design & API Stability

* Keep changes focused and minimal
* Write clear docstrings for new functionality
* Avoid breaking existing public APIs without prior discussion
* Maintain inference-only guarantees
* Preserve backward compatibility whenever possible

---

## Security

If you discover a security vulnerability, do **not** open a public issue. Instead, contact the maintainers privately with detailed information.

---

## License

By contributing to Viyog, you agree that your contributions will be licensed under the *MIT** used by this project.
