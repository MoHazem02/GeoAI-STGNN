# Contributing to GeoAI-STGNN

Thank you for your interest in contributing to GeoAI-STGNN! We welcome contributions from the community to help improve this spatio-temporal GNN framework for satellite change detection.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## 📜 Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## 🤝 How to Contribute

There are several ways you can contribute to this project:

1. **Bug Reports**: Submit detailed bug reports when you encounter issues
2. **Feature Requests**: Propose new features or improvements
3. **Code Contribudata: tions**: Fix bugs, implement new features, or improve existing code
4. **Documentation**: Improve documentation, add examples, or fix typos
5. **Testing**: Add test cases or help improve test coverage

## 🛠️ Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/GeoAI-STGNN.git
   cd GeoAI-STGNN
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```
5. Install pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 📥 Pull Request Process

1. Create a fork of the repository
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them with clear, descriptive messages
4. Add tests for your changes if applicable
5. Ensure all tests pass:
   ```bash
   python -m pytest tests/ -v
   ```
6. Update documentation if needed
7. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Open a pull request against the main repository

### Pull Request Guidelines

- Keep pull requests focused on a single feature or bugfix
- Include a clear description of the changes
- Reference any related issues
- Ensure your code follows the project's coding standards
- Write tests for new functionality
- Update documentation as needed

## 📝 Coding Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where possible
- Write docstrings for all public classes and methods
- Keep functions and classes focused on a single responsibility

### Example Code Style

```python
def calculate_uncertainty(
    probs: torch.Tensor,
    strategy: str = "entropy",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Computes node-level uncertainty scores from predicted class probabilities.

    Args:
        probs: Softmax class probabilities [N, C].
        strategy: Sampling method ("entropy", "margin", "least_confidence").
        eps: Numerical stability epsilon.

    Returns:
        uncertainty_scores: Per-node scalar uncertainty [N].

    Raises:
        ValueError: If unsupported strategy is provided.
    """
    if strategy == "entropy":
        # Implementation here
        pass
    # ... rest of implementation
```

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## 🧪 Testing

We use pytest for testing. All contributions should include appropriate tests.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_data_generation.py -v

# Run tests with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow the naming convention `test_*.py`
- Use descriptive test function names
- Test both normal and edge cases
- Mock external dependencies when appropriate

## 📚 Documentation

### Docstrings

All public classes, methods, and functions should have docstrings following the Google Python Style Guide format.

### README Updates

Update the README.md if your changes affect:
- Installation process
- Usage instructions
- Command-line arguments
- Core functionality

### New Features

When adding new features, include:
- Documentation in the `docs/` directory
- Examples in the documentation
- Updates to the README if appropriate

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Clear title** describing the issue
2. **Detailed description** of the problem
3. **Steps to reproduce** the issue
4. **Expected behavior** vs actual behavior
5. **Environment information** (OS, Python version, package versions)
6. **Screenshots** if applicable
7. **Code snippets** that demonstrate the issue

### Issue Templates

We provide issue templates for:
- Bug reports
- Feature requests
- Documentation improvements

Please use the appropriate template when creating issues.

## 🔄 Release Process

(For maintainers only)

1. Update version number in `setup.py`
2. Update CHANGELOG.md
3. Create a release branch
4. Run all tests
5. Create a GitHub release
6. Publish to PyPI (if applicable)

## 📞 Getting Help

If you need help with contributing, feel free to:
- Open an issue with the "question" label
- Contact the maintainers directly
- Join our community discussion (if available)

Thank you for contributing to GeoAI-STGNN!
