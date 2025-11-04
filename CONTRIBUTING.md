# Contributing to Bi-MEMIT

Thank you for your interest in contributing to Bi-MEMIT! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Environment Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/Bi_Memit.git
   cd Bi_Memit
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

### Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

3. Run tests to ensure everything works:
   ```bash
   pytest tests/
   ```

4. Push your branch and create a pull request

## 📝 Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks with:
```bash
pre-commit run --all-files
```

### Code Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- Add tests for new functionality

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_memit.py

# Run with coverage
pytest --cov=src tests/

# Run only fast tests
pytest -m "not slow"
```

### Writing Tests

- Write unit tests for all new functions
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Add integration tests for end-to-end workflows

## 📖 Documentation

### Updating Documentation

- Update docstrings for any modified functions
- Add examples to the `examples/` directory
- Update relevant markdown files in `docs/`
- Ensure all links work correctly

### Building Documentation Locally

```bash
cd docs
pip install -r requirements.txt
make html
```

## 🐛 Reporting Issues

When reporting bugs, please include:

- Python version and operating system
- Bi-MEMIT version
- Minimal code example that reproduces the issue
- Full error message and traceback
- Any relevant configuration files

## 💡 Feature Requests

Before proposing new features:

1. Check existing issues and pull requests
2. Discuss the feature in an issue first
3. Consider backward compatibility
4. Think about the API design

## 📋 Pull Request Process

1. Ensure your PR addresses an existing issue
2. Update documentation if needed
3. Add tests for new functionality
4. Ensure all CI checks pass
5. Request review from maintainers

### PR Checklist

- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All CI checks pass
- [ ] PR description explains the changes

## 🏷️ Release Process

Releases are managed by maintainers and follow semantic versioning:

- **Patch** (0.0.X): Bug fixes
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes

## 👥 Community Guidelines

- Be respectful and inclusive
- Help newcomers get started
- Provide constructive feedback
- Focus on the technical merits of ideas
- Follow our Code of Conduct

## 🆘 Getting Help

- Check the documentation first
- Search existing issues
- Ask questions in GitHub Discussions
- Join our community chat (if available)

Thank you for contributing to Bi-MEMIT! 🎉