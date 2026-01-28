# Contributing to Computer Vision Project

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed and what behavior you expected
* Include screenshots if relevant
* Include your environment details (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A detailed description of the proposed enhancement
* Explain why this enhancement would be useful
* List any alternative solutions you've considered

### Pull Requests

* Fill in the required template
* Follow the Python style guide (PEP 8)
* Include appropriate test cases
* Update documentation as needed
* Ensure all tests pass

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/Muhammad-Farooq-13/OPencv.git
cd OPencv
```

3. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a branch:
```bash
git checkout -b feature/your-feature-name
```

## Style Guide

### Python Style

* Follow PEP 8
* Use meaningful variable names
* Add docstrings to all functions and classes
* Keep functions focused and small
* Write unit tests for new features

### Git Commit Messages

* Use present tense ("Add feature" not "Added feature")
* Use imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit first line to 72 characters
* Reference issues and pull requests after the first line

Example:
```
Add feature to extract SIFT features

- Implement SIFT feature extraction
- Add unit tests
- Update documentation

Fixes #123
```

## Testing

Run tests before submitting:

```bash
pytest
pytest --cov=src --cov-report=html
```

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Community

* Be respectful and inclusive
* Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
* Help others when you can

## Questions?

Feel free to open an issue or contact:
* Muhammad Farooq - mfarooqshafee333@gmail.com

Thank you for contributing! 🙏
