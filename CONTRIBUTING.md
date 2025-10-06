# Contributing to Plant Disease Classification Project

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🎯 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, PyTorch version)
- Error messages and stack traces

### Suggesting Enhancements

For feature requests:
- Describe the feature clearly
- Explain the use case
- Provide examples if possible
- Consider implementation complexity

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style guidelines
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   python -m pytest tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new model architecture"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

## 📝 Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Add type hints where applicable

Example:
```python
def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Accuracy as percentage
    """
    return np.mean(predictions == targets) * 100
```

### Documentation

- Update README.md for major changes
- Add inline comments for complex logic
- Include examples in docstrings
- Update configuration files if needed

### Commit Messages

Use conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest --cov=.
```

### Adding Tests

- Write tests for new features
- Ensure tests are reproducible
- Use descriptive test names
- Test edge cases

## 📚 Adding New Models

To add a new model architecture:

1. **Add to `model_lib.py`**
   ```python
   def YourModelName(num_classes=38):
       """Your model description"""
       # Model implementation
       return model
   ```

2. **Update `get_all_models()` function**
   ```python
   models['YourModelName'] = YourModelName(num_classes)
   ```

3. **Update `config.py`**
   ```python
   AVAILABLE_MODELS = [
       # ... existing models
       "YourModelName"
   ]
   ```

4. **Test the model**
   ```python
   python test_new_model.py
   ```

5. **Update documentation**

## 🔬 Adding Analysis Features

For new analysis features:

1. **Add functions to `utils.py`**
2. **Create visualization in appropriate notebook**
3. **Document the metric/visualization**
4. **Add example usage**

## 🌿 Project Structure Guidelines

```
New features should follow existing structure:
- Models → model_lib.py
- Training → train_analyze_lib.py
- Analysis → analyze_and_results/
- Utilities → utils.py or specialized files
```

## 📊 Data Contributions

### Dataset Improvements
- Suggest additional plant species
- Report labeling errors
- Propose augmentation strategies

### Evaluation Metrics
- Propose new evaluation metrics
- Suggest visualization improvements
- Add statistical tests

## 🤝 Code Review Process

All PRs will be reviewed for:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Compatibility

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Questions?

- Create an issue for general questions
- Email: nguyenhuudatgm1@gmail.com
- Join discussions in GitHub Discussions

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to making plant disease detection more accessible and accurate!