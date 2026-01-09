# Grid Search with Custom Refit Strategy

A practical guide to implementing custom model selection strategies in scikit-learn's GridSearchCV.

## 📋 Overview

This repository demonstrates how to go beyond simple metric maximization when selecting models during hyperparameter tuning. Instead of just picking the model with the highest score, we implement a sophisticated selection strategy that balances:

- **Precision** (minimum threshold enforcement)
- **Recall** (performance consideration)
- **Prediction speed** (efficiency optimization)

## 🎯 Why Custom Refit Strategies?

By default, GridSearchCV selects models based on a single metric. However, real-world applications often require:

- Balancing multiple performance metrics
- Meeting minimum quality thresholds
- Optimizing for deployment constraints (speed, memory)
- Applying domain-specific selection criteria

This notebook shows you how to implement these complex selection strategies.

## 📊 Problem Description

We tackle a binary classification problem: identifying whether a handwritten digit image represents the number "8" or not, using the scikit-learn digits dataset.

The custom refit strategy implements a three-stage selection process:
1. **Quality Filter**: Keep only models with precision > 0.98
2. **Performance Filter**: Select models within 1 std of the best recall
3. **Efficiency Selection**: Choose the fastest model from remaining candidates

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.7
scikit-learn >= 0.24
pandas >= 1.0
matplotlib >= 3.0
jupyter >= 1.0
```

## 📚 Additional Resources

- [scikit-learn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [User Guide: Tuning Hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html)
- [Original Example from scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html)


⭐ If you found this helpful, please consider giving it a star!
