# t-SNE Visualization of the MNIST Digits Dataset

An illustrated exploration of the t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm using the scikit-learn digits dataset. This project creates beautiful animated visualizations showing how t-SNE progressively clusters similar digits together in 2D space.

## 🎯 Overview

This notebook demonstrates:
- Loading and preprocessing the MNIST digits dataset
- Implementing t-SNE dimensionality reduction
- Creating custom gradient descent to capture intermediate states
- Generating animated visualizations of the clustering process
- Understanding similarity matrices and probability distributions in t-SNE

## 📊 Sample Output

The animation shows how t-SNE progressively separates the 10 digit classes (0-9) from a random initialization into distinct, well-separated clusters.

![t-SNE Animation](tsne_evolution.gif)



## 📚 What's Inside

### Key Components

1. **Data Loading**: Uses scikit-learn's built-in digits dataset (1,797 samples, 64 features)

2. **Custom Gradient Descent**: A monkey-patched version of sklearn's gradient descent that captures intermediate embedding positions for animation

3. **Visualization Functions**: 
   - `scatter_df()`: Creates beautiful scatter plots with labeled clusters
   - `mplfig_to_npimage()`: Converts matplotlib figures to numpy arrays for video generation. This was created because the original is no longer in the moviepy library at the time of this writing.

4. **Animation Generation**: Creates frame-by-frame animations showing the evolution of the t-SNE embedding

### Important Parameters

- `max_iter=5000`: Number of optimization iterations (increase for better separation)
- `learning_rate=200.0`: Controls the step size during optimization
- `perplexity=30.0`: Balances local vs global structure (typical range: 5-50)
- `fps=20`: Frames per second for the output animation

## 🔧 Key Fixes & Improvements

This implementation includes several important fixes:

1. **Parameter Update**: Uses `max_iter` instead of deprecated `n_iter` (scikit-learn 1.0+)
2. **Coordinate Consistency**: Ensures animation frames match the final output orientation
3. **Complete Convergence**: Runs enough iterations (5000+) for full cluster separation
4. **Proper Frame Timing**: Correct FPS calculation ensures all frames are shown

## 📖 Tutorial Reference

This project is based on the excellent tutorial:
- [An Illustrated Introduction to the t-SNE Algorithm](https://www.oreilly.com/content/an-illustrated-introduction-to-the-t-sne-algorithm/) by O'Reilly

## 🎓 Learning Objectives

- Understand how t-SNE reduces high-dimensional data to 2D/3D
- Visualize the optimization process of dimensionality reduction
- Learn about probability distributions and KL divergence in t-SNE
- Create publication-quality visualizations and animations

## 🛠️ Customization

### Adjust Animation Speed
```python
fps = 30  # Faster playback
fps = 10  # Slower playback
```

### Change Number of Iterations
```python
max_iter = 3000   # Faster computation, less separation
max_iter = 10000  # Slower computation, better separation
```

### Modify Visualization Style
```python
# In scatter_df function
s=20  # Increase point size
alpha=0.8  # Adjust transparency
```

## 📝 Output Files

The notebook generates:
- `tsne_evolution.gif` or `.mp4`: Animation of the clustering process
- `images/`: Directory containing intermediate visualizations

## ⚠️ Common Issues

### Animation ends before clusters separate
**Solution**: Increase `max_iter` to 5000+ iterations

### Coordinate flip in animation
**Solution**: Use `X_proj` instead of `X_iter[..., -1]` when initializing the plot

## 🙏 Acknowledgments

- O'Reilly Media for the original tutorial
- scikit-learn team for the excellent ML library
- The t-SNE algorithm creators: Laurens van der Maaten and Geoffrey Hinton

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: The animation generation can take several minutes depending on `max_iter` and your hardware.
