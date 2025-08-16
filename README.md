# Neural Network from Scratch using NumPy
A comprehensive implementation of a 2-layer feedforward neural network built from scratch using only NumPy, trained on the sklearn digits dataset for handwritten digit recognition.

## Overview
This project demonstrates a complete understanding of neural network fundamentals by implementing forward propagation, backpropagation, and gradient descent algorithms from scratch. The network successfully classifies handwritten digits (0-9) with high accuracy, showcasing the power of deep learning principles without relying on high-level frameworks.

## Key Highlights
- **From Scratch Implementation**: Complete neural network built using only NumPy
- **Mathematical Foundation**: Implements core algorithms (forward/backpropagation, gradient descent)
- **High Performance**: Achieves 96.94% test accuracy on digit recognition
- **Educational Focus**: Clear, well-documented code for learning purposes
- **Comprehensive Analysis**: Detailed evaluation with visualizations
- **Production Ready**: Includes prediction function for new digit images

## Dataset
**Source**: sklearn.datasets - Digits Dataset

### Dataset Statistics:
- **Total Samples**: 1,797 handwritten digit images
- **Image Dimensions**: 8×8 grayscale images (64 features when flattened)
- **Classes**: 10 digits (0-9)
- **Train/Test Split**: 80% training (1,437 samples) / 20% testing (360 samples)
- **Data Range**: Pixel values normalized between 0-16

### Key Features Used:
- **Input Features**: 64 pixel values (8×8 flattened image)
- **Target Variable**: Single digit class (0-9)
- **Data Preprocessing**: One-hot encoding for multi-class classification

## Features

### Core Functionality
- **Digit Classification**: Predict handwritten digits from 8×8 pixel images
- **Multi-class Support**: Handle all 10 digit classes simultaneously
- **Performance Analytics**: Comprehensive model evaluation with multiple metrics
- **Loss Visualization**: Training loss progression over epochs
- **Prediction Analysis**: Visual comparison of correct vs incorrect predictions
- **Results Export**: Detailed confusion matrix and classification report

### Technical Features
- **Forward Propagation**: Manual implementation of matrix operations
- **Backpropagation**: Chain rule implementation for gradient computation
- **Activation Functions**: ReLU for hidden layer, Softmax for output
- **Mini-batch Training**: Efficient batch gradient descent
- **Loss Function**: Cross-entropy loss with numerical stability
- **Parameter Updates**: Vanilla gradient descent optimization

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. **Install required packages**
   ```bash
   pip install numpy scikit-learn matplotlib
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook neural_network_from_scratch.ipynb
   ```

## Usage

### Quick Start
**Basic Usage**: Run all cells in the Jupyter notebook to train the model and see results

**Predict New Digit**:
```python
# Example prediction for a new 8x8 image
new_digit = np.array([[0,  0,  5, 13,  9,  1,  0,  0],
                     [0,  0, 13, 15, 10, 15,  5,  0],
                     [0,  3, 15,  2,  0, 11,  8,  0],
                     [0,  4, 12,  0,  0,  8,  8,  0],
                     [0,  5,  8,  0,  0,  9,  8,  0],
                     [0,  4, 11,  0,  1, 12,  7,  0],
                     [0,  2, 14,  5, 10, 12,  0,  0],
                     [0,  0,  6, 13, 10,  0,  0,  0]])

# Flatten and predict
flattened_digit = new_digit.flatten().reshape(1, -1)
prediction = nn.predict(flattened_digit)
print(f"Predicted digit: {prediction[0]}")
```

**Access Results**: View detailed analysis including confusion matrix and sample predictions

### Advanced Usage
- **Custom Architecture**: Modify `hidden_size` parameter for different network sizes
- **Hyperparameter Tuning**: Adjust `learning_rate`, `epochs`, and `batch_size`
- **Evaluation**: Use built-in evaluation functions for detailed performance analysis
- **Visualization**: Generate custom plots for loss curves and prediction samples

## Model Performance

### Performance Metrics
| Metric | Score | Description |
|--------|--------|-------------|
| **Test Accuracy** | **96.94%** | Overall classification accuracy |
| **Training Accuracy** | **100.00%** | Perfect training set performance |
| **Loss Reduction** | **99.94%** | From 2.2650 to 0.0013 |
| **Training Epochs** | **1000** | Complete convergence achieved |

### Performance Analysis
- **High Accuracy**: 96.94% test accuracy exceeds target of ~90%
- **No Overfitting**: Good generalization despite 100% training accuracy
- **Fast Convergence**: Loss stabilizes after ~300 epochs
- **Consistent Performance**: Reliable predictions across all digit classes

### Model Strengths
- **Mathematical Precision**: Implements core algorithms correctly
- **Numerical Stability**: Handles edge cases in softmax and cross-entropy
- **Efficient Training**: Mini-batch approach for scalable learning
- **Interpretable Results**: Clear visualization of learning progress

## Results & Visualizations

### Training Progress
![Training Loss](path/to/training_loss_plot.png)

**Key Insights**:
- Rapid initial loss reduction (2.27 → 0.10 in first 100 epochs)
- Smooth convergence without oscillations
- Final loss of 0.0013 indicates excellent learning
- No signs of overfitting or underfitting

### Sample Predictions
![Sample Predictions](path/to/sample_predictions.png)

**Prediction Analysis**:
- **Correct Predictions**: Clear, well-formed digits recognized accurately
- **Incorrect Predictions**: Ambiguous or poorly formed digits (rare cases)
- **Confidence Levels**: High softmax probabilities for correct predictions

### Confusion Matrix
![Confusion Matrix](path/to/confusion_matrix.png)

**Classification Insights**:
- Strong diagonal pattern indicates good classification
- Minimal confusion between dissimilar digits
- Slight confusion between similar digits (6/8, 3/5)

## 🔧 Technical Implementation

### Architecture Overview
```
Input (64) → Hidden Layer (32) → Output (10)
    ↓            ↓                  ↓
  Pixel      ReLU Activation    Softmax
  Values                       Probabilities
```

### Key Components

**Mathematical Operations**:
```python
# Forward Propagation
Z1 = X @ W1 + b1          # Linear transformation
A1 = ReLU(Z1)             # Non-linear activation
Z2 = A1 @ W2 + b2         # Output layer
A2 = Softmax(Z2)          # Probability distribution

# Backpropagation
dZ2 = A2 - Y              # Output layer gradient
dW2 = A1.T @ dZ2 / m      # Weight gradients
dZ1 = dZ2 @ W2.T * ReLU'(Z1)  # Hidden layer gradient
dW1 = X.T @ dZ1 / m       # Input weight gradients
```

**Activation Functions**:
- **ReLU**: `f(x) = max(0, x)` - Introduces non-linearity
- **Softmax**: `f(x) = exp(x) / Σexp(x)` - Probability distribution
- **Cross-entropy**: `L = -Σ(y*log(ŷ))` - Multi-class loss function

### Algorithm Choice Rationale
- **2-Layer Architecture**: Sufficient complexity for digit recognition
- **ReLU Activation**: Prevents vanishing gradients, computationally efficient
- **Softmax Output**: Perfect for multi-class probability distribution
- **Cross-entropy Loss**: Standard choice for classification tasks
- **Mini-batch GD**: Balance between computational efficiency and convergence stability

## Project Structure
```
neural-network-from-scratch/
│
├── README.md                           # Project documentation
├── neural_network_from_scratch.ipynb   # Main implementation notebook
├── neural_network_from_scratch.py      # Standalone Python script
├── images/                             # Generated visualizations
│   ├── training_loss_plot.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── sample_digits.png
└── requirements.txt                    # Python dependencies
```

## Learning Outcomes

### Mathematical Concepts Demonstrated
1. **Forward Propagation**: Matrix multiplication chains
2. **Backpropagation**: Chain rule for gradient computation
3. **Gradient Descent**: Parameter optimization
4. **Activation Functions**: Non-linear transformations
5. **Loss Functions**: Error quantification
6. **Vectorization**: Efficient NumPy operations

### Programming Skills Showcased
- **Object-Oriented Design**: Clean, modular neural network class
- **NumPy Mastery**: Advanced array operations and broadcasting
- **Mathematical Implementation**: Translating equations to code
- **Data Visualization**: Matplotlib for comprehensive analysis
- **Code Documentation**: Clear comments and docstrings

## Future Enhancements

### Potential Improvements
- **Deep Networks**: Add more hidden layers
- **Advanced Optimizers**: Implement Adam, RMSprop, or momentum
- **Regularization**: Add L2 regularization or dropout
- **Batch Normalization**: Improve training stability
- **Learning Rate Scheduling**: Adaptive learning rates
- **Cross-validation**: More robust evaluation methodology

### Extension Ideas
- **Different Datasets**: CIFAR-10, Fashion-MNIST
- **Convolutional Layers**: CNN implementation from scratch
- **Different Architectures**: Autoencoder, multi-layer perceptrons
- **Performance Optimization**: GPU acceleration with CuPy

## Requirements
```
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

## Acknowledgments
- **sklearn**: For providing the digits dataset
- **NumPy**: For efficient numerical computations
- **Matplotlib**: For comprehensive data visualization

---

**NOTE**: This is an educational/portfolio project demonstrating deep understanding of neural network fundamentals. The implementation serves as a learning tool for understanding the mathematical foundations of deep learning without relying on high-level frameworks like TensorFlow or PyTorch. This project showcases technical competency in machine learning theory, mathematical implementation, and scientific programming.
