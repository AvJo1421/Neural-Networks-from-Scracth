# 🧠 Neural Network from Scratch – Binary Classification with NumPy

## 📌 Overview
This project implements a **feedforward neural network from scratch** using only **NumPy** to perform a **binary classification task**. The goal is to deeply understand the internal mechanics of neural networks — including **forward propagation**, **backpropagation**, and **manual weight updates** — without relying on high-level libraries like TensorFlow or PyTorch.

---

## 🧠 Key Concepts

- Feedforward neural network (fully connected)
- Binary classification (labels: 0 or 1)
- Sigmoid activation function for hidden and output layers
- Binary cross-entropy loss
- Gradient descent for training
- Manual implementation of forward and backward passes

---

## 🧪 Dataset
- Input features: real-valued numeric inputs (e.g., velocity, distance)
- Output: binary label (0 = fail, 1 = success)
- Stored in a `.csv` file and loaded using NumPy

---

## 🔧 Technologies Used
- Python 3.x
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 📂 Files

| File | Description |
|------|-------------|
| `CE889_Jupyterfile_for_training_neural_network.ipynb` | Main notebook with complete neural network implementation and training loop |
| `your_dataset.csv` (not included) | CSV file with training data (features + labels) — expected shape: (n_samples, n_features + 1) |

---

## 🚀 How It Works

1. **Initialize weights and biases** for each layer
2. **Forward propagation** using sigmoid activation:
   - \( a = \sigma(Wx + b) \)
3. **Compute binary cross-entropy loss**
4. **Backpropagation** to compute gradients:
   - Derivative of sigmoid and BCE used to propagate error
5. **Update weights manually** using gradient descent
6. Repeat over epochs and monitor accuracy/loss

---

## 📈 Output
- Training accuracy and loss over epochs (plotted)
- Final predictions vs actual labels
- Confusion matrix for model evaluation

---

## 📊 Sample Visualization

```python
plt.plot(losses)
plt.title("Training Loss Over Epochs")
