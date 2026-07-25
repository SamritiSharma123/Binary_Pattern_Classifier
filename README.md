# Binary Pattern Classifier using Perceptron

## Project Overview

The **Binary Pattern Classifier** is a machine learning mini project developed using **PyTorch** and **Streamlit**. The project implements a single-layer perceptron to classify binary inputs based on a selected logic gate (AND or OR). After training, the model is deployed through an interactive Streamlit interface that allows users to provide binary inputs and obtain predictions along with the learned model parameters.
<img width="1735" height="625" alt="image" src="https://github.com/user-attachments/assets/a607f470-5ab0-4446-a0cb-9b7a1ee95796" />
<img width="1807" height="741" alt="image" src="https://github.com/user-attachments/assets/d4a8f68c-8337-43d9-80f5-c920382aaf4a" />
<img width="1893" height="757" alt="image" src="https://github.com/user-attachments/assets/e65c3b70-e6eb-45e1-b042-9d2936aaae70" />
<img width="1865" height="416" alt="image" src="https://github.com/user-attachments/assets/be070337-77e7-49fa-bb22-9fb37c9576e2" />



The project demonstrates the complete workflow of building a simple neural network, training it using gradient descent, saving the trained model, and deploying it as a user-friendly web application.

---

## Objectives

The objectives of this project are:

- Train a perceptron using binary logic gate datasets.
- Understand the working of a single-layer neural network.
- Implement Binary Cross Entropy Loss for binary classification.
- Compare model predictions with expected outputs.
- Display the learned weights and bias after training.
- Develop an interactive Streamlit application for real-time prediction.

---

## Features

- Binary Pattern Classification
- Supports AND and OR Logic Gates
- Single Layer Perceptron using PyTorch
- Binary Cross Entropy Loss Function
- Stochastic Gradient Descent Optimizer
- Interactive Streamlit User Interface
- Real-Time Prediction
- Sigmoid Probability Display
- Learned Weights and Bias Display
- Truth Table Visualization
- Trained Model Saving

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| PyTorch | Neural Network Development |
| Streamlit | Web Application Framework |
| NumPy | Numerical Computation |
| Pandas | Data Handling |
| Matplotlib | Visualization (Optional) |

---

## Project Structure

```text
Binary_Pattern_Classifier/
│
├── app.py
├── train_model.py
├── saved_model.pth
├── requirements.txt
├── README.md
└── assets/
```

---

## Working Principle

The project follows the following steps:

1. Prepare the binary logic gate dataset.
2. Build a single-layer perceptron model.
3. Apply the Sigmoid activation function.
4. Compute Binary Cross Entropy Loss.
5. Train the model using Gradient Descent.
6. Save the trained model.
7. Load the model in the Streamlit application.
8. Accept user input.
9. Predict the output.
10. Display probability, predicted class, weights, and bias.

---

## Dataset

### AND Gate

| Input 1 | Input 2 | Output |
|----------|----------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

### OR Gate

| Input 1 | Input 2 | Output |
|----------|----------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

---

## Model Architecture

```
Input Layer (2 Neurons)
        │
        ▼
Linear Layer (2 → 1)
        │
        ▼
Sigmoid Activation
        │
        ▼
Binary Output
```

---

## Loss Function

The project uses **Binary Cross Entropy Loss (BCELoss)** for binary classification.

```
criterion = nn.BCELoss()
```

---

## Optimizer

The model is trained using **Stochastic Gradient Descent (SGD)**.

```
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/your-username/Binary-Pattern-Classifier.git
```

Move into the project directory.

```bash
cd Binary-Pattern-Classifier
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1

Train the model.

```bash
python train_model.py
```

This generates the trained model file.

```
saved_model.pth
```

### Step 2

Launch the Streamlit application.

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## Application Workflow

1. Select the logic gate.
2. Enter two binary inputs.
3. Click the **Predict** button.
4. View the predicted output.
5. Observe the Sigmoid probability.
6. Display the learned weights and bias.
7. Compare with the truth table.

---

## Sample Output

```
Selected Logic Gate

AND

Input 1 : 1
Input 2 : 0

Predicted Probability

0.0438

Predicted Class

0

Learned Weights

[[7.28 7.15]]

Bias

[-10.62]
```

---

## Evaluation Criteria

| Criterion | Weight |
|------------|--------|
| Correct Training Implementation | 40% |
| Working Streamlit Interface | 30% |
| Weight and Bias Display | 15% |
| Code Quality and Documentation | 15% |

---

## Learning Outcomes

After completing this project, the following concepts are understood:

- Perceptron Architecture
- Binary Classification
- Linear Layer
- Sigmoid Activation Function
- Binary Cross Entropy Loss
- Gradient Descent
- Backpropagation
- Model Training
- Model Deployment using Streamlit

---

## Future Enhancements

- Support XOR classification using a multilayer neural network.
- Add training accuracy and loss visualization.
- Include model performance metrics.
- Allow custom datasets.
- Save prediction history.
- Deploy the application on Streamlit Community Cloud.

---

## Conclusion

This project demonstrates the implementation of a Binary Pattern Classifier using a single-layer perceptron in PyTorch. The model successfully learns linearly separable logic gates such as AND and OR. The trained model is integrated with a Streamlit interface, enabling users to perform real-time predictions while visualizing the learned parameters. The project provides a practical understanding of neural network training, binary classification, and model deployment.

---

## Author

**Name:** Samriti Sharma

**Course:** Bachelor of Engineering (Computer Science and Engineering)

**Project:** Binary Pattern Classifier using Perceptron

**Frameworks:** PyTorch, Streamlit
