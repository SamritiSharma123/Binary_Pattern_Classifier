import streamlit as st
import torch
import pandas as pd

from model import Perceptron

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Binary Pattern Classifier",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("Binary Pattern Classifier")

st.sidebar.markdown("""
### Project Information

This application demonstrates a **Single Layer Perceptron**
trained using PyTorch.

Supported Logic Gates

- AND
- OR
- NAND
- NOR
- XOR

Framework

- PyTorch
- Streamlit

Model

- Single Layer Perceptron
""")

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("Binary Pattern Classifier using Perceptron")

st.write(
"""
This application classifies binary patterns using
a trained Perceptron model.

Select a logic gate, enter two binary inputs,
and click **Predict**.
"""
)

# -----------------------------------------------------
# Select Logic Gate
# -----------------------------------------------------

gate = st.selectbox(
    "Select Logic Gate",
    (
        "AND",
        "OR",
        "NAND",
        "NOR",
        "XOR"
    )
)

# -----------------------------------------------------
# Load Corresponding Model
# -----------------------------------------------------

model = Perceptron()

if gate == "AND":
    model.load_state_dict(torch.load("saved_model_and.pth", map_location="cpu"))

elif gate == "OR":
    model.load_state_dict(torch.load("saved_model_or.pth", map_location="cpu"))

elif gate == "NAND":
    model.load_state_dict(torch.load("saved_model_nand.pth", map_location="cpu"))

elif gate == "NOR":
    model.load_state_dict(torch.load("saved_model_nor.pth", map_location="cpu"))

elif gate == "XOR":
    model.load_state_dict(torch.load("saved_model_xor.pth", map_location="cpu"))

model.eval()

# -----------------------------------------------------
# User Input
# -----------------------------------------------------

st.subheader("Binary Inputs")

col1, col2 = st.columns(2)

with col1:
    input1 = st.selectbox("Input 1", [0, 1])

with col2:
    input2 = st.selectbox("Input 2", [0, 1])

# -----------------------------------------------------
# Predict Button
# -----------------------------------------------------

if st.button("Predict"):

    x = torch.tensor([[float(input1), float(input2)]])

    with torch.no_grad():

        probability = model(x)

        prediction = torch.round(probability)

    st.success(
        f"Predicted Class : {int(prediction.item())}"
    )

    st.info(
        f"Probability : {probability.item():.4f}"
    )
 # -----------------------------------------------------
# Learned Parameters
# -----------------------------------------------------

st.markdown("---")
st.subheader("Learned Parameters")

weights = model.linear.weight.detach().numpy()

bias = model.linear.bias.detach().numpy()

col1, col2 = st.columns(2)

with col1:
    st.write("### Learned Weights")
    st.dataframe(weights)

with col2:
    st.write("### Learned Bias")
    st.dataframe(bias)

# -----------------------------------------------------
# Truth Tables
# -----------------------------------------------------

st.markdown("---")
st.subheader("Truth Table")

truth_tables = {

    "AND": pd.DataFrame({
        "Input 1":[0,0,1,1],
        "Input 2":[0,1,0,1],
        "Expected Output":[0,0,0,1]
    }),

    "OR": pd.DataFrame({
        "Input 1":[0,0,1,1],
        "Input 2":[0,1,0,1],
        "Expected Output":[0,1,1,1]
    }),

    "NAND": pd.DataFrame({
        "Input 1":[0,0,1,1],
        "Input 2":[0,1,0,1],
        "Expected Output":[1,1,1,0]
    }),

    "NOR": pd.DataFrame({
        "Input 1":[0,0,1,1],
        "Input 2":[0,1,0,1],
        "Expected Output":[1,0,0,0]
    }),

    "XOR": pd.DataFrame({
        "Input 1":[0,0,1,1],
        "Input 2":[0,1,0,1],
        "Expected Output":[0,1,1,0]
    })

}

st.table(truth_tables[gate])

# -----------------------------------------------------
# Model Status
# -----------------------------------------------------

st.markdown("---")
st.subheader("Model Status")

if gate == "XOR":

    st.warning("""
Single-layer Perceptrons cannot correctly learn the XOR gate because XOR is not linearly separable.

Therefore, the predictions for XOR may not always match the expected output.
A Multi-Layer Perceptron (MLP) is required to classify XOR perfectly.
""")

else:

    st.success("Model trained successfully.")

# -----------------------------------------------------
# Current Gate
# -----------------------------------------------------

st.info(f"Current Logic Gate : {gate}")

# -----------------------------------------------------
# Prediction Example
# -----------------------------------------------------

st.markdown("---")
st.subheader("Current Input")

st.write(f"Input 1 : {input1}")

st.write(f"Input 2 : {input2}")
# -----------------------------------------------------
# Evaluate Model on Complete Truth Table
# -----------------------------------------------------

st.markdown("---")
st.subheader("Model Evaluation")

x_test = torch.tensor([[0.,0.],
                       [0.,1.],
                       [1.,0.],
                       [1.,1.]])

if gate == "AND":

    y_test = torch.tensor([[0.],[0.],[0.],[1.]])

elif gate == "OR":

    y_test = torch.tensor([[0.],[1.],[1.],[1.]])

elif gate == "NAND":

    y_test = torch.tensor([[1.],[1.],[1.],[0.]])

elif gate == "NOR":

    y_test = torch.tensor([[1.],[0.],[0.],[0.]])

elif gate == "XOR":

    y_test = torch.tensor([[0.],[1.],[1.],[0.]])


criterion = torch.nn.BCELoss()

with torch.no_grad():

    output = model(x_test)

    loss = criterion(output, y_test)

    predicted = torch.round(output)

    accuracy = (predicted == y_test).float().mean()*100


col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="Training Accuracy",
        value=f"{accuracy.item():.2f}%"
    )

with col2:

    st.metric(
        label="Binary Cross Entropy Loss",
        value=f"{loss.item():.5f}"
    )

# -----------------------------------------------------
# Prediction Table
# -----------------------------------------------------

st.markdown("---")
st.subheader("Prediction on Complete Dataset")

result = pd.DataFrame({

    "Input 1":[0,0,1,1],

    "Input 2":[0,1,0,1],

    "Expected Output":y_test.numpy().flatten(),

    "Predicted Probability":
        output.numpy().flatten().round(4),

    "Predicted Class":
        predicted.numpy().flatten()

})

st.dataframe(result, use_container_width=True)

# -----------------------------------------------------
# Summary
# -----------------------------------------------------

st.markdown("---")

st.subheader("Project Summary")

st.write("""
This project demonstrates the implementation of a **Single Layer Perceptron**
using **PyTorch** for binary pattern classification.

The perceptron successfully learns the **AND**, **OR**, **NAND**, and **NOR**
logic gates because they are **linearly separable**.

The **XOR** gate is **not linearly separable**, therefore a single-layer
perceptron cannot learn it perfectly. A Multi-Layer Perceptron (MLP)
is required to classify XOR correctly.
""")

# -----------------------------------------------------
# Technologies Used
# -----------------------------------------------------

st.markdown("---")

st.subheader("Technologies Used")

tech = pd.DataFrame({

    "Technology":[
        "Python",
        "PyTorch",
        "Streamlit",
        "Pandas"
    ],

    "Purpose":[
        "Programming Language",
        "Model Training",
        "Web Interface",
        "Data Handling"
    ]

})

st.table(tech)

# -----------------------------------------------------
# Footer
# -----------------------------------------------------

st.markdown("---")

st.caption(
    "Binary Pattern Classifier using Perceptron | "
    "Developed using PyTorch and Streamlit"
)