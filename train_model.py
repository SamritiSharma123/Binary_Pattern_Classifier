# train_model.py

import torch
import torch.nn as nn
from model import Perceptron


# -------------------------------------
# Logic Gate Datasets
# -------------------------------------

datasets = {

    "AND": {
        "x": torch.tensor([[0.,0.],
                           [0.,1.],
                           [1.,0.],
                           [1.,1.]]),

        "y": torch.tensor([[0.],
                           [0.],
                           [0.],
                           [1.]])
    },

    "OR": {

        "x": torch.tensor([[0.,0.],
                           [0.,1.],
                           [1.,0.],
                           [1.,1.]]),

        "y": torch.tensor([[0.],
                           [1.],
                           [1.],
                           [1.]])
    },

    "NAND": {

        "x": torch.tensor([[0.,0.],
                           [0.,1.],
                           [1.,0.],
                           [1.,1.]]),

        "y": torch.tensor([[1.],
                           [1.],
                           [1.],
                           [0.]])
    },

    "NOR": {

        "x": torch.tensor([[0.,0.],
                           [0.,1.],
                           [1.,0.],
                           [1.,1.]]),

        "y": torch.tensor([[1.],
                           [0.],
                           [0.],
                           [0.]])
    },

    "XOR": {

        "x": torch.tensor([[0.,0.],
                           [0.,1.],
                           [1.,0.],
                           [1.,1.]]),

        "y": torch.tensor([[0.],
                           [1.],
                           [1.],
                           [0.]])
    }

}


# -------------------------------------
# Train Every Gate
# -------------------------------------

for gate_name, data in datasets.items():

    print("\n========================================")
    print(f"Training {gate_name} Gate")
    print("========================================")

    x = data["x"]
    y = data["y"]

    model = Perceptron()

    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epochs = 1000

    for epoch in range(epochs):

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

    with torch.no_grad():

        prediction = model(x)

        predicted = torch.round(prediction)

        accuracy = (predicted == y).float().mean() * 100

    filename = f"saved_model_{gate_name.lower()}.pth"

    torch.save(model.state_dict(), filename)

    print(f"\nGate : {gate_name}")

    print(f"Loss : {loss.item():.6f}")

    print(f"Accuracy : {accuracy.item():.2f}%")

    print("\nPredictions")

    print(prediction)

    print("\nRounded Predictions")

    print(predicted)

    print("\nWeights")

    print(model.linear.weight.data)

    print("\nBias")

    print(model.linear.bias.data)

    print(f"\nModel saved as {filename}")


print("\n========================================")
print("All Models Trained Successfully")
print("========================================")