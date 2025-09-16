# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The model consists of three fully connected layers where the first layer takes one input and maps it to eight neurons, the second layer maps eight neurons to ten, and the final layer maps ten neurons to a single output. ReLU activation is applied in the hidden layers to introduce non-linearity, while the output layer remains linear to suit regression tasks. The training process uses Mean Squared Error (MSE) as the loss function, since it is commonly used for measuring errors in continuous value predictions. For optimization, the RMSProp optimizer is applied to adjust the weights efficiently and speed up convergence. During training, the model undergoes forward propagation to generate predictions, calculates the loss by comparing predictions with the target values, and applies backpropagation to update weights. The loss values are stored in a history dictionary, and after training, the model is evaluated on test data to compute test loss, which indicates how well the model generalizes. Finally, the training loss is visualized using a loss curve, which shows how the error decreases over epochs. A smooth decreasing curve indicates effective learning of the model.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

### Neural Network Model:
<img width="1384" height="878" alt="Screenshot 2025-09-16 104428" src="https://github.com/user-attachments/assets/23036215-0bf2-41e5-a20e-cc43cdc7201f" />




## PROGRAM
### Name: MOHANRAM GUNASEKAR
### Register Number: 212223240095
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple 2-hidden-layer network
        self.net = nn.Sequential(
            nn.Linear(1, 16),  # Input -> first hidden (16 neurons)
            nn.ReLU(),
            nn.Linear(16, 8),  # first hidden -> second hidden (8 neurons)
            nn.ReLU(),
            nn.Linear(8, 1)    # second hidden -> output (1 neuron)
        )
        self.history = {'loss': []}

    def forward(self, x):
        return self.net(x)

ai_brain  = NeuralNet()
criterion = nn.MSELoss()                   # Regression loss
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss    = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")

loss_df = pd.DataFrame(ai_brain.history)
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
scaled_input = torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)
prediction = ai_brain(scaled_input).item()
print(f"Prediction for Input=9: {prediction:.2f}")


```
## Dataset Information
<img width="179" height="510" alt="image" src="https://github.com/user-attachments/assets/d31273bf-8bdd-4bdb-835d-e0f04a38896e" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="776" height="581" alt="Screenshot 2025-09-16 105510" src="https://github.com/user-attachments/assets/d757ab2f-f407-41da-9973-4e04a12c6fbd" />


<img width="307" height="42" alt="Screenshot 2025-09-16 105517" src="https://github.com/user-attachments/assets/4ea53d8d-7c3d-49a0-a42a-fbf9f4294c9e" />


## RESULT
To develop a neural network regression model for the given dataset is excuted sucessfully.
