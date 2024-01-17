import torch
from torch import nn

weight = 12
bias = 0.12

# Normalize the input data
x_train = torch.tensor([[2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
x_train_normalized = (x_train - x_train.mean()) / x_train.std()

y_train = x_train_normalized * weight + bias

class Linear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output)

    def forward(self, x):
        return self.linear(x)

model = Linear(1, 1)
learning_rate = 0.01
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 400
for epoch in range(epochs):
    model.train()
    y_pred = model(x_train_normalized)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_pred = model(x_train_normalized)

        losses = loss_fn(y_pred, y_train)
        if epoch % 10 == 0:
            print(f"Epoch : {epoch} , Loss :{losses}")

print(f"Model Parameters :{list(model.parameters())}")
