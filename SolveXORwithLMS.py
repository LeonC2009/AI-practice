import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype = torch.float32)

class XORNET(nn.Module):
    def __init__(self):
        super(XORNET, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = XORNET()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

print("\nResults:")
with torch.no_grad():
    for i in range(len(X)):
        output = model(X[i:i+1])
        print(f"Input: {X[i].numpy()}, Target: {y[i].item()}, Output: {output.item():.4f}")