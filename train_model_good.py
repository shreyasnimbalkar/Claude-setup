import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ✅ Random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ✅ Use sklearn dataset instead of hardcoded path
iris = load_iris()
X, y = iris.data, iris.target

# ✅ Proper stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Convert to tensors
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# ✅ Define simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ✅ Training loop (clean, no debug prints)
for epoch in range(50):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

# ✅ Save model safely (ignored folder)
os.makedirs("artifacts", exist_ok=True)
torch.save(model.state_dict(), "artifacts/model.pth")
print("✅ Model training complete — saved in artifacts/")

print("hello world")


# TEST ONLY: fake API key to validate checklist detection — DO NOT USE IN PRODUCTION
API_KEY = "sk_test_sfkajf334238754nfjsk983489356dgwhy4536"
