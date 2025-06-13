# %% 
from sklearn.datasets import make_s_curve
import torch
device = 'xpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import random
from scipy.ndimage import rotate
import torch.nn.functional as F

# Define the MLP-based energy-based model
class EnergyBasedModel(nn.Module):
    def __init__(self, input_dim):
        super(EnergyBasedModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 211),
            nn.SiLU(),
            nn.Linear(211, 1)
        )

    def forward(self, x, penultimate=False, flattened=False):
        x = x.view(x.size(0), -1)  # Flatten the input
        if not penultimate:
            return self.network(x)
        else:
            for i, layer in enumerate(self.network):
                x = layer(x)
                if i == len(self.network) - 2:
                    break
            return x

import torch.nn as nn

# Define the CNN + MLP-based energy-based model
class FlattenedCNN(nn.Module):
    def __init__(self):
        super(FlattenedCNN, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 11x11 → 11x11
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 11x11 → 6x6
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 6x6 → 3x3
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 3x3 → 3x3
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # 3x3 → 2x2
            nn.Flatten()  # 256 * 2 * 2 = 1024
        )

        self.fc1 = nn.Linear(1024, 211)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(211, 1)

    def forward(self, x, penultimate=False, flattened=True):
        x = x.view(x.size(0), 1, 11, 11)  # from (B, 121)
        x = self.conv_stack(x)
        x = self.fc1(x)
        x = self.activation(x)

        if penultimate:
            return x
        else:
            return self.fc2(x)

    def forward2(self, x, penultimate=False, flattened=True):
        x = x.view(x.size(0), 1, 7, 7)  # from (B, 49)
        x = F.interpolate(x, size=(11, 11), mode='bilinear', align_corners=False)
        return self.forward(x, penultimate=penultimate, flattened=False)

# Training function
def train_energy_based_model(model, dataloader, optimizer, sigma, epochs=10):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in dataloader:
            noise = torch.randn_like(data) * sigma

            optimizer.zero_grad()
            data2 = data + noise
            data2.requires_grad = True

            energy = model(data2)
            # differentiate with respect to the input
            grad_energy = torch.autograd.grad(energy.sum(), data2, create_graph=True)[0]
            loss = criterion(grad_energy, - noise/sigma**2)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Main script
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    X1 = make_s_curve(n_samples=10000, noise=0.01)[0][:, [0, 2]]

    # X1 = torch.randn(5000, 2)*.53 + 2
    # X1 = torch.vstack([X1, torch.randn(5000, 2)*.53 - 2])
    X1 = torch.tensor(X1, dtype=torch.float32).to(device)
    # X0 = torch.rand(10000, 2, device=device)*5 - 2.5
    X0 = X1 + torch.randn_like(X1)*.2
    X = torch.vstack([X0, X1])
    
    dataloader = DataLoader(X, batch_size=777, shuffle=True)
    # Model, optimizer
    model = EnergyBasedModel(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_energy_based_model(model, dataloader, optimizer, epochs=103)

    print("Training complete.")

    # save the model
    torch.save(model.state_dict(), "sm_model.pth")
# %%

def vismodel(model, X = None):
    # plot the contour of the model
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    Xmarks, Ymarks = torch.meshgrid(x, y)
    pts = torch.vstack([Xmarks.flatten(), Ymarks.flatten()]).T.to(device)
    Z = model(pts).reshape(Xmarks.shape).cpu().detach().numpy()

    plt.contourf(Xmarks, Ymarks, Z, levels=100)
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c='r', s=1, alpha = 1)
    # X = torch.randn(10000, 2) + 2
    # X, _ = make_moons(n_samples=10000, noise=.05)
    # if X is not None:
        # plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c='r', s=1, alpha = .1)
    # plt.show()

# %%

    fx = model(torch.randn(1000, 2).to(device), penultimate = True)
    print(torch.linalg.eigvals(torch.cov(fx.T)))
    