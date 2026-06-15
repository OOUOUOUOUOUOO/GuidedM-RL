import torch
import numpy as np
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.model(x)


class AeroModel:

    def __init__(self, weight_path, device="cpu"):

        self.device = torch.device(device)

        self.model = Net().to(self.device)

        checkpoint = torch.load(weight_path, map_location=self.device)

        self.y_mean = checkpoint["y_mean"].numpy()
        self.y_std = checkpoint["y_std"].numpy()

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

    def predict(self, alpha, beta, delta):

        delta = np.asarray(delta, dtype=np.float32)

        inp = np.array(
            [
                alpha,
                beta,
                delta[0],
                delta[1],
                delta[2],
                delta[3],
            ],
            dtype=np.float32,
        )

        inp = torch.from_numpy(inp).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)

        out = out.cpu().numpy()

        out = out * self.y_std + self.y_mean

        out = out.squeeze(0)

        return out