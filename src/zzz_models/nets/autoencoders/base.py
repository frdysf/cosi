import torch
import torch.nn as nn


class BaseAutoencoder(nn.Module):
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.extract_features(x)
        z = self.encode(x)
        x_hat = self.decode(z)
        return x, x_hat, z
