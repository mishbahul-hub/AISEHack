import torch
import torch.nn as nn
import torch.nn.functional as F



class ViT2D(nn.Module):
    """Vision Transformer for spatio-temporal regression.

    This model treats each spatial location as a token and encodes the temporal
    history at that location into a token embedding. A standard Transformer
    encoder processes the tokens, and a small head projects the output into the
    desired output time horizon.

    Forward Input
    -------------
    x : torch.Tensor  
        Shape (batch, time_in, nx, ny, features)

    Returns
    -------
    out : torch.Tensor  
        Shape (batch, nx, ny, time_out)
    """

    def __init__(
        self,
        time_in,
        features,
        time_out,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.0,
    ):
        super().__init__()

        self.time_in = time_in
        self.features = features
        self.time_out = time_out

        # Embed the (time_in x features) vector at each spatial location into a token.
        self.patch_embed = nn.Linear(time_in * features, embed_dim)

        # Positional embedding derived from normalized (x,y) coordinates.
        self.pos_proj = nn.Linear(2, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Decode each token into the output time horizon.
        self.head = nn.Linear(embed_dim, time_out)

    def forward(self, x):
        # x: (b, t, nx, ny, f)
        b, t, nx, ny, f = x.shape

        # Flatten spatial dimensions into tokens: (b, nx*ny, t*f)
        x = x.permute(0, 2, 3, 1, 4).reshape(b, nx * ny, t * f)
        x = self.patch_embed(x)

        # Positional encoding using normalized coordinates (same for all batches/time)
        grid = self.get_grid(b, t, nx, ny, x.device)  # (b, t, nx, ny, 2)
        pos = grid[:, 0]  # (b, nx, ny, 2)
        pos = pos.reshape(b, nx * ny, 2)
        pos = self.pos_proj(pos)

        x = x + pos

        x = self.transformer(x)
        x = self.norm(x)

        x = self.head(x)  # (b, nx*ny, time_out)
        x = x.view(b, nx, ny, self.time_out)
        return x

    def get_grid(self, b, t, nx, ny, device):
        """Generates normalized spatial coordinate grid for positional encoding."""

        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)

        gridx = gridx.view(1, 1, nx, 1, 1).repeat(b, t, 1, ny, 1)
        gridy = gridy.view(1, 1, 1, ny, 1).repeat(b, t, nx, 1, 1)

        return torch.cat((gridx, gridy), dim=-1)


# Backwards-compatible alias for existing training scripts
FNO2D = ViT2D
