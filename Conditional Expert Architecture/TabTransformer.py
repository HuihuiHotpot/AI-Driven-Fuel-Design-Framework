import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    def __init__(self, num_categories, category_embedding_dim,
                 d_model, nhead, num_transformer_layers, dropout=0.1):
        super().__init__()

        self.category_embeddings = nn.ModuleList([
            nn.Embedding(num_cat, category_embedding_dim) for num_cat in num_categories
        ])

        self.category_projection = nn.Linear(category_embedding_dim, d_model)

        self.continuous_projection = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self, category_features, continuous_features):
        category_tokens = torch.stack([
            self.category_projection(embedding(category_features[:, i]))
            for i, embedding in enumerate(self.category_embeddings)
        ], dim=1)  # [batch_size, num_categories, d_model]

        continuous_tokens = self.continuous_projection(
            continuous_features.unsqueeze(-1)  # [batch_size, num_continuous, 1]
        )  # [batch_size, num_continuous, d_model]

        tokens = torch.cat([category_tokens, continuous_tokens], dim=1)  # [batch_size, num_tokens, d_model]

        contextual_embeddings = self.transformer_encoder(tokens)  # [batch_size, num_tokens, d_model]

        return contextual_embeddings