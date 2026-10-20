import torch
import torch.nn as nn
import math

class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size=30002, d_model=256, nhead=4, num_layers=2, dim_feedforward=1024, num_classes=2, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 3. Classifier
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] (input_ids)
        """
        # x shape: [batch_size, seq_len]
        
        # Create padding mask: [batch_size, seq_len]
        # Assuming 0 is padding index
        padding_mask = (x == 0)
        
        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer
        # output: [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Global Average Pooling (ignoring padding would be better, but simple mean is standard baseline)
        # Alternatively, take the first token if we used CLS, but we didn't.
        # Let's use mean over non-padding tokens for better accuracy.
        
        # Mask out padding for mean calculation
        mask = (~padding_mask).float().unsqueeze(-1) # [batch, seq_len, 1]
        x = x * mask
        sum_x = x.sum(dim=1) # [batch, d_model]
        token_counts = mask.sum(dim=1).clamp(min=1) # [batch, 1]
        mean_x = sum_x / token_counts
        
        # Classifier
        logits = self.classifier(mean_x)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SentimentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets):
        return self.criterion(logits, targets)
