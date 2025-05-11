import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution with batchnorm and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ChannelAttention(nn.Module):
    """Simplified channel attention with fewer parameters."""
    def __init__(self, total_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(total_channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(total_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, total_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, seq_len, total_channels)
        batch_size, seq_len, total_channels = x.size()
        x = x.transpose(1, 2)  # (batch_size, total_channels, seq_len)
        avg_out = self.avg_pool(x).squeeze(-1)  # (batch_size, total_channels)
        weights = self.fc(avg_out).unsqueeze(-1)  # (batch_size, total_channels, 1)
        x = x * weights
        return x.transpose(1, 2)  # (batch_size, seq_len, total_channels)

class TemporalAttention(nn.Module):
    """Lightweight temporal self-attention with adaptive dimensions."""
    def __init__(self, total_channels, num_heads=2, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        projected_dim = total_channels // 2
        
        # Dynamically adjust dimensions to be divisible by num_heads
        self.adjusted_proj_dim = math.ceil(projected_dim / num_heads) * num_heads
        self.d_k = self.adjusted_proj_dim // num_heads

        self.q_proj = nn.Linear(total_channels, self.adjusted_proj_dim)
        self.k_proj = nn.Linear(total_channels, self.adjusted_proj_dim)
        self.v_proj = nn.Linear(total_channels, self.adjusted_proj_dim)
        self.out_proj = nn.Linear(self.adjusted_proj_dim, total_channels)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project and split into heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention mechanism
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Combine heads and project back
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.adjusted_proj_dim)
        return self.out_proj(attn_output)

# class SelfAttentionHAR(nn.Module):
#     def __init__(self, input_shape, num_classes, config):
#         super(SelfAttentionHAR, self).__init__()
#         self.seq_len = input_shape[2]
#         self.total_channels = input_shape[3]  # Should be 27 in test case

#         # Fixed convolutional layers (input channels match)
#         self.conv_layers = nn.Sequential(
#             SeparableConv1d(self.total_channels, 32, kernel_size=5),
#             SeparableConv1d(32, 64, kernel_size=3),
#             SeparableConv1d(64, self.total_channels, kernel_size=3),
#             SeparableConv1d(self.total_channels, self.total_channels, kernel_size=3)
#         )

#         self.channel_attention = ChannelAttention(self.total_channels, reduction_ratio=8)
#         self.temporal_attention = TemporalAttention(self.total_channels, num_heads=2)

#         # Final layers
#         self.fc = nn.Linear(self.total_channels, num_classes)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         # Input: (batch_size, 1, seq_len, total_channels)
#         x = x.squeeze(1)  # (batch_size, seq_len, total_channels)
        
#         # Process through convolutional layers
#         x = x.transpose(1, 2)  # (batch_size, total_channels, seq_len)
#         x = self.conv_layers(x)
#         x = x.transpose(1, 2)  # (batch_size, seq_len, total_channels)

#         # Attention processing
#         x = self.channel_attention(x)
#         x = self.temporal_attention(x)

#         # Aggregate and classify
#         x = x.mean(dim=1)  # Average over sequence
#         x = self.dropout(x)
#         return self.fc(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embed size must be divisible by num_heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.size()

        # Linear transformations for Q, K, V
        Q = self.query(x)  # Shape: (batch_size, seq_length, embed_size)
        K = self.key(x)    # Shape: (batch_size, seq_length, embed_size)
        V = self.value(x)  # Shape: (batch_size, seq_length, embed_size)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)  # Shape: (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads and pass through final linear layer
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, embed_size)
        out = self.fc_out(out)

        return out


import torch
import torch.nn as nn

class SelfAttentionHAR(nn.Module):
    def __init__(self, input_shape, num_classes, config):
        super(SelfAttentionHAR, self).__init__()
        _, _, seq_len, in_channels = input_shape
        
        # Feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5,  # Depthwise
                     padding=2, groups=in_channels),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),  # Pointwise
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(in_channels, 32, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=2)
        
        # Classification
        self.flatten_size = seq_len * 32
        self.fc = nn.Linear(self.flatten_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input: (batch, 1, seq_len, channels)
        x = x.squeeze(1).transpose(1, 2)  # (B, C, T)
        
        # Feature learning
        x = self.conv(x)                   # (B, C, T)
        x = x.transpose(1, 2)              # (B, T, C)
        
        # Sequence modeling
        x, _ = self.lstm(x)                # (B, T, 32)
        x = x.permute(1, 0, 2)             # (T, B, 32)
        x, _ = self.attention(x, x, x)     # (T, B, 32)
        x = x.permute(1, 0, 2)             # (B, T, 32)
        
        # Final classification
        x = x.reshape(x.size(0), -1)       # (B, T*32)
        x = self.dropout(x)
        return self.fc(x)                  # (B, num_classes)





# class SelfAttentionHAR(nn.Module):
#     def __init__(self, input_shape, num_classes, config):
#         super(SelfAttentionHAR, self).__init__()
#         self.seq_len = input_shape[2]
#         self.total_channels = input_shape[3]
#         self.hidden_size = 128  # Unified dimension

#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             SeparableConv1d(self.total_channels, 32, kernel_size=5),
#             SeparableConv1d(32, 64, kernel_size=5),
#             SeparableConv1d(64, 64, kernel_size=5),
#             SeparableConv1d(64, self.hidden_size, kernel_size=5)
#         )

#         self.channel_attention = ChannelAttention(self.hidden_size, reduction_ratio=8)
#         self.attention = MultiHeadAttention(embed_size=self.hidden_size, num_heads=2)
#         self.fc = nn.Linear(self.hidden_size, num_classes)

#     def forward(self, x):
#         # Input: (batch_size, 1, seq_len, total_channels)
#         x = x.squeeze(1)  # (batch_size, seq_len, total_channels)
        
#         # Conv processing
#         x = x.transpose(1, 2)  # (batch_size, total_channels, seq_len)
#         x = self.conv_layers(x)  # (batch_size, hidden_size, seq_len)
#         x = x.transpose(1, 2)    # (batch_size, seq_len, hidden_size)

#         # Attention processing
#         x = self.channel_attention(x)
#         x = self.attention(x)
        
#         # Aggregate and classify
#         x = x.mean(dim=1)  # (batch_size, hidden_size)
#         return self.fc(x)  # (batch_size, num_classes)



# class SelfAttentionHAR(nn.Module):
#     def __init__(self, input_shape, num_classes, config):
#         super(SelfAttentionHAR, self).__init__()
#         self.seq_len = input_shape[2]
#         self.total_channels = input_shape[3]
#         self.hidden_size = 128
#         self.inter_channel = 128

#         # Convolutional layers with kernel_size=5
#         self.conv_layers = nn.Sequential(
#             SeparableConv1d(self.total_channels, 32, kernel_size=5),
#             SeparableConv1d(32, 64, kernel_size=5),
#             SeparableConv1d(64, 64, kernel_size=5),
#             SeparableConv1d(64, self.inter_channel, kernel_size=5)
#         )

#         self.channel_attention = ChannelAttention(self.inter_channel, reduction_ratio=8)
#         #self.temporal_attention = TemporalAttention(self.hidden_size, num_heads=2)

#         # Attention mechanism
#         #self.attention = Attention(hidden_size)
#         # Multi-head self-attention
#         self.attention = MultiHeadAttention(embed_size=self.hidden_size, num_heads=2)

#         # Fully connected layer for regression output
#         self.fc = nn.Linear(self.hidden_size, num_classes)



#         # New linear layers after temporal attention
#         #self.post_attention = nn.Sequential(
#         #    nn.Dropout(0.3),
#         #    nn.Linear(self.total_channels, 64),
#         #    nn.ReLU(),
#         #    nn.Dropout(0.2)
#         #)
        
#         #self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # Input: (batch_size, 1, seq_len, total_channels)
#         x = x.squeeze(1)  # (batch_size, seq_len, total_channels)
        
#         # Conv processing
#         x = x.transpose(1, 2)
#         x = self.conv_layers(x)
#         x = x.transpose(1, 2)

#         # Attention processing
#         x = self.channel_attention(x)
#         #x = self.temporal_attention(x)
#         weighted_output = self.attention(x)

#         final_output = weighted_output[:, -1, :]  # Shape: (batch_size, hidden_size)
#         # Fully connected layer for regression
#         output = self.fc(final_output)  # Shape: (batch_size, output_size)

#         # New classification head
#         #x = self.post_attention(x)  # (batch_size, seq_len, 64)
#         #x = x.mean(dim=1)  # (batch_size, 64)
#         return output




# Test the model
if __name__ == "__main__":
    # Example parameters
    batch_size = 32
    seq_len = 100
    total_channels = 27  # e.g., 9 sensors * 3 channels (PAMAP2-like)
    num_classes = 12
    input_shape = (batch_size, 1, seq_len, total_channels)

    model = SelfAttentionHAR(input_shape, num_classes, total_channels)
    x = torch.randn(*input_shape)
    output = model(x)
    print("Output shape:", output.shape)  # Expected: (32, 12)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", total_params)