import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import

class Attention(nn.Module):
    """Simple attention mechanism for temporal weighting"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attn_weights = F.softmax(self.attention(lstm_output).squeeze(-1), dim=1)  # Added dim for softmax
        return torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)

class ChannelAttention(nn.Module):
    """Simple channel attention mechanism"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        weights = self.fc(self.pool(x).view(b, c))  # (B, C)
        return x * weights.unsqueeze(-1)  # (B, C, T)

class SeparableConvBlock(nn.Module):
    """Depthwise separable convolution block with BN and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv1d(in_channels, in_channels, kernel_size, 
                     padding=kernel_size//2, groups=in_channels),
            # Pointwise
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
            #nn.MaxPool1d(2)
        )
        
    def forward(self, x):
        return self.conv(x)


class LSTM_Attention(nn.Module):
    def __init__(self, input_shape, num_classes, config):
        super(LSTM_Attention, self).__init__()
        _, _, self.seq_len, self.input_dim = input_shape
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["nb_layers_lstm"]
        self.drop_prob = config["drop_prob"]
        self.conv_filter_num = config["conv_filter_num"]


        # Conv blocks with channel attention
        self.conv_blocks = nn.Sequential(
            SeparableConvBlock(self.input_dim, self.conv_filter_num, 5),
            SeparableConvBlock(self.conv_filter_num, self.conv_filter_num, 5),
            ChannelAttention(self.conv_filter_num, reduction=8),
            SeparableConvBlock(self.conv_filter_num, self.conv_filter_num, 5),
            SeparableConvBlock(self.conv_filter_num, self.conv_filter_num, 5),
            ChannelAttention(self.conv_filter_num, reduction=8)
        )
        


        self.lstm = nn.LSTM(
            input_size=self.conv_filter_num,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.drop_prob if self.num_layers > 1 else 0
        )
        self.attention = Attention(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)  # (B, C, T)
        x = self.conv_blocks(x)           # (B, 64, T)
        x = x.transpose(1, 2)             # (B, T, 64)
        
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        return self.fc(attn_out)


# Test execution
if __name__ == "__main__":
    # Configuration
    config = {
        "hidden_size": 64,
        "nb_layers_lstm": 1,
        "drop_prob": 0.3
    }
    
    # Test parameters
    batch_size = 32
    seq_len = 100
    features = 27
    num_classes = 12
    input_shape = (batch_size, 1, seq_len, features)
    
    # Initialize model
    model = LSTM_Attention(  # Fixed class name
        input_shape=input_shape,
        num_classes=num_classes,
        config=config
    )
    
    # Test forward pass
    x = torch.randn(*input_shape)
    output = model(x)
    print("Output shape:", output.shape)  # Should be (32, 12)
    total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", total_params)