import torch
import torch.nn as nn



### Following code is DeepConvGRU (TinyHAR++), now comment out (line 6 to line 255) for an ablation study. 
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (dilation * (kernel_size - 1) + 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, (kernel_size, 1),
            padding=(padding, 0),  # Maintain temporal length
            dilation=(dilation, 1),  # Add dilation here
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, use_maxpool=True, shortcut=True):
        super().__init__()
        self.use_maxpool = use_maxpool
        self.shortcut = shortcut
        
        # Main convolution path
        conv_layers = [
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if self.use_maxpool:
            conv_layers.append(nn.MaxPool2d((2, 1)))
        
        self.conv = nn.Sequential(*conv_layers)

        # Shortcut path
        self.F_shortcut = self._create_shortcut(in_channels, out_channels)

    def _create_shortcut(self, in_channels, out_channels):
        layers = []
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, (1, 1)))
            layers.append(nn.BatchNorm2d(out_channels))
        if self.use_maxpool:
            layers.append(nn.MaxPool2d((2, 1)))  # Always apply pooling if used in main path
        if not layers:
            return nn.Identity()
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.shortcut:
            return self.conv(x) + self.F_shortcut(x)
        else:
            return self.conv(x)



class TinierHAR_Model(nn.Module):
    def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
        super(TinierHAR_Model, self).__init__()
        self.input_channels = input_shape[3]
        self.seq_length = input_shape[2]
        self.nb_conv_blocks = config["nb_conv_blocks"]  # Get number of blocks from config
        self.nb_units_gru = config["nb_units_gru"]
        self.nb_filters = config["nb_filters"]
        self.drop_prob = config["drop_prob"]
        self.nb_classes = nb_classes

        # Build conv blocks dynamically
        conv_blocks = []
        # First two blocks with maxpool
        conv_blocks.append(
            ConvBlock(1, self.nb_filters, kernel_size=5, dilation=1, 
                     use_maxpool=True, shortcut=True)
        )
        conv_blocks.append(
            ConvBlock(self.nb_filters, 2*self.nb_filters, kernel_size=5, 
                     dilation=1, use_maxpool=True, shortcut=True)
        )
        
        # Additional blocks without maxpool
        for _ in range(self.nb_conv_blocks):
            conv_blocks.append(
                ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5,
                         dilation=1, use_maxpool=False, shortcut=True)
            )
            
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Auto-configure GRU input dimension
        with torch.no_grad():
            dummy = torch.randn(1, 1, self.seq_length, self.input_channels)
            out = self.conv_blocks(dummy)
            gru_input_dim = out.size(1) * out.size(3)

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=self.nb_units_gru,
            bidirectional=True,
            batch_first=True
        )

        self.attention = nn.Linear(2*self.nb_units_gru, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2*self.nb_units_gru, self.nb_classes)
        )
        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        x = self.conv_blocks(x)
        B, C, T, C_in = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        x = self.dropout(x)
        x, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        return self.classifier(x)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

