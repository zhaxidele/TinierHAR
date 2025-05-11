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


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, out_channels, kernel_size),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, (3, 1), stride=(2, 1), padding=(1, 0)),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, (1, 1), stride=(2, 1)),
#             nn.BatchNorm2d(out_channels)
#         ) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.conv(x) + self.shortcut(x)


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, out_channels, kernel_size, dilation),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 1))  # Downsample temporal dimension by 2
#         )
        
#         # Shortcut path with matching downsampling
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, (1, 1)),
#             nn.BatchNorm2d(out_channels),
#             nn.MaxPool2d((2, 1))  # Match temporal downsampling
#         ) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.conv(x) + self.shortcut(x)

# class ConvBlock_1(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         self.conv = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, out_channels, kernel_size),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 1))  # Downsample temporal dimension by 2
#         )
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, (1, 1)),
#             nn.BatchNorm2d(out_channels),
#             nn.MaxPool2d((2, 1))  # Match temporal downsampling
#         ) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.conv(x) + self.shortcut(x)






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




# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1, use_maxpool=True, maxpool_axis=0):
#         super().__init__()
#         self.use_maxpool = use_maxpool
#         self.maxpool_axis = maxpool_axis
        
#         # Define maxpool layer based on maxpool_axis
#         self.pool_layer = nn.MaxPool2d((2, 1)) if maxpool_axis == 0 else nn.MaxPool2d((1, 2))

#         # Main convolution path
#         conv_layers = [
#             DepthwiseSeparableConv(in_channels, out_channels, kernel_size, dilation),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         if self.use_maxpool:
#             conv_layers.append(self.pool_layer)
        
#         self.conv = nn.Sequential(*conv_layers)

#         # Shortcut path
#         self.shortcut = self._create_shortcut(in_channels, out_channels)

#     def _create_shortcut(self, in_channels, out_channels):
#         layers = []
#         if in_channels != out_channels:
#             layers.append(nn.Conv2d(in_channels, out_channels, (1, 1)))
#             layers.append(nn.BatchNorm2d(out_channels))
#         if self.use_maxpool:
#             layers.append(self.pool_layer)  # Use the same pooling as main path
#         if not layers:
#             return nn.Identity()
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         return self.conv(x) + self.shortcut(x)



# class DeepConvGRU(nn.Module):
#     def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
#         super(DeepConvGRU, self).__init__()
#         self.input_channels = input_shape[3]  # Number of sensor channels
#         self.seq_length = input_shape[2] 
#         self.nb_conv_block = config["nb_conv_blocks"]
#         self.nb_units_gru = config["nb_units_gru"]
#         self.nb_filters = config["nb_filters"]
#         self.drop_prob = config["drop_prob"]
#         self.nb_classes     = nb_classes
        
#         self.conv_blocks = nn.Sequential(
#             #ConvBlock(1, self.nb_filters, kernel_size=5, dilation=1),       # (B, self.nb_filters, T/2, C)
#             #ConvBlock(self.nb_filters, 2*self.nb_filters, kernel_size=5, dilation=1),     # (B, 2*self.nb_filters, T/4, C)
#             #ConvBlock_1(2*self.nb_filters, 2*self.nb_filters, kernel_size=5),
#             #ConvBlock_1(2*self.nb_filters, 2*self.nb_filters, kernel_size=5)
#             ConvBlock(1, self.nb_filters, kernel_size=5, dilation=1, use_maxpool=True, shortcut=True),
#             ConvBlock(self.nb_filters, 2*self.nb_filters, kernel_size=5, dilation=1, use_maxpool=True, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5, dilation=1, use_maxpool=False, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5, dilation=1, use_maxpool=False, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5, dilation=1, use_maxpool=False, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5, dilation=1, use_maxpool=False, shortcut=True)
#         )
        
#         # # Calculate GRU input dimension
#         # with torch.no_grad():
#         #     dummy = torch.randn(1, 1, self.seq_length, self.input_channels)
#         #     out = self.conv_blocks(dummy)
#         #     self.gru_input_dim = out.size(1) * out.size(3)  # Channels × input_channels

#         # self.gru = nn.GRU(self.gru_input_dim, self.nb_units_gru, batch_first=True)
#         # self.fc = nn.Linear(self.nb_units_gru, nb_classes)
#         # # define dropout layer
#         # self.dropout = nn.Dropout(self.drop_prob)


#         # Auto-configure GRU input dimension
#         with torch.no_grad():
#             dummy = torch.randn(1, 1, self.seq_length, self.input_channels)
#             out = self.conv_blocks(dummy)
#             gru_input_dim = out.size(1) * out.size(3)  # Channels × input_channels
#         # Bidirectional GRU with compressed hidden size
#         self.gru = nn.GRU(
#             input_size=gru_input_dim,
#             hidden_size=self.nb_units_gru,  # Each direction: 16 → total 32
#             bidirectional=True,
#             batch_first=True
#         )


#         # Attention-based temporal pooling
#         self.attention = nn.Linear(2*self.nb_units_gru, 1)  # 2*16 for bidirectional
#         self.classifier = nn.Sequential(
#             #nn.Dropout(0.5),
#             nn.Linear(2*self.nb_units_gru, self.nb_classes)
#         )

#         self.dropout = nn.Dropout(self.drop_prob)


#     def forward(self, x):
#         # Input shape: (B, 1, T, C)
#         x = self.conv_blocks(x)
        
#         # Reshape for GRU: (B, C, T', C_in) → (B, T', C*C_in)
#         B, C, T, C_in = x.shape
#         x = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        
#         x = self.dropout(x)



#         # Bidirectional GRU processing
#         x, _ = self.gru(x)  # (B,T,32)
        
#         # Learnable temporal aggregation
#         attn_weights = torch.softmax(self.attention(x), dim=1)
#         x = torch.sum(attn_weights * x, dim=1)  # (B,32)
#         # Use last hidden state for classification
#         #x = x[:, -1, :]  # Take final timestep output
        
#         return self.classifier(x)


#         # # GRU processing
#         # x, _ = self.gru(x)
#         # x = x[:, -1, :]  # Last timestep
        
#         # return self.fc(x)
    
#     def number_of_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)



class DeepConvGRU(nn.Module):
    def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
        super(DeepConvGRU, self).__init__()
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


# #### Ablation study 1: remove the convblocks of TinyHAR++:
# #class DeepConvGRU_NoConvBlocks(nn.Module):
# class DeepConvGRU(nn.Module):
#     def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
#         super(DeepConvGRU, self).__init__()
#         self.input_channels = input_shape[3]  # Number of sensor channels (C)
#         self.seq_length = input_shape[2]      # Original sequence length (T)
#         self.nb_units_gru = config["nb_units_gru"]
#         self.drop_prob = config["drop_prob"]
#         self.nb_classes = nb_classes

#         # Direct GRU processing of raw input
#         self.gru = nn.GRU(
#             input_size=self.input_channels,  # Now matches sensor channels (C)
#             hidden_size=self.nb_units_gru,
#             bidirectional=True,
#             batch_first=True
#         )

#         self.attention = nn.Linear(2*self.nb_units_gru, 1)
#         self.classifier = nn.Sequential(
#             nn.Linear(2*self.nb_units_gru, self.nb_classes)
#         )
#         self.dropout = nn.Dropout(self.drop_prob)

#     def forward(self, x):
#         # Input shape: (B, 1, T, C)
#         # Remove channel dimension and reshape for GRU
#         x = x.squeeze(1)  # (B, T, C)
        
#         #x = self.dropout(x)
        
#         # GRU processing - input shape: (B, T, C)
#         x, _ = self.gru(x)  # Output shape: (B, T, 2*hidden_size)
        
#         # Attention pooling
#         attn_weights = torch.softmax(self.attention(x), dim=1)  # (B, T, 1)
#         x = torch.sum(attn_weights * x, dim=1)  # (B, 2*hidden_size)
        
#         return self.classifier(x)






# #### Ablation study 2: remove the GRU block of TinyHAR++:
# #class DeepConvGRU_NoConvBlocks(nn.Module):
# class DeepConvGRU(nn.Module):
#     def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
#         super(DeepConvGRU, self).__init__()
#         self.input_channels = input_shape[3]  # Number of sensor channels
#         self.seq_length = input_shape[2]
#         self.nb_filters = config["nb_filters"]
#         self.drop_prob = config["drop_prob"]
#         self.nb_classes = nb_classes

#         self.conv_blocks = nn.Sequential(
#             ConvBlock(1, self.nb_filters, kernel_size=5, dilation=1, 
#                      use_maxpool=True, shortcut=True),
#             ConvBlock(self.nb_filters, 2*self.nb_filters, kernel_size=5, 
#                      dilation=1, use_maxpool=True, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5,
#                      dilation=1, use_maxpool=False, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5,
#                      dilation=1, use_maxpool=False, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5,
#                      dilation=1, use_maxpool=False, shortcut=True),
#             ConvBlock(2*self.nb_filters, 2*self.nb_filters, kernel_size=5,
#                      dilation=1, use_maxpool=False, shortcut=True)
#         )

#         # Global temporal-spatial pooling
#         self.pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Reduces T and C dimensions to 1x1
#             nn.Flatten()
#         )
        
#         self.classifier = nn.Linear(2*self.nb_filters, self.nb_classes)
#         self.dropout = nn.Dropout(self.drop_prob)

#     def forward(self, x):
#         # Input shape: (B, 1, T, C)
#         x = self.conv_blocks(x)  # (B, 2*nb_filters, T', C')
        
#         # Global average pooling
#         x = self.pool(x)  # (B, 2*nb_filters)
#         x = self.dropout(x)
        
#         return self.classifier(x)

#     def number_of_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)