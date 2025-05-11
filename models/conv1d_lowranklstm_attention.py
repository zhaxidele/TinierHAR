import torch
import torch.nn as nn
import torch.optim as optim


# class ChannelSqueezeExcitation(nn.Module):
#     """Channel attention branch (similar to SE block)."""
#     def __init__(self, channels, reduction_ratio=16):
#         super().__init__()
#         self.gap = nn.AdaptiveAvgPool1d(1)  # Global average pooling
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction_ratio, channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # Squeeze
#         batch, channels, _ = x.size()
#         squeezed = self.gap(x).view(batch, channels)
#         # Excitation
#         weights = self.fc(squeezed).view(batch, channels, 1)
#         return x * weights  # Channel-wise reweighting

# class SpatialSqueezeExcitation(nn.Module):
#     """Spatial attention branch."""
#     def __init__(self, channels):
#         super().__init__()
#         self.spatial_conv = nn.Conv1d(channels, 1, kernel_size=1)  # 1x1 conv
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Compute spatial attention weights
#         spatial_weights = self.spatial_conv(x)  # Shape: (batch, 1, seq_len)
#         spatial_weights = self.sigmoid(spatial_weights)
#         return x * spatial_weights  # Spatial reweighting

# class ConcurrentSpatialChannelSELayer(nn.Module):
#     """Combines channel and spatial attention (scSE block)."""
#     def __init__(self, channels, reduction_ratio=16):
#         super().__init__()
#         self.cSE = ChannelSqueezeExcitation(channels, reduction_ratio)
#         self.sSE = SpatialSqueezeExcitation(channels)

#     def forward(self, x):
#         # Apply channel and spatial attention separately
#         cSE_out = self.cSE(x)
#         sSE_out = self.sSE(x)
#         # Combine outputs (summation)
#         return cSE_out + sSE_out



class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv1D, self).__init__()
        self.out_channels = out_channels
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,  # Depthwise: out_channels = in_channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Groups = in_channels for depthwise
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,  # Pointwise: combine channels
            kernel_size=1,  # 1x1 convolution
        )
        self.batchnorm =  nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise convolution
        x = self.pointwise(x)  # Pointwise convolution
        x = self.batchnorm(x)  # Pointwise convolution
        x = self.relu(x)
        return x


# class DepthwiseConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super(DepthwiseConv1D, self).__init__()
#         self.depthwise = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=in_channels,  # Depthwise: out_channels = in_channels
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             groups=in_channels,  # Groups = in_channels for depthwise
#         )
#         self.pointwise = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,  # Pointwise: combine channels
#             kernel_size=1,  # 1x1 convolution
#         )

#     def forward(self, x):
#         x = self.depthwise(x)  # Depthwise convolution
#         x = self.pointwise(x)  # Pointwise convolution
#         return x



class Attention(nn.Module):  ## simple additive attention 
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)  # Shape: (batch_size, sequence_length, 1)
        weighted_output = torch.sum(attention_weights * x, dim=1)  # Shape: (batch_size, hidden_size)
        return weighted_output



# class ResidualDepthwiseConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super(ResidualDepthwiseConv1D, self).__init__()
#         # First DepthwiseConv1D layer
#         self.conv1 = DepthwiseConv1D(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
        
#         # Batch normalization and activation after the first convolution
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu1 = nn.ReLU()

#         # Second DepthwiseConv1D layer
#         self.conv2 = DepthwiseConv1D(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#         # Batch normalization after the second convolution
#         self.bn2 = nn.BatchNorm1d(out_channels)

#         # Skip connection: 1x1 convolution to match dimensions if needed
#         self.skip_conv = None
#         if in_channels != out_channels or stride != 1:
#             self.skip_conv = nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=stride,
#                 padding=0,
#             )
#             self.skip_bn = nn.BatchNorm1d(out_channels)

#         # Final activation
#         self.relu2 = nn.ReLU()

#     def forward(self, x):
#         identity = x  # Save the input for the skip connection

#         # First convolution
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)

#         # Second convolution
#         out = self.conv2(out)
#         out = self.bn2(out)

#         # Skip connection
#         if self.skip_conv is not None:
#             identity = self.skip_conv(identity)
#             identity = self.skip_bn(identity)

#         # Add skip connection to the output
#         out += identity
#         out = self.relu2(out)  # Final activation

#         return out



# Define the LowRankLSTM (same as before)
class LowRankLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, rank, num_layers=1, bias=True):
        super(LowRankLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_layers = num_layers
        self.bias = bias

        # Low-rank factorization for input-to-hidden weights
        self.U_i = nn.Parameter(torch.Tensor(input_size, rank))
        self.V_i = nn.Parameter(torch.Tensor(rank, hidden_size * 4))
        self.U_h = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.V_h = nn.Parameter(torch.Tensor(rank, hidden_size * 4))

        # Bias terms
        if bias:
            self.bias_i = nn.Parameter(torch.Tensor(hidden_size * 4))
            self.bias_h = nn.Parameter(torch.Tensor(hidden_size * 4))
        else:
            self.register_parameter('bias_i', None)
            self.register_parameter('bias_h', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.U_i)
        nn.init.kaiming_normal_(self.V_i)
        nn.init.kaiming_normal_(self.U_h)
        nn.init.kaiming_normal_(self.V_h)
        if self.bias:
            nn.init.zeros_(self.bias_i)
            nn.init.zeros_(self.bias_h)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state

        # Ensure h_prev is 2D: (batch_size, hidden_size)
        if h_prev.dim() == 3:
            h_prev = h_prev.squeeze(0)
        h_prev = h_prev.double()   ##  float32 to float64

        # Low-rank factorization for input-to-hidden transformation
        W_i = torch.mm(self.U_i, self.V_i)  # Shape: (input_size, hidden_size * 4)
        W_h = torch.mm(self.U_h, self.V_h)  # Shape: (hidden_size, hidden_size * 4)

        # Linear transformations
        gates_i = torch.mm(x, W_i) + self.bias_i  # Shape: (batch_size, hidden_size * 4)
        gates_h = torch.mm(h_prev, W_h) + self.bias_h  # Shape: (batch_size, hidden_size * 4)

        # Combine the gates
        gates = gates_i.float() + gates_h.float()

        # Split the combined gates into input, forget, cell, and output gates
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

        # Apply activations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)

        # Update cell state and hidden state
        c_next = f_gate * c_prev + i_gate * c_gate
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next

        
class Conv1D_LowRankLSTM_Attention(nn.Module):
    def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
        super(Conv1D_LowRankLSTM_Attention, self).__init__()
        self.input_size = input_shape[3]  # Number of sensor channels
        self.hidden_size = config["hidden_size"]
        self.rank = config["rank"]
        self.output_size = nb_classes
        self.num_layers = config["nb_layers_lstm"]
        self.drop_prob = config["drop_prob"]
        


        # Initial convolution
        # self.initial_conv = nn.Conv1d(
        #     in_channels=self.input_size,
        #     out_channels=64,
        #     kernel_size=3,
        #     padding=1,
        # )
        # self.initial_bn = nn.BatchNorm1d(64)
        # self.initial_relu = nn.ReLU()

        # # Residual blocks
        # self.residual_block1 = ResidualDepthwiseConv1D(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     padding=1,
        # )
        # self.residual_block2 = ResidualDepthwiseConv1D(
        #     in_channels=64,
        #     out_channels=128,
        #     kernel_size=3,
        #     padding=1,
        # )


        # Three layers of Depthwise Conv1D
        self.conv1 = DepthwiseConv1D(in_channels=self.input_size, out_channels=32, kernel_size=7, padding=4)
        self.conv2 = DepthwiseConv1D(in_channels=32, out_channels=32, kernel_size=7, padding=4)
        self.conv3 = DepthwiseConv1D(in_channels=32, out_channels=32, kernel_size=7, padding=4)
        self.conv4 = DepthwiseConv1D(in_channels=32, out_channels=32, kernel_size=7, padding=4)


        # Concurrent Spatial and Channel SE (scSE) block
        #self.scse1 = ConcurrentSpatialChannelSELayer(channels=64)
        #self.scse2 = ConcurrentSpatialChannelSELayer(channels=128)


        # Low-rank LSTM
        self.low_rank_lstm1 = LowRankLSTM(input_size=32, hidden_size=self.hidden_size, rank=self.rank, num_layers=self.num_layers)
        self.low_rank_lstm2 = LowRankLSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, rank=self.rank)

        # Attention mechanism
        self.attention = Attention(self.hidden_size)
        #self.scse3 = ConcurrentSpatialChannelSELayer(channels=self.hidden_size)


        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)

        # Fully connected layer for regression output
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        #self.fc = nn.Sequential(
        #    nn.Linear(self.hidden_size, 64),
        #    nn.LeakyReLU(inplace=True),
        #    nn.Linear(64, self.output_size),
        #)

    def forward(self, x):

        x = x.squeeze(1) ## (B, L, C)

        # Input shape: (batch_size, sequence_length, input_channels)
        batch_size, sequence_length, input_channels = x.size()

        # Reshape for Conv1D: (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)

        # Initial convolution
        # x = self.initial_conv(x)
        # x = self.initial_bn(x)
        # x = self.initial_relu(x)

        # # Residual blocks
        # x = self.residual_block1(x)
        # x = self.scse1(x)
        # x = self.residual_block2(x)
        # x = self.scse2(x)

        # Apply three layers of Residual Depthwise Conv1D
        x = self.conv1(x)  # Shape: (batch_size, 32, sequence_length)
        x = self.conv2(x)  # Shape: (batch_size, 64, sequence_length)
        x = self.conv3(x)  # Shape: (batch_size, 64, sequence_length)
        x = self.conv4(x)  # Shape: (batch_size, 128, sequence_length)


        x = self.dropout(x)

        # Reshape back to (batch_size, sequence_length, 128) for LSTM
        x = x.permute(0, 2, 1)

        # Initialize hidden and cell states for both LSTM layers
        h1 = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c1 = torch.zeros(batch_size, self.hidden_size).to(x.device)
        h2 = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c2 = torch.zeros(batch_size, self.hidden_size).to(x.device)

        lstm2_outputs = []
        for t in range(sequence_length):
            x_t = x[:, t, :]  # (batch_size, 32)
            # First LSTM layer
            h1, c1 = self.low_rank_lstm1(x_t, (h1, c1))
            h1 = h1.double()   ##  float32 to float64
            # Second LSTM layer takes output of first as input
            h2, c2 = self.low_rank_lstm2(h1, (h2, c2))
            h2 = h2.double()   ##  float32 to float64
            lstm2_outputs.append(h2)

        # Stack outputs from second LSTM: (batch_size, sequence_length, hidden_size)
        h = torch.stack(lstm2_outputs, dim=1)
        h = h[:, -1, :]  # Take last timestep: (batch_size, hidden_size)

        h = self.dropout(h)

        output = self.fc(h)  # Shape: (batch_size, output_size)
        return output

        #weighted_output = self.scse3(lstm_outputs)
        ## if using simple additive attenton:
        #weighted_output = self.attention(lstm_outputs)  # Shape: (batch_size, hidden_size)
        # Fully connected layer for regression
        #output = self.fc(weighted_output)  # Shape: (batch_size, output_size)
        
        #return output

