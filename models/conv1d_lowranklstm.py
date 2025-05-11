import torch
import torch.nn as nn
import torch.optim as optim

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
        self.batchnorm1 =  nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise convolution
        x = self.pointwise(x)  # Pointwise convolution
        x = self.batchnorm1(x)  
        x = self.relu(x)
        return x


class Conv1D_LowRankLSTM(nn.Module):
    def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
    #def __init__(self, input_size, hidden_size, rank, num_layers=1, output_size=3):
        super(Conv1D_LowRankLSTM, self).__init__()
        #self.input_size = input_size  # Number of sensor channels
        self.input_size = input_shape[3]  # Number of sensor channels
        self.hidden_size = config["hidden_size"]
        self.rank = config["rank"]
        self.output_size = nb_classes
        self.num_layers = config["nb_layers_lstm"]
        self.drop_prob = config["drop_prob"]

        # Three layers of Depthwise Conv1D
        self.conv1 = DepthwiseConv1D(in_channels=self.input_size, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = DepthwiseConv1D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = DepthwiseConv1D(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = DepthwiseConv1D(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        # Low-rank LSTM
        #self.low_rank_lstm = LowRankLSTM(input_size=32, hidden_size=self.hidden_size, rank=self.rank, num_layers=self.num_layers )
        # Two LowRankLSTM layers
        self.low_rank_lstm1 = LowRankLSTM(input_size=32, hidden_size=self.hidden_size, rank=self.rank)
        self.low_rank_lstm2 = LowRankLSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, rank=self.rank)

        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)

        # Fully connected layer for regression output
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        #print("x shape: ", x.shape)
        #x shape: (B, 1, L, C)

        x = x.squeeze(1) ## (B, L, C)
        #x shape: (batch_size, input_size, sequence_length)
        #x = x.permute(0, 2, 1)

        # x shape: (batch_size, sequence_length, input_size(channel))
        batch_size, sequence_length, input_size = x.size()

        # Reshape x for Conv1D: (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)

        # Apply three layers of Depthwise Conv1D
        x = self.conv1(x)  # Shape: (batch_size, 32, sequence_length)
        x = self.conv2(x)  # Shape: (batch_size, 64, sequence_length)
        x = self.conv3(x)  # Shape: (batch_size, 64, sequence_length)
        x = self.conv4(x)  # Shape: (batch_size, 128, sequence_length)

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
    
    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Define the LowRankLSTM (same as before)
class LowRankLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, rank, bias=True):
        super(LowRankLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
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

        #print(f"x dtype: {x.dtype}")
        #print(f"W_i dtype: {W_i.dtype}")
        #print(f"h_prev dtype: {h_prev.dtype}")
        #print(f"W_h dtype: {W_h.dtype}")
        #print(f"self.bias_i dtype: {self.bias_i.dtype}")
        #print(f"self.bias_h dtype: {self.bias_h.dtype}")
        #print(f"self.bias_h dtype: {self.bias_h.dtype}")

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


'''
# Example usage for sensor data
input_size = 24  # Number of sensor channels
hidden_size = 64  # Hidden state size
rank = 16  # Rank for low-rank factorization
output_size = 3  # Output size (e.g., 3 continuous values for regression)
batch_size = 128  # Batch size
sequence_length = 200  # Number of samples (time steps)

# Create the model
model = Conv1D_LowRankLSTM(input_size, hidden_size, rank, output_size=output_size)

# Create some dummy sensor data
x = torch.randn(batch_size, sequence_length, input_size)  # Shape: (batch_size, sequence_length, input_size)

# Forward pass
output = model(x)
print("Output shape:", output.shape)  # Should be (batch_size, output_size)
'''