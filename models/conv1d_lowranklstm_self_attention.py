import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv1D, self).__init__()
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

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise convolution
        x = self.pointwise(x)  # Pointwise convolution
        return x


class ResidualDepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResidualDepthwiseConv1D, self).__init__()
        self.depthwise = DepthwiseConv1D(in_channels, out_channels, kernel_size, stride, padding)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.depthwise(x)
        out += self.shortcut(x)  # Residual connection
        return out


class Attention(nn.Module):  ## simple additive attention 
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)  # Shape: (batch_size, sequence_length, 1)
        weighted_output = torch.sum(attention_weights * x, dim=1)  # Shape: (batch_size, hidden_size)
        return weighted_output


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


class Conv1D_LowRankLSTM_Self_Attention(nn.Module):
    def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
        super(Conv1D_LowRankLSTM_Self_Attention, self).__init__()
        self.input_size = input_shape[3]  # Number of sensor channels
        self.hidden_size = config["hidden_size"]
        self.rank = config["rank"]
        self.output_size = nb_classes
        self.num_layers = config["nb_layers_lstm"]
        

        # Three layers of Depthwise Conv1D
        self.conv1 = DepthwiseConv1D(in_channels=self.input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = DepthwiseConv1D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = DepthwiseConv1D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Three layers of Residual Depthwise Conv1D
        #self.conv1 = ResidualDepthwiseConv1D(in_channels=self.input_size, out_channels=32, kernel_size=3, padding=1)
        #self.conv2 = ResidualDepthwiseConv1D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        #self.conv3 = ResidualDepthwiseConv1D(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Low-rank LSTM
        self.low_rank_lstm = LowRankLSTM(input_size=128, hidden_size=self.hidden_size, rank=self.rank, num_layers=self.num_layers)

        # Attention mechanism
        #self.attention = Attention(hidden_size)
        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_size=self.hidden_size, num_heads=2)

        # Fully connected layer for regression output
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):

        x = x.squeeze(1) ## (B, L, C)

        # Input shape: (batch_size, sequence_length, input_channels)
        batch_size, sequence_length, input_channels = x.size()

        # Reshape for Conv1D: (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)

        # Apply three layers of Residual Depthwise Conv1D
        x = self.conv1(x)  # Shape: (batch_size, 32, sequence_length)
        x = self.conv2(x)  # Shape: (batch_size, 64, sequence_length)
        x = self.conv3(x)  # Shape: (batch_size, 128, sequence_length)

        # Reshape back to (batch_size, sequence_length, 128) for LSTM
        x = x.permute(0, 2, 1)

        # Initialize hidden and cell states for LSTM
        h = torch.zeros(batch_size, self.low_rank_lstm.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.low_rank_lstm.hidden_size).to(x.device)

        # Process each time step in the sequence with LowRankLSTM
        lstm_outputs = []
        for t in range(sequence_length):
            x_t = x[:, t, :]  # Get the t-th time step
            h, c = self.low_rank_lstm(x_t, (h, c))
            h = h.double()   ##  float32 to float64
            lstm_outputs.append(h)
        #lstm_outputs = lstm_outputs.double()   ##  float32 to float64

        # Stack LSTM outputs and apply attention
        lstm_outputs = torch.stack(lstm_outputs, dim=1)  # Shape: (batch_size, sequence_length, hidden_size)
        
        ## if using simple additive attenton:
        #weighted_output = self.attention(lstm_outputs)  # Shape: (batch_size, hidden_size)
        # Fully connected layer for regression
        #output = self.fc(weighted_output)  # Shape: (batch_size, output_size)

        
        ## if using multi-head self-attention:
        weighted_output = self.attention(lstm_outputs)  # Shape: (batch_size, sequence_length, hidden_size)
        # Use the final time step's output for regression
        final_output = weighted_output[:, -1, :]  # Shape: (batch_size, hidden_size)
        # Fully connected layer for regression
        output = self.fc(final_output)  # Shape: (batch_size, output_size)


        
        return output


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