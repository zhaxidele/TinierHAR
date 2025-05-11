import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

class SelfAttention_interaction(nn.Module):
    """
    Self-Attention mechanism for channel interaction.
    """
    def __init__(self, sensor_channel, n_channels):
        super(SelfAttention_interaction, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))
    
    def forward(self, x):
        # Input dimensions: (batch_size, sensor_channel, feature_dim)
        f, g, h = self.query(x), self.key(x), self.value(x)
        
        # Calculate attention weights
        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)
        
        # Compute the output
        o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
        o = o.permute(0, 2, 1).contiguous()
        
        return o

class Identity(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


crosschannel_interaction = {
    "attn": SelfAttention_interaction  # applied
}


class FC(nn.Module):
    """
    Simple fully connected layer.
    """
    def __init__(self, channel_in, channel_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(channel_in, channel_out)

    def forward(self, x):
        x = self.fc(x)
        return x

crosschannel_aggregation = {
    "FC": FC    ### applied
}


class temporal_LSTM(nn.Module):
    """
    LSTM for temporal interaction.
    """
    def __init__(self, sensor_channel, filter_num):
        super(temporal_LSTM, self).__init__()
        self.lstm = nn.LSTM(filter_num, 
                            filter_num, 
                            batch_first=True)
    
    def forward(self, x):
        # Input: (batch_size, length, filter_num)
        outputs, h = self.lstm(x)
        return outputs


temporal_interaction = {
    "lstm": temporal_LSTM    ## applied
}


## for original TinyHAR
class Temporal_Weighted_Aggregation(nn.Module):
    """
    Temporal weighted aggregation mechanism.
    """
    def __init__(self, sensor_channel, hidden_dim):
        super(Temporal_Weighted_Aggregation, self).__init__()
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.weighs_activation = nn.Tanh()
        self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
        self.sm = torch.nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Input: (batch_size, sensor_channel, feature_dim)
        out = self.weighs_activation(self.fc_1(x))
        out = self.fc_2(out).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        context = x[:, -1, :] + self.gamma * context
        return context


# ## for ablation study of removing part1
# class Temporal_Weighted_Aggregation(nn.Module):
#     def __init__(self, hidden_dim):  # Sensor_channel removed
#         #super().__init__()
#         super(Temporal_Weighted_Aggregation, self).__init__()
#         self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.activation = nn.Tanh()
#         self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#         self.gamma = nn.Parameter(torch.tensor([0.]))

#     def forward(self, x):
#         # Input: (batch, temporal_length, hidden_dim)
#         scores = self.activation(self.fc_1(x))  # (B, L, H)
#         scores = self.fc_2(scores).squeeze(-1)  # (B, L)
#         weights = self.softmax(scores).unsqueeze(-1)  # (B, L, 1)
#         context = torch.sum(weights * x, dim=1)  # (B, H)
#         return x[:, -1, :] + self.gamma * context




temmporal_aggregation = {
    "tnaive": Temporal_Weighted_Aggregation    ## applied
}

class TinyHAR_Model(nn.Module):
    def __init__(
        self,
        input_shape,
        number_class,
        filter_num,
        nb_conv_layers=4,
        filter_size=5,
        cross_channel_interaction_type="attn",  # attn, transformer, identity
        cross_channel_aggregation_type="FC",  # filter, naive, FC
        temporal_info_interaction_type="lstm",  # gru, lstm, attn, transformer, identity
        temporal_info_aggregation_type="tnaive",  # naive, filter, FC
        dropout=0.1,
        activation="ReLU",
    ):
        super(TinyHAR_Model, self).__init__()
        
        self.cross_channel_interaction_type = cross_channel_interaction_type
        self.cross_channel_aggregation_type = cross_channel_aggregation_type
        self.temporal_info_interaction_type = temporal_info_interaction_type
        self.temporal_info_aggregation_type = temporal_info_aggregation_type
        
        """
        PART 1: Channel wise Feature Extraction
        Input: (Batch, filter_num, length, Sensor_channel)
        Output: (Batch, filter_num, downsampling_length, Sensor_channel)
        """
        filter_num_list = [1]
        filter_num_step = int(filter_num / nb_conv_layers)
        for i in range(nb_conv_layers - 1):
            filter_num_list.append(filter_num)
        filter_num_list.append(filter_num)

        layers_conv = []
        for i in range(nb_conv_layers):
            in_channel = filter_num_list[i]
            out_channel = filter_num_list[i + 1]
            if i % 2 == 1:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (filter_size, 1), (2, 1)),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channel)))
            else:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (filter_size, 1), (1, 1)),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channel)))
        self.layers_conv = nn.ModuleList(layers_conv)

        # Determine downsampling length
        downsampling_length = self.get_the_shape(input_shape)
        
        """
        PART 2: Cross Channel interaction
        Options: attn, transformer, identity
        Output: (Batch, filter_num, downsampling_length, Sensor_channel)
        """
        self.channel_interaction = crosschannel_interaction[cross_channel_interaction_type](input_shape[3], filter_num)
        
        """
        PART 3: Cross Channel Fusion
        Options: filter, naive, FC
        Output: (Batch, downsampling_length, filter_num)
        """
        if cross_channel_aggregation_type == "FC":
            self.channel_fusion = crosschannel_aggregation[cross_channel_aggregation_type](input_shape[3] * filter_num, 2 * filter_num)
        self.activation = nn.ReLU()

        """
        PART 4: Temporal information Extraction
        Options: gru, lstm, attn, transformer, identity
        Output: (Batch, downsampling_length, filter_num)
        """
        self.temporal_interaction = temporal_interaction[temporal_info_interaction_type](input_shape[3], 2 * filter_num)
        
        """
        PART 5: Temporal information Aggregation
        Options: naive, filter, FC
        Output: (Batch, downsampling_length, filter_num)
        """
        self.dropout = nn.Dropout(dropout)
        self.temporal_fusion = temmporal_aggregation[temporal_info_aggregation_type](input_shape[3], 2 * filter_num)
            
        # PART 6: Prediction
        self.prediction = nn.Linear(2 * filter_num, number_class)

    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)
        for layer in self.layers_conv:
            x = layer(x)
        return x.shape[2]
        
    def forward(self, x):
        # Input: (Batch, filter_num, length, Sensor_channel)
        for layer in self.layers_conv:
            x = layer(x)

        x = x.permute(0, 3, 2, 1)  # B x C x L* x F*
        
        """ Cross channel interaction """
        x = torch.cat(
            [self.channel_interaction(x[:, :, t, :]).unsqueeze(3) for t in range(x.shape[2])],
            dim=-1,
        )
        x = self.dropout(x)

        """ Cross channel fusion """
        #if self.cross_channel_aggregation_type == "FC":
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.activation(self.channel_fusion(x))
        
        """ Temporal interaction """
        x = self.temporal_interaction(x)
        
        """ Temporal fusion """
        x = self.temporal_fusion(x)
        
        y = self.prediction(x)
        return y




#### Ablation study TinyHAR_Model_Without_PART1 
# class TinyHAR_Model(nn.Module):
#     def __init__(
#         self,
#         input_shape,          # (batch, 1, time_steps, sensor_channels)
#         number_class,
#         filter_num=1,         # Forced to 1 to match raw input’s channel dimension
#         cross_channel_interaction_type="attn",
#         cross_channel_aggregation_type="FC",
#         temporal_info_interaction_type="lstm",
#         temporal_info_aggregation_type="tnaive",
#         dropout=0.1,
#     ):
#         #super().__init__()
#         super(TinyHAR_Model, self).__init__()


#         # --- Hyperparameters ---
#         _, _, self.raw_time_steps, self.sensor_channels = input_shape
#         self.filter_num = 1  # Raw input has 1 channel (PART1 removed)
#         self.hidden_dim = 2 * self.filter_num  # To match later layers

#         # --- PART 2: Cross Channel Interaction ---
#         self.channel_interaction = crosschannel_interaction[cross_channel_interaction_type](
#             self.sensor_channels, self.filter_num
#         )

#         # --- PART 3: Cross Channel Fusion ---
#         if cross_channel_aggregation_type == "FC":
#             # Input: (batch, time_steps, sensor_channels * filter_num=1)
#             self.channel_fusion = crosschannel_aggregation[cross_channel_aggregation_type](
#                 self.sensor_channels * self.filter_num, self.hidden_dim
#             )
#         self.activation = nn.ReLU()

#         # --- PART 4: Temporal Interaction ---
#         self.temporal_interaction = temporal_interaction[temporal_info_interaction_type](
#             self.sensor_channels, self.hidden_dim
#         )

#         # --- PART 5: Temporal Aggregation ---
#         self.temporal_fusion = temmporal_aggregation[temporal_info_aggregation_type](
#             self.hidden_dim  # Match LSTM’s hidden_dim
#         )

#         # --- PART 6: Prediction ---
#         self.dropout = nn.Dropout(dropout)
#         self.prediction = nn.Linear(self.hidden_dim, number_class)

#     def forward(self, x):
#         # Input: (batch, 1, raw_time_steps, sensor_channels)
#         B, _, T, C = x.shape

#         # --- PART 2: Cross Channel Interaction ---
#         # Reshape to mimic PART1’s output (B, C, T, filter_num=1)
#         x = x.permute(0, 3, 2, 1).contiguous()  # (B, C, T, 1)

#         # Apply interaction across sensor channels for each timestep
#         x = torch.cat(
#             [self.channel_interaction(x[:, :, t, :]).unsqueeze(2) for t in range(T)],
#             dim=2,
#         )  # (B, C, T, 1)

#         # --- PART 3: Cross Channel Fusion ---
#         x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, C, 1)
#         x = x.reshape(B, T, -1)  # (B, T, C * 1)
#         x = self.activation(self.channel_fusion(x))  # (B, T, hidden_dim=2)

#         # --- PART 4: Temporal Interaction (LSTM) ---
#         x = self.temporal_interaction(x)  # (B, T, hidden_dim=2)

#         # --- PART 5: Temporal Aggregation ---
#         x = self.temporal_fusion(x)  # (B, hidden_dim=2)

#         # --- PART 6: Prediction ---
#         return self.prediction(self.dropout(x))



# #### Ablation study TinyHAR_Model_Without_PART2
# class TinyHAR_Model(nn.Module):
#     def __init__(
#         self,
#         input_shape,
#         number_class,
#         filter_num,
#         nb_conv_layers=4,
#         filter_size=5,
#         cross_channel_interaction_type="attn",
#         cross_channel_aggregation_type="FC",
#         temporal_info_interaction_type="lstm",
#         temporal_info_aggregation_type="tnaive",
#         dropout=0.1,
#     ):
#         #super().__init__()
#         super(TinyHAR_Model, self).__init__()

#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction (Unchanged)
#         ########################################################################
#         filter_num_list = [1] + [filter_num] * nb_conv_layers
#         self.layers_conv = nn.ModuleList()
#         for i in range(nb_conv_layers):
#             in_channel = filter_num_list[i]
#             out_channel = filter_num_list[i + 1]
#             stride = (2, 1) if i % 2 == 1 else (1, 1)
#             self.layers_conv.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channel, out_channel, (filter_size, 1), stride),
#                     nn.ReLU(inplace=True),
#                     nn.BatchNorm2d(out_channel),
#                 )
#             )
        
#         # Calculate downsampled temporal length
#         self.downsampling_length = self.get_the_shape(input_shape)
#         self.sensor_channels = input_shape[3]  # Original sensor channels

#         ########################################################################
#         # PART 3: Cross Channel Fusion (Adjusted for missing PART2)
#         ########################################################################
#         if cross_channel_aggregation_type == "FC":
#             # Input: (sensor_channels * filter_num) → Output: 2 * filter_num
#             self.channel_fusion = crosschannel_aggregation[cross_channel_aggregation_type](
#                 self.sensor_channels * filter_num, 2 * filter_num
#             )
#         self.activation = nn.ReLU()

#         ########################################################################
#         # PART 4: Temporal Interaction (Unchanged)
#         ########################################################################
#         self.temporal_interaction = temporal_interaction[temporal_info_interaction_type](
#             self.sensor_channels, 2 * filter_num
#         )

#         ########################################################################
#         # PART 5: Temporal Aggregation (Adjusted)
#         ########################################################################
#         # Use hidden_dim = 2 * filter_num (matches LSTM output)
#         self.temporal_fusion = temmporal_aggregation[temporal_info_aggregation_type](2 * filter_num)
        
#         ########################################################################
#         # PART 6: Prediction (Unchanged)
#         ########################################################################
#         self.dropout = nn.Dropout(dropout)
#         self.prediction = nn.Linear(2 * filter_num, number_class)

#     def get_the_shape(self, input_shape):
#         x = torch.rand(input_shape)
#         for layer in self.layers_conv:
#             x = layer(x)
#         return x.shape[2]

#     def forward(self, x):
#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction
#         ########################################################################
#         for layer in self.layers_conv:
#             x = layer(x)  # Output shape: (B, F, L*, C)

#         ########################################################################
#         # PART 3: Cross Channel Fusion (Direct reshape, skip PART2)
#         ########################################################################
#         # Reshape to (B, L*, C, F) → Flatten to (B, L*, C*F)
#         x = x.permute(0, 3, 2, 1)  # (B, C, L*, F)
#         x = x.permute(0, 2, 1, 3)  # (B, L*, C, F)
#         x = x.reshape(x.size(0), x.size(1), -1)  # (B, L*, C*F)
#         x = self.activation(self.channel_fusion(x))  # (B, L*, 2F)

#         ########################################################################
#         # PART 4: Temporal Interaction (LSTM)
#         ########################################################################
#         x = self.temporal_interaction(x)  # (B, L*, 2F)

#         ########################################################################
#         # PART 5: Temporal Aggregation
#         ########################################################################
#         x = self.temporal_fusion(x)  # (B, 2F)

#         ########################################################################
#         # PART 6: Prediction
#         ########################################################################
#         return self.prediction(self.dropout(x))




# #### Ablation study TinyHAR_Model_Without_PART3
# class TinyHAR_Model(nn.Module):
#     def __init__(
#         self,
#         input_shape,
#         number_class,
#         filter_num,
#         nb_conv_layers=4,
#         filter_size=5,
#         cross_channel_interaction_type="attn",
#         cross_channel_aggregation_type="FC",
#         temporal_info_interaction_type="lstm",
#         temporal_info_aggregation_type="tnaive",
#         dropout=0.1,
#     ):
#         #super().__init__()
#         super(TinyHAR_Model, self).__init__()
#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction (Unchanged)
#         ########################################################################
#         filter_num_list = [1] + [filter_num] * nb_conv_layers
#         self.layers_conv = nn.ModuleList()
#         for i in range(nb_conv_layers):
#             in_channel = filter_num_list[i]
#             out_channel = filter_num_list[i + 1]
#             stride = (2, 1) if i % 2 == 1 else (1, 1)
#             self.layers_conv.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channel, out_channel, (filter_size, 1), stride),
#                     nn.ReLU(inplace=True),
#                     nn.BatchNorm2d(out_channel),
#                 )
#             )
        
#         # Calculate downsampled temporal length
#         self.downsampling_length = self.get_the_shape(input_shape)
#         self.sensor_channels = input_shape[3]  # Original sensor channels

#         ########################################################################
#         # PART 2: Cross Channel Interaction (Unchanged)
#         ########################################################################
#         self.channel_interaction = crosschannel_interaction[cross_channel_interaction_type](
#             self.sensor_channels, filter_num
#         )

#         ########################################################################
#         # PART 4: Temporal Interaction (Adjusted input dimension)
#         ########################################################################
#         # LSTM input_dim = sensor_channels * filter_num (instead of 2*filter_num)
#         self.temporal_interaction = temporal_interaction[temporal_info_interaction_type](
#             self.sensor_channels, self.sensor_channels * filter_num  # Now matches PART2's output
#         )

#         ########################################################################
#         # PART 5: Temporal Aggregation (Adjusted hidden_dim)
#         ########################################################################
#         # Use hidden_dim = sensor_channels * filter_num
#         self.temporal_fusion = temmporal_aggregation[temporal_info_aggregation_type](
#             self.sensor_channels * filter_num
#         )

#         ########################################################################
#         # PART 6: Prediction (Adjusted input dimension)
#         ########################################################################
#         self.dropout = nn.Dropout(dropout)
#         self.prediction = nn.Linear(self.sensor_channels * filter_num, number_class)

#     def get_the_shape(self, input_shape):
#         x = torch.rand(input_shape)
#         for layer in self.layers_conv:
#             x = layer(x)
#         return x.shape[2]

#     def forward(self, x):
#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction
#         ########################################################################
#         for layer in self.layers_conv:
#             x = layer(x)  # (B, F, L*, C)

#         ########################################################################
#         # PART 2: Cross Channel Interaction
#         ########################################################################
#         x = x.permute(0, 3, 2, 1)  # (B, C, L*, F)
#         x = torch.cat(
#             [self.channel_interaction(x[:, :, t, :]).unsqueeze(2) for t in range(x.shape[2])],
#             dim=2,
#         )  # (B, C, L*, F)

#         ########################################################################
#         # Skip PART3: Directly reshape for temporal interaction
#         ########################################################################
#         x = x.permute(0, 2, 1, 3)  # (B, L*, C, F)
#         x = x.reshape(x.size(0), x.size(1), -1)  # (B, L*, C*F)

#         ########################################################################
#         # PART 4: Temporal Interaction (LSTM with adjusted input_dim)
#         ########################################################################
#         x = self.temporal_interaction(x)  # (B, L*, C*F)

#         ########################################################################
#         # PART 5: Temporal Aggregation
#         ########################################################################
#         x = self.temporal_fusion(x)  # (B, C*F)

#         ########################################################################
#         # PART 6: Prediction
#         ########################################################################
#         return self.prediction(self.dropout(x))






# #### Ablation study TinyHAR_Model_Without_PART4
# class TinyHAR_Model(nn.Module):
#     def __init__(
#         self,
#         input_shape,
#         number_class,
#         filter_num,
#         nb_conv_layers=4,
#         filter_size=5,
#         cross_channel_interaction_type="attn",
#         cross_channel_aggregation_type="FC",
#         temporal_info_interaction_type="lstm",
#         temporal_info_aggregation_type="tnaive",
#         dropout=0.1,
#     ):
#         #super().__init__()
#         super(TinyHAR_Model, self).__init__()

#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction (Unchanged)
#         ########################################################################
#         filter_num_list = [1] + [filter_num] * nb_conv_layers
#         self.layers_conv = nn.ModuleList()
#         for i in range(nb_conv_layers):
#             in_channel = filter_num_list[i]
#             out_channel = filter_num_list[i + 1]
#             stride = (2, 1) if i % 2 == 1 else (1, 1)
#             self.layers_conv.append(
#                 nn.Sequential(
#                      nn.Conv2d(in_channel, out_channel, (filter_size, 1), stride),
#                      nn.ReLU(inplace=True),
#                      nn.BatchNorm2d(out_channel),
#                  )
#             )
        
#         self.downsampling_length = self.get_the_shape(input_shape)
#         self.sensor_channels = input_shape[3]

#         ########################################################################
#         # PART 2: Cross Channel Interaction (Unchanged)
#         ########################################################################
#         self.channel_interaction = crosschannel_interaction[cross_channel_interaction_type](
#             self.sensor_channels, filter_num
#         )

#         ########################################################################
#         # PART 3: Cross Channel Fusion (Unchanged)
#         ########################################################################
#         if cross_channel_aggregation_type == "FC":
#             self.channel_fusion = crosschannel_aggregation[cross_channel_aggregation_type](
#                 self.sensor_channels * filter_num, 2 * filter_num
#             )
#         self.activation = nn.ReLU()

#         ########################################################################
#         # PART 5: Temporal Aggregation (Adjusted input dimension)
#         ########################################################################
#         # Input shape: (B, L*, 2F) → Same as PART3 output
#         self.temporal_fusion = temmporal_aggregation[temporal_info_aggregation_type](2 * filter_num)

#         ########################################################################
#         # PART 6: Prediction (Unchanged)
#         ########################################################################
#         self.dropout = nn.Dropout(dropout)
#         self.prediction = nn.Linear(2 * filter_num, number_class)

#     def get_the_shape(self, input_shape):
#         x = torch.rand(input_shape)
#         for layer in self.layers_conv:
#             x = layer(x)
#         return x.shape[2]

#     def forward(self, x):
#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction
#         ########################################################################
#         for layer in self.layers_conv:
#             x = layer(x)  # (B, F, L*, C)

#         ########################################################################
#         # PART 2: Cross Channel Interaction
#         ########################################################################
#         x = x.permute(0, 3, 2, 1)  # (B, C, L*, F)
#         x = torch.cat(
#             [self.channel_interaction(x[:, :, t, :]).unsqueeze(2) for t in range(x.shape[2])],
#             dim=2,
#         )  # (B, C, L*, F)

#         ########################################################################
#         # PART 3: Cross Channel Fusion
#         ########################################################################
#         x = x.permute(0, 2, 1, 3)  # (B, L*, C, F)
#         x = x.reshape(x.size(0), x.size(1), -1)  # (B, L*, C*F)
#         x = self.activation(self.channel_fusion(x))  # (B, L*, 2F)

#         ########################################################################
#         # Skip PART4: Directly pass to PART5
#         ########################################################################
#         x = self.temporal_fusion(x)  # (B, 2F)

#         ########################################################################
#         # PART 6: Prediction
#         ########################################################################
#         return self.prediction(self.dropout(x))









# #### Ablation study TinyHAR_Model_Without_PART5
# class TinyHAR_Model(nn.Module):
#     def __init__(
#         self,
#         input_shape,
#         number_class,
#         filter_num,
#         nb_conv_layers=4,
#         filter_size=5,
#         cross_channel_interaction_type="attn",
#         cross_channel_aggregation_type="FC",
#         temporal_info_interaction_type="lstm",
#         temporal_info_aggregation_type="tnaive",
#         dropout=0.1,
#     ):
#         #super().__init__()
#         super(TinyHAR_Model, self).__init__()

#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction (Unchanged)
#         ########################################################################
#         filter_num_list = [1] + [filter_num] * nb_conv_layers
#         self.layers_conv = nn.ModuleList()
#         for i in range(nb_conv_layers):
#             in_channel = filter_num_list[i]
#             out_channel = filter_num_list[i + 1]
#             stride = (2, 1) if i % 2 == 1 else (1, 1)
#             self.layers_conv.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channel, out_channel, (filter_size, 1), stride),
#                     nn.ReLU(inplace=True),
#                     nn.BatchNorm2d(out_channel),
#                 )
#             )
        
#         self.downsampling_length = self.get_the_shape(input_shape)
#         self.sensor_channels = input_shape[3]

#         ########################################################################
#         # PART 2: Cross Channel Interaction (Unchanged)
#         ########################################################################
#         self.channel_interaction = crosschannel_interaction[cross_channel_interaction_type](
#             self.sensor_channels, filter_num
#         )

#         ########################################################################
#         # PART 3: Cross Channel Fusion (Unchanged)
#         ########################################################################
#         if cross_channel_aggregation_type == "FC":
#             self.channel_fusion = crosschannel_aggregation[cross_channel_aggregation_type](
#                 self.sensor_channels * filter_num, 2 * filter_num
#             )
#         self.activation = nn.ReLU()

#         ########################################################################
#         # PART 4: Temporal Interaction (Unchanged)
#         ########################################################################
#         self.temporal_interaction = temporal_interaction[temporal_info_interaction_type](
#             self.sensor_channels, 2 * filter_num
#         )

#         ########################################################################
#         # PART 6: Prediction (Unchanged)
#         ########################################################################
#         self.dropout = nn.Dropout(dropout)
#         self.prediction = nn.Linear(2 * filter_num, number_class)

#     def get_the_shape(self, input_shape):
#         x = torch.rand(input_shape)
#         for layer in self.layers_conv:
#             x = layer(x)
#         return x.shape[2]

#     def forward(self, x):
#         ########################################################################
#         # PART 1: Channel-wise Feature Extraction
#         ########################################################################
#         for layer in self.layers_conv:
#             x = layer(x)  # (B, F, L*, C)

#         ########################################################################
#         # PART 2: Cross Channel Interaction
#         ########################################################################
#         x = x.permute(0, 3, 2, 1)  # (B, C, L*, F)
#         x = torch.cat(
#             [self.channel_interaction(x[:, :, t, :]).unsqueeze(2) for t in range(x.shape[2])],
#             dim=2,
#         )  # (B, C, L*, F)

#         ########################################################################
#         # PART 3: Cross Channel Fusion
#         ########################################################################
#         x = x.permute(0, 2, 1, 3)  # (B, L*, C, F)
#         x = x.reshape(x.size(0), x.size(1), -1)  # (B, L*, C*F)
#         x = self.activation(self.channel_fusion(x))  # (B, L*, 2F)

#         ########################################################################
#         # PART 4: Temporal Interaction (LSTM)
#         ########################################################################
#         x = self.temporal_interaction(x)  # (B, L*, 2F)

#         ########################################################################
#         # Skip PART5: Use last timestep for prediction
#         ########################################################################
#         x = x[:, -1, :]  # (B, 2F)

#         ########################################################################
#         # PART 6: Prediction
#         ########################################################################
#         return self.prediction(self.dropout(x))