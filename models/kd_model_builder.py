# ---- import models ---------------
from models.Attend import AttendDiscriminate
from models.SA_HAR import SA_HAR
from models.deepconvlstm import DeepConvLSTM
from models.conv1d_lowranklstm import Conv1D_LowRankLSTM
from models.conv1d_lowranklstm_attention import Conv1D_LowRankLSTM_Attention
from models.conv1d_lowranklstm_self_attention import Conv1D_LowRankLSTM_Self_Attention
from models.deepconvlstm_attn import DeepConvLSTM_ATTN
from models.crossatten.model import Cross_TS,TSTransformer_Basic
from models.TinyHAR import TinyHAR_Model
from models.SA_HAR import SA_HAR
from models.mcnn import MCNN
from models.mlp import MLP
from models.self_attention_har import SelfAttentionHAR
from dataloaders.utils import PrepareWavelets,FiltersExtention
# ------- import other packages ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Wavelet_learnable_filter(nn.Module):
    def __init__(self, args, f_in):
        super(Wavelet_learnable_filter, self).__init__()
        self.args = args
        if args.windowsize%2==1:
            self.Filter_ReplicationPad2d = nn.ReflectionPad2d((int((args.windowsize-1)/2),int((args.windowsize-1)/2),0,0))
            raw_filter = np.zeros((1,1,1,args.windowsize))
            raw_filter[0,0,0,int((args.windowsize-1)/2)] = 1
        else:
            self.Filter_ReplicationPad2d = nn.ReflectionPad2d((int(args.windowsize/2),int(args.windowsize/2),0,0))
            raw_filter = np.zeros((1,1,1,args.windowsize))
            raw_filter[0,0,0,int(args.windowsize/2)] = 1
        raw_filter = torch.tensor(raw_filter)
        SelectedWavelet = PrepareWavelets(K=args.number_wavelet_filtering, length=args.windowsize, seed=self.args.seed)
        ScaledFilter = FiltersExtention(SelectedWavelet)
        ScaledFilter = torch.cat((raw_filter,ScaledFilter),0)

        #print("debug: ", ScaledFilter.shape[0], "   ", f_in)


        self.wavelet_conv = nn.Conv2d(1, f_in, 
                                      (1,ScaledFilter.shape[3]),
                                      stride=1, bias=False, padding='valid') 
        # TODO shut down
        if not args.wavelet_filtering_learnable and f_in==ScaledFilter.shape[0]:
            print("clone the  wavefiler weight")
            self.wavelet_conv.weight.data.copy_(ScaledFilter)                                        
        self.wavelet_conv.weight.requires_grad = True
        if self.args.wavelet_filtering_layernorm:
            print("wavelet layernorm")
            self.layer_norm = nn.LayerNorm(self.args.windowsize, elementwise_affine=False)
                                                               
    def forward(self,x):
        # input shape B 1 L  C  
        x = x.permute(0,1,3,2)
        x = self.Filter_ReplicationPad2d(x)
        x = self.wavelet_conv(x)[:,:,:,:self.args.windowsize]

        if self.args.wavelet_filtering_layernorm:
            x = self.layer_norm(x)

        x = x.permute(0,1,3,2)
        return x

class model_builder_s(nn.Module):
    """
    
    """
    def __init__(self, args, input_f_channel = None):
        super(model_builder_s, self).__init__()

        self.args = args
        if input_f_channel is None:
            f_in  = self.args.f_in
        else:
            f_in  = input_f_channel


        if self.args.model_type_s == "conv1d_lowranklstm":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["conv1d_lowranklstm"]
            print(config)
            self.model_s  = Conv1D_LowRankLSTM((1,f_in, self.args.input_length, self.args.c_in ), 
                                       self.args.num_classes,
                                       config["filter_scaling_factor"],
                                       config)
            print("Build the Conv1D_LowRankLSTM student model!")
        else:
            self.model_s = Identity()
            print("Build the None model!")


    def forward(self,x):
        #if self.first_conv ：
        #    x = self.pre_conv(x)

        if self.args.wavelet_filtering:
            x = self.wave_conv(x)
            if self.args.wavelet_filtering_regularization:
                x = x * self.gamma
        y = self.model_s(x)
        return y


class model_builder_t(nn.Module):
    """
    
    """
    def __init__(self, args, input_f_channel = None):
        super(model_builder_t, self).__init__()

        self.args = args
        if input_f_channel is None:
            f_in  = self.args.f_in
        else:
            f_in  = input_f_channel


        if self.args.model_type_t == "deepconvlstm":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm"]
            print(config)
            self.model  = DeepConvLSTM((1,f_in, self.args.input_length, self.args.c_in ), 
                                       self.args.num_classes,
                                       config["filter_scaling_factor"],
                                       config)
            print("Build the DeepConvLSTM teacher model!")

        else:
            self.model = Identity()
            print("Build the None model!")


    def forward(self,x):

        if self.args.wavelet_filtering:
            x = self.wave_conv(x)
            if self.args.wavelet_filtering_regularization:
                x = x * self.gamma
        y = self.model(x)
        return y



