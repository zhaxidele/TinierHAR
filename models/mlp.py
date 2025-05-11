import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes,
                 filter_scaling_factor,
                 config):
                 #numNueronsFCL1        = 128,
                 #numNueronsFCL2        = 64,
                 #):
       
        super(MLP, self).__init__()
        self.numNueronsFCL1 = int(config["numNueronsFCL1"]*config["filter_scaling_factor"])
        self.numNueronsFCL2 = int(config["numNueronsFCL2"]*config["filter_scaling_factor"])     
        self.numNueronsFCL3 = int(config["numNueronsFCL3"]*config["filter_scaling_factor"])     
        #self.nb_channels = input_shape[3]
        self.nb_input = input_shape[3]*input_shape[2]*input_shape[1]*input_shape[0]
        self.nb_classes = nb_classes

        self.activation = nn.ReLU() 
        self.fc_L1 = nn.Linear(self.nb_input, self.numNueronsFCL1)
        self.flatten = nn.Flatten()
        self.fc_L2 = nn.Linear(self.numNueronsFCL1, self.numNueronsFCL2)
        self.fc_L3 = nn.Linear(self.numNueronsFCL2, self.numNueronsFCL3)
        # define classifier
        self.fc_prediction = nn.Linear(self.numNueronsFCL3, self.nb_classes)


    def forward(self, x):

        x = self.flatten(x)
        x = self.activation(self.fc_L1(x)) 
        x = self.activation(self.fc_L2(x))
        x = self.activation(self.fc_L3(x))
        out = self.fc_prediction(x)    

        return out

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)