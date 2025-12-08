import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib as plt

class InputAttentionEncoder(nn.Module):
    def __init__(self, N, M, T, stateful=False, device='cpu'):
        """
        :param: N: int
            number of time serieses / stocks
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T
        self.device = device
        self.to(device)
        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)
        
        #equation 8 matrices
        
        self.W_e = nn.Linear(2*self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)
    
    def forward(self, inputs):
        #inputs: [batch_size, T, N]
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M))
        
        #initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.M)).to(self.device)
        s_tm1 = torch.zeros((inputs.size(0), self.M)).to(self.device)
        
        # Calculate Input Attention
        self.input_attention_weights = []
        for t in range(self.T):
            #concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1).to(self.device) # [embedding size, 2M]
            #attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1).to(self.device)
            y = self.U_e(inputs.permute(0, 2, 1)).to(self.device)
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))
        
            #normalize attention weights (equation 9)
            if e_k_t.dim() == 1:
                e_k_t = e_k_t.unsqueeze(0)
            alpha_k_t = F.softmax(e_k_t, dim=1)
            self.input_attention_weights.append(alpha_k_t.detach().cpu())
            
            #weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :] 
    
            #calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))
            
            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs
    
class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T, stateful=False, device='cpu'):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.M = M # Encoder units
        self.P = P # Decoder units
        self.T = T # number of timestamps
        self.stateful = stateful 
        self.device = device
        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P) #input size 1 since we have Close price
         
        #equation 12 matrices
        self.W_d = nn.Linear(2*self.P, self.M) # [2P, M]
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias = False)
        
        #equation 15 matrix
        self.w_tilda = nn.Linear(self.M + 1, 1)
        
        #equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)
        
    def forward(self, encoded_inputs, y):
        encoded_inputs = encoded_inputs.to(self.device)
        #initializing hidden states
        y = y.to(self.device)
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(self.device) #embedding size x decoder units
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(self.device)
        self.temporal_attention_weights = []
        for t in range(self.T): # for each timestamp
            #concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1).to(self.device) # [embedding size, 2P]
            #temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1).to(self.device)
            y1 = self.U_d(encoded_inputs).to(self.device)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)
            
            #normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1).to(self.device)  # [embedding size, M]
            self.temporal_attention_weights.append(beta_i_t.detach().cpu())
            
            #create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1).to(self.device)  # [embedding size, M]
            
            #concatenate c_t and y_t
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1).to(self.device)  # [embedding size, M + 1]
            #create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat)
            #calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
            
        #concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        #calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1
    

class DARNN(nn.Module):
    def __init__(self, N, M, P, T, stateful_encoder=False, stateful_decoder=False, device='cpu'):
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder, device)
        self.decoder = TemporalAttentionDecoder(M, P, T, stateful_decoder, device)
    def forward(self, X_history, y_history):
        out = self.decoder(self.encoder(X_history), y_history)
        return out


class MultiStockDARNN(nn.Module):
    def __init__(self, N, M, P, T, num_stocks, stateful_encoder=False, stateful_decoder=False, device='cpu'):
        super(MultiStockDARNN, self).__init__()
        self.device = device
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder, self.device)
        # Create a decoder for each stock
        self.decoders = nn.ModuleList([
            TemporalAttentionDecoder(M, P, T, stateful_decoder, self.device) 
            for _ in range(num_stocks)
        ])
        self.to(self.device)
        
    def forward(self, X_history, y_histories):
        # X_history: [batch_size, T, N]
        # y_histories: [batch_size, T, num_stocks]
        
        encoded = self.encoder(X_history).to(self.device)
        
        # Get prediction for each stock using its dedicated decoder
        outputs = []
        for i, decoder in enumerate(self.decoders):
            # Extract the i-th stock's history
            y_history_i = y_histories[:, :, i:i+1]
            output_i = decoder(encoded, y_history_i)
            outputs.append(output_i)
            
        # Stack all predictions
        return torch.cat(outputs, dim=1)  # [batch_size, num_stocks]


class DARNNDataset():

    def __init__(self, dataset):
        super().__init__(dataset)
    
    def __getitem__(self, idx):
        data_sample = self.dataset[idx] 
        x = data_sample.x.to(device)
        target = data_sample.y.to(device)
        
        
        
        num_nodes = x.shape[0]
        N = num_nodes  # number of nodes
        M = 1  # number of encoder LSTM units
        P = 1  # number of decoder LSTM units
        T = int(x.shape[1] / 5) - 1
        
        
        
        
        # number of time steps
        X = x.view(num_nodes, 5, T+1).permute(0, 2, 1)  # [num_nodes, T]
        X = X[:, :, :1].squeeze()
        #X = X.unsqueeze(0)  # batch size 1
        X = X.permute(0, 2, 1)  # [batch_size, T, N]
        
        y_target = torch.zeros((1, T, num_nodes), dtype=torch.long)
        for t in range(T):
            y_target[0, t, :] = (X[0, t, :] > X[0, t+1, :]).long() 

        X = X[:, :-1, :]  # use all but last timestep as input
    
        return X, y_target, target
