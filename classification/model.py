# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, data_size, device, args):
        super(RNN, self).__init__()
        self.init_lin_h = nn.Linear(args.noise_dim, args.latent_dim)   # hidden state
        self.init_lin_c = nn.Linear(args.noise_dim, args.latent_dim)   # cell state
        self.init_input = nn.Linear(args.noise_dim, args.latent_dim)

        self.rnn = nn.LSTM(args.latent_dim, args.latent_dim, args.num_rnn_layer, dropout=args.dropout_rate)
        
        # Transforming LSTM output to vector shape
        self.lin_transform_down = nn.Sequential(
                            nn.Linear(args.latent_dim, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, data_size*data_size))   # data_size*128+128+128+1 maybe the weights of predicted MLP
                            
        # Transforming vector to LSTM input shape
        self.lin_transform_up = nn.Sequential(
                            nn.Linear(data_size*data_size, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, args.latent_dim))
        
        self.num_rnn_layer = args.num_rnn_layer
        self.data_size = data_size
        self.device = device

    def nn_construction(self, E):
        # m_1, m_2, bias = E[:, :self.data_size**2], E[:, self.data_size**2:self.data_size**2+self.data_size], E[:, -1:]
        m_1 = E[:, :self.data_size*128]
        #m_2 = E[:, self.data_size*128:(self.data_size*128+128*128)]
        m_2 = E[:, (self.data_size*128):(self.data_size*128+128)]
        b_1 = E[:, (self.data_size*128+128):(self.data_size*128+128+128)]
        #b_2 = E[:, (self.data_size*128+128*128+128+128):(self.data_size*128+128*128+128+128+128)]
        b_2 = E[:, -1:]
        return [m_1.view(-1, 128), m_2.view(-1, 1)], [b_1.view(-1, 128), b_2]
    
    def forward(self, X, z, E=None, hidden=None):
        if hidden == None and E == None:
            init_c, init_h = [], []
            for _ in range(self.num_rnn_layer):
                init_c.append(torch.tanh(self.init_lin_h(z)))
                init_h.append(torch.tanh(self.init_lin_c(z)))
            # Initialize hidden inputs for the LSTM
            hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        
            # Initialize an input for the LSTM
            inputs = torch.tanh(self.init_input(z))
        else:
            inputs = self.lin_transform_up(E)

        out, hidden = self.rnn(inputs.unsqueeze(0), hidden)

        E = self.lin_transform_down(out.squeeze(0))   # the weights of the predicted MLP
        # relu = nn.ReLU()
        # E = relu(E)  # This will replace all negative elements with zero
        # E = F.relu(E.clone())  # This creates a new tensor

        '''
        m_1, m_2, bias = self.nn_construction(E)
        
        pred = torch.relu(torch.mm(X, m_1))
        pred = torch.sigmoid(torch.add(torch.mm(pred, m_2), bias))
        '''
        
        # m_list, bias_list = self.nn_construction(E)
        # pred = X
        # for i, m in enumerate(m_list):
        #     if i != len(m_list)-1:
        #         pred = torch.relu(torch.add(torch.mm(pred, m), bias_list[i]))
        #     else:
        #         pred = torch.sigmoid(torch.add(torch.mm(pred, m), bias_list[i]))
        
        return E, hidden#, pred


class RNN_prelim(nn.Module):
    def __init__(self, min_data_len, data_size, device, args):
        super(RNN_prelim, self).__init__()
        self.init_lin_h = nn.Linear(args.noise_dim, args.latent_dim)   # hidden state
        self.init_lin_c = nn.Linear(args.noise_dim, args.latent_dim)   # cell state
        self.init_input = nn.Linear(args.noise_dim, args.latent_dim)

        self.rnn = nn.LSTM(args.latent_dim, args.latent_dim, args.num_rnn_layer, dropout=args.dropout_rate)
        
        # Transforming LSTM output to vector shape
        self.lin_transform_down = nn.Sequential(
                            nn.Linear(args.latent_dim, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, min_data_len*data_size))   # data_size*128+128+128+1 maybe the weights of predicted MLP
                            
        # Transforming vector to LSTM input shape
        self.lin_transform_up = nn.Sequential(
                            nn.Linear(min_data_len*data_size, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, args.latent_dim))
        
        self.num_rnn_layer = args.num_rnn_layer
        self.data_size = data_size
        self.device = device

    def nn_construction(self, E):
        # m_1, m_2, bias = E[:, :self.data_size**2], E[:, self.data_size**2:self.data_size**2+self.data_size], E[:, -1:]
        m_1 = E[:, :self.data_size*128]
        #m_2 = E[:, self.data_size*128:(self.data_size*128+128*128)]
        m_2 = E[:, (self.data_size*128):(self.data_size*128+128)]
        b_1 = E[:, (self.data_size*128+128):(self.data_size*128+128+128)]
        #b_2 = E[:, (self.data_size*128+128*128+128+128):(self.data_size*128+128*128+128+128+128)]
        b_2 = E[:, -1:]
        return [m_1.view(-1, 128), m_2.view(-1, 1)], [b_1.view(-1, 128), b_2]
    
    def forward(self, X, z, E=None, hidden=None):
        if hidden == None and E == None:
            init_c, init_h = [], []
            for _ in range(self.num_rnn_layer):
                init_c.append(torch.tanh(self.init_lin_h(z)))
                init_h.append(torch.tanh(self.init_lin_c(z)))
            # Initialize hidden inputs for the LSTM
            hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        
            # Initialize an input for the LSTM
            inputs = torch.tanh(self.init_input(z))
        else:
            inputs = self.lin_transform_up(E)

        out, hidden = self.rnn(inputs.unsqueeze(0), hidden)

        E = self.lin_transform_down(out.squeeze(0))   # the weights of the predicted MLP
        # relu = nn.ReLU()
        # E = relu(E)  # This will replace all negative elements with zero
        # E = F.relu(E.clone())  # This creates a new tensor

        '''
        m_1, m_2, bias = self.nn_construction(E)
        
        pred = torch.relu(torch.mm(X, m_1))
        pred = torch.sigmoid(torch.add(torch.mm(pred, m_2), bias))
        '''
        
        # m_list, bias_list = self.nn_construction(E)
        # pred = X
        # for i, m in enumerate(m_list):
        #     if i != len(m_list)-1:
        #         pred = torch.relu(torch.add(torch.mm(pred, m), bias_list[i]))
        #     else:
        #         pred = torch.sigmoid(torch.add(torch.mm(pred, m), bias_list[i]))
        
        return E, hidden#, pred


class MLP_Elec2(nn.Module):
    def __init__(self, input_size):
        super(MLP_Elec2, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, 128)         # Second hidden layer with 128 neurons
        self.fc3 = nn.Linear(128, 1)           # Output layer
        self.relu = nn.ReLU()                  # ReLU activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply first hidden layer and ReLU activation
        x = self.relu(self.fc2(x))  # Apply second hidden layer and ReLU activation
        x = self.fc3(x)              # Apply output layer
        x = self.sigmoid(x)
        
        return x
    

class MLP_ONP(nn.Module):
    def __init__(self, input_size):
        super(MLP_ONP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)  # First hidden layer
        self.fc2 = nn.Linear(200, 200)         # Second hidden layer
        self.fc3 = nn.Linear(200, 1)           # Output layer
        self.relu = nn.ReLU()                  # ReLU activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply first hidden layer and ReLU activation
        x = self.relu(self.fc2(x))  # Apply second hidden layer and ReLU activation
        x = self.fc3(x)              # Apply output layer
        x = self.sigmoid(x)
        
        return x
    
n_neurons = 128
class MLP_Shuttle(nn.Module):
    def __init__(self, input_size):
        super(MLP_Shuttle, self).__init__()
        self.fc1 = nn.Linear(input_size, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, n_neurons)
        self.fc6 = nn.Linear(n_neurons, n_neurons)
        self.fc7 = nn.Linear(n_neurons, n_neurons)
        self.fc8 = nn.Linear(n_neurons, n_neurons)
        self.fc9 = nn.Linear(n_neurons, n_neurons)
        self.fc10 = nn.Linear(n_neurons, 1)
        self.relu = nn.ReLU()                  # ReLU activation function
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.fc10(x)              # Apply output layer
        # x = self.softmax(x)
        x = self.sigmoid(x)
        
        return x
    

class MLP_Moons(nn.Module):
    # def __init__(self, input_size):
    #     super(MLP_Moons, self).__init__()
    #     self.fc1 = nn.Linear(input_size, 50)  # First hidden layer
    #     self.fc2 = nn.Linear(50, 50)         # Second hidden layer
    #     self.fc3 = nn.Linear(50, 1)           # Output layer
    #     self.relu = nn.ReLU()                  # ReLU activation function
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     x = self.relu(self.fc1(x))  # Apply first hidden layer and ReLU activation
    #     x = self.relu(self.fc2(x))  # Apply second hidden layer and ReLU activation
    #     x = self.fc3(x)              # Apply output layer
    #     x = self.sigmoid(x)
        
    #     return x
    def __init__(self, input_size):
        super(MLP_Moons, self).__init__()
        self.fc1 = nn.Linear(input_size, n_neurons)  # First hidden layer
        self.fc2 = nn.Linear(n_neurons, n_neurons)         # Second hidden layer
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, 1)           # Output layer
        self.relu = nn.ReLU()                  # ReLU activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply first hidden layer and ReLU activation
        x = self.relu(self.fc2(x))  # Apply second hidden layer and ReLU activation
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)              # Apply output layer
        x = self.sigmoid(x)
        
        return x


n_neurons = 50
class MLP_Moons_test(nn.Module):
    def __init__(self, input_size):
        super(MLP_Moons_test, self).__init__()
        self.fc1 = nn.Linear(input_size, n_neurons)  # First hidden layer
        self.fc2 = nn.Linear(n_neurons, n_neurons)         # Second hidden layer
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, 1)           # Output layer
        self.relu = nn.ReLU()                  # ReLU activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply first hidden layer and ReLU activation
        x = self.relu(self.fc2(x))  # Apply second hidden layer and ReLU activation
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)              # Apply output layer
        x = self.sigmoid(x)
        
        return x