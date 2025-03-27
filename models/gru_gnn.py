import math
import torch.nn.functional as F
from .gnn_layers import *
from .motion_models import *
import torch
from mamba_ssm import Mamba


class DyT(nn.Module):
    def __init__(self):
        super(DyT, self).__init__()
        self.lag1 = nn.Parameter(torch.empty(1).uniform_(-1e-2, 1e-2))
        self.lag2 = nn.Parameter(torch.empty(1).uniform_(-1e-2, 1e-2))
        self.lag3 = nn.Parameter(torch.empty(1).uniform_(-1e-2, 1e-2))
    def forward(self, x):
        x = torch.tanh(self.lag1 * x)
        return self.lag2*x + self.lag3

class MambaCell(nn.Module):

    def __init__(self, input_size=8, hidden_size=64, n_heads=3,
                 n_layers=1, dropout=0.1, gnn_layer="graphconv", edge_dim=None):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = hidden_size  # Need 3 hidden states for all GRU gates
        self.Wx = create_sequential_gnn(input_size=input_size,
                                        output_size=gate_size*2,
                                        hidden_size=hidden_size,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        layers=n_layers,
                                        activation='elu',
                                        gnn_layer=gnn_layer,
                                        edge_dim=edge_dim)
        self.bias = nn.Parameter(torch.empty(hidden_size*2).uniform_(-1e-2, 1e-2))
        self.reset_parameters()

    def reset_parameters(self):

        std = 1.0 / (math.sqrt(self.hidden_size))
        init_params = (p for name, p in self.named_parameters() if
                       not str.endswith(name, "log_edge_bw"))
        for p in init_params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -std, std)


    def forward(self, x, edge_index, h=None, edge_attr=None):

        wrx, wzx = torch.split(self.Wx(x, edge_index, edge_attr),
                                    self.hidden_size, dim=1)
        br, bz= torch.split(self.bias, self.hidden_size, dim=0)
        return wrx+br,wzx+bz


class GRUGNNCell(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, n_heads=3,
                 n_layers=1, dropout=0.1, gnn_layer="graphconv", edge_dim=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size =  hidden_size  # Need 3 hidden states for all GRU gates

        self.Wh = create_sequential_gnn(input_size=input_size,
                                        output_size=gate_size,
                                        hidden_size=hidden_size,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        layers=n_layers,
                                        activation='elu',
                                        gnn_layer=gnn_layer,
                                        edge_dim=edge_dim
                                        )


        self.bias = nn.Parameter(torch.empty(gate_size).uniform_(-1e-2, 1e-2))  

        self.hi =  nn.Sequential(nn.Linear(hidden_size,hidden_size),
                                  nn.SiLU(),
                                  nn.Linear(hidden_size,hidden_size),
                                  nn.SiLU(),
                                  nn.Linear(hidden_size,hidden_size),
                                  )
        self.hi2 =  nn.Sequential(nn.Linear(hidden_size,hidden_size),
                                  nn.Tanh(),
                                  nn.Linear(hidden_size,hidden_size),
                                  nn.Tanh(),
                                  nn.Linear(hidden_size,hidden_size),
                                  )


        self.lin2= nn.Linear(2*hidden_size,hidden_size)
        self.lin1= nn.Linear(hidden_size,hidden_size)
        self.tan = nn.Sequential(  nn.LayerNorm(hidden_size),
           nn.Tanh(),)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (math.sqrt(self.hidden_size))
        # Exclude edge bws from initialization
        init_params = (p for name, p in self.named_parameters() if
                       not str.endswith(name, "log_edge_bw"))
        for p in init_params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -std, std)

  
    def forward(self, x, edge_index, h=None, edge_attr=None):


        haha= self.hi(h)
        wrh = torch.tanh(self.bias+ self.Wh(x, edge_index,edge_attr))
        wrr =  self.hi2(x)
        hidden = h+ self.lin2(torch.cat((h,self.lin1(self.dropout(self.tan((wrr+wrh)*haha)))),-1))
 
        return hidden


class GRUGNNEncoder(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, n_heads=3, n_layers=1,
                 n_mixtures=7, static_f_dim=6, dropout=0.1, gnn_layer="graphconv",
                 init_static=False, use_edge_features=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_static = init_static
        self.dropout = nn.Dropout(p=dropout)
        self.static_feature_dim = static_f_dim

        if self.init_static:
            # GNN model to learn init states
            self.init_gnn = create_sequential_gnn(input_size=self.static_feature_dim,
                                                  output_size=hidden_size,
                                                  hidden_size=hidden_size,
                                                  n_heads=n_heads,
                                                  dropout=dropout,
                                                  layers=n_layers,
                                                  activation='elu',
                                                  gnn_layer=gnn_layer)
        else:
            init_std = 1.0 / (math.sqrt(hidden_size))
            self.init_state_param = nn.Parameter(torch.empty(hidden_size).uniform_(
                -1e-2, 1e-2))

        edge_dim = 1 if use_edge_features else None
        self.gru_cell = MambaCell(input_size, hidden_size, n_heads, n_layers, dropout,
                                   gnn_layer, edge_dim=edge_dim)

        self.lin3 = nn.Sequential(
          Mamba(
                d_model=hidden_size, # Model dimension d_model
                d_state=10,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=4,    # Block expansion factor
            ),
           nn.LayerNorm(hidden_size),
           nn.Tanh(),
        )

        self.lin4 = nn.Sequential(
         Mamba(
                d_model=hidden_size, # Model dimension d_model
                d_state=10,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=4,    # Block expansion factor
            ),
         nn.LayerNorm(hidden_size),
         nn.Tanh(),
        )
      
        self.lin5 = nn.Sequential(
         Mamba(
                d_model=hidden_size, # Model dimension d_model
                d_state=10,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=4,    # Block expansion factor
            ),        
        )
        self.mixture = nn.Linear(hidden_size, n_mixtures)  # <-- mixture weights

    def init_hidden(self, data, batch_size):
        if self.init_static:
            init_gnn_input = extract_static_features(data)
            # No edge features
            return self.init_gnn(init_gnn_input, data.edge_index[0], None)
        else:
            return self.init_state_param.repeat(batch_size, 1)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_features
        batch_size, _, _ = x.size()
        batch_size, seq_len, _ = x.size()
        hidden = self.init_hidden(data, batch_size)
        output = [hidden]
        output2 = [hidden]

        for x_i, ei_i, ef_i in zip(x.transpose(0, 1), edge_index, edge_features):
            hidden,hidden2 = self.gru_cell(x_i, ei_i, hidden, edge_attr=ef_i.to(torch.float32))
            output.append(hidden)
            output2.append(hidden2)

        output = torch.stack(output, dim=1)
        output_inv = torch.flip(output, dims=[1])
        output2 = torch.stack(output2, dim=1)
        output= self.lin5(self.lin3(output)+self.dropout(self.lin4(output_inv*output2)))
        mixture_w = self.mixture(self.dropout(torch.tanh(output[:,-1])))#g
        return output, mixture_w


class GRUGNNDecoder(nn.Module):
    def __init__(self, motion_model, max_length=10, hidden_size=64,
                 n_heads=3, n_layers=1, static_f_dim=6, alpha=0.2, dropout=0.1,
                 gnn_layer="graphconv", init_static=False):
        super().__init__()
        self.gru_cell = GRUGNNCell(hidden_size, hidden_size, n_heads, n_layers, dropout,
                                   gnn_layer,1)
        self.alpha = alpha
        self.init_static = init_static
        self.static_feature_dim = static_f_dim

        if init_static:
            self.init_combiner = nn.Linear(hidden_size, hidden_size, bias=True)
            self.init_gnn = create_sequential_gnn(input_size=self.static_feature_dim,
                                                  output_size=hidden_size,
                                                  hidden_size=hidden_size,
                                                  n_heads=n_heads,
                                                  dropout=dropout,
                                                  layers=1,
                                                  activation='elu',
                                                  gnn_layer=gnn_layer)

        self.motion_model = motion_model
        self.input_size = motion_model.n_states
        self.output_size = motion_model.n_inputs
        self.mixtures = motion_model.mixtures
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=dropout)

        # Temporal attention weight calculations
        self.embedding = nn.Linear(self.input_size*self.mixtures, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        # Scale GRU outputs
        self.generator = nn.Linear(hidden_size, hidden_size * 3)

        # Motion model inputs
        self.controller =  nn.Linear(hidden_size, 128*self.output_size * self.mixtures)

        # # To generate process noise
        self.sig_1 = nn.Linear(hidden_size, self.mixtures)
        self.sig_2 = nn.Linear(hidden_size, self.mixtures)
        self.rho = nn.Linear(hidden_size, self.mixtures)
   
        


    def process_noise_matrix(self, x1, x2, x3, batch_size):
        sig1 = torch.clamp(F.softplus(self.sig_1(x1)),0,100)
        sig2 = torch.clamp(F.softplus(self.sig_2(x2)),0,100)
        rho = F.softsign(self.rho(x3))

        q_t = torch.zeros(batch_size, self.mixtures, self.output_size,
                          self.output_size, device=x1.device)

        q_t[..., 0, 0] = torch.pow(sig1, 2)
        q_t[..., 1, 1] = torch.pow(sig2, 2)
        q_t[..., 0, 1] = q_t[..., 1, 0] = sig1 * sig2 * rho
        return q_t

    def forward(self, data):
        x, hidden, encoder_out, edge_index,edge_index2,  past_state, static_features= data
        batch_size = x.size(0)
        # print(past_state.reshape(batch_size,-1).shape,x.shape,self.mixtures,self.input_size )
        embedded = self.embedding(past_state.reshape(batch_size,-1))#good
        embedded = self.dropout(embedded) #good

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_out)

        output = torch.cat((embedded, attn_applied[:, 0]), 1)
        output = self.attn_combine(output)
        output = F.leaky_relu(output, self.alpha)

        hidden = self.gru_cell(output, edge_index, hidden, edge_index2)

        output = self.dropout(self.generator(hidden))#g

        x1, x2, x3 = torch.split(output, self.hidden_size, dim=-1)
        process_noise = self.process_noise_matrix(x1, x2, x3, batch_size)
        model_input =self.dropout(self.controller(hidden)).view(batch_size,#g
                                               self.mixtures,
                                              128* self.output_size)
        
        next_state, model_input = self.motion_model(past_state,model_input,
                                                    static_features)

        return next_state, model_input, process_noise, hidden

    def get_initial_state(self, last_enc_state, data):
        if self.init_static:
            static_features = extract_static_features(data)  # (B, 6)
            edge_index = data.edge_index[-1]  # Use last graph from encoder
            graph_combined = self.init_gnn(static_features, edge_index, None)  # No edge features
            combined_repr = nn.functional.elu(graph_combined)
            return last_enc_state + self.init_combiner(combined_repr)
        else:
            return last_enc_state
