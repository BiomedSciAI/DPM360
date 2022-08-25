#!/usr/bin/env python3


import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

from lightsaber import constants as C
from lightabser.model_lib.pt_sota_models import LSTMDroput as ld


# *****************************************************************************
#                          Main codes
# *****************************************************************************
class BetterLSTM(nn.LSTM):
    def __init__(self, 
                 *args, 
                 dropouti: float=0.,
                 dropoutw: float=0., 
                 dropouto: float=0.,
                 batch_first: bool=True, 
                 unit_forget_bias: bool=True, 
                 debug: bool=False,
                 **kwargs):
        """BetterLSTM: LSTM with better defaults. 
        ref: https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
        
        Modified by: Prithwish Chakraborty
        
        Parameters:
        -----------
        dropouti: 
            input dropout. (Default 0)
        dropoutw: 
            weight dropout. (Default 0)
        dropouto: 
            output dropout. (Default 0)
        batch_first: bool
            Default: True
        unit_forget_bias: bool
            Default: True
        debug: bool
            Default: False
        """ + super().__doc__
        
        dropout = kwargs.pop('dropout', None)
        if dropout is not None:
            # If dropout provided use the same value for all dropouts
            dropoutw = dropout
            dropouti = dropout
            dropouto = dropout

        super().__init__(*args, **kwargs, batch_first=batch_first)
        # additions for better LSTM
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalLockedDropout(dropouti,
                                                   batch_first=batch_first)
        self.output_drop = VariationalLockedDropout(dropouto,
                                                    batch_first=batch_first)
        self.state_drop = LockedDropout(dropoutw, batch_first=batch_first)
        if not debug:
            print("initializing weights")
            self._init_weights()

    def get_mask(self):
        payload = (self.input_drop.mask,
                   self.output_drop.mask,
                   self.state_drop.mask)
        return payload

    def set_mask(self, payload):
        input_mask, output_mask, state_mask = payload

        self.input_drop.set_mask(input_mask)
        self.output_drop.set_mask(output_mask)
        self.state_drop.set_mask(state_mask)

    def reset_mask(self):
        self.input_drop.reset_mask()
        self.output_drop.reset_mask()
        self.state_drop.reset_mask()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                _masked_weights = self.state_drop.forward(param.data)
                getattr(self, name).data = _masked_weights
                # getattr(self, name).data = \
                #     torch.nn.functional.dropout(param.data, p=self.dropoutw,
                #                                 training=self.training).contiguous()
        return

    # def __repr__(self):
    #     s = """BetterLSTM
    #     {}
    #     """.format("\n\t".join([f"{name}" for name, param in self.named_parameters()
    #                             if param.requires_grad]))
    #     return s

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class BetterLSTMLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 use_ln=True,
                 use_bn=False,
                 act=nn.ReLU,
                 dropout=None,
                 unit_forget_bias=True,
                 batch_first=True,):

        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = dropout

        # Layer parameters
        self._use_ln = use_ln
        self._use_bn = use_bn

        self.act = act

        self.base_model = ld.BetterLSTM(input_size=self._input_size,
                                        hidden_size=self._hidden_size,
                                        dropout=self._dropout)
        self.tx = self._get_layer_tx()

    def reset_mask(self):
        self.base_model.reset_mask()

    def _get_layer_tx(self):
        tx = []
        if self._use_ln:
            tx.append(nn.LayerNorm(self._hidden_size))
        if self._use_bn:
            tx.append(nn.BatchNorm1d(self._hidden_size))
        tx.append(self.act())
        return nn.Sequential(*tx)

    def forward(self, x, hidden=None, lengths=None):
        out, hidden = self.base_model(x, hidden)
        out = self.tx(out)
        return out, hidden


class BetterLSTMNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 use_ln=True,
                 use_bn=False,
                 use_residual=False,
                 act='ReLU',
                 dropout=None,
                 unit_forget_bias=True,
                 batch_first=True,):
        super().__init__()
        self.batch_first = batch_first

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._dropout = dropout

        self._num_layers = num_layers
        self._use_ln = use_ln
        self._use_bn = use_bn
        self._use_residual = use_residual

        self.act = getattr(nn, act)

        self.build_model()
        pass

    def reset_mask(self):
        if isinstance(self.seq_model, nn.Sequential):
            for layer in self.seq_model:
                if hasattr(layer, 'reset_mask'):
                    layer.reset_mask()
        else:
            self.seq_model.reset_mask()

    def on_load_checkpoint(self, checkpoint):
        self.reset_mask()

    def on_save_checkpoint(self, checkpoint):
        if self.training:
            self.reset_mask()

    def _get_layer(self, input_size, hidden_size, residual=False):
        layer = BetterLSTMLayer(input_size, hidden_size,
                                use_ln=self._use_ln,
                                use_bn=self._use_bn,
                                act=self.act)
        if residual:
            layer = ResidualBlock(layer)
        return layer

    def build_model(self):
        seq_model = self._get_layer(self._input_size, self._hidden_size, residual=self._use_residual)

        if self._num_layers > 1:
            #  raise NotImplementedError("Doesnt work yet. nn.Sequential cannot take multiple variables")
            #TODO: https://discuss.pytorch.org/t/nn-sequential-layers-forward-with-multiple-inputs-error/35591/7
            import ipdb; ipdb.set_trace()  # BREAKPOINT
            seq_model = [seq_model]
            for n in range(1, self._num_layers):
                _model = [self._get_layer(self._hidden_size, self._hidden_size,
                                          residual=self._use_residual)]
                seq_model += _model
            #  seq_model = nn.Sequential(*seq_model)
            seq_model = CombineModel(*seq_model)
        if self._output_size is None:
            op_model = nn.Identity
        else:
            op_model = FFBlock(self._hidden_size, self._output_size)

        self.seq_model = seq_model
        self.op_model = op_model
        return

    def init_hidden(self, batch_size, device='cpu'):
        # RNN takes hidden as (n_layers * n_dir, bs, hs)
        h_0 = T.rand(1, batch_size, self._hidden_size).to(device)  
        c_0 = T.rand(1, batch_size, self._hidden_size).to(device)
        return (h_0, c_0)

    def forward(self, x, hidden=None, lengths=None):
        if self._num_layers > 1:
            import ipdb; ipdb.set_trace()  # BREAKPOINT
        seq_out, seq_state = self.seq_model.forward(x, hidden)
        out = self.op_model.forward(seq_out)
        return out, seq_state
    
