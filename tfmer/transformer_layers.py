# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1, 
                 pe='rpe_gau', D_std_gamma=[6.3,1.4,2.0], mod_D=None, mod_src='Q',
                 qkv_context=[0,0,0], qkv_fuse_type=None):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        
        self.qkv_context = qkv_context
        self.qkv_fuse_type = qkv_fuse_type
        kernel_size = 5
        if qkv_context[0] == 1:
            self.q_context_layer = nn.Conv1d(size, size, kernel_size, padding=kernel_size//2, 
                                             groups=size, bias=False)
            if qkv_fuse_type == 'gate':
                self.q_gate_layer = nn.Linear(2*size, 1)
        if qkv_context[1] == 1:
            self.k_context_layer = nn.Conv1d(size, size, kernel_size, padding=kernel_size//2, 
                                             groups=size, bias=False)
            if qkv_fuse_type == 'gate':
                self.k_gate_layer = nn.Linear(2*size, 1)
        if qkv_context[2] == 1:
            self.v_context_layer = nn.Conv1d(size, size, kernel_size, padding=kernel_size//2, 
                                             groups=size, bias=False)
            if qkv_fuse_type == 'gate':
                self.v_gate_layer = nn.Linear(2*size, 1)
        
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.pe = pe
        self.max_rpe_len = 16
        if 'rpe' in self.pe:
            #learnable relative positional encoding
            self.rpe_layer = nn.Embedding(2*self.max_rpe_len+1, 2*self.head_size)

        self.D, self.std, self.gamma = D_std_gamma
        if mod_D is not None:
            if mod_D in ['head_specific']:
                self.mod_D_layer = nn.Linear(size, num_heads)
            elif mod_D in ['head_share', 'nostat']:
                self.mod_D_layer = nn.Linear(size, 1)
            if isinstance(self.mod_D_layer, nn.Sequential):
                l = self.mod_D_layer[-1]
            else:
                l = self.mod_D_layer
            l.weight.data.zero_()
            if l.bias is not None:
                l.bias.data.zero_()
        self.learn_cen = False
        if self.learn_cen:
            self.cen_layer = nn.Linear(size, 1)
            self.cen_layer.weight.data.zero_()
            self.cen_layer.bias.data.zero_()
        self.mod_D = mod_D
        self.mod_src = mod_src

    def forward(self, key: Tensor, value: Tensor, query: Tensor, mask: Tensor = None, need_att=False):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = key.size(0)
        T = key.size(1)
        num_heads = self.num_heads
        x_norm = key
        
        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(key)
        v = self.v_layer(value)
        q = self.q_layer(query)  #[B,T,512]
        ori_q = q
        
        #local context
        if self.qkv_context[0] == 1:
            if self.qkv_fuse_type == 'gate':
                q_context = self.q_context_layer(x_norm.transpose(1,2)).transpose(1,2)
                gate_q = torch.sigmoid(self.q_gate_layer(torch.cat([q, q_context], dim=-1)))
                q = (1-gate_q)*q + gate_q*q_context
            else:
                q = self.q_context_layer(q.transpose(1,2)).transpose(1,2)
        if self.qkv_context[1] == 1:
            if self.qkv_fuse_type == 'gate':
                k_context = self.k_context_layer(x_norm.transpose(1,2)).transpose(1,2)
                gate_k = torch.sigmoid(self.k_gate_layer(torch.cat([k, k_context], dim=-1)))
                k = (1-gate_k)*k + gate_k*k_context
            else:
                k = self.k_context_layer(k.transpose(1,2)).transpose(1,2)
        if self.qkv_context[2] == 1:
            if self.qkv_fuse_type == 'gate':
                v_context = self.v_context_layer(x_norm.transpose(1,2)).transpose(1,2)
                gate_v = torch.sigmoid(self.v_gate_layer(torch.cat([v, v_context], dim=-1)))
                v = (1-gate_v)*v + gate_v*v_context
            else:
                v = self.v_context_layer(v.transpose(1,2)).transpose(1,2)
        
        # Modulate D
        if self.mod_D is None:
            m_D = torch.tensor(self.D).cuda()
        elif self.mod_D in ['head_share', 'head_specific']:
            if self.mod_src == 'Q':
                off = self.mod_D_layer(q).squeeze()
            elif self.mod_src == 'K':
                off = self.mod_D_layer(k).squeeze()
            elif self.mod_src == 'ori_Q':
                off = self.mod_D_layer(ori_q).squeeze()
            m_D = self.D + 2 * self.std * torch.tanh(off/self.gamma)
        elif self.mod_D == 'nostat':
            assert self.mod_src == 'Q'
            off = self.mod_D_layer(q).squeeze()
            # get frame length for each sample in the batch
            max_T = mask.shape[-1]
            f_len = max_T - mask.sum(dim=(-2,-1))
            m_D = (f_len * torch.sigmoid(off/self.gamma).transpose(0,-1)).transpose(0,-1)
        else:
            off = self.mod_D_layer(q.transpose(1,2)).squeeze()  #[B,T]
            m_D = self.D + 2 * self.std * torch.tanh(off/self.gamma)
        
        # learn center position
        cen_off = None
        if self.learn_cen:
            max_T = mask.shape[-1]
            f_len = max_T - mask.sum(dim=(-2,-1))
            cen_off = self.cen_layer(q)
            cen_off = (f_len * torch.sigmoid(cen_off/self.gamma).transpose(0,-1)).transpose(0,-1)  #[B,T,1]

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)  #[B,8,T,64]

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len [B,8,T,T]
        scores = torch.matmul(q, k.transpose(2, 3))
        
        #learable rpe
        if 'rpe' in self.pe:
            distance = -self.get_dist(T)  #[M,M], int
            distance = distance.clamp(-self.max_rpe_len, self.max_rpe_len) + self.max_rpe_len
            rpe_k, rpe_v = self.rpe_layer(distance).chunk(2, dim=-1)  #[T,T,64]
            scores += torch.einsum('...qd,qkd->...qk', q, rpe_k)
            
        #gaussian distance, unlearnable, for local heads only
        if 'gau' in self.pe:
            distance = self.get_dist(T, cen_off).float()  #[T,T] or [B,1,T,T]
            distance = distance.expand_as(scores)
            if self.mod_src in ['Q', 'ori_Q']:
                if self.mod_D in ['head_specific']:
                    m_D = m_D.expand_as(scores.permute(3,0,2,1)).permute(1,3,2,0)
                else:
                    m_D = m_D.expand_as(scores.permute(1,3,0,2)).permute(2,0,3,1)
            elif self.mod_src == 'K':
                if self.mod_D == 'head_specific':
                    m_D = m_D.expand_as(scores.permute(2,0,3,1)).permute(1,3,0,2)
                else:
                    m_D = m_D.expand_as(scores.permute(1,2,0,3)).permute(2,0,1,3)
            scores -= distance**2 / (m_D**2/2)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, T]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, T, C]
        context = torch.matmul(attention, v)
        if 'rpe' in self.pe:
            context += torch.einsum("...qk,qkd->...qd", attention, rpe_v)
        
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)
        if not need_att:
            return output, None
        else:
            return output, m_D  #attention[0,0,...]
    
    def get_dist(self, T, cen_off=None):
        indices = torch.arange(T).unsqueeze(1).expand(-1, T)
        if cen_off is not None:
            cen_off = cen_off.expand(-1,-1,T).unsqueeze(1)  #[B,1,T,T]
            return (cen_off - indices.transpose(0,1).cuda())
        return (indices - indices.transpose(0,1)).cuda()  #[T,T]


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, output_size, dropout=0.1, layer_type='linear'):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size) if layer_type=='linear' else nn.Conv1d(input_size, ff_size, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, output_size) if layer_type=='linear' else nn.Conv1d(ff_size, output_size, 5, padding=2),
            nn.Dropout(dropout),
        )
        self.layer_type = layer_type

    def forward(self, x):
        x_norm = self.layer_norm(x)
        if self.layer_type == 'tcn':
            return self.pwff_layer(x_norm.transpose(1,2)).transpose(1,2) + x
        return self.pwff_layer(x_norm) + x


class DepthwiseConv1d(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=5, groups=None, dropout=0.1, comb_conv='gate'):
        super(DepthwiseConv1d, self).__init__()
        if groups is None:
            groups = inchannels
        self.comb_conv = comb_conv
        self.conv_layer = nn.Sequential(nn.Conv1d(inchannels, outchannels, kernel_size, 
                                                  padding=kernel_size//2, groups=groups, bias=False),
                                        nn.ReLU(),
                                        nn.Dropout(dropout))
        if comb_conv not in ['gate', 'linear', 'add']:
            self.layer_norm = nn.LayerNorm(inchannels, eps=1e-6)
    
    def forward(self, x, mask=None):
        if self.comb_conv in ['gate', 'linear', 'add']:
            # parallel
            if mask is not None:
                return self.conv_layer(x.transpose(1,2)).mul(mask<=0).transpose(1,2)
            else:
                return self.conv_layer(x.transpose(1,2)).transpose(1,2)
        else:
            # cascaded
            x_norm = self.layer_norm(x)
            if mask is not None:
                return self.conv_layer(x_norm.transpose(1,2)).mul(mask<=0).transpose(1,2) + x
            else:
                return self.conv_layer(x_norm.transpose(1,2)).transpose(1,2) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, size]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]
    

class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1,
        pe='rpe_gau', D_std_gamma=[6.3,1.4,2.0], mod_D=None, mod_src='Q', comb_conv=None,
        qkv_context=[0,0,0], qkv_fuse_type=None, need_cls_token=None
    ):
        """
        A single Transformer layer.
        :param size: hidden_size
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout, pe, D_std_gamma, mod_D, mod_src,
                                                qkv_context, qkv_fuse_type)        
        
        self.comb_conv = comb_conv
        if self.comb_conv is not None:
            self.depthwise_conv = DepthwiseConv1d(size, size, 5, size, dropout, comb_conv)
            if self.comb_conv == 'gate':
                self.gate_layer = nn.Linear(2*size, 1)
                self.sigmoid = nn.Sigmoid()
            elif self.comb_conv == 'linear':
                self.linear_fuse = nn.Linear(2*size, size)
                
        # self.d_conv = DepthwiseConv1d(size, size, 5, size, dropout, 'cas_bef_san')
        self.layer_type = 'linear'
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, output_size=size, dropout=dropout, layer_type=self.layer_type
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.need_cls_token = need_cls_token

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor, need_att=False) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        if self.comb_conv == 'cas_bef_san':
            if self.need_cls_token == 'sen':
                sen_token, fea = x.split([1, x.shape[1]-1], dim=1)
                fea = self.depthwise_conv(fea)
                x = torch.cat([sen_token, fea], dim=1)
            else:
                x = self.depthwise_conv(x)
        
        #MHA
        x_norm = self.layer_norm(x)
        h, att = self.src_src_att(x_norm, x_norm, x_norm, mask, need_att)
        
        if self.comb_conv not in ['cascade', 'gate', 'linear', 'add']:
            h = self.dropout(h) + x
            gate = None
        
        elif self.comb_conv == 'gate':
            #it is parallel, thus share the same norm layer
            h = self.dropout(h)
            h_conv = self.depthwise_conv(x_norm)
            gate = torch.cat([h, h_conv], dim=-1)  #[B,MAX_T,2C]
            gate = self.sigmoid(self.gate_layer(gate))  #[B,MAX_T,1]
            h = (1-gate)*h + gate*h_conv + x  #[B,MAX_T,C]
            
        elif self.comb_conv == 'linear':
            h = self.dropout(h)
            h_conv = self.depthwise_conv(x_norm)
            h = self.linear_fuse(torch.cat([h, h_conv], dim=-1)) + x
            gate = None
            
        elif self.comb_conv == 'add':
            h = self.dropout(h)
            h_conv = self.depthwise_conv(x_norm)
            h = h + h_conv + x
            gate = None
        
        elif self.comb_conv == 'cascade':
            h = self.dropout(h) + x
            h = self.depthwise_conv(h)
            gate = None
        
        o = self.feed_forward(h)
        return o, att


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o
