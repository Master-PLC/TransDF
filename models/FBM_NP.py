from typing import Optional

import torch.nn as nn
from layers.FBM_backbone import backbone_new_NPatchTST
from layers.PatchTST_layers import series_decomp
from torch import Tensor


class Model(nn.Module):
    def __init__(
        self, configs, 
        max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None, norm: str = 'BatchNorm', 
        attn_dropout: float = 0., act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None, 
        attn_mask: Optional[Tensor] = None, res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False, 
        pe: str = 'zeros', learn_pe: bool = True, pretrain_head: bool = False, head_type = 'flatten', **kwargs
    ):
        super().__init__()

        verbose = configs.verbose

        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        kernel_size = configs.kernel_size

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff

        individual = configs.individual
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        patch_len = configs.patch_len
        patch_num = configs.patch_num
        stride = configs.stride
        padding_patch = configs.padding_patch

        sr = context_window
        ts = 1.0 / sr

        # model
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.linear = nn.Linear(context_window, target_window)
            self.model_trend = backbone_new_NPatchTST(
                c_in=c_in, context_window = context_window, target_window=target_window, patch_num=patch_num, stride=stride, 
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, 
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, 
                padding_patch = padding_patch, pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs
            )
            self.model_res = backbone_new_NPatchTST(
                c_in=c_in, context_window = context_window, target_window=target_window, patch_num=patch_num, stride=stride, 
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, 
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, 
                padding_patch = padding_patch, pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs
            )

        else:
            self.model = backbone_new_NPatchTST(
                c_in=c_in, context_window = context_window, target_window=target_window, patch_num=patch_num, stride=stride, 
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, 
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, 
                padding_patch = padding_patch, pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs
            )

    def forward(self, x, *args, **kwargs):
        # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x
