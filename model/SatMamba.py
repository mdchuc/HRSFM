from functools import partial

import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model

from util.pos_embed import get_2d_sincos_pos_embed
import model.Mamba2Helper as Mamba2Helper

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



class MaskedAutoencoderMamba(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_chans: int = 3,
                 embed_dim: int = 768, 
                 depth: int = 12, 
                 decoder_embed_dim: int = 512, 
                 decoder_depth: int = 8, 
                 norm_pix_loss: bool = True, 
                 if_pos_encod: bool = False, 
                 ssm_cfg: dict = {"layer": "Mamba2"},
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm: bool = True,
                 residual_in_fp32: bool = True,
                 d_intermediate: int = 0,
                 attn_layer_idx: list = [],
                 attn_cfg: dict = {},
                 d_state: int = 64,
                 headdim: int = 96,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 

        self.in_c = in_chans
        self.residual_in_fp32 = residual_in_fp32
        self.if_pos_encod = if_pos_encod
        self.depth = depth 
        self.decoder_depth = decoder_depth
        self.expand_factor = 4  # full 4 ways traversal
        self.initializer_cfg = initializer_cfg
        self.d_intermediate = d_intermediate

        print("d_state: ", d_state)
        print("headdim: ", headdim)
        
        print(f"Use {ssm_cfg['layer']}")

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.if_pos_encod:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList(
            [
                Mamba2Helper.create_block(
                    embed_dim,
                    d_intermediate,
                    d_state,
                    headdim,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(self.depth*self.expand_factor)
            ]
        )

        # encoder output head norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        if self.if_pos_encod:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.decoder_blocks = nn.ModuleList(
            [
                Mamba2Helper.create_block(
                    decoder_embed_dim,
                    d_intermediate,
                    d_state,
                    headdim=128,  # fix number of heads to 8 
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(self.decoder_depth*self.expand_factor)
            ]
        )

        self.norm_f_decoder = (nn.LayerNorm if not rms_norm else RMSNorm)(
            decoder_embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.if_pos_encod:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Linear layers
        self.decoder_embed.apply(Mamba2Helper.segm_init_weights)
        self.decoder_pred.apply(Mamba2Helper.segm_init_weights)

        # initialize Mamba blocks
        self.apply(
            partial(
                self._init_weights,
                n_layer=self.depth+self.decoder_depth,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )

    def _init_weights(self, 
                      module, 
                      n_layer,
                      initializer_range=0.02, # Now only used for embedding layer.
                      rescale_prenorm_residual=True,
                      n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        if isinstance(module, nn.Linear):
            # Following official Mamba implementation
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        # Mamba blocks ann layers
        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.blocks + self.decoder_blocks) 
        }

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def swap_order(self, x, gather_idx=None):
        """
        Row-major order to/from Column-major order 
        x: B, L, D
        gather_idx: B, L
        """
        if gather_idx is not None:
            swapped_x = torch.gather(x, dim=1, index=gather_idx.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            return swapped_x
        else:
            # matrix transposed then swap
            B, L, D = x.shape
            grid_size = int(L**0.5)
            reverted_mx = x[:, 1:, :].reshape(B, grid_size, grid_size, D)  # without cls token
            reverted_mx = reverted_mx.permute(0, 2, 1, 3).reshape(B, -1, D)
            swapped_x = torch.cat([x[:, :1, :], reverted_mx], dim=1)
            return swapped_x
    
    def scan_merge_row_order(self, x_row_major, x_col_major, col_2_row_idx=None):
        x_row_major_from_col = self.swap_order(x_col_major, col_2_row_idx)
        x_combined = (x_row_major + x_row_major_from_col) / 2
        return x_combined

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # B, C, H, W = x.shape
        B, L, C = x.shape
        H = W = int(L**0.5)
        len_keep = int(L * (1 - mask_ratio))

        # Generate random masks for each batch element
        masks = torch.rand(B, L, device=x.device)
        ids_shuffle = masks.argsort(dim=1)
        ids_shuffle_sort, _ = ids_shuffle[:, :len_keep].sort(dim=1)
        ids_shuffle = torch.cat((ids_shuffle_sort, ids_shuffle[:, len_keep:]), dim=1)
        ids_restore_row_major = torch.argsort(ids_shuffle, dim=1)
        masked_indices = ids_shuffle[:, len_keep:]
        masks = torch.zeros_like(masks, dtype=torch.bool).scatter_(1, masked_indices, True)
        masks = masks.view(B, H, W)

        # Get the indices of unmasked elements
        unmasked_indices = torch.nonzero(~masks, as_tuple=True)

        num_rows, num_cols = H, W

        index_row_major = (unmasked_indices[1] * num_cols + unmasked_indices[2])
        index_col_major = (unmasked_indices[2] * num_rows + unmasked_indices[1])
        index_row_major = index_row_major.reshape(B, len_keep)
        index_col_major = index_col_major.reshape(B, len_keep)

        row_2_col_idx = index_col_major.argsort(dim=1)
        col_2_row_idx = row_2_col_idx.argsort(dim=1)
        
        # keep the first subset
        ids_keep_row_major = index_row_major.reshape(B, len_keep)
        x_row_major = torch.gather(x, dim=1, index=ids_keep_row_major.unsqueeze(-1).repeat(1, 1, C))

        mask = masks.reshape(B, -1).int()

        return x_row_major, mask, ids_restore_row_major, row_2_col_idx, col_2_row_idx

    def forward_encoder(self, x, mask_ratio, scan=4, inference_params=None):

        # embed patches
        x = self.patch_embed(x)
        B = x.shape[0]
        
        # add pos embed w/o cls token
        if self.if_pos_encod:
            x = x + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio  
        x_row_major, mask, ids_restore, row_2_col_idx, col_2_row_idx = self.random_masking(x, mask_ratio)

        # append cls token
        # modify col_major_idx to include cls token at the beginning
        cls_pos = torch.zeros(B, 1, dtype=torch.int64, device=x.device)
        row_2_col_idx = torch.cat((cls_pos, row_2_col_idx+1), dim=1)
        col_2_row_idx = torch.cat((cls_pos, col_2_row_idx+1), dim=1)
        if self.if_pos_encod:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:
            cls_token = self.cls_token
        cls_tokens = cls_token.expand(B, -1, -1)
        # 
        x_row_major = torch.cat((cls_tokens, x_row_major), dim=1)
        x_col_major = self.swap_order(x_row_major, row_2_col_idx)
        
        # apply Mamba blocks
        row_hidden_states = x_row_major
        row_residual = None
        col_hidden_states = x_col_major
        col_residual = None
        
        # Mamba blocks
        # get 4 layers in a single for-loop
        n_layers = len(self.blocks) // 4

        if scan == 1:
            for i in range(n_layers):
                # row-major order
                row_hidden_states, row_residual = self.blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )

        elif scan == 2:
            for i in range(n_layers):
                # row-major order
                row_hidden_states_f, row_residual_f = self.blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2

        elif scan == 3:
            for i in range(n_layers):
                # row-major order
                row_hidden_states_f, row_residual_f = self.blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2
                
                # column-major order
                col_hidden_states, col_residual = self.blocks[i * 4 + 2](
                    col_hidden_states, col_residual, inference_params=inference_params
                )

                # cross merge row and col hidden states and residuals
                row_hidden_states = self.scan_merge_row_order(row_hidden_states, col_hidden_states, col_2_row_idx)
                row_residual = self.scan_merge_row_order(row_residual, col_residual, col_2_row_idx)

                if i < n_layers-1:
                    # extract col hidden states and residual out of row ones
                    col_hidden_states = self.swap_order(row_hidden_states, row_2_col_idx)
                    col_residual = self.swap_order(row_residual, row_2_col_idx)

        else:  # always full scan
            for i in range(n_layers):
                
                # row-major order
                row_hidden_states_f, row_residual_f = self.blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2

                # column-major order
                col_hidden_states_f, col_residual_f = self.blocks[i * 4 + 2](
                    col_hidden_states, col_residual, inference_params=inference_params
                )
                col_hidden_states_b, col_residual_b = self.blocks[i * 4 + 3](
                    col_hidden_states.flip([1]), None if col_residual == None else col_residual.flip([1]), inference_params=inference_params
                )
                col_hidden_states = (col_hidden_states_f + col_hidden_states_b.flip([1])) / 2  
                col_residual = (col_residual_f + col_residual_b.flip([1])) / 2

                # cross merge row and col hidden states and residuals
                row_hidden_states = self.scan_merge_row_order(row_hidden_states, col_hidden_states, col_2_row_idx)
                row_residual = self.scan_merge_row_order(row_residual, col_residual, col_2_row_idx)
                
                if i < n_layers-1:
                    # extract col hidden states and residual out of row ones
                    col_hidden_states = self.swap_order(row_hidden_states, row_2_col_idx)
                    col_residual = self.swap_order(row_residual, row_2_col_idx)
        
        hidden_states = row_hidden_states
        residual = row_residual
        
        # final norm after all Mamba blocks
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        return hidden_states, mask, ids_restore

    def forward_decoder(self, x, ids_restore, scan=4, inference_params=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  # learnable parameters
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        if self.if_pos_encod:
            x = x + self.decoder_pos_embed

        # apply Mamba blocks
        row_hidden_states = x
        row_residual = None
        col_hidden_states = self.swap_order(x)
        col_residual = None

        # Mamba blocks
        # get 4 layers in a single for-loop
        n_layers = len(self.decoder_blocks) // 4

        if scan == 1:
            for i in range(n_layers):
                # row-major order
                row_hidden_states, row_residual = self.decoder_blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )

        elif scan == 2:
            for i in range(n_layers):
                # row-major order
                row_hidden_states_f, row_residual_f = self.decoder_blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.decoder_blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2

        elif scan == 3:
            for i in range(n_layers):
                # row-major order
                row_hidden_states_f, row_residual_f = self.decoder_blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.decoder_blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2
                
                # column-major order
                col_hidden_states, col_residual = self.decoder_blocks[i * 4 + 2](
                    col_hidden_states, col_residual, inference_params=inference_params
                )
                
                # cross merge row and col hidden states and residuals
                row_hidden_states = self.scan_merge_row_order(row_hidden_states, col_hidden_states)
                row_residual = self.scan_merge_row_order(row_residual, col_residual)
                
                if i < n_layers-1:
                    # extract col hidden states and residual out of row ones
                    col_hidden_states = self.swap_order(row_hidden_states)
                    col_residual = self.swap_order(row_residual)
        
        else:
            for i in range(n_layers):
                # row-major order
                row_hidden_states_f, row_residual_f = self.decoder_blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.decoder_blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2

                # column-major order
                col_hidden_states_f, col_residual_f = self.decoder_blocks[i * 4 + 2](
                    col_hidden_states, col_residual, inference_params=inference_params
                )
                col_hidden_states_b, col_residual_b = self.decoder_blocks[i * 4 + 3](
                    col_hidden_states.flip([1]), None if col_residual == None else col_residual.flip([1]), inference_params=inference_params
                )
                col_hidden_states = (col_hidden_states_f + col_hidden_states_b.flip([1])) / 2  
                col_residual = (col_residual_f + col_residual_b.flip([1])) / 2

                # cross merge row and col hidden states and residuals
                row_hidden_states = self.scan_merge_row_order(row_hidden_states, col_hidden_states)
                row_residual = self.scan_merge_row_order(row_residual, col_residual)

                if i < n_layers-1:
                    # extract col hidden states and residual out of row ones
                    col_hidden_states = self.swap_order(row_hidden_states)
                    col_residual = self.swap_order(row_residual)
        
        hidden_states = row_hidden_states
        residual = row_residual

        # final norm after all Mamba blocks
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f_decoder(residual.to(dtype=self.norm_f_decoder.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f_decoder.weight,
                self.norm_f_decoder.bias,
                eps=self.norm_f_decoder.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f_decoder, RMSNorm)
            )

        # predictor projection
        x = self.decoder_pred(hidden_states)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = imgs[:, :3, :, :]
        # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # target = self.patchify(target, self.patch_embed.patch_size[0], 3)
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, scan=4, inference_params=None):
        
        latent, mask, ids_restore = self.forward_encoder(x=imgs, mask_ratio=mask_ratio, scan=scan, inference_params=inference_params)
        pred = self.forward_decoder(x=latent, ids_restore=ids_restore, scan=scan, inference_params=inference_params)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        # pred = self.unpatchify(pred, 16, 3)
        return loss, pred, mask
    
    def fuse_norm_encoder(self, x, res=None, norm=False):

        if not norm:
            return x

        if not self.fused_add_norm:
            residual = (x + res) if res is not None else x
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                x,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=res,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        
        return hidden_states
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: list[int],
        reshape: bool = True,
        norm: bool = False,
        scan: int = 4,
        inference_params=None
    ):
        """Modified from timm.VisionTransformer.get_intermediate_layers"""
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        if self.if_pos_encod:
            x = x + self.pos_embed[:, 1:, :]
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:
            cls_token = self.cls_token

        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Mamba blocks
        row_hidden_states = x
        row_residual = None
        col_hidden_states = self.swap_order(x)
        col_residual = None

        # Mamba blocks
        features = [self.fuse_norm_encoder(row_hidden_states, norm=norm)]

        # get 4 layers in a single for-loop
        n_layers = len(self.blocks) // 4

        if scan == 1:
            for i in range(n_layers):
                # row-major order
                row_hidden_states, row_residual = self.blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                features.append(self.fuse_norm_encoder(row_hidden_states, row_residual, norm=norm))

        elif scan == 2:
            for i in range(n_layers):
                # row-major order
                row_hidden_states_f, row_residual_f = self.blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2

        else:  # always full scan
            for i in range(n_layers):
                
                # row-major order
                row_hidden_states_f, row_residual_f = self.blocks[i * 4](
                    row_hidden_states, row_residual, inference_params=inference_params
                )
                row_hidden_states_b, row_residual_b = self.blocks[i * 4 + 1](
                    row_hidden_states.flip([1]), None if row_residual == None else row_residual.flip([1]), inference_params=inference_params
                )
                row_hidden_states = (row_hidden_states_f + row_hidden_states_b.flip([1])) / 2  
                row_residual = (row_residual_f + row_residual_b.flip([1])) / 2
                
                # column-major order
                col_hidden_states_f, col_residual_f = self.blocks[i * 4 + 2](
                    col_hidden_states, col_residual, inference_params=inference_params
                )
                col_hidden_states_b, col_residual_b = self.blocks[i * 4 + 3](
                    col_hidden_states.flip([1]), None if col_residual == None else col_residual.flip([1]), inference_params=inference_params
                )
                col_hidden_states = (col_hidden_states_f + col_hidden_states_b.flip([1])) / 2  
                col_residual = (col_residual_f + col_residual_b.flip([1])) / 2

                # cross merge row and col hidden states and residuals
                row_hidden_states = self.scan_merge_row_order(row_hidden_states, col_hidden_states)
                row_residual = self.scan_merge_row_order(row_residual, col_residual)

                if i < n_layers-1:
                    # extract col hidden states and residual out of row ones
                    col_hidden_states = self.swap_order(row_hidden_states)
                    col_residual = self.swap_order(row_residual)

                features.append(self.fuse_norm_encoder(row_hidden_states, row_residual, norm=norm))
            
        # Remove cls token from intermediate features
        features = [feat[:, 1:, :] for feat in features]

        if reshape:
            grid_size = self.patch_embed.grid_size
            features = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in features
            ]

        features = [features[i] for i in n]
        return features


# entry function
def create_model(**kwargs):
    model = MaskedAutoencoderMamba(**kwargs)
    return model