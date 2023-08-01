import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from .cross_modality import ScoringModule
from clip import x_clip
from models.cross_modality import Image_to_text_attention, FeedForwardBlock
from einops import rearrange
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionEmbedScale(nn.Module):
    def __init__(self, cfg, image_encoder, text_encoder = None, return_attention = False):
        super().__init__()
        self.cfg = cfg

        self.image_encoder = image_encoder
        # self.text_encoder = text_encoder
        self.return_attention = return_attention

        if text_encoder is None:
            if cfg.models.positional:
                self.text_encoder = x_clip.TextTransformer_noemb(dim = cfg.models.emb_dim, num_tokens = 50000, max_seq_len=512, depth = cfg.models.transformer_depth , dim_head = cfg.dataset.feat_dim//cfg.models.heads, heads = cfg.models.heads, return_attention = cfg.models.attention_mean)
            else:
                self.text_encoder = x_clip.Transformer(cfg.models.emb_dim, depth = cfg.models.transformer_depth , dim_head = cfg.dataset.feat_dim//cfg.models.heads, heads = cfg.models.heads, add_cls=cfg.models.cls_token, return_attention = True)
        else:
            self.text_encoder = text_encoder

        self.image_encoder_train = False
        if cfg.models.img_transformer_depth > 0:
            self.image_encoder_train = x_clip.Transformer(cfg.models.emb_dim, depth = cfg.models.img_transformer_depth , dim_head = cfg.dataset.feat_dim//cfg.models.img_heads, heads = cfg.models.img_heads, add_cls=False, return_attention = cfg.models.attention_mean)

        self.img_embedder = MLP(cfg.dataset.feat_dim, cfg.models.emb_dim, num_layers = cfg.models.img_layers, dropout= True, norm = True,
                            layers = [cfg.models.hidden_dim])


        self.aux_embedder = MLP(cfg.dataset.emb_dim, cfg.models.emb_dim, num_layers=cfg.models.emb_layers, 
                            dropout=True, norm = True, layers = [cfg.models.hidden_dim])

        self.decoder = Image_to_text_attention(cfg.models.emb_dim, 1, False, cfg.models.image_tokens, True)
        self.full_attention = x_clip.Transformer(cfg.models.emb_dim, depth = 1 , dim_head = cfg.dataset.feat_dim//cfg.models.img_heads, heads = cfg.models.img_heads, add_cls=True, return_attention = cfg.models.attention_mean)
        self.score = nn.Linear(cfg.models.emb_dim, 1)


    def forward(self, img, aux, aux_mask, labels):
        feat = self.image_encoder(img)
        if self.cfg.models.freeze_image:
            feat = feat.detach()


        img = self.img_embedder(feat)

        if self.cfg.models.img_transformer_depth > 0:
            img, _ = self.image_encoder_train(img)
        
        
        aux = self.aux_embedder(aux)

        token_embeddings, text_attention  = self.text_encoder(aux, aux_mask)

        cls_emb = token_embeddings[:, 0, :]
        cls_img = img[:, 0, :]
        

        loss = 0.0
        cross_out = None

        imgx, textx,  attention, t2i_attention = self.decoder(img[:, 1:, :], (token_embeddings[:, 1:, :]), aux_mask, max_pool = self.cfg.models.max_pool, pool = False)
        I, T = imgx.shape[0], imgx.shape[1]
        imgx = rearrange(imgx, 'i t p f -> (i t) p f')
        textx = rearrange(textx, 'i t a f -> (i t) a f')
        attention_input = torch.cat([imgx, textx], 1)
        cross_out, _ = self.full_attention(attention_input)
        cross_out = cross_out[:, 0, :]
        cross_out = rearrange(cross_out, '(i t) f -> i t f', i = I)


        cross_out = self.score(cross_out).squeeze(-1)
        cross_loss = self.cfg.models.decode * F.cross_entropy(cross_out, labels)
        loss += cross_loss

        if self.cfg.models.lambda_cls > 0:
            out = torch.matmul(cls_img, cls_emb.T)
            loss += self.cfg.models.lambda_cls * F.cross_entropy(out, labels)
        else:
            out = cross_out



        if self.return_attention:
            return loss, out, attention, t2i_attention
        return loss, out, cross_out