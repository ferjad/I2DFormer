import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from .common import MLP
from clip import x_clip
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

class I2DAttention(nn.Module):
    def __init__(self, embed_size, heads, cls_present, img_tokens, t2i_attention = False):
        super(I2DAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.cls_present = cls_present
        self.img_tokens = img_tokens

        assert (
            self.head_dim * heads == embed_size
        ), f"Embedding size needs to be divisible by heads {self.head_dim} {heads} {embed_size}"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.t2i_attention = t2i_attention
        if self.t2i_attention:
            self.text_values = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.fc_out_text = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, image, text, text_mask=None, topk = -1, max_pool = False, pool = True):
        # Get number of training examples
        I = image.shape[0]
        T = text.shape[0]
        text_len, image_len = text.shape[1], image.shape[1]
        
        # Reshape into heads
        image = image.reshape(I, image_len, self.heads, self.head_dim)
        text = text.reshape(T, text_len, self.heads, self.head_dim)

        query = self.queries(image)
        key = self.keys(text)
        value = self.values(text)
        # Calculating similarity of every image token to every text token
        energy = torch.einsum('tkhd, iqhd -> ithqk', [key, query])
        
        # Masking out non existing language tokensfloat("-1e20")
        if text_mask is not None:
            if self.cls_present:
                text_mask = F.pad(text_mask, (1, 0), value = True)
            energy = rearrange(energy, 'i t h q k -> t k i h q')
            # energy[~text_mask] = float(0)
            energy[~text_mask] = torch.finfo(torch.float32).min
            energy = rearrange(energy, 't k i h q -> i t h q k')        
        
        # Energy is [image, documents, head, patches, words]
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        text_attention = torch.softmax(energy / (self.embed_size ** (1 / 2)) , dim = -2)

        out = torch.einsum('ithqk, tkhd -> iqthd', [attention, value])
        out = rearrange(out, 'i q t h d -> i t q (h d)')
        out = self.fc_out(out)
        if pool:
            if max_pool:
                out = torch.max(out, 2).values
            else:
                out = out.mean(2)

        out_text = None
        if self.t2i_attention:
            text_values = self.text_values(image)
            out_text = torch.einsum('ithqk, iqhd -> tkihd' ,[text_attention, text_values])
            out_text = rearrange(out_text, 't k i h d -> i t k (h d)')
            out_text = self.fc_out_text(out_text)
            if pool:
                if max_pool:
                    out_text = torch.max(out_text, 2).values
                else:
                    out_text = out_text.mean(2)
        return out, out_text, attention, text_attention # Only returning cls token

class I2DFormer(nn.Module):
    def __init__(self, cfg, image_encoder, text_encoder = None, return_attention = False):
        super().__init__()
        self.cfg = cfg

        self.image_encoder = image_encoder
        self.return_attention = return_attention

        if text_encoder is None:
            if cfg.models.positional:
                self.text_encoder = x_clip.TextTransformer_noemb(dim = cfg.models.emb_dim, num_tokens = 50000, max_seq_len=cfg.models.max_seq_len, depth = cfg.models.transformer_depth , dim_head = cfg.dataset.feat_dim//cfg.models.heads, heads = cfg.models.heads, return_attention = False)
            else:
                self.text_encoder = x_clip.Transformer(cfg.models.emb_dim, depth = cfg.models.transformer_depth , dim_head = cfg.dataset.feat_dim//cfg.models.heads, heads = cfg.models.heads, add_cls=cfg.models.cls_token, return_attention = True)
        else:
            self.text_encoder = text_encoder

        self.image_encoder_train = False
        if cfg.models.img_transformer_depth > 0:
            self.image_encoder_train = x_clip.Transformer(cfg.models.emb_dim, depth = cfg.models.img_transformer_depth , dim_head = cfg.dataset.feat_dim//cfg.models.img_heads, heads = cfg.models.img_heads, add_cls=False, return_attention = False)

        self.img_embedder = MLP(cfg.dataset.feat_dim, cfg.models.emb_dim, num_layers = cfg.models.img_layers, dropout= True, norm = True,
                            layers = [cfg.models.hidden_dim])


        self.aux_embedder = MLP(cfg.dataset.emb_dim, cfg.models.emb_dim, num_layers=cfg.models.emb_layers, 
                            dropout=True, norm = True, layers = [cfg.models.hidden_dim])

        if self.cfg.models.decode > 0.0 or self.cfg.models.t2i_attention > 0.0:
            self.decoder = I2DAttention(cfg.models.emb_dim, 1, False, cfg.models.image_tokens, cfg.models.t2i_attention > 0.0)
            self.cross_score = nn.Linear(cfg.models.emb_dim, 1)
        if cfg.models.t2i_attention > 0.0:
            self.text_score = nn.Linear(cfg.models.emb_dim, 1)

    def forward(self, img, aux, aux_mask, labels):
        feat = self.image_encoder(img)
        if self.cfg.models.freeze_image:
            feat = feat.detach()


        img = self.img_embedder(feat)

        if self.cfg.models.img_transformer_depth > 0:
            img, _ = self.image_encoder_train(img)
        
        
        aux = self.aux_embedder(aux)
        token_embeddings, text_attention  = self.text_encoder(aux, aux_mask)
        # img, token_embeddings = map(l2norm, (img, token_embeddings))

        cls_emb = token_embeddings[:, 0, :]
        cls_img = img[:, 0, :]
        

        loss = 0.0
        cross_out = None
        if self.cfg.models.cross_cls > 0:
            cross_out, attention = self.cross_score(img[:, 1:, :], token_embeddings[:, 1:, :], aux_mask, 1)
            cross_loss = self.cfg.models.cross_cls * F.cross_entropy(cross_out, labels)
            loss += cross_loss

        if self.cfg.models.decode > 0.0 or self.cfg.models.t2i_attention > 0.0:
            imgx = img
            imgx, textx,  attention, t2i_attention = self.decoder(imgx[:, 1:, :], (token_embeddings[:, 1:, :]), aux_mask, max_pool = self.cfg.models.max_pool)
            imgo = imgx
            cross_out = self.cross_score(imgo).squeeze(-1)
            cross_loss = self.cfg.models.decode * F.cross_entropy(cross_out, labels)
            loss += cross_loss

        if self.cfg.models.lambda_cls > 0:
            out = torch.matmul(cls_img, cls_emb.T)
            loss += self.cfg.models.lambda_cls * F.cross_entropy(out, labels)
        else:
            out = cross_out

        if self.cfg.models.t2i_attention > 0.0:
            out_text = self.text_score(textx).squeeze(-1)
            text_cross_loss = self.cfg.models.t2i_attention * F.cross_entropy(out_text, labels)
            loss += text_cross_loss
            if self.cfg.models.decode == 0.0:
                cross_out = out_text


        if self.return_attention:
            return loss, out, attention, t2i_attention
        return loss, out, cross_out