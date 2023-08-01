import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from clip import x_clip
from einops import rearrange, reduce, repeat
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttPool(nn.Module):
    def __init__(self, embed_dim, num_heads = 1, tokens = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.probe = nn.Parameter(torch.zeros(1, 1, tokens, embed_dim))
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
 
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=4 * embed_dim, act_layer=nn.GELU, drop=0)
        
    def forward(self, x, mask = None):
        n, m, l , d = x.shape
        probe = self.probe.tile((n, m, 1, 1))
        
        query = self.query(probe)
        key, value = self.key(x), self.value(x)
 
        attention = torch.einsum('nmqd, nmkd -> nmqk', [query, key])
        if mask is not None:
            # mask is m k
            attention = rearrange(attention, 'n m q k -> m k n q')
            attention[~mask] = max_neg_value(attention.dtype)
            attention = rearrange(attention, 'm k n q -> n m q k')
 
        attention = torch.softmax(attention / (self.embed_dim ** (1 / 2)), dim=-1)
        x = torch.einsum('nmqk, nmkd -> nmqd', [attention, value])
 
        y = self.norm(x)
        x = x + self.mlp(y)
        return x[:, :, 0, :]

class AttPoolcls(nn.Module):
    def __init__(self, embed_dim, tokens = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokens = tokens
        self.probe = nn.Parameter(torch.zeros(1, tokens, embed_dim))
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
 
        self.mlp = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        n, l , d = x.shape
        probe = self.probe.tile((n, 1, 1))
        
        query = self.query(probe)
        key, value = self.key(x), self.value(x)

        attention = torch.einsum('nqd, nkd -> nqk', [query, key])
 
        attention = torch.softmax(attention / (self.embed_dim ** (1 / 2)), dim=-1)
        x = torch.einsum('nqk, nkd -> nqd', [attention, value])
 
        x = self.mlp(x)
        if self.tokens > 1:
            return x
        return x[:, 0, :]

class I2DAttention(nn.Module):
    def __init__(self, embed_size, heads, cls_present, img_tokens, pooling_type = None):
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
        self.pooling_type = pooling_type
        if self.pooling_type == 'map':
            self.pool = AttPool(embed_size)
    
    
    def forward(self, image, text, text_mask=None, topk = -1):
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
        
        # Masking out non existing language tokens
        if text_mask is not None:
            if self.cls_present:
                text_mask = F.pad(text_mask, (1, 0), value = True)
            energy = rearrange(energy, 'i t h q k -> t k i h q')
            energy[~text_mask] = torch.finfo(torch.float32).min
            energy = rearrange(energy, 't k i h q -> i t h q k')        
        
        # Energy is [image, documents, head, patches, words]
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        text_attention = torch.softmax(energy / (self.embed_size ** (1 / 2)) , dim = -2)

        out = torch.einsum('ithqk, tkhd -> iqthd', [attention, value])
        out = rearrange(out, 'i q t h d -> i t q (h d)')
        out = self.fc_out(out)
        if self.pooling_type is not None:
            if self.pooling_type == 'max':
                out = torch.max(out, 2).values
            elif self.pooling_type == 'mean':
                out = out.mean(2)
            elif self.pooling_type == 'map':
                out = self.pool(out)
            else:
                raise Exception("Invalid Pooling selected")

        out_text = None
        return out, out_text, attention, text_attention

class I2MVFormer(nn.Module):
    def __init__(self, cfg, image_encoder, text_encoder = None, return_attention = False):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = image_encoder
        self.return_attention = return_attention

        if text_encoder is None:
            self.text_encoder = x_clip.PerceiverTransformer(cfg.models.emb_dim, depth = cfg.models.transformer_depth , positional = cfg.models.positional, dim_head = cfg.dataset.feat_dim//cfg.models.heads, heads = cfg.models.heads, add_cls=cfg.models.fine_grained_cls + 1, return_attention = True)
        else:
            self.text_encoder = text_encoder
        
        self.text_encoder = nn.DataParallel(self.text_encoder)
        
        self.image_encoder_train = False
        if cfg.models.img_transformer_depth > 0:
            self.image_encoder_train = x_clip.PerceiverTransformer(cfg.models.emb_dim, depth = cfg.models.transformer_depth , positional = cfg.models.positional, dim_head = cfg.dataset.feat_dim//cfg.models.heads, heads = cfg.models.heads, add_cls=cfg.models.fine_grained_cls + 1, return_attention = True)

        self.img_embedder = MLP(cfg.dataset.feat_dim, cfg.models.emb_dim, num_layers = cfg.models.img_layers, dropout= True, norm = True,
                            layers = [cfg.models.hidden_dim])


        self.aux_embedder = MLP(cfg.dataset.emb_dim, cfg.models.emb_dim, num_layers=cfg.models.emb_layers, 
                            dropout=True, norm = True, layers = [cfg.models.hidden_dim])

        if self.cfg.models.decode > 0.0:
            self.decoder = I2DAttention(cfg.models.emb_dim, 1, False, cfg.models.image_tokens, pooling_type=cfg.models.pooling_type)
            self.cross_score = nn.Linear(cfg.models.emb_dim, 1)
        if self.cfg.models.t2i_attention > 0.0:
            self.decoder_d2i = I2DAttention(cfg.models.emb_dim, 1, False, cfg.models.image_tokens, pooling_type=cfg.models.pooling_type)
            self.cross_score_d2i = nn.Linear(cfg.models.emb_dim, 1)
        if self.cfg.models.pooling_type == 'mapa':
            self.cls_pooler = AttPoolcls(cfg.models.emb_dim)

        if self.cfg.models.multipool:
            self.multipool = x_clip.PerceiverTransformer(cfg.models.emb_dim, depth = cfg.models.transformer_depth , positional = False, dim_head = cfg.dataset.feat_dim//cfg.models.heads, heads = cfg.models.heads, add_cls=cfg.models.fine_grained_cls, return_attention = False)

    def forward(self, img, aux, aux_mask, labels):
        attentions = []
        feat = self.image_encoder(img)
        if self.cfg.models.freeze_image:
            feat = feat.detach()


        img = self.img_embedder(feat)

        cls_img = img[:, 0, :]
        if self.cfg.models.img_transformer_depth > 0:
            img, _ = self.image_encoder_train(img[:,1:,:])
            img = img[:, :self.cfg.models.fine_grained_cls, :]
        
        aux = self.aux_embedder(aux)
        
        num_articles = aux.shape[1]
        aux = rearrange(aux, 'c a w e -> (c a) w e')
        aux_mask = rearrange(aux_mask, 'c a w -> (c a) w')
        token_embeddings, svsummaryattention  = self.text_encoder(aux, aux_mask)
        attentions.append(svsummaryattention)
        token_embeddings = rearrange(token_embeddings, '(c a) w e -> c a w e', a = num_articles)

        cls_emb = token_embeddings[:, :, 0, :]
        if self.cfg.models.pooling_type == 'mapa':
            cls_emb = self.cls_pooler(cls_emb)
        else:
            cls_emb = cls_emb.mean(1)
        
        att_emb = token_embeddings[:, :, 1:self.cfg.models.fine_grained_cls+1, :]
        att_emb = rearrange(att_emb, 'c a w e -> c (a w) e')
        if self.cfg.models.multipool:
            att_emb, mvsummaryattention = self.multipool(att_emb)
            attentions.append(mvsummaryattention)
            att_emb = att_emb[:, 0:self.cfg.models.fine_grained_cls, :]

        loss = 0.0
        cross_out = None

        if self.cfg.models.t2i_attention > 0.0:
            imgx = img
            imgx, textx,  attention, t2i_attention = self.decoder_d2i(att_emb, imgx[:, 1:, :], text_mask = None)
            imgo = imgx.transpose(0,1)
            cross_out = self.cross_score_d2i(imgo).squeeze(-1)
            cross_loss = self.cfg.models.t2i_attention * F.cross_entropy(cross_out, labels)
            loss += cross_loss

        if self.cfg.models.decode > 0.0:
            imgx = img
            imgx, textx,  attention, t2i_attention = self.decoder(imgx[:, 1:, :], att_emb, text_mask = None)
            imgo = imgx
            cross_out = self.cross_score(imgo).squeeze(-1)
            cross_loss = self.cfg.models.decode * F.cross_entropy(cross_out, labels)
            loss += cross_loss
            attentions.append(attention)

        if self.cfg.models.lambda_cls > 0:
            out = torch.matmul(cls_img, cls_emb.T)
            loss += self.cfg.models.lambda_cls * F.cross_entropy(out, labels)
        else:
            out = cross_out

        if self.return_attention:
            return loss, out, attentions, t2i_attention
        return loss, out, cross_out
