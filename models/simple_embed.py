import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleEmbed(nn.Module):
    def __init__(self, cfg, image_encoder, text_encoder = None):
        super().__init__()
        self.cfg = cfg

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.img_embedder = MLP(cfg.dataset.feat_dim, cfg.models.emb_dim, num_layers = cfg.models.img_layers, dropout= True, norm = True,
                            layers = [cfg.models.hidden_dim])


        self.aux_embedder = MLP(cfg.dataset.emb_dim, cfg.models.emb_dim, num_layers=cfg.models.emb_layers, 
                            dropout=True, norm = True, layers = [cfg.models.hidden_dim])


    def forward(self, img, aux, labels):
        feat = self.image_encoder(img)
        if self.cfg.models.image_tokens:
            feat = feat[:, 0, :]
        if self.cfg.models.freeze_image:
            feat = feat.detach()
        img = self.img_embedder(feat)
        
        if self.text_encoder is not None:
            mask = (aux != 0)
            # print(aux.shape, mask.shape)
            aux = self.text_encoder(aux, mask)[:, 0, :]
            # print(aux.shape)
            # aux = aux.sum(1) / mask.sum(1).view(mask.shape[0], aux.shape[-1])
        emb = self.aux_embedder(aux)

        out = torch.matmul(img, emb.T)
        loss = F.cross_entropy(out, labels)

        # top1acc = (torch.topk(out, 1, dim = 1) == labels).mean()
        # top5acc = (torch.eq(torch.topk(out, 5, dim = 1), labels).any(dim = 1)).mean()

        return loss, out