from models.vit import vit_base
import torch
import torch.nn as nn
from torchvision import models
from utils.text_encoder import WordEmbeddings
from utils.utils import View
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from os.path import join as ospj
import numpy as np
from clip.x_clip import TextTransformer
from tqdm import tqdm
import torch.nn.functional as F
import nltk
# Model imports
from models.i2dformer import I2DFormer # Submitted
from models.i2mvformer import I2MVFormer
from sklearn.feature_extraction.text import TfidfVectorizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(cfg, return_attention = False):
    image_encoder = get_image_encoder(cfg)
    if cfg.models.name == 'i2dformer':
        model = I2DFormer(cfg, image_encoder, return_attention = return_attention)
    elif cfg.models.name == 'i2mvformer':
        model = I2MVFormer(cfg, image_encoder, return_attention = return_attention)
    else:
        raise Exception('Invalid model')
    return model

class View(nn.Module):
    def __init__(self, feats):
        super(View, self).__init__()
        self.feats = feats

    def forward(self, x):
        b = x.shape[0]
        pooled = F.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1).unsqueeze(1)
        pre_pooled =  x.view(b, -1, self.feats)
        return torch.cat([pooled, pre_pooled], 1)

def get_image_encoder(cfg):
    if cfg.models.image_encoder == 'vit':
        model = vit_base()
        try:
            ckpt = torch.load(cfg.models.vit_base)
            model.load_state_dict(ckpt)
        except:
            print('Could not load pretrained ViT, please check path and try again. Image Extractor is randomly initialized!')
    else:
        model = nn.Sequential()
    return model


def get_text_encoder(cfg):
    check = cfg.models.text_encoder
    print(f'Selected text encoder is {check}')
    if check == 'glove':
        model = WordEmbeddings(cfg.models.glove_path, cfg.models.stanza_path)
        embfunction = get_embeddings_glove
    elif check == 'glove_im':
        model = WordEmbeddings(cfg.models.glove_path, cfg.models.stanza_path)
        embfunction = get_embeddings_glove_imagenet
    elif check == 'glove_list':
        model = WordEmbeddings(cfg.models.glove_path, cfg.models.stanza_path)
        embfunction = get_embeddings_glove_list
    elif check == 'none':
        model, embfunction = None, None
    elif check == 'sentencebert':
        print('Loading sentence bert')
        model = SentenceTransformer(cfg.models.text_encoder_path)
        embfunction = None
    elif check == 'hf' :
        print('Hugging face ', cfg.models.text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(cfg.models.text_encoder_path)
        text_encoder = AutoModel.from_pretrained(cfg.models.text_encoder_path)
        model = [text_encoder, tokenizer]
        embfunction = get_embeddings_hf
    elif check == 'hftokens' :
        print('Hugging face ', cfg.models.text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(cfg.models.text_encoder_path)
        text_encoder = AutoModel.from_pretrained(cfg.models.text_encoder_path)
        model = [text_encoder, tokenizer]
        embfunction = get_embeddings_hf_tokens
    elif check == 'bert' :
        print('Bert')
        tokenizer = AutoTokenizer.from_pretrained(ospj(cfg.models.text_encoder_path, 'tokenizer'))
        text_encoder = AutoModel.from_pretrained(ospj(cfg.models.text_encoder_path, 'model'))
        model = [text_encoder, tokenizer]
    elif check == 'scratch':
        print('scratch')
        model = TextTransformer(dim = 128, num_tokens = 50000, max_seq_len=1822, depth = 4, heads = 2)
        embfunction = get_embeddings_hf_fromscratch
    return model, embfunction

def get_embeddings_hf(cfg, dataset, text_encoder):
        text_model, tokenizer = text_encoder
        text_model = text_model.to(device)
        
        embeddings  = []
        masks = []
        max_len = 0
        for class_name in tqdm(dataset.class_names, desc='Max len'):
            if cfg.dataset.name == 'CUB':
                class_name = class_name.split('.')[1]
            article = dataset.articles[class_name]
            with torch.no_grad():
                encoded_input = tokenizer(article, return_tensors='pt')['input_ids']
                if encoded_input.shape[1] > max_len:
                    max_len = encoded_input.shape[1]
        for class_name in tqdm(dataset.class_names, desc='Embeddings'):
            if cfg.dataset.name == 'CUB':
                class_name = class_name.split('.')[1]
            article = dataset.articles[class_name]
            
            # print(class_name)
            with torch.no_grad():
                current_emb = []
                
                encoded_input = tokenizer(article, padding = 'max_length', max_length = max_len, return_tensors='pt')
                
                tokens = encoded_input['input_ids'].to(device)
                mask = encoded_input['attention_mask'].to(device)
                
                emb = text_model(input_ids = tokens, attention_mask = mask)['last_hidden_state']
                
                # We dont want cls token from text transformer
                emb = emb[:,1:-1,:]
                mask = mask[:, 1:-1]

                embeddings.append(emb)
                masks.append(mask.bool())
            
        embeddings = torch.cat(embeddings, 0)
        masks = torch.cat(masks, 0)
        text_model = text_model.cpu()
        return embeddings, masks

def get_embeddings_glove(cfg, dataset, text_encoder):
        embeddings  = []
        max_seq = 0
        for class_name in dataset.class_names:
            if cfg.dataset.name == 'CUB':
                class_name = class_name.split('.')[1]
            article = dataset.articles[class_name]
            emb = text_encoder.token_embeddings(article)
            if emb.shape[1] > max_seq:
                max_seq = emb.shape[1]
            embeddings.append(emb)

        out_embeddings = []
        out_masks = []
        # Creating concat tensor and mask
        for emb in embeddings:
            seq_len = emb.shape[1]
            if seq_len < max_seq:
                padding = max_seq - seq_len
                emb = np.concatenate([emb, np.zeros((1, padding, 300))], 1)
                mask = np.concatenate([np.ones((1, seq_len)), np.zeros((1, padding))], 1) 
            out_embeddings.append(emb)
            out_masks.append(mask)
        out_embeddings = np.concatenate(out_embeddings, 0)
        out_masks = np.concatenate(out_masks, 0)
        return torch.from_numpy(out_embeddings).float(), torch.from_numpy(out_masks).bool()

def get_embeddings_glove_list(cfg, dataset, text_encoder):
        embeddings  = []
        max_seq = 0
        for class_name in dataset.class_names:
            if cfg.dataset.name == 'CUB':
                class_name = class_name.split('.')[1]
            article = dataset.articles[class_name][:cfg.dataset.num_articles]
            if not cfg.models.includewiki:
                article = article[1:]
            if cfg.dataset.concat:
                article = [' '.join(article)]
            emb = []
            for a in article:
                emb_l = text_encoder.token_embeddings(a)
                if emb_l.shape[1] > max_seq:
                    max_seq = emb_l.shape[1]
                emb.append(emb_l)
            embeddings.append(emb)

        # These are the ones we export
        out_embeddings = []
        out_masks = []
        # Creating concat tensor and mask
        for emb_list in embeddings:
            # These are the per class one
            emb_class_list = []
            out_class_mask = []
            for emb in emb_list:
                seq_len = emb.shape[1]
                if seq_len < max_seq:
                    padding = max_seq - seq_len
                    emb = np.concatenate([emb, np.zeros((1, padding, 300))], 1)
                    mask = np.concatenate([np.ones((1, seq_len)), np.zeros((1, padding))], 1) 
                emb_class_list.append(emb)
                out_class_mask.append(mask)
            # concatenate per class
            emb_class_list = np.expand_dims(np.concatenate(emb_class_list, 0), 0)
            out_class_mask = np.expand_dims(np.concatenate(out_class_mask, 0), 0)
            # append to global list
            out_embeddings.append(emb_class_list)
            out_masks.append(out_class_mask)
        out_embeddings = np.concatenate(out_embeddings, 0)
        out_masks = np.concatenate(out_masks, 0)
        return torch.from_numpy(out_embeddings).float(), torch.from_numpy(out_masks).bool()

def get_embeddings_glove_imagenet(cfg, dataset, text_encoder):
        embeddings  = []
        max_seq = 0
        for idx, article in enumerate(dataset.articles):
            print(idx)
            emb = text_encoder.token_embeddings(article)
            if emb.shape[1] > max_seq:
                max_seq = emb.shape[1]
            embeddings.append(emb)

        out_embeddings = []
        out_masks = []
        # Creating concat tensor and mask
        for emb in embeddings:
            seq_len = emb.shape[1]
            if seq_len < max_seq:
                padding = max_seq - seq_len
                emb = np.concatenate([emb, np.zeros((1, padding, 300))], 1)
                mask = np.concatenate([np.ones((1, seq_len)), np.zeros((1, padding))], 1) 
            out_embeddings.append(emb)
            out_masks.append(mask)
        out_embeddings = np.concatenate(out_embeddings, 0)
        out_masks = np.concatenate(out_masks, 0)
        return torch.from_numpy(out_embeddings).float(), torch.from_numpy(out_masks).bool()

def get_embeddings_hf_fromscratch(cfg, dataset, text_encoder):
        text_model, tokenizer = text_encoder
        text_model = text_model.to(device)
        
        embeddings  = []
        masks = []
        for class_name in tqdm(dataset.class_names, desc='Embeddings'):
            class_name = class_name.split('.')[1]
            article = dataset.articles[class_name]
            
            # print(class_name)
            with torch.no_grad():
                current_emb = []
                
                encoded_input = tokenizer(article, padding = 'max_length', max_length = 512, return_tensors='pt')
                tokens = encoded_input['input_ids'].to(device)
                token_mask = encoded_input['attention_mask'].to(device)
                mask = (tokens!= 1)
                if not cfg.models.train_text_encoder:
                    emb = text_model(input_ids = tokens, attention_mask = token_mask)['last_hidden_state']
                else:
                    emb = tokens

                embeddings.append(emb)
                masks.append(mask)
            
        embeddings = torch.cat(embeddings, 0)
        masks = torch.cat(masks, 0)
        text_model = text_model.cpu()
        return embeddings, masks

def get_embeddings_hf_tokens(cfg, dataset, text_encoder):
        text_model, tokenizer = text_encoder
        text_model = text_model.to(device)
        
        embeddings  = []
        masks = []
        max_len = 0
        for class_name in tqdm(dataset.class_names, desc='Max len'):
            if cfg.dataset.name == 'CUB':
                class_name = class_name.split('.')[1]
            article = dataset.articles[class_name]
            with torch.no_grad():
                encoded_input = tokenizer(article, return_tensors='pt')['input_ids'][:,1:-1]
                if encoded_input.shape[1] > max_len:
                    max_len = encoded_input.shape[1]


        
        for class_name in tqdm(dataset.class_names, desc='Embeddings'):
            if cfg.dataset.name == 'CUB':
                class_name = class_name.split('.')[1]
            article = dataset.articles[class_name]
            sentences = nltk.tokenize.sent_tokenize(article)
            with torch.no_grad():
                article_emb, article_mask = [], []
                for sentence in sentences:
                    encoded_input = tokenizer(sentence, return_tensors='pt')
                    
                    tokens = encoded_input['input_ids'].to(device)
                    mask = encoded_input['attention_mask'].to(device)
                    
                    emb = text_model(input_ids = tokens, attention_mask = mask)['last_hidden_state']
                    
                    # We dont want cls token from text transformer
                    emb = emb[:,1:-1,:]
                    mask = mask[:, 1:-1]
                    article_emb.append(emb[0])
                    article_mask.append(mask[0])
                article_emb = torch.cat(article_emb)
                article_mask = torch.cat(article_mask)
                seq_len = article_emb.shape[0]
                if seq_len < max_len:
                    padding = max_len - seq_len
                    emb = torch.cat([article_emb, torch.zeros((padding, article_emb.shape[1]), device = device)]).unsqueeze(0)
                    mask = torch.cat([article_mask, torch.zeros(padding, device = device)]).unsqueeze(0)
                else:
                    emb = article_emb.unsqueeze(0)
                    mask = article_mask.unsqueeze(0)
                embeddings.append(emb)
                masks.append(mask.bool())
            
        embeddings = torch.cat(embeddings, 0)
        masks = torch.cat(masks, 0)
        text_model = text_model.cpu()
        return embeddings, masks
