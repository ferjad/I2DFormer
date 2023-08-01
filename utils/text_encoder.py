import re
import tqdm
import gensim
import stanza
import logging
import numpy as np
from pathlib import Path

import pickle
import re
from nltk import tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

def return_enc(x, mask):
    return x

def load_embeddings(emb_file):
    if re.match(r".*\.txt$", str(emb_file)):
        return load_txt_embeddings(emb_file)
    elif re.match(r".*\.bin$", str(emb_file)):
        return load_bin_embeddings(emb_file)
    else:
        raise ValueError(f'Do not know how to read the file: "{emb_file}"')


def load_bin_embeddings(emb_file):
    if 'glove' not in emb_file:
        m = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=True)
    else:
        m = gensim.models.KeyedVectors.load(emb_file)
    return m, m.vector_size


def load_txt_embeddings(emb_file):
    embeddings = {}
    n_expected_tokens = None
    dim = None
    logging.info(f'Loading embeddings "{emb_file}"')
    with open(emb_file, 'r') as f:

        if re.match(r"^enwiki_.*d\.txt", str(emb_file)):
            n_expected_tokens, dim = (float(x) for x in f.readline().rstrip())

        for line in tqdm.tqdm(f, unit='tokens'):
            if not line.strip():
                continue
            token, emb_str = line.split(' ', 1)
            emb = [float(x) for x in emb_str.split(' ')]

            if dim is None:
                dim = len(emb)
            else:
                assert len(emb) == dim

            embeddings[token] = np.array(emb)

    if n_expected_tokens:
        assert len(embeddings) == n_expected_tokens

    logging.info(f'Loaded: {len(embeddings)} embedding tokens.')
    return embeddings, dim


class WordEmbeddings:

    def __init__(self, emb_path, stanza_path):
        self.emb_file = emb_path
        self.emb, self.emb_dim = load_embeddings(self.emb_file)

        # stanza.download('en')
        self.tokenizer = stanza.Pipeline('en', processors='tokenize', model_dir=stanza_path)

    def tokenize(self, text):
        doc = self.tokenizer(text)
        for s in doc.sentences:
            for token in s.tokens:
                yield token.text

    def __call__(self, text):
        text = text.lower()

        # Compute moving average
        i = 0
        t = 0
        mean_feats = np.zeros(self.emb_dim)
        for token in self.tokenize(text):
            if token in self.emb:
                mean_feats += (self.emb[token] - mean_feats) / (i+1)
                i += 1
            t += 1

        if i < 1:
            raise RuntimeError(f'Could not embed any tokens, {text}')
        print(f'Encoded tokens: {i}/{t}={i/t}')

        return mean_feats

    def token_embeddings(self, text, return_tokens = False):
        text = text.lower()
        embeddings = []
        found = 0
        total = 0
        found_tokens = []
        for token in self.tokenize(text):
            if token in self.emb:
                embeddings.append(np.expand_dims(self.emb[token], 0))
                found += 1
                found_tokens.append(token)
            total += 1
        if found < 1:
            raise RuntimeError(f'Could not embed any tokens, {text}')
        print(f'Encoded tokens: {found}/{total}={found/total}')
        embeddings = np.concatenate(embeddings, 0)
        if return_tokens:
            return np.expand_dims(embeddings, 0), found_tokens
        return np.expand_dims(embeddings, 0)