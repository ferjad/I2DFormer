"""Basic experiments configuration

For different tasks, a specific configuration might be created by importing this basic config.

"""

from yacs.config import CfgNode as CN

cfg = CN()
cfg.log_dir = ''
cfg.cluster = True

cfg.dataset = CN()
cfg.dataset.name = 'CUB'
cfg.dataset.root = 'datasets/CUB_200_2011/images'
cfg.dataset.train_split = 'datasets/xlsa17/data/CUB/trainclasses1.txt'
cfg.dataset.eval_split = 'datasets/xlsa17/data/CUB/valclasses1.txt'
cfg.dataset.articles = 'articles/cub_visual.pkl'
cfg.dataset.filelist = 'xlsa17/data/CUB/res101.mat'
cfg.dataset.feats_file = 'feature_word_embedding_CUB/res101_finetune.mat'
cfg.dataset.split_file = 'xlsa17/data/CUB/att_splits.mat'
cfg.dataset.feat_dim = 2048
cfg.dataset.emb_dim = 300
cfg.dataset.pretrained_feats = False
cfg.dataset.mode = 'gzsl'
cfg.dataset.num_articles = 4
cfg.dataset.concat = False

cfg.models = CN()
cfg.models.name = 'i2mvformer'
cfg.models.load = 'no'
cfg.models.emb_dim = 300
cfg.models.img_layers = 1
cfg.models.hidden_dim = 512
cfg.models.emb_layers = 1
cfg.models.glove_path = 'pretrained/glove.840B.300d.txt'
cfg.models.text_encoder_path = ''
cfg.models.stanza_path = 'stanza_resources'
cfg.models.vit_base = 'dino_vitbase16_pretrain.pth'
cfg.models.resnet_path = ''
cfg.models.image_encoder = 'vit'
cfg.models.text_encoder = 'glove_list'
cfg.models.freeze_image = True
cfg.models.freeze_image_after = 9999
cfg.models.transformer_depth = 1
cfg.models.heads = 1
cfg.models.cls_token = False
cfg.models.train_image_after = 9999
cfg.models.image_tokens = False
cfg.models.lambda_cls = 1.0
cfg.models.cross_cls = 0.0
cfg.models.decode = 0.0
cfg.models.positional = False
cfg.models.img_transformer_depth = 0
cfg.models.img_heads = 1
cfg.models.embed_classname = False
cfg.models.max_pool = False
cfg.models.t2i_attention = 0.0
cfg.models.fine_grained_cls = None
cfg.models.pooling_type = 'max'
cfg.models.multipool = False
cfg.models.includewiki = True
cfg.models.unseen_calibration = None
cfg.models.max_seq_len = 700

cfg.train = CN()
cfg.train.imagefinetune = False
cfg.train.ee_diff = False
cfg.train.epochs = 50
cfg.train.eval_every = 1
cfg.train.batch_size = 16
cfg.train.lr = 1e-3
cfg.train.ee_last = False
cfg.train.image_size = 224
cfg.train.train_split = 'trainval'
cfg.train.eval_split = 'test'