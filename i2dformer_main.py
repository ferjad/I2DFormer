#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import numpy as np
import tqdm
import torchvision.models as tmodels
from tqdm import tqdm
import os
from os.path import join as ospj
import itertools
import glob
import csv
import shutil
import tempfile
import argparse
import nltk
print('Importing tf')
# import tensorflow as tf
from tensorflow.io.gfile import GFile
print('Imported tf')
from gensim.models.keyedvectors import KeyedVectors
from scipy.stats import hmean
from einops import rearrange

#Local imports
from data.CUB import CUB_dataset
from models.cross_modality import CrossModal
from models.all_token_transformer import MultiTrans
from models.simple_embed_att import AttentionEmbed
from utils.setup_encoders import get_text_encoder, get_model
from flags import cfg
from clip import x_clip
from utils.utils import save_checkpoint, compute_per_class_acc, Result

nltk.data.path += ['pretrained/nltk_data']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_zsl, max_gzsl, max_cross_zsl, max_cross_gzsl = 0, 0, 0, 0

    
def main():
    global max_acc
    parser = argparse.ArgumentParser(description='ZSL')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='/workdir/finegrainedwiki/configs/run_local.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.log_dir
    os.makedirs(output_dir, exist_ok = True)
    try:
        # shutil.copy2(args.config_file, os.path.join(output_dir, 'config.yaml'))
        with GFile(os.path.join(output_dir, 'config.yaml'), "w") as f:
            f.write(cfg.dump())   # save config to file
    except:
        print(f'Copy operation threw an error')
    # local_log = tempfile.mkdtemp()
    # print(f'Temp tensorboard to {local_log}')

    tb_folder = output_dir
    if cfg.cluster:
        tb_folder = 'gs://bucket/'+ '/'.join(output_dir.split('/')[3:])
    print(f'Tensorboard folder is {tb_folder}')
    writer = SummaryWriter(log_dir=tb_folder)

    ### Init datasets
    trainset = CUB_dataset(cfg, 'train')
    valset_unseen = CUB_dataset(cfg, 'test_unseen')
    valset_seen = CUB_dataset(cfg, 'test_seen')

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size = cfg.train.batch_size,
                                              shuffle = True,
                                              num_workers = 8)

    valloader_unseen = torch.utils.data.DataLoader(valset_unseen, 
                                              batch_size = cfg.train.batch_size,
                                              shuffle = False,
                                              num_workers = 8)

    valloader_seen = torch.utils.data.DataLoader(valset_seen, 
                                              batch_size = cfg.train.batch_size,
                                              shuffle = False,
                                              num_workers = 8)
    
    model_text_encoder = None
    
    # Checking for cached aux
    print('Looking up cache info')
    fn = cfg.dataset.articles.split('/')[-1].split('.')[0]
    if cfg.models.includewiki:
        if cfg.dataset.concat:
            lookup_name = f'cached_aux_{cfg.dataset.name}_{fn}_concat_n_{cfg.dataset.num_articles}.pkl'
        else:
            lookup_name = f'cached_aux_{cfg.dataset.name}_{fn}.pkl'
    else:
        if cfg.dataset.concat:
            lookup_name = f'cached_aux_{cfg.dataset.name}_{fn}_concat_n_{cfg.dataset.num_articles}_nowiki.pkl'
        else:
            lookup_name = f'cached_aux_{cfg.dataset.name}_{fn}_nowiki.pkl'
    print(f'Looking up aux information from {lookup_name}')
    try:
        with GFile(ospj('gs://bucket/ferjad/data/finegrainedwiki',lookup_name), 'rb') as f:
            seen_aux, seen_aux_mask, unseen_aux, unseen_aux_mask = torch.load(f)
        raise Exception('Going to manual loading')
    except:
        print('Loading aux from text')
        text_encoder, get_embeddings = get_text_encoder(cfg)
        seen_aux, seen_aux_mask = get_embeddings(cfg, trainset, text_encoder)
        unseen_aux, unseen_aux_mask = get_embeddings(cfg, valset_unseen, text_encoder)
        with GFile(ospj('gs://bucket/ferjad/data/finegrainedwiki',lookup_name), 'wb') as f:
            torch.save([seen_aux, seen_aux_mask, unseen_aux, unseen_aux_mask], f)
    unseen_mask = valset_unseen.unseen_mask
    
    seen_aux, seen_aux_mask = seen_aux[:, :cfg.dataset.num_articles, :, :], seen_aux_mask[:, :cfg.dataset.num_articles, :]
    unseen_aux, unseen_aux_mask = unseen_aux[:, :cfg.dataset.num_articles, :, :], unseen_aux_mask[:, :cfg.dataset.num_articles, :]
    if cfg.dataset.concat:
        seen_aux, unseen_aux = rearrange(seen_aux, 'c a w e -> c (a w) e'), rearrange(unseen_aux, 'c a w e -> c (a w) e')
        seen_aux_mask, unseen_aux_mask = rearrange(seen_aux_mask, 'c a w -> c (a w)'), rearrange(unseen_aux_mask, 'c a w -> c (a w)')
    seen_aux, unseen_aux = seen_aux.to(device), unseen_aux.to(device)
    seen_aux_mask, unseen_aux_mask = seen_aux_mask.to(device), unseen_aux_mask.to(device)
    cfg.dataset.emb_dim = unseen_aux.shape[-1]

    print(f'Seen Aux {seen_aux.shape} Unseen aux {unseen_aux.shape} Unseen mask {unseen_mask.shape} Total Unseen {unseen_mask.sum()}')



    model = get_model(cfg)
    model = model.to(device)
    results_zsl = Result()
    results_c_zsl = Result()
    results_gzsl = Result()
    results_c_gzsl = Result()
    if cfg.models.load != 'no':
        print(f'Loading ckpt from {cfg.models.load}')
        ckpt = torch.load(cfg.models.load)
        model.load_state_dict(ckpt['net'])

    if cfg.train.imagefinetune:
        optimize_parameters = []
        print('Finetuning the image backbone only')
        for name, parameters in model.named_parameters():
            if 'image_encoder' in name:
                optimize_parameters.append(parameters)
    elif cfg.train.ee_diff:
        cfg.models.freeze_image = False
        backbone_parameters = []
        model_parameters = []
        print('Learning full model at diff lr')
        for name, parameters in model.named_parameters():
            if 'image_encoder' in name:
                backbone_parameters.append(parameters)
            else:
                model_parameters.append(parameters)
        optimize_parameters = [{'params' : backbone_parameters, 'lr' : 5e-6},
                               {'params' : model_parameters,}]
    elif cfg.train.ee_last:
        cfg.models.freeze_image = False
        backbone_parameters = []
        finetuned = model.module.image_encoder
        for p in finetuned.parameters():
            p.requires_grad = False
        for c in list(list(finetuned.children())[-3].children())[-2:]:
            for p in c.parameters():
                p.requires_grad = True
                backbone_parameters.append(p)
        print(f'Backbone parameters {len(backbone_parameters)}')
        optimize_parameters = model.parameters()
    else:
        optimize_parameters = model.parameters()

    optimizer = torch.optim.Adam(optimize_parameters, lr = cfg.train.lr)

    print(model)
    
    
    

    for epoch in tqdm(range(cfg.train.epochs)):
        model.train()
        train(model, seen_aux, seen_aux_mask, trainloader, optimizer, writer, epoch)
        if not cfg.models.freeze_image and epoch > cfg.models.freeze_image_after:
            print('Freezing image extractor')
            cfg.models.freeze_image = True

        # if epoch > cfg.models.train_image_after:
        #     print('Training image extractor')
        #     cfg.models.freeze_image = False
            
        if epoch % cfg.train.eval_every == 0:
            print(f'Evaluating {epoch} Epoch')
            model.eval()
            with torch.no_grad():
                evaluate(cfg, model, unseen_aux, unseen_aux_mask, unseen_mask, valloader_unseen, valloader_seen, writer, epoch, results_zsl, results_gzsl, results_c_zsl, results_c_gzsl)
                
            writer.flush()
            
            print(f'Experiment: {output_dir.split("/")[-1]}')    
    print(f'Max acc {max_cross_gzsl}')

def train(model, seen_aux, seen_aux_mask, trainloader, optimizer, writer, epoch):
    train_loss = 0.0
    preds, preds_cross, labels = [], [], []
    for img, label in tqdm(trainloader):
        img, label = img.to(device), label.to(device)
    
        loss, out, out_cross = model(img, seen_aux, seen_aux_mask, label)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out = torch.argmax(out,  1)
        if out_cross is not None:
            out_cross = torch.argmax(out_cross,  1)
            preds_cross.append(out_cross.cpu().numpy())
        preds.append(out.cpu().numpy())
        labels.append(label.cpu().numpy())
        # break
        
    train_loss = train_loss / len(trainloader)
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    if len(preds_cross) > 0:
        preds_cross = np.concatenate(preds_cross)
    
    top1_correct, top1_pcorrect = compute_per_class_acc(labels, preds, len(trainloader.dataset.class_names))
    
    print(f'Top 1 {top1_correct}')
    print(f'Top1 per class {top1_pcorrect}')
    
    writer.add_scalar('train/loss', loss, epoch)
    writer.add_scalar('train/top1', top1_correct, epoch)
    writer.add_scalar('train/top1p', top1_pcorrect, epoch)
    if len(preds_cross) > 0:
        top1_c_correct, top1_c_pcorrect = compute_per_class_acc(labels, preds_cross, len(trainloader.dataset.class_names))
        print(f'Top 1 cross {top1_c_correct}')
        print(f'Top1 cross per class {top1_c_pcorrect}')
        writer.add_scalar('train/top1_c', top1_c_correct, epoch)
        writer.add_scalar('train/top1p_c', top1_c_pcorrect, epoch)

def evaluate(cfg, model, unseen_aux, unseen_aux_mask, unseen_mask, valloader_unseen, valloader_seen, writer, epoch, results_zsl, results_gzsl, results_c_zsl, results_c_gzsl):
    global max_zsl, max_gzsl, max_cross_zsl, max_cross_gzsl
    
    def get_predictions(data_loader):
        unseen_loss = 0.0
        preds, preds_cross, labels = [], [], []
        for img, label in tqdm(data_loader):
            img, label = img.to(device), label.to(device)

            loss, out, out_cross = model(img, unseen_aux, unseen_aux_mask, label)
            unseen_loss += loss.item()

            # out = torch.argmax(out, dim = 1)
            preds.append(out.cpu())
            labels.append(label.cpu())
            if out_cross is not None:
                # out_cross = np.argmax(out_cross, axis = 1)
                preds_cross.append(out_cross.cpu())
        preds, labels = torch.cat(preds), torch.cat(labels)
        if len(preds_cross) > 0:
            preds_cross = torch.cat(preds_cross)
        else:
            preds_cross = None
        return preds, preds_cross, labels.numpy() 

        
    preds_unseen, preds_cross_unseen, labels_unseen = get_predictions(valloader_unseen)
    preds_seen, preds_cross_seen, labels_seen = get_predictions(valloader_seen) 
    
    # Unseen performance
    zsl_preds = preds_unseen.clone()
    zsl_preds[:, unseen_mask] += 1000
    
    zsl_preds = torch.argmax(zsl_preds, 1).numpy()
    # print(zsl_preds.shape, labels_unseen.shape)
    _, zsl_top1_correct = compute_per_class_acc(labels_unseen, zsl_preds, len(valloader_unseen.dataset.class_names))
    results_zsl.update(epoch, zsl_top1_correct)

    # Seen performance
    seen_preds = preds_seen.clone()
    seen_preds[:, unseen_mask] -= 1000
    seen_preds = torch.argmax(seen_preds, 1).numpy()
    _, seen_top1_correct = compute_per_class_acc(labels_seen, seen_preds, len(valloader_unseen.dataset.class_names))

    def calibrated_stacking(preds_seen, preds_unseen):
        preds_seen = F.softmax(preds_seen, 1)
        preds_unseen = F.softmax(preds_unseen, 1)
        best_hm, best_seen, best_unseen, best_bias = 0, 0, 0, 0
        for a in tqdm(range(0,100), desc = ' '):
            bias = a / 100.0
            preds_seen_biased = preds_seen.clone()
            preds_unseen_biased = preds_unseen.clone()
            preds_seen_biased[:, unseen_mask] += bias
            preds_unseen_biased[:, unseen_mask] += bias
            preds_seen_biased = torch.argmax(preds_seen_biased, 1).numpy()
            preds_unseen_biased = torch.argmax(preds_unseen_biased, 1).numpy()
            _, acc_seen = compute_per_class_acc(labels_seen, preds_seen_biased, len(valloader_unseen.dataset.class_names))
            _, acc_unseen = compute_per_class_acc(labels_unseen, preds_unseen_biased, len(valloader_unseen.dataset.class_names))
            if acc_seen > 0 and acc_unseen > 0:
                hm = hmean([acc_seen, acc_unseen])
            else:
                hm = 0
            if hm > best_hm:
                best_hm = hm
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_bias = bias
        return best_hm, best_seen, best_unseen, best_bias

    best_hm, best_seen, best_unseen, best_bias = calibrated_stacking(preds_seen, preds_unseen)
    results_gzsl.update_gzsl(epoch, best_unseen, best_seen, best_hm)
    # unseen_loss = unseen_loss / len(valloader)
    
    if zsl_top1_correct > max_zsl:
        max_zsl = zsl_top1_correct
        print(f'New best zsl found: {max_zsl}')
        save_checkpoint(model, 'bestzsl', cfg.log_dir)
    if best_hm > max_gzsl:
        max_gzsl = best_hm
        print(f'New best gzsl found: {max_gzsl}')
        save_checkpoint(model, 'bestgzsl', cfg.log_dir)
    # print(f'UnSeen loss {unseen_loss}')
    print(f'Epoch {epoch}')
    # print(f'ZSL performance: {zsl_top1_correct}')
    # print(f'GZSL performance u {best_unseen} s {best_seen} h {best_hm}')
    # writer.add_scalar('eval_unseen/loss', unseen_loss, epoch)
    writer.add_scalar('eval_unseen/zsl', results_zsl.best_acc, epoch)
    writer.add_scalar('eval_unseen/seen', seen_top1_correct, epoch)
    writer.add_scalar('eval_unseen/gzsl_unseen', results_gzsl.best_acc_U, epoch)
    writer.add_scalar('eval_unseen/gzsl_seen', results_gzsl.best_acc_S, epoch)
    writer.add_scalar('eval_unseen/gzsl_hm', results_gzsl.best_acc, epoch)
    print(f'Best Bias {best_bias}')
    print(f'Best ZSL {results_zsl.best_acc}')
    print(f'Best GZSL u {results_gzsl.best_acc_U} s {results_gzsl.best_acc_S} h {results_gzsl.best_acc}')
    if preds_cross_unseen is not None:
        # Unseen performance
        zsl_preds = preds_cross_unseen.clone()
        zsl_preds[:, unseen_mask] += 1000
        zsl_preds = torch.argmax(zsl_preds, 1).numpy()
        _, zsl_top1_correct = compute_per_class_acc(labels_unseen, zsl_preds, len(valloader_unseen.dataset.class_names))
        results_c_zsl.update(epoch, zsl_top1_correct)

        # Seen performance
        seen_preds = preds_cross_seen.clone()
        seen_preds[:, unseen_mask] -= 1000
        seen_preds = torch.argmax(seen_preds, 1).numpy()
        _, seen_top1_correct = compute_per_class_acc(labels_seen, seen_preds, len(valloader_unseen.dataset.class_names))

        best_hm, best_seen, best_unseen, best_bias = calibrated_stacking(preds_cross_seen, preds_cross_unseen)
        results_c_gzsl.update_gzsl(epoch, best_unseen, best_seen, best_hm)
        
        if zsl_top1_correct > max_cross_zsl:
            max_cross_zsl = zsl_top1_correct
            print(f'Cross New ZSL found: {max_cross_zsl}')
        if best_hm > max_cross_gzsl:
            max_cross_gzsl = best_hm
            print(f'Cross New best GZSL found: {max_cross_gzsl}')

        # print(f'ZSL performance: {zsl_top1_correct}')
        # print(f'GZSL performance u {best_unseen} s {best_seen} h {best_hm}')
        writer.add_scalar('eval_unseen/c_zsl', results_c_zsl.best_acc, epoch)
        writer.add_scalar('eval_unseen/c_seen', seen_top1_correct, epoch)
        writer.add_scalar('eval_unseen/c_gzsl_unseen', results_c_gzsl.best_acc_U, epoch)
        writer.add_scalar('eval_unseen/c_gzsl_seen', results_c_gzsl.best_acc_S, epoch)
        writer.add_scalar('eval_unseen/c_gzsl_hm', results_c_gzsl.best_acc, epoch)
        print(f'Best Bias {best_bias}')
        print(f'Cross Best ZSL {results_c_zsl.best_acc}')
        print(f'Cross Best GZSL u {results_c_gzsl.best_acc_U} s {results_c_gzsl.best_acc_S} h {results_c_gzsl.best_acc}')

try:
    main()
except KeyboardInterrupt:
    print(f'Best accuracy achieved {max_zsl}')
