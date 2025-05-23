import random
import numpy as np
import os

import torch
import pickle
import argparse
from page import PAGE
from utils import MaskedBCELoss2
from dataset import get_dataloaders
from sklearn.metrics import f1_score, classification_report
from transformers import AdamW, get_constant_schedule, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")


def train(model, model_path, train_loader, dev_loader, dd_loader, ie_loader,
          loss_fn, optimizer, n_epochs, log, accumulate_step, scheduler):
    model.train()
    best_f1_score_ece = 0.
    best_f1_score_ece_dd = 0.
    best_f1_score_ece_ie = 0.

    best_report_ece = None
    best_report_ece_dd = None
    best_report_ece_ie = None

    step = 0

    for e in range(n_epochs):
        ece_prediction_list = []
        ece_prediction_mask = []
        ece_label_list = []
        ece_loss = 0.
        ece_total_sample = 0

        for data in train_loader:
      
            input_ids, attention_mask, clen, mask, adj_index, label, ece_pair, _ = data
            
            
            mask = mask.cuda()
            label = label.cuda()
            ece_pair = ece_pair.cuda()

            input_ids = [t.cuda() for t in input_ids]
            attention_mask = [t.cuda() for t in attention_mask]
  
            adj = mask.clone()
            adj = adj.long().cuda()
            rel_adj = mask.clone()
            rel_adj = rel_adj.long().cuda()
            prediction = model(input_ids, attention_mask, mask, adj, label+1)
            
            ece_label_list.append(torch.flatten(ece_pair.data).cpu().numpy())
            ece_prediction_mask.append(torch.flatten(mask.data).cpu().numpy())
            ece_samples = mask.data.sum().item()
            ece_total_sample += ece_samples
            # print('prediction:',prediction)
            # print('ece_pair:',ece_pair)
            # breakpoint()
            loss = loss_fn(ece_pair, prediction, mask)
            ece_loss = ece_loss + loss.item() * ece_samples

            ece_prediction = torch.gt(prediction.data, 0.5).long()
            ece_prediction_list.append(torch.flatten(ece_prediction.data).cpu().numpy())

            if accumulate_step > 1:
                loss = loss / accumulate_step
            model.zero_grad()
            loss.backward()
            if (step + 1) % accumulate_step == 0:
                optimizer.step()
                scheduler.step()
            step += 1

        ece_prediction_mask = np.concatenate(ece_prediction_mask)
        ece_label_list = np.concatenate(ece_label_list)

        ece_loss = ece_loss / ece_total_sample
        ece_prediction = np.concatenate(ece_prediction_list)

        fscore_ece = f1_score(ece_label_list, ece_prediction,
                              average='macro', sample_weight=ece_prediction_mask)
        log_line = f'[Train] Epoch {e+1}: ECE loss: {round(ece_loss, 6)}, Fscore: {round(fscore_ece, 4)}'
        print(log_line)
        log.write(log_line + '\n')

        dev_fscores, dev_reports = valid('DEV', model, dev_loader, log)
        dd_fscores, dd_reports = valid('DD_test', model, dd_loader, log)
        ie_fscores, ie_reports = valid('IE_test', model, ie_loader, log)


        ece_final_fscore_dev = dev_fscores[0]
        ece_final_fscore_dd = dd_fscores[0]
        ece_final_fscore_ie = ie_fscores[0]
        if best_f1_score_ece < ece_final_fscore_dev:
            best_f1_score_ece = ece_final_fscore_dev
            best_f1_score_ece_dd = ece_final_fscore_dd
            best_f1_score_ece_ie = ece_final_fscore_ie
            
            best_report_ece = dev_reports
            best_report_ece_dd = dd_reports
            best_report_ece_ie = ie_reports
            
            torch.save(model, model_path)
            # torch.save(model.state_dict(), model_path)


    log_line = f'[FINAL--DEV]: best_Fscore: {round(best_f1_score_ece, 4)}'
    print(log_line)
    log.write('\n\n' + log_line + '\n\n')
    log.write(best_report_ece + '\n')

    log_line = f'[FINAL--DD_test]: best_Fscore: {round(best_f1_score_ece_dd, 4)}'
    print(log_line)
    log.write('\n\n' + log_line + '\n\n')
    log.write(best_report_ece_dd + '\n')
    
    log_line = f'[FINAL--IE_test]: best_Fscore: {round(best_f1_score_ece_ie, 4)}'
    print(log_line)
    log.write('\n\n' + log_line + '\n\n')
    log.write(best_report_ece_ie + '\n')
    log.close()

    return best_f1_score_ece, best_f1_score_ece_dd,best_f1_score_ece_ie


def valid(valid_type, model, data_loader, log):
    model.eval()
    ece_prediction_list = []
    ece_prediction_mask = []
    ece_label_list = []
    ece_total_sample = 0

    with torch.no_grad():
        for data in data_loader:

            input_ids, attention_mask, clen, mask, adj_index, label, ece_pair, _  = data

            attention_mask = [t.cuda() for t in attention_mask]
            mask = mask.cuda()
            label = label.cuda()
            ece_pair = ece_pair.cuda()
            input_ids = [t.cuda() for t in input_ids]
         
            adj = mask.clone()
            adj = adj.long().cuda()
            rel_adj = mask.clone()
            rel_adj = rel_adj.long().cuda()
            
            prediction = model(input_ids, attention_mask, mask, adj,label+1)

            ece_prediction = torch.flatten(torch.gt(prediction.data, 0.5).long()).cpu().numpy()
            ece_prediction_list.append(ece_prediction)

            ece_label_list.append(torch.flatten(ece_pair.data).cpu().numpy())
            ece_prediction_mask.append(torch.flatten(mask.data).cpu().numpy())
            ece_samples = mask.data.sum().item()
            ece_total_sample += ece_samples

    ece_prediction_mask = np.concatenate(ece_prediction_mask)
    ece_label_list = np.concatenate(ece_label_list)
    ece_prediction = np.concatenate(ece_prediction_list)
    fscore_ece = f1_score(ece_label_list,
                          ece_prediction,
                          average='macro',
                          sample_weight=ece_prediction_mask)

    log_line = f'[{valid_type}]: Fscore: {round(fscore_ece, 4)}'
    
    reports = classification_report(ece_label_list,
                                    ece_prediction,
                                    target_names=['neg', 'pos'],
                                    sample_weight=ece_prediction_mask,
                                    digits=4)
    print(log_line)
    
    log.write(log_line + '\n')
    fscores = [fscore_ece]
    model.train()

    return fscores, reports


def main(args, seed=0, index=0):
    
    print(seed)
    print(args)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_path = args.save_dir

    batch_size = args.batch_size
    lr = args.lr
    model_name = 'model_' + str(index) + '.pkl'
    model_name = os.path.join(model_path, model_name)
    log_name = 'log_' + str(index) + '.txt'
    log_path = os.path.join(model_path, log_name)

    log = open(log_path, 'w')
    log.write(str(args) + '\n\n')
    seed_num = 'seed = ' + str(seed)
    log.write(seed_num)
    model_size = args.model_size
    valid_shuffle = args.valid_shuffle
    window = args.window
    max_len = args.max_len
    posi_dim = args.posi_dim

    train_loader, dev_loader, dd_loader, ie_loader = get_dataloaders(model_size, batch_size, valid_shuffle)
    n_epochs = args.n_epochs
    weight_decay = args.weight_decay
    utter_dim = args.utter_dim
    accumulate_step = args.accumulate_step
    scheduler_type = args.scheduler
    emotion_dim = args.emotion_dim
    

    nhead = args.nhead
    ff_dim = args.ff_dim
    att_dropout = args.att_dropout
    mlp_dropout = args.mlp_dropout
    pag_dropout = args.pag_dropout
    num_bases = args.num_bases
    
    emo_emb = pickle.load(open('emotion_embeddings.pkl', 'rb'), encoding='latin1')
    model = PAGE(utter_dim, emo_emb,emotion_dim,att_dropout,mlp_dropout,pag_dropout,ff_dim,nhead,window,num_bases,max_len,posi_dim)

    loss_fn = MaskedBCELoss2()
      
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},  
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} 
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    if scheduler_type == 'linear':
        num_conversations = len(train_loader.dataset)
        if (num_conversations * n_epochs) % (batch_size * accumulate_step) == 0:
            num_training_steps = (num_conversations * n_epochs) / (batch_size * accumulate_step)
        else:
            num_training_steps = (num_conversations * n_epochs) // (batch_size * accumulate_step) + 1
        num_warmup_steps = int(num_training_steps * args.warmup_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        scheduler = get_constant_schedule(optimizer)
    model = model.cuda()
    
    dev_fscore, dd_fscore ,ie_fscore= train(model, model_name, train_loader, dev_loader, dd_loader, ie_loader,
                                    loss_fn, optimizer, n_epochs, log, accumulate_step, scheduler)
    return dev_fscore, dd_fscore, ie_fscore,model_path

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', type=str, required=False, default='0')  
    parser.add_argument('--n_epochs', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--accumulate_step', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=3e-5)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-3)
    parser.add_argument('--model_size', type=str, required=False, default='base')
    parser.add_argument('--valid_shuffle', action='store_false', help='whether to shuffle dev and test sets')
    parser.add_argument('--scheduler', type=str, required=False, default='constant')
    parser.add_argument('--warmup_rate', type=float, required=False, default=0.1)
    parser.add_argument('--emotion_dim', type=int, required=False, default=200)
    parser.add_argument('--window', type=int, required=False, default=2)
    parser.add_argument('--max_len', type=int, required=False, default=10)
    parser.add_argument('--posi_dim', type=int, required=False, default=100)
    parser.add_argument('--pag_dropout', required=False, type=float, default=0.1)
    parser.add_argument('--att_dropout', required=False, type=float, default=0.1)
    parser.add_argument('--mlp_dropout', type=float, required=False, default=0.1)
    parser.add_argument('--nhead', required=False, type=int, default=6)
    parser.add_argument('--ff_dim', required=False, type=int, default=128)
    parser.add_argument('--utter_dim', type=int, required=False, default=300)
    parser.add_argument('--num_bases', type=int, required=False, default=2)
    parser.add_argument('--seed', nargs='+', type=int, required=False, default=[0])
    parser.add_argument('--index', nargs='+', type=int, required=False, default=[0])
    parser.add_argument('--save_dir', type=str, required=False, default='saves')


    args_for_main = parser.parse_args()
    seed_list = args_for_main.seed
    index_list = args_for_main.index
    dev_fscore_list = []
    dd_fscore_list = []
    ie_fscore_list = []
    
    model_dir = ''
    for sd, idx in zip(seed_list, index_list):
        dev_f1, dd_f1, ie_f1, model_dir = main(args_for_main, sd, idx)
        dev_fscore_list.append(dev_f1)
        dd_fscore_list.append(dd_f1)
        ie_fscore_list.append(ie_f1)

    dev_fscore_mean = np.round(np.mean(dev_fscore_list) * 100, 2)
    dev_fscore_std = np.round(np.std(dev_fscore_list) * 100, 2)

    dd_fscore_mean = np.round(np.mean(dd_fscore_list) * 100, 2)
    dd_fscore_std = np.round(np.std(dd_fscore_list) * 100, 2)

    ie_fscore_mean = np.round(np.mean(ie_fscore_list) * 100, 2)
    ie_fscore_std = np.round(np.std(ie_fscore_list) * 100, 2)

    logs_path = model_dir + '/log_metrics_' + str(index_list[0]) + '-' + str(index_list[-1]) + '.txt'
    logs = open(logs_path, 'w')

    logs.write(str(args_for_main) + '\n\n')

    log_lines = f'DEV fscore: {dev_fscore_mean}(+-{dev_fscore_std})'
    print(log_lines)
    logs.write(log_lines + '\n')
    log_lines = f'DD fscore: {dd_fscore_mean}(+-{dd_fscore_std})'
    print(log_lines)
    log_lines = f'IE fscore: {ie_fscore_mean}(+-{ie_fscore_std})'
    print(log_lines)
    logs.write(log_lines + '\n')
    logs.close()
