import math
import argparse
import logging
import os
import re
import string
import sys
import pickle
from typing import Optional, Tuple
from tqdm import tqdm 
import numpy as np
import statistics

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from bert_score import score as BERTScore
from sacrebleu.metrics import BLEU

from transformers import pipeline
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.building_utils import build_model, deploy_model

from inputters import inputters
from confidence_intervals import get_bootstrap_indices, get_conf_int

sys.path.append("..")
sys.path.insert(0, '../PAGE')
from IntentClassifier_head.pplm_classification_head import ClassificationHead

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DISCRIMINATOR_MODELS_PARAMS = {
    "Strategy": {
        "path": "../StrategyClassifier_head/output_blenderbot_test/ESC_classifier_head_epoch_10.pt",
        "class_size": 8,
        "embed_size": 512,
        "pretrained_model": "../GenerationModel/DATA/strat_pp.strat/2024-04-21230454.3e-05.16.1gpu/epoch-2.bin",
        "class_vocab": {"Self-disclosure": 0, "Question": 1, "Restatement or Paraphrasing": 2,
                        "Affirmation and Reassurance": 3, "Reflection of feelings": 4,
                        "Information": 5, "Providing Suggestions": 6, "Others": 7}
    },
}

def encode_page(dialog, tokenizer, device):
    dialog_len = len(dialog)
    
    tk_id = []
    at_mk = []
    spk = []
    spk_flag = False
    for index, u in enumerate(dialog): 
        encoded_output = tokenizer(u)
        tid = encoded_output.input_ids
        atm = encoded_output.attention_mask
        tk_id.append(torch.tensor(tid, dtype=torch.long, device=device))
        at_mk.append(torch.tensor(atm, device=device))
        if spk_flag == False:
            s = 0
            spk.append(s)
            spk_flag = True
        else:
            s = 1
            spk.append(s)
            spk_flag = False
    
    spk = torch.tensor(spk)
    same_spk = spk.unsqueeze(1) == spk.unsqueeze(0)
    other_spk = same_spk.eq(False).long().tril(0)
    same_spk = same_spk.long().tril(0)
    msk = torch.ones_like(same_spk, dtype=torch.long).tril(0)
    
    adj = torch.stack([same_spk, other_spk], dim=0)
    tk_id = pad_sequence(tk_id, batch_first=True, padding_value=1)
    at_mk = pad_sequence(at_mk, batch_first=True, padding_value=0)
    
    return tk_id, at_mk, msk, adj, dialog_len

def predict_page(PAGE, histories, tokenizer_page, device):
    tk_id, at_mk, msk, adj, d_len = encode_page(histories, tokenizer_page, device)     
    msk = msk.unsqueeze(0).cuda()
    tk_id = tk_id.unsqueeze(0)
    at_mk = at_mk.unsqueeze(0)
    adj = msk.clone()
    adj = adj.long().cuda()
    label_ = 1 # burden parameter
    prediction = PAGE(tk_id, at_mk, msk, adj, label_) 
    
    return prediction


def get_user_emotional_state(usr_eds, speaker_roles):
   
    user_emotional_state = 'neutral'
    for i, usr_ed in enumerate(usr_eds):
        if usr_ed == None: # skip sys_ed
            continue
        for j ,(ed, role) in reversed(list(enumerate(zip(usr_ed, speaker_roles[0])))):
            if role == 'sys' and ed == 1: # user is in the inter-personal state
                user_emotional_state = 'inter-personal' # system should be in the self-contagion state
                return user_emotional_state
            
            elif role == 'usr' and ed == 1: # user is in the self-contagion state
                user_emotional_state = 'self-contagion' # system should be in the inter-personal state
                            
    return user_emotional_state

def get_system_emotional_state(sys_eds, speaker_roles):
    sys_emotional_state = 'neutral'
    for i, sys_ed in enumerate(sys_eds):
        if sys_ed == None: # skip sys_ed
            continue
        for j ,(ed, role) in reversed(list(enumerate(zip(sys_ed, speaker_roles[0])))):            
            if role == 'usr' and ed == 1: # user is in the self-contagion state
                sys_emotional_state = 'inter-personal' # system should be in the inter-personal state
                return sys_emotional_state
            elif role == 'sys' and ed == 1: # user is in the inter-personal state
                sys_emotional_state = 'self-contagion' # system should be in the self-contagion state
                            
    return sys_emotional_state

def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for r in res:
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16) # dict1 scores from each sentence.
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n 
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len

def get_classifier(
        name: Optional[str],
        device: str,
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)

    resolved_archive_file = params["path"]

    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    return classifier


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print('-------------')
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))    

def f1_score_ml(pred, label):
    acc = 0       
    for label_n in label:
        if label_n in pred:
            acc += 1

    return acc/len(label)

def classifying_emotion(dec_sentence, classifier, device):
    temp = None
    respon_list=[]
    respon_keys=[]
    
    respon_split = re.split(r'([.!?])', dec_sentence)

    for i, respon in enumerate(respon_split):
        respon = respon.strip()
        if respon in string.punctuation:
            try:
                temp = temp + respon
            except:
                continue
            respon_list.append(temp)
            temp = None
        elif respon == '':
            continue
        elif (i+1) == len(respon_split):
            respon_list.append(respon)
        else:
            temp = respon
    
    for respon in respon_list:  
        prediction = classifier(respon)
        for pred in prediction[0]:
            respon_keys.append(pred['label'])
            
    return set(respon_keys)


def classifying_strategy(dec_sentence, model, tokenizer, strategy_classifier, device):
    temp = None
    respon_list=[]
    respon_keys=[]
    
    bos = torch.tensor([tokenizer.bos_token_id], device=device, dtype=torch.long).unsqueeze(0)
    respon_split = re.split(r'([.!?])', dec_sentence)

    for i, respon in enumerate(respon_split):
        respon = respon.strip()
        if respon in string.punctuation:
            try:
                temp = temp + respon
            except:
                continue
            respon_list.append(temp)
            temp = None
        elif respon == '':
            continue
        elif (i+1) == len(respon_split):
            respon_list.append(respon)
        else:
            temp = respon
    
    for i, respon in enumerate(respon_list):
        pert_response = tokenizer.encode(respon)
        pert_response = torch.tensor(pert_response, device=device, dtype=torch.long).unsqueeze(0)
           
        encoder_outputs = model.model.encoder(
            input_ids=pert_response,
            attention_mask=torch.ones_like(pert_response),
            return_dict=True,
        ) 
        
        decoder_outputs = model.model.decoder(
            input_ids=torch.cat((bos,pert_response), 1),
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=torch.ones_like(pert_response),
            output_hidden_states=True,
            return_dict=True,
        )     
        response_hidden = torch.mean(decoder_outputs.last_hidden_state,dim=1)

        response_pred = strategy_classifier(response_hidden)    
        class_pred = torch.argmax(response_pred).item()
        
        strategydict = DISCRIMINATOR_MODELS_PARAMS['Strategy']['class_vocab']
        class_pred_key = list(strategydict.keys())[list(strategydict.values()).index(class_pred)]
        
        respon_keys.append(class_pred_key)
    
    return set(respon_keys)

def update_counts(target_dict, state):
    if state == 'neutral':
        target_dict['neutral'] += 1
    elif state == 'self-contagion':
        target_dict['self-contagion'] += 1
    elif state == 'inter-personal':
        target_dict['inter-personal'] += 1
        
def calculate_stage_acc(acc_dict, count_dict):
    avg_acc = {}
    for key in acc_dict:
        if count_dict[key] > 0:
            avg_acc[key] = round((acc_dict[key] / count_dict[key]) * 100, 2) # emo+str
        else:
            avg_acc[key] = 0  # Avoid division by zero
    return avg_acc
          
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_input_file', type=str, nargs='+', required=True)    
    parser.add_argument('-pred_file', type=str, required=True) # preficted result file
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_src_turn", type=int, default=None)
    parser.add_argument("--max_decoder_input_length", type=int, default=256)
    parser.add_argument("--max_knowledge_length", type=int, default=None)
    parser.add_argument('--label_num', type=int, default=None)
    parser.add_argument('--multi_knl', action='store_true', help='allow candidate knowledge items')
    parser.add_argument('--only_encode', action='store_true', help='only do encoding')
    parser.add_argument("--infer_batch_size", type=int, default=1)
    
    parser.add_argument("--num_bootstraps", type=int, default=1000)
    
    parser.add_argument("--for_test_run", action="store_true")
    parser.add_argument("--valid", action="store_true")
    
    args = parser.parse_args()

    # open a new file for bleu output result
    if args.valid:
        out_dir = 'Evaluation/valid/' + os.path.basename(args.pred_file)
    else:
        out_dir = 'Evaluation/' + os.path.basename(args.pred_file)
        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
     
    # set the device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
    
    logging.info('initializing cuda...')
    _ = torch.tensor([1.], device=device)
    
    # set logger
    logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    names = {
    'inputter_name':'strat_pp',
    'config_name':'strat',
    }
    
    # Set seeds
    set_seed(args.seed)
    
    # load pretrained model
    load_checkpoint = 'DATA/strat_pp.strat/2024-04-21230454.3e-05.16.1gpu/epoch-2.bin'
    logger.info('Loading checkpoint: {}\n'.format(load_checkpoint))
    tokenizer, model = build_model(checkpoint=load_checkpoint, **names)
    model = deploy_model(model, args)
    
    model.to(device)
    model.eval()
    
    strategy_classifier = get_classifier('Strategy', device)
    
    finetune_generation_model = 'SamLowe/roberta-base-go_emotions'
    emotion_classifier = pipeline("text-classification", model=finetune_generation_model, top_k=1)
    
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    
    inputter = inputters['strat_pp']()
    dataloader_kwargs = {
        'max_src_turn': args.max_src_turn,
        'max_input_length': args.max_input_length,
        'max_decoder_input_length': args.max_decoder_input_length,
        'max_knowledge_length': args.max_knowledge_length,
        'label_num': args.label_num,
        'multi_knl': args.multi_knl,
        'only_encode': args.only_encode,
        'infer_batch_size': args.infer_batch_size,
    }
    
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
    
    tokenizer.add_bos_token = True
    
    for infer_idx, infer_input_file in enumerate(args.infer_input_file):
        infer_dataloader = inputter.infer_dataloader(
            infer_input_file,
            tokenizer,
            **dataloader_kwargs
        )
    
    # load PAGE
    PAGE = torch.load('../PAGE/saves_best/model_0.pkl')
    PAGE.to(device)
    PAGE.eval()
    
    # Freeze PAGE
    for param in PAGE.parameters():
        param.requires_grad = False
    
    # load tokenizer for page
    tokenizer_page = AutoTokenizer.from_pretrained("roberta-base")
      
    # predict
    preds = []
    preds_line = []
    preds_line_tokenize = []
    with open(args.pred_file, 'r', encoding='utf-8') as fpred:
        for line in fpred:
            line = line.replace('[SEP]','').strip()
            line = line.replace('<|endoftext|>','').strip()
            # line = line.split(')',1)[-1]
            preds_line.append(line)
            
            # For distinct-n
            words = word_tokenize(line)
            preds_line_tokenize.append(words)
        preds.append(preds_line)
    
    count = 0
    total_strategy_acc = 0
    total_emotion_acc = 0
    turn_str_acc = []
    turn_emo_acc = []
    
    str_acc_self = 0
    str_acc_inter = 0
    str_acc_neutral = 0
    emo_acc_self = 0
    emo_acc_inter = 0
    emo_acc_neutral = 0
    
    self_count = 0
    inter_count = 0
    neutral_count = 0
    
    exp_acc = {'self-contagion':0, 'inter-personal':0, 'neutral':0}
    cmf_acc = {'self-contagion':0, 'inter-personal':0, 'neutral':0}
    sug_acc = {'self-contagion':0, 'inter-personal':0, 'neutral':0}
    
    exp_count = {'self-contagion':0, 'inter-personal':0, 'neutral':0}
    cmf_count = {'self-contagion':0, 'inter-personal':0, 'neutral':0}
    sug_count = {'self-contagion':0, 'inter-personal':0, 'neutral':0}
    
    exp_flag = False 
    cmf_flag = False
    sug_flag = False
    
    exp = ['Question', 'Restatement or Paraphrasing','Reflection of feelings', 'Self-disclosure']
    cmf = ['Affirmation and Reassurance', 'Reflection of feelings', 'Self-disclosure']
    sug = ['Providing Suggestions', 'Information', 'Affirmation and Reassurance', 'Self-disclosure']
    
    labels_line = []
    
    for batch, posts, references, sample_ids, histories, speaker_roles in tqdm(infer_dataloader, mininterval=2, desc='  - (Evaluation) -  ', leave=False):    
        response = preds_line[count]
        strategy = [tokenizer.decode(batch['strat_id'] + len(tokenizer) - 8).strip('[]')]
        context = posts[0]
        label = references[0]
        labels_line.append(label)
        
        prediction = predict_page(PAGE, histories, tokenizer_page, device)
        ed_prediction = torch.gt(prediction.data, 0.5).long()[0] # d_dim x d_dim, emotion evidence
        
        usr_eds = []
        for i, role in reversed(list(enumerate(speaker_roles[0]))):
            if role == 'usr':
                usr_eds.append(ed_prediction[i])
            elif role == 'sys' and usr_eds:
                break
            else:
                usr_eds.append(None)
                            
        user_emotional_state = get_user_emotional_state(usr_eds, speaker_roles)

        histories.append(response)
        speaker_roles[0].append('sys')

        sys_prediction = predict_page(PAGE, histories, tokenizer_page, device)
        sys_ed_prediction = torch.gt(sys_prediction.data, 0.5).long()[0] # d_dim x d_dim, emotion evidence
        
        sys_eds = []
        for i, role in reversed(list(enumerate(speaker_roles[0]))):
            if role == 'sys':
                sys_eds.append(sys_ed_prediction[i])
            elif role == 'usr' and sys_eds:
                break
            else:
                sys_eds.append(None)

        system_emotional_state = get_system_emotional_state(sys_eds, speaker_roles)
        
        count += 1
        if user_emotional_state == 'neutral':
            neutral_count += 1
        elif user_emotional_state == 'self-contagion':
            self_count += 1
        elif user_emotional_state == 'inter-personal':
            inter_count += 1
            
        for gold_k in strategy:
            if gold_k in exp:
                exp_flag = True
                update_counts(exp_count, system_emotional_state)
        
            if gold_k in cmf:
                cmf_flag = True
                update_counts(cmf_count, system_emotional_state)
        
            if gold_k in sug:
                sug_flag = True
                update_counts(sug_count, system_emotional_state)
           
        respon_str = classifying_strategy(response, model, tokenizer, strategy_classifier, device)
        
        str_flag = False
        for respon_k in respon_str:
            for gold_k in strategy:
                if respon_k == gold_k or gold_k == 'Others':
                    total_strategy_acc += 1
                    if user_emotional_state == 'self-contagion':
                        str_acc_self += 1
                    elif user_emotional_state == 'inter-personal':
                        str_acc_inter += 1
                    elif user_emotional_state == 'neutral':
                        str_acc_neutral += 1

                    if exp_flag:
                        update_counts(exp_acc, system_emotional_state)
                    if cmf_flag:
                        update_counts(cmf_acc, system_emotional_state)
                    if sug_flag:
                        update_counts(sug_acc, system_emotional_state)

                    str_flag = True
                    break
            else:
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break
        
        if str_flag:
            turn_str_acc.append(1)
        else:
            turn_str_acc.append(0)
            
        context_emotions = classifying_emotion(context, emotion_classifier, device)
        respon_emotions = classifying_emotion(response, emotion_classifier, device)
        gold_emotions = classifying_emotion(label, emotion_classifier, device)

        emo_flag = False
        for respon_e in respon_emotions:
            for gold_e in gold_emotions:
                if respon_e == gold_e:
                    total_emotion_acc += 1
                    if user_emotional_state == 'self-contagion':
                        emo_acc_self += 1
                    elif user_emotional_state == 'inter-personal':
                        emo_acc_inter += 1
                    elif user_emotional_state == 'neutral':
                        emo_acc_neutral += 1

                    # if exp_flag:
                    #     update_counts(exp_acc, system_emotional_state)
                    # if cmf_flag:
                    #     update_counts(cmf_acc, system_emotional_state)
                    # if sug_flag:
                    #     update_counts(sug_acc, system_emotional_state)

                    emo_flag = True
                    break
            else:
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break
        
        if emo_flag:
            turn_emo_acc.append(1)
        else:
            turn_emo_acc.append(0)
           
        exp_flag = False
        cmf_flag = False
        sug_flag = False    

        if count % 500 == 0 or count ==1 or args.for_test_run:
            print('\n-------------start-----')
            print('context:',context)
            print('user_emotional_state:',user_emotional_state)
            print()
            print('gold_res:',label)
            print('response:',response)
            print()
            print('context_emotions:',context_emotions)
            print('emotion_gold:{}, emotion_classified:{}'.format(gold_emotions, respon_emotions))
            print()
            # str_cls = list(str_dict.keys())[list(str_dict.values()).index(str_pred.item())]
            print('strategy_gold: {}, strategy_classified: {}'.format(strategy, respon_str))
            print()
            if args.for_test_run:
                breakpoint()
            
    avg_emotion_acc = round((total_emotion_acc/count)*100,2)
    print('avg emotion Accuracy: {}'.format(avg_emotion_acc))
    avg_strategy_acc = round((total_strategy_acc/count)*100,2)
    print('avg strategy Accuracy: {}'.format(avg_strategy_acc))
    
    #### Confidence intervals for evaluation in machine learning
    # Percentage for the confidence interval
    alpha = 5 
    metric_values_str = []
    metric_values_emo = []

    with open('Evaluation/DPPLM_eval_result.pkl', 'wb')as f:
        pickle.dump({'str':metric_values_str, 'emo':metric_values_emo}, f)
        
    # Number of bootstrap samples to use (the run time will be proportional to this number). We set it to
    # 50/alpha*100 to get enough samples in the tails.
    num_bootstraps = args.num_bootstraps
    num_samples = len(preds_line)
    
    for nb in tqdm(np.arange(num_bootstraps) , mininterval=2, desc='  - (Bootstraps Evaluation) -  ', leave=False):
        indices = get_bootstrap_indices(num_samples, conditions=None, random_state=nb)
        
        emo_boot = []
        str_boot = []

        for i in indices:
            emo_boot.append(turn_emo_acc[i])
            str_boot.append(turn_str_acc[i])
                
        strategy_acc = sum(str_boot)/len(str_boot)
        emotion_acc = sum(emo_boot)/len(emo_boot)
        
        metric_values_str.append(round(strategy_acc*100, 2))
        metric_values_emo.append(round(emotion_acc*100, 2))
        
    boots_str = get_conf_int(metric_values_str, alpha)
    boots_emo = get_conf_int(metric_values_emo, alpha)
    print('---bootstrap---')
    print(boots_emo)
    print(boots_str)
    print('---stdev---')
    print(statistics.stdev(metric_values_emo))
    print(statistics.stdev(metric_values_str))
    print('---Emotional Dynamics---')
    print('self_count: ',self_count)
    print('inter_count: ',inter_count)
    print('neutral_count: ',neutral_count)
    print()
    avg_emo_acc_self = round((emo_acc_self/self_count)*100,2)
    avg_emo_acc_inter = round((emo_acc_inter/inter_count)*100,2)
    avg_emo_acc_neutral = round((emo_acc_neutral/neutral_count)*100,2)
    print('emotion accuracy')
    print('[self-contagion]: {}'.format(avg_emo_acc_self))
    print('[inter-personal]: {}'.format(avg_emo_acc_inter))
    print('[neutral]: {}'.format(avg_emo_acc_neutral))
    print()
    avg_str_acc_self = round((str_acc_self/self_count)*100,2)
    avg_str_acc_inter = round((str_acc_inter/inter_count)*100,2)
    avg_str_acc_neutral = round((str_acc_neutral/neutral_count)*100,2)
    print('strategy accuracy')
    print('[self-contagion]: {}'.format(avg_str_acc_self))
    print('[inter-personal]: {}'.format(avg_str_acc_inter))
    print('[neutral]: {}'.format(avg_str_acc_neutral)) 
    print('========System==============') 
    
    avg_exp_acc = calculate_stage_acc(exp_acc, exp_count)
    avg_cmf_acc = calculate_stage_acc(cmf_acc, cmf_count)
    avg_sug_acc = calculate_stage_acc(sug_acc, sug_count)

    print('Exploration performance: {}'.format(avg_exp_acc))
    print('Comforting performance: {}'.format(avg_cmf_acc))
    print('Suggestion performance: {}'.format(avg_sug_acc))
    print()
    print('Explration count: {}'.format(exp_count))
    print('Comforting count: {}'.format(cmf_count))
    print('Suggestion count: {}'.format(sug_count))
    print('======================') 
    # calculate SacreBleu score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(preds_line, [labels_line])
    print('SacreBleu:', bleu_score)
    
    print(len(preds_line))
    print(len(labels_line))
    # bertS = BERTScore()     
    P, R, F1 = BERTScore(preds_line, labels_line, lang='en')
    print('Bert_score:', torch.mean(F1))
   
    # calculate distinct-n score
    dist1_avg, dist2_avg, dist1_avg_all, dist2_avg_all, avg_len = get_dist(preds_line_tokenize)

    print('Distinct_1: ', dist1_avg_all * 100)
    print('Distinct_2: ', dist2_avg_all * 100)
    
    # write into files
    with open('Evaluation/{}'.format(args.pred_file), 'w+', encoding='utf-8') as f:
        f.write('output data: {}\n'.format(str(args.pred_file)))
        f.write('SacreBleu: {}\n'.format(bleu_score))
        f.write('Bert_score: {}\n'.format(torch.mean(F1)))
        f.write('Distinct_1: {}\n'.format(dist1_avg_all * 100)) 
        f.write('Distinct_2: {}\n'.format(dist2_avg_all * 100))
        f.write('emotion accuracy: {}\n'.format(avg_emotion_acc))
        f.write('strategy accuracy: {}\n'.format(avg_strategy_acc))
        f.write('=========Bootstraps Evaluation========\n')
        f.write('emotion: {}\n'.format(boots_emo))
        f.write('strategy: {}\n'.format(boots_str))
        f.write('=========Standard Deviation========\n')
        f.write('emotion: {}\n'.format(statistics.stdev(metric_values_emo)))
        f.write('strategy: {}\n'.format(statistics.stdev(metric_values_str)))  
        f.write('=========Emotional Dynamics========\n')
        f.write('emotion accuracy\n')
        f.write('[self-contagion]: {}\n'.format(avg_emo_acc_self))
        f.write('[inter-personal]: {}\n'.format(avg_emo_acc_inter))
        f.write('[neutral]: {}\n'.format(avg_emo_acc_neutral))
        f.write('strategy accuracy\n')
        f.write('[self-contagion]: {}\n'.format(avg_str_acc_self))
        f.write('[inter-personal]: {}\n'.format(avg_str_acc_inter))
        f.write('[neutral]: {}\n'.format(avg_str_acc_neutral))
        f.write('=========Stage Performance========\n')
        f.write('Exploration: {}\n'.format(avg_exp_acc))
        f.write('Comforting: {}\n'.format(avg_cmf_acc))
        f.write('Action: {}\n'.format(avg_sug_acc))
        f.write('=========Stage count========\n')
        f.write('Exploration count: {}\n'.format(exp_count))
        f.write('Comforting count: {}\n'.format(cmf_count))
        f.write('Suggestion count: {}\n'.format(sug_count))

    
if __name__ == '__main__':
    main()