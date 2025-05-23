#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import logging
import argparse
import os
import sys
from typing import Optional, Tuple
import re
import string
import numpy as np
from difflib import SequenceMatcher
from tqdm import trange

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from utils.building_utils import boolean_string, build_model, deploy_model
from inputters import inputters
from inputters.inputter_utils import _norm

from transformers import logging as tf_logging
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.generation.utils import top_k_top_p_filtering

sys.path.append("..") 
sys.path.insert(0, '../PAGE')

from StrategyClassifier_head.pplm_classification_head import ClassificationHead

tf_logging.set_verbosity_error()

### about loss record
file_info = None
strategy_loss_record_list=[]
eng_loss_record_list=[]
emo_loss_record_list=[]
kl_loss_record_list=[]
total_loss_record_list=[]
iteration_num_record_list=[]

strat_pred_acc = 0

TYPE_ENGAGEMENT = 1
TYPE_strategy = 2
TYPE_EMOTION = 3
PPLM_ALL = 4
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

DISCRIMINATOR_MODELS_PARAMS = {
    "Engagement": { # Not used
        "path": "../EngagementClassifier_head/output_master/NSP_classifier_head_epoch_5.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"0": 0, "1": 1},
        "default_class": 0,
        "pretrained_model": "../DialoGPT/model-medium", 
    },
    "Strategy": {
        "path": "../StrategyClassifier_head/output_blenderbot_test/ESC_classifier_head_epoch_10.pt",
        "class_size": 8,
        "embed_size": 512,
        "pretrained_model": "../GenerationModel/DATA/strat_pp.strat/2024-04-21230454.3e-05.16.1gpu/epoch-2.bin",
        "class_vocab": {"Self-disclosure": 0, "Question": 1, "Restatement or Paraphrasing": 2,
                        "Affirmation and Reassurance": 3, "Reflection of feelings": 4,
                        "Information": 5, "Providing Suggestions": 6, "Others": 7}
    },
    "Emotion_GO": {
        "path": "../EmotionClassifier_head/output_blenderbot_test/GO_classifier_head_epoch_10.pt",
        "class_size": 28,
        "embed_size": 512,
        "pretrained_model": "../GenerationModel/DATA/strat_pp.strat/2024-04-21230454.3e-05.16.1gpu/epoch-2.bin",
        "class_vocab": {"surprise": 0, "love": 1, "grief": 2, "fear": 3, "optimism": 4,
                        "pride": 5, "disappointment": 6, "anger": 7, "nervousness": 8, "disgust": 9,
                        "excitement": 10, "remorse": 11, "embarrassment": 12, "disapproval": 13, "amusement": 14,
                        "caring": 15, "sadness": 16, "admiration": 17, "annoyance": 18, "approval": 19,
                        "curiosity": 20, "desire": 21, "relief": 22, "confusion": 23, "gratitude": 24,
                        "joy": 25, "neutral": 26, "realization": 27}
    }
}

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

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

def get_user_emotional_state(usr_eds, speaker_roles, emo_weight, str_weight, prediction): #TODO
    
    emo_w = 0
    str_w = 0
    user_emotional_state = 'neutral'
    for i, usr_ed in enumerate(usr_eds):
        if usr_ed == None: # skip sys_ed
            continue
        
        self_p = 0
        inter_p = 0
        for j, role in enumerate(speaker_roles[0]):
            role_index = i + 1
            p = round(prediction[0][-role_index][j].item(), 2)
            if role == 'usr':
                self_p = max(self_p, p)
            elif role == 'sys':
                inter_p = max(inter_p, p)

        if inter_p >= 0.5: # user is in the inter-personal state
            user_emotional_state = 'inter-personal' # system should be in the self-contagion state
            
            if self_p >= 0.5: #hybrid
                emo_weight = 0.5 # 
                str_weight = 1 # 
                
                emo_w = emo_weight * self_p
                str_w = str_weight * inter_p
            else:
                emo_weight = 0.5 # 
                str_weight = 1.0 # 
                
                emo_w = emo_weight * inter_p
                str_w = str_weight * inter_p
                            
            return user_emotional_state, emo_w, str_w
        
        elif self_p >= 0.5: # user is in the self-contagion state
            user_emotional_state = 'self-contagion' # system should be in the inter-personal state
            emo_weight = 0.5 # 
            str_weight = 0
            
            emo_w = emo_weight * self_p
            str_w = str_weight * self_p
        else:
            emo_w = self_p
            str_w = inter_p
                                                
    return user_emotional_state, emo_w, str_w
     
def RSA_inference(log_score, worldpriors, bad_words_mask, top_k, top_p):
    beta = 0.9
    alpha = 6
    
    worldprior_t = worldpriors.repeat(1,log_score.size(1),1).transpose(dim0=1, dim1=2).contiguous()

    # S_0 for L_1
    _literal_speaker = log_score.clone() # (1, perturb_num, vocab)
    _literal_speaker, _literal_s_next_token_idxs = torch.max(_literal_speaker, dim=-1, keepdim=True)
    
    # S_0 for the actual given persona (bsz, vocab)
    speaker_prior = log_score.select(1, 0)  # target persona is always index 0

    # S_0 for L_0
    # (bsz, vocab, world_cardinality)
    log_score = log_score.transpose(dim0=1, dim1=2).contiguous()
    log_score = log_score * beta
                
    # L_0 \propto S_0 * p(i)
    # worldprior should be broadcasted to all the tokens
    # (bsz, vocab, world_cardinality)
    listener_posterior = (log_score + worldprior_t) - torch.logsumexp(log_score + worldprior_t, 2, keepdim=True)

    # (bsz, vocab)
    listener_score = listener_posterior.select(2, 0)  # target persona is always index 0
    listener_score = listener_score * alpha

    speaker_posterior = (listener_score + speaker_prior) - torch.logsumexp(listener_score + speaker_prior, 1, keepdim=True)
    
    pert_logits = speaker_posterior
    
    pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
    pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
    rsa_probs = F.softmax(pert_logits, dim=-1)
    
    worldpriors = listener_posterior[:,:,0]
    
    return rsa_probs, worldpriors
    
def classifying_emotion(dec_sentence, model, tokenizer, emotion_classifier, device):
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
            input_ids=bos,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=torch.ones_like(pert_response),
            output_hidden_states=True,
            return_dict=True,
        )
        response_hidden = torch.mean(decoder_outputs.last_hidden_state,dim=1)

        response_pred = emotion_classifier(response_hidden)    
        class_pred = torch.argmax(response_pred).item() 
        
        emotiondict = DISCRIMINATOR_MODELS_PARAMS['Emotion_GO']['class_vocab']
        class_pred_key = list(emotiondict.keys())[list(emotiondict.values()).index(class_pred)]
        
        respon_keys.append(class_pred_key)
    
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

def preprocess_detect(inputs_id, device):
    segment_ids = torch.tensor([[0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    input_mask = torch.tensor([[1 if word_id==1 else 0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    return segment_ids, input_mask

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def _initialize_worldpriors_unigram(pertrub_num):
    """
    initialize the world prior with a uniform distribution
    """
    torch_dtype=torch.float
    
    ones = torch.ones(1, pertrub_num, dtype=torch_dtype, requires_grad=False).cuda()
    uniform_world_prior = torch.log(ones / pertrub_num)
    world_priors = uniform_world_prior.detach()

    return world_priors

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

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

def add_func(L1, L2):
    return [
            [m1 + m2 for (m1, m2) in zip(l1, l2)]
            for (l1, l2) in zip(L1, L2)
    ]

def perturb_hidden(
        past,
        model,
        last,
        decoder_input_ids,
        encoder_outputs=None,
        encoder_attention_mask=None,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        strategy_prediction_model=None,
        target_strategy=None,
        strategy_classifier=None,
        nsp_classifier=None,
        emo_classifier=None,
        user_emotional_state=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        sample=True,
        gamma=1.5,
        emo_w=1,
        str_w=1,
        device='cuda',
        verbosity_level=REGULAR,
        tokenizer=None,
        strategy_tokenizer=None,
        context_t=None,
        last_response=None
):
    # Generate inital perturbed past
    grad_accumulator = [
        [
        (np.zeros(p.shape).astype("float32"))        
        for p in p_layer
        ]
        for p_layer in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, _, curr_length, _ = past[0][0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_strategy_shape = (
                tuple(past[0][0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0][0].shape[-1:])
        )

        zeros_key_strategy_shape = (
                tuple(past[0][0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0][0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_strategy_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 3, 2)
        ones_mask = ones_mask.permute(0, 1, 3, 2)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_strategy_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0][0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    
    log_probs_record = []

    eng_loss_record = 0.0
    strategy_loss_record = 0.0
    emo_loss_record = 0.0
    kl_loss_record = 0.0
    total_loss_record = 0.0
    
    iteration_stop = False
    
    for i in range(num_iterations):
        if iteration_stop:
            break
        
        if verbosity_level >= VERBOSE:
            print("\nIteration ", i + 1)
        
        curr_perturbation = [
                              [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in p_layer
            ]
            for p_layer in grad_accumulator
        ]
        
        # Compute hidden using perturbed past
        perturbed_past = add_func(past, curr_perturbation)   
        _, _, curr_length, _ = past[0][0].shape
        
        output = model.model.decoder(input_ids=last,
                                        encoder_hidden_states=encoder_outputs[0],
                                        encoder_attention_mask=encoder_attention_mask,
                                        past_key_values=perturbed_past,
                                        output_hidden_states=True,
                                        return_dict=True)
        all_logits = model.lm_head(output.last_hidden_state) + model.final_logits_bias

        hidden = output.last_hidden_state     
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach() 
        
        logits = all_logits[:, -1, :] #[1,1,50257]
        probs = F.softmax(logits, dim=-1)

        if sample:
            next_token = torch.multinomial(probs.detach(), num_samples=1)
        else:
            _, next_token = torch.topk(probs, k=1, dim=-1)
            
        respon = torch.cat((last_response, next_token), dim=1)     
        if verbosity_level >= VERBOSE:
            print('respon(unperterbed):', tokenizer.decode(respon[0])) 

        loss = 0.0
        loss_list = [] 
        
        ce_loss = torch.nn.CrossEntropyLoss()
        bce_loss = torch.nn.BCEWithLogitsLoss()
        mse_loss = torch.nn.MSELoss()
        
        ### ===engaging attribute model start=== 
        eng_loss = 0
        if loss_type == TYPE_ENGAGEMENT: # TODO, not included for ALL fow now.
            
            #system
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                                                        past_key_values=curr_unpert_past,
                                                        inputs_embeds=inputs_embeds,
                                                        output_hidden_states=True,
                                                        return_dict=False
                )
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)

            prediction = nsp_classifier(new_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))
            
            # user
            class_label = 0 # next sentence prediction postive target 
            label = torch.tensor(prediction.shape[0] * [class_label],
                      device=device,
                      dtype=torch.long)
            
            eng_loss = ce_loss(prediction, label)

            # # # weight eng_loss
            # eng_loss = torch.mul(eng_loss, 2, out=None)
            
            if verbosity_level >= VERY_VERBOSE:
                print('--------')
                print('class_pred:{}'.format(torch.argmax(prediction).item()))
                print(" pplm_eng_loss:", eng_loss.data.cpu().numpy())
                eng_loss_record += np.round(eng_loss.data.cpu().numpy(),3)          
                loss_list.append(eng_loss)
        ### ===engaging attribute model end=== 
        
        ### ===strategy attribute model start================================== 
        strategy_loss = 0
        if (loss_type == TYPE_strategy or loss_type == PPLM_ALL) and target_strategy and str_w != 0:      

            strdict = DISCRIMINATOR_MODELS_PARAMS['Strategy']['class_vocab']
            label_vector = [0.0] * len(strdict)
            for i, label_m in enumerate(strdict):
                for t_strat in target_strategy:
                    if SequenceMatcher(None, label_m, t_strat).ratio() > 0.7 :
                        label_vector[i] = 1.0
            
            # strategy classification
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings(len(tokenizer))
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                output = model.model.decoder(inputs_embeds=inputs_embeds,
                                            encoder_hidden_states=encoder_outputs[0],
                                            encoder_attention_mask=encoder_attention_mask,
                                            past_key_values=curr_unpert_past,
                                            output_hidden_states=True,
                                            return_dict=True)
                
                curr_hidden = output.last_hidden_state
                predict_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)

            classification = strategy_classifier(predict_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))
            
            label = torch.tensor([label_vector], device=device, dtype=torch.float)
            strategy_loss = bce_loss(classification, label)

            # # # weight strategy 
            strategy_loss = torch.mul(strategy_loss, str_w, out=None)
            
            if verbosity_level >= VERY_VERBOSE: 
                print('\n---strategy attribute model info-----')
                intdict = DISCRIMINATOR_MODELS_PARAMS['Strategy']['class_vocab']
                strategy_cls = list(intdict.keys())[list(intdict.values()).index(torch.argmax(classification).item())]
                print('str_w:', str_w)
                print('class_target:{},\n class_pred:{}\n'.format(target_strategy, strategy_cls))
                print("pplm_strategy_loss:", strategy_loss.data.cpu().numpy())
                strategy_loss_record += np.round(strategy_loss.data.cpu().numpy(),3)  
                loss_list.append(strategy_loss)
                # breakpoint()
                        
        ### ===emotion attribute model start===================================
        emo_loss = 0
        if (loss_type == TYPE_EMOTION or loss_type == PPLM_ALL) and emo_w != 0:
            #system
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings(len(tokenizer))
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                output = model.model.decoder(inputs_embeds=inputs_embeds,
                                            encoder_hidden_states=encoder_outputs[0],
                                            encoder_attention_mask=encoder_attention_mask,
                                            past_key_values=curr_unpert_past,
                                            output_hidden_states=True,
                                            return_dict=True)
                
                curr_hidden = output.last_hidden_state
                predict_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)
                
            prediction = emo_classifier(predict_accumulated_hidden /
                                        (curr_length + 1 + horizon_length))
            
            # emotion soft-target            
            output = model.model.decoder(input_ids=context_t,
                                        encoder_hidden_states=encoder_outputs[0],
                                        encoder_attention_mask=encoder_attention_mask,
                                        output_hidden_states=True,
                                        return_dict=True)
            context_hidden = torch.mean(output.last_hidden_state,dim=1)
            
            context_pred = emo_classifier(context_hidden)
            emo_loss = mse_loss(prediction, context_pred)
             
            # # # weight emo_loss
            emo_loss = torch.mul(emo_loss, emo_w, out=None)
            
            if verbosity_level >= VERY_VERBOSE:
                print('\n---emotion attribute model info-----')
                print('user_self-contagion:{}'.format(user_emotional_state))
                print('emo_w:{}'.format(emo_w))
                emodict = DISCRIMINATOR_MODELS_PARAMS['Emotion_GO']['class_vocab']
                emo_target = list(emodict.keys())[list(emodict.values()).index(torch.argmax(context_pred).item())]
                emo_cls = list(emodict.keys())[list(emodict.values()).index(torch.argmax(prediction).item())]  
                print('class_target:{}, class_pred:{}\n'.format(emo_target, emo_cls))
                print("pplm_emo_loss:", emo_loss.data.cpu().numpy())
                emo_loss_record += np.round(emo_loss.data.cpu().numpy(),3)          
                loss_list.append(emo_loss)
            
        ### === calculating Kullback–Leibler Divergence loss ==================
        kl_loss = 0.0
         
        KLD = nn.KLDivLoss(reduction="batchmean")
        log_output = F.log_softmax(logits, dim=-1) #[1,50257]
        
        if not iteration_stop:
            log_probs_record.append(log_output.detach()) #for RSA

        #Sample a batch of distributions. Usually this would come from the dataset
        target = F.softmax(unpert_logits[:, -1, :], dim=-1)
        kl_loss = KLD(log_output, target)

        kl_loss = torch.mul(kl_loss, 0.01, out=None)
                    
        if verbosity_level >= VERY_VERBOSE:
            print('--------')
            print('kl_loss', kl_loss.data.cpu().numpy())
            kl_loss_record += np.round(kl_loss.data.cpu().numpy(),3)
                             
        # calculating total loss
        if loss_type == TYPE_ENGAGEMENT:
            loss += eng_loss
            loss += kl_loss  
        elif loss_type == TYPE_strategy:
            loss += strategy_loss
            loss += kl_loss  
        elif loss_type == TYPE_EMOTION:
            loss += emo_loss
            loss += kl_loss   
        elif loss_type == PPLM_ALL:
            loss += strategy_loss
            # loss += eng_loss # TODO, not included for now.
            loss += emo_loss
            loss += kl_loss
        else:
            loss += kl_loss
                  
        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print('--------')
            print("total loss: ", loss.data.cpu().numpy())
            total_loss_record += np.round(loss.data.cpu().numpy(),3)
        
        # compute gradients
        loss.backward()

        # gradient checking
        # for p_layer in curr_perturbation:
        #     for p_ in p_layer:
        #         print(p_.grad , end=' ')
        #         break
        #     break
        
        if grad_norms is not None:
            grad_norms = [
                    [
                    torch.max(grad, torch.norm(p_.grad * window_mask))
                    for grad, p_ in zip(grads, p_layer)
                    ]
                    for grads, p_layer in zip(grad_norms, curr_perturbation)
            ]
        else:       
            grad_norms = [
                    [
                    torch.norm(p_.grad * window_mask) + SMALL_CONST
                    for p_ in p_layer[:2]
                    ]
                    for p_layer in curr_perturbation
            ]
            
        # normalize gradients
        grad = [
                [
                -stepsize *
                (p_.grad * window_mask / grad ** gamma).data.cpu().numpy()
                for grad, p_ in zip(grads, p_layer[:2])
                ]
                for grads, p_layer in zip(grad_norms, curr_perturbation)
        ]
        
        # accumulate gradient
        grad_accumulator = add_func(grad, grad_accumulator)
        
        # reset gradients, just to make sure
        for p_layer in curr_perturbation:
            for p_ in p_layer[:2]:
                p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_layer in past:
            new_past.append([])
            for p_ in p_layer:
                new_past[-1].append(p_.detach())
        past = new_past
    
    if verbosity_level >= VERBOSE:
        print('strategy_loss_record: ',strategy_loss_record)
        print('eng_loss_record: ',eng_loss_record)
        print('emo_loss_record: ',emo_loss_record)
        print('kl_loss_record: ',kl_loss_record)
        print()
        strategy_loss_record_list.append(strategy_loss_record)
        eng_loss_record_list.append(eng_loss_record)
        emo_loss_record_list.append(emo_loss_record)
        kl_loss_record_list.append(kl_loss_record)
        total_loss_record_list.append(total_loss_record)
    
    # apply the accumulated perturbations to the past
    grad_accumulator = [
                        [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in p_layer
        ]
        for p_layer in grad_accumulator
    ]
    pert_past = add_func(past, grad_accumulator)
    
    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter, log_probs_record


def full_text_generation(
        model,
        tokenizer,
        strategy_tokenizer=None,
        batch=None,
        max_length=100,
        min_length=10,
        num_samples=1,
        device="cuda",
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        sample=True,
        rsa=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        emo_w=1,
        str_w=1,
        verbosity_level=REGULAR,
        loss_type=None,
        strategy_prediction_model=None,
        strategy_classifier=None,
        nsp_classifier=None,
        emo_classifier=None,
        user_emotional_state=None,
        gold=False,
        joint=False,
        **kwargs
):

    # # # Generating the original responses without perturbation
    # unpert_gen_tok_text = user_prefix + original_response
    unpert_response, _, context = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        batch=batch,
        gold=gold,
        joint=joint,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        max_length=max_length,
        min_length=min_length,
        sample=sample,
        perturb=False, # without perturbation
        verbosity_level=verbosity_level
    )
    
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_responses = []
    losses_in_time = []
    
    # we first use last_response to perturb the current hidden, then use the perturbed hidden to generate the next word
    for i in range(num_samples):
        # use pert_sen_past to generate a complete sentence
        # here para_perturb = false, which means this function will use para_past = pert_sen_past  to generate a complete sentence without perturbation in word level
        # if para_perturb = true, this function will perturb in word level(original author design)
        
        if loss_type != 0 and (emo_w !=0 or str_w !=0):
            # # # Generating the responses with perturbation
            pert_response, loss_in_time, _ = generate_text_pplm(
                model=model,
                tokenizer=tokenizer,
                strategy_tokenizer=strategy_tokenizer,
                batch=batch,
                gold=gold,
                joint=joint,
                device=device,
                perturb=True, # with perturbation 
                strategy_prediction_model=strategy_prediction_model,
                strategy_classifier=strategy_classifier,
                nsp_classifier=nsp_classifier,
                emo_classifier=emo_classifier,
                user_emotional_state=user_emotional_state,
                loss_type=loss_type,
                max_length=max_length,
                min_length=min_length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample=sample,
                rsa=rsa,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                emo_w=emo_w,
                str_w=str_w,
                verbosity_level=verbosity_level,
                last_response=unpert_response
            )
        else:
            pert_response = unpert_response
            loss_in_time = []
            
        pert_responses.append(pert_response)
        losses_in_time.append(loss_in_time)

        # print('pert_gen_tok_text: {}'.format(pert_gen_tok_text))
        # print('pert_response: {}'.format(pert_response))
        
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    return context, unpert_response, pert_responses, losses_in_time

def generate_text_pplm(
        model,
        tokenizer,
        strategy_tokenizer=None,
        batch=None,
        gold=None,
        joint=None,
        past=None,
        device="cuda",
        perturb=True,
        strategy_prediction_model=None,
        strategy_classifier=None,
        nsp_classifier=None,
        emo_classifier=None,
        user_emotional_state=None,
        loss_type=0,
        max_length=100,
        min_length=10,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        sample=True,
        rsa=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        emo_w=1,
        str_w=1,
        verbosity_level=REGULAR,
        last_response=None,
):
    output_so_far = None
    system_response =None

    output_so_far = batch['input_ids']
    decoder_input_ids = batch['decoder_input_ids']
    attention_mask = batch['attention_mask']
    
    context_t = batch['input_ids'].clone()
    context = tokenizer.decode(context_t[0]).split(tokenizer.eos_token)[-2].strip()
    context_t = torch.tensor(tokenizer.encode(context), device=device).unsqueeze(0)
    
    if len(tokenizer) > tokenizer.vocab_size:
        bad_words_ids = [i for i in range(tokenizer.vocab_size, len(tokenizer))]
        bad_words_ids.append(3) # __unk__
        ones_mask = torch.ones(len(tokenizer)).to(device)
        ones_mask[bad_words_ids] = 0
        bad_words_mask = (ones_mask == 0)
       
    last = None
    grad_norms = None
    loss_in_time = []
    
    if verbosity_level >= VERBOSE:
        range_func = trange(max_length, ascii=True)
    else:
        range_func = range(max_length)
    
    if rsa:
        worldprior_initial = True
    
    encoder_outputs = model.model.encoder(
        input_ids=output_so_far,
        attention_mask=attention_mask,
        return_dict=True,
    )
    decoder_outputs = model.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=attention_mask,
        return_dict=True,
    )
    
    # strategy prediction
    if gold:
        strat_id = batch['strat_id'].unsqueeze(-1) + len(tokenizer) - 8
        decoder_input_ids = torch.cat([decoder_input_ids, strat_id], dim=-1)
        last = strat_id
        target_strategy = []
        target_strategy.append(tokenizer.decode(strat_id[0]))   

    elif joint:
        lm_logits = model.lm_head(decoder_outputs.last_hidden_state) + model.final_logits_bias
        encoded_info = {}
        model.predict_strategy(lm_logits, encoded_info)   
        strat_id = encoded_info['pred_strat_id'].unsqueeze(-1) + len(tokenizer) - 8
        
        decoder_input_ids = torch.cat([decoder_input_ids, strat_id], dim=-1)
        last = strat_id
        
        if not perturb and encoded_info['pred_strat_id_top1'] == batch['strat_id'].item():
            global strat_pred_acc
            strat_pred_acc += 1
        
        if perturb:
            target_strategy = []
            pred_strat_ids = encoded_info['pred_strat_id_top3'] + len(tokenizer) - 8
            for pred_strat_id in pred_strat_ids[0]:
                target_strategy.append(tokenizer.decode([pred_strat_id]))
  
    else:
        target_strategy = None
        last = decoder_input_ids
    
    for i in range_func:
        '''
        Get past/probs for current output, except for last word
        "past" are the precomputed key and value hidden states of the attention blocks
        Note that GPT takes 2 inputs: past + current_token
        '''

        # decoder_input_ids = torch.cat((torch.tensor([[tokenizer.bos_token_id]]).to(device), output_so_far), 1)
        # run model forward to obtain unperturbed past
        if past is None and output_so_far is not None:
            # last = decoder_input_ids[:, -1:]
            if output_so_far.shape[1] > 1:             
                _, past = model.model.decoder(
                    input_ids=decoder_input_ids[:, :1],
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    return_dict=False,
                )                             
                
        unpert_last_hidden_state, unpert_past, _ = model.model.decoder(
                                                    input_ids=decoder_input_ids,
                                                    encoder_hidden_states=encoder_outputs[0],
                                                    encoder_attention_mask=attention_mask,
                                                    output_hidden_states=True,
                                                    return_dict=False,
                                                    )
        unpert_logits = model.lm_head(unpert_last_hidden_state) + model.final_logits_bias
        
        # check if we are above grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary (unperturb or perturb)
        if not perturb or num_iterations == 0:
            pert_past = past

        else:       
            accumulated_hidden = unpert_last_hidden_state[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            
            # shared world O initilization
            log_probs_record = torch.tensor([]) 
            
            if past is not None:
                pert_past, _, grad_norms, loss_this_iter, log_probs_record = perturb_hidden(
                    past,
                    model,
                    last,
                    decoder_input_ids,
                    encoder_outputs=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    strategy_prediction_model=strategy_prediction_model,
                    target_strategy=target_strategy,
                    strategy_classifier=strategy_classifier,
                    nsp_classifier=nsp_classifier,
                    emo_classifier=emo_classifier,
                    user_emotional_state=user_emotional_state,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    sample=sample,
                    gamma=gamma,
                    emo_w=emo_w,
                    str_w=str_w,
                    device=device,
                    verbosity_level=verbosity_level,
                    tokenizer=tokenizer,
                    strategy_tokenizer=strategy_tokenizer,
                    context_t=context_t,
                    last_response=last_response
                )

                log_probs_record = torch.cat(log_probs_record, 0) 
                
                loss_in_time.append(loss_this_iter)
                                
            else:
                pert_past = past
        
        # # # generating actual output token
        pert_last_hidden_state, past, _ = model.model.decoder(
                                                input_ids=last,
                                                encoder_hidden_states=encoder_outputs[0],
                                                encoder_attention_mask=attention_mask,
                                                past_key_values=pert_past,
                                                output_hidden_states=True,
                                                return_dict=False,
                                                )
        pert_logits = model.lm_head(pert_last_hidden_state) + model.final_logits_bias
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_logits_ = pert_logits.clone()
        
        # Fuse the modified model and original model
        if rsa and perturb:    
            ## S_0
            log_pert_probs = F.log_softmax(pert_logits, dim=-1) #[1,50257]
            log_unpert_probs = F.log_softmax(unpert_logits[:, -1, :], dim=-1)
            log_pert_probs = ((log_pert_probs * gm_scale) + (
                    log_unpert_probs * (1 - gm_scale)))  # + SMALL_CONST
                  
            log_score = torch.cat((log_pert_probs, log_probs_record.to(device)),0).unsqueeze(0) #S_0 [1,perturb_num,50257]
                      
            if worldprior_initial:
                worldpriors = _initialize_worldpriors_unigram(log_pert_probs.size(1))
                worldprior_initial = False  
                
            pert_probs, worldpriors = RSA_inference(log_score, worldpriors, bad_words_mask, top_k, top_p)    
                
        elif perturb:
            pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
            pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
            pert_probs = F.softmax(pert_logits, dim=-1)
            
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST   
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST
            
            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)
                       
        else:
            pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
            pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
            pert_probs = F.softmax(pert_logits, dim=-1)

        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)
                
        # if tokenizer.decode(last.tolist()[0]) == '__unk__': # TODO
        #     pert_probs = pert_probs.masked_fill(bad_words_mask, float("-inf"))
        #     pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)
        #     pert_probs = F.softmax(pert_probs, dim=-1)    
        #     if sample:
        #         last = torch.multinomial(pert_probs, num_samples=1)
        #     else:
        #         _, last = torch.topk(pert_probs, k=1, dim=-1)
                
        if last.tolist()[0][0] == tokenizer.eos_token_id:
            if decoder_input_ids.size(1)-2 <= min_length:
                pert_logits_[:, tokenizer.eos_token_id] = -float("inf")
                pert_logits = pert_logits_.masked_fill(bad_words_mask, float("-inf"))
                pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
                pert_probs = F.softmax(pert_logits, dim=-1) 
                
                if sample:
                    last = torch.multinomial(pert_probs, num_samples=1)
                else:
                    _, last = torch.topk(pert_probs, k=1, dim=-1)
            else:    
                # ***avoid system_response = None***
                if system_response is None:
                    system_response = context_t
                break

        if last.tolist()[0][0] <= len(tokenizer):
            #update system_response
            decoder_input_ids = (
                last if decoder_input_ids is None
                else torch.cat((decoder_input_ids, last), dim=1)
            )
            #update system_response
            system_response = (
                last if system_response is None
                else torch.cat((system_response, last), dim=1)
            )
        else:
            print(last.tolist()[0][0])
            name = input('pause of word_id out of 50256: ')
            print('continue: ', name)
            break
        
        last_response = system_response
        if verbosity_level > REGULAR:
            decode_response = tokenizer.decode(system_response.tolist()[0])
            print('system_response(perturbed)--------------:')
            print(decode_response)
            print()
            # breakpoint()

    return system_response, loss_in_time, context


def run_pplm_example(
        config_name=None,
        inputter_name=None,
        seed=42,
        load_checkpoint=None,
        fp16=False,
        max_input_length=256, 
        max_src_turn=None,
        max_decoder_input_length=50,
        max_knowledge_length=None,
        label_num=None,
        multi_knl=None,
        only_encode=None,
        only_generate=None,
        chinese=None,
        add_nlg_eval=None,
        infer_batch_size=1,
        infer_input_file=None,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        use_gpu=False,
        max_length=100,
        min_length=10,
        num_samples=1,
        stepsize=0.02,
        sample=True,
        rsa=False,
        page=False,
        gold=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        emo_weight=0.25,
        str_weight=1,
        no_cuda=False,
        verbosity='regular',
        out_dir=None,
        for_test_run=False,
        valid=False,
        attribute_type=None,
        joint=False
):
    
    # Set seeds
    set_seed(args.seed)
    
    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
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
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
    }
    
    # load pretrained model
    logger.info('Loading checkpoint: {}\n'.format(load_checkpoint))
    tokenizer, model = build_model(checkpoint=args.load_checkpoint, **names)
    model = deploy_model(model, args)
    
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Number of parameter = {}'.format(total_params))
    
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, opt_level="O1")
    
    model.eval()
    model.to(device)
    
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    
    if page:
        # load PAGE
        PAGE = torch.load('../PAGE/saves_best/model_0.pkl')
        PAGE.eval()
        PAGE.to(device)
        
        # Freeze PAGE
        for param in PAGE.parameters():
            param.requires_grad = False
        
        # load tokenizer for page
        tokenizer_page = AutoTokenizer.from_pretrained("roberta-base")

    inputter = inputters[args.inputter_name]()
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
    
    # Set output path 
    logger.info("Output dir: {}".format(out_dir))

    global file_info
    if rsa and page:
        out_name = '/DPPRSA_blenderbot_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight) 
    elif rsa:
        out_name = '/PPRSA_blenderbot_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight)
    elif page:
        out_name = '/DPPLM_blenderbot_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight)
    elif attribute_type == 'all':
        out_name = '/PPLM_blenderbot_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight) + '_sw' + str(str_weight) 
    elif attribute_type == 'emotion':
        out_name = '/PPLM_blenderbot_' + attribute_type + '_x' + str(num_iterations) + '_ew' + str(emo_weight)
    elif attribute_type == 'strategy':
        out_name = '/PPLM_blenderbot_' + attribute_type + '_x' + str(num_iterations) + '_sw' + str(str_weight)
    else:
        out_name = '/BlenderBot-Joint-pplm'
        num_iterations = 0
    
    if gold:
        out_name += '_g' 
        joint = True
    elif joint:
        out_name += '_j'
        
    if valid:
        out_dir = 'valid'
        if not sample:
            out_dir += '/greedy/'
    
    if for_test_run:
        pass
    elif out_dir != None:
        if not os.path.exists('output/{}'.format(out_dir)):
            logger.info("Create dir: {}".format(out_dir))
            os.makedirs('output/{}'.format(out_dir))
        
        file_pert = open('output/' + out_dir + '/' + out_name + '.txt', 'w+', encoding='utf-8')
        file_info = open('output/' + out_dir + '/' + out_name + '_loss.txt', 'w+', encoding='utf-8')
    else:
        file_pert = open('output/' + out_name + '.txt', 'w+', encoding='utf-8')
        file_info = open('output/' + out_name + '_info.txt', 'w+', encoding='utf-8')
         
    # # attribute model control
    emo_attribute_flag = False
    str_attribute_flag = False
    nsp_attribute_flag = False
    
    strategy_classifier = None
    strategy_tokenizer = None
    strategy_prediction_model = None
    nsp_classifier = None
    emo_classifier = None
    
    if attribute_type == 'all':
        print("***** All atribute model activated *****")
        loss_type = 4
        emo_attribute_flag = True
        str_attribute_flag = True
        nsp_attribute_flag = False # TODO, not used for now.   
    elif attribute_type == 'engagemnt':
        print("***** Engagement atribute model activated *****")
        loss_type = 1
        nsp_attribute_flag = True       
    elif attribute_type == 'strategy': 
        print("***** Empathetic strategy atribute model activated *****")
        loss_type = 2 
        str_attribute_flag = True
    elif attribute_type == 'emotion':
        print("***** Emotion atribute model activated *****")
        loss_type = 3
        emo_attribute_flag = True
    else:
        print("***** No atribute model *****")
        loss_type = 0
        num_iterations = 0

    if rsa:
        print("***** Rational Speech Act activated *****")
        
    if page:
        print("***** Emotion Dynamic activated *****")
        
    if emo_attribute_flag:
        emo_classifier = get_classifier(
            'Emotion_GO',
            device
        )
    
    if str_attribute_flag:
        strategy_classifier = get_classifier(
            'Strategy',
            device
        )
    
    if nsp_attribute_flag:
        nsp_classifier = get_classifier(
            'Engagement',
            device
        )
        
    # # # === begin time ====
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    sentence_count = 0
    neutral_count = 0
    self_count = 0
    inter_count = 0
    
    # file_emo = open('emotional_state2.txt', 'w+', encoding='utf-8')
    
    for batch, posts, references, sample_ids, histories, speaker_roles in infer_dataloader:    
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        
        if page:
            tk_id, at_mk, msk, adj, d_len = encode_page(histories, tokenizer_page, device)
            breakpoint()
            msk = msk.unsqueeze(0).cuda()
            tk_id = tk_id.unsqueeze(0)
            at_mk = at_mk.unsqueeze(0)
            adj = msk.clone()
            adj = adj.long().cuda()
            
            label = 1 # burden parameter
            prediction = PAGE(tk_id, at_mk, msk, adj, label) 
            ed_prediction = torch.gt(prediction.data, 0.5).long()[0] # d_dim x d_dim, emotion evidence
            
            usr_eds = []
            for i, role in reversed(list(enumerate(speaker_roles[0]))):
                if role == 'usr':
                    usr_eds.append(ed_prediction[i])
                elif role == 'sys' and usr_eds:
                    break
                else:
                    usr_eds.append(None)
                                
            user_emotional_state, emo_w, str_w = get_user_emotional_state(
                                            usr_eds, speaker_roles, emo_weight, str_weight, prediction)
    
            if user_emotional_state == 'neutral': #TODO
                neutral_count += 1
                emo_w = emo_weight
                str_w = str_weight
            elif user_emotional_state == 'self-contagion':
                self_count += 1
            elif user_emotional_state == 'inter-personal':
                inter_count += 1
                
            # file_emo.write(user_emotional_state + '\n')    
                
        else:
            user_emotional_state = 'neutral'
            emo_w = emo_weight 
            str_w = str_weight
        
        if for_test_run == True:
            strategy_loss_record_list.clear()
            eng_loss_record_list.clear()
            emo_loss_record_list.clear()
            kl_loss_record_list.clear()
            total_loss_record_list.clear()
            iteration_num_record_list.clear()
            # # early stop for test
            # if sentence_count == 10:
            #     break
         
        logging.disable(logging.WARNING)
         
        sentence_count += 1
        if sentence_count %500 == 0 or sentence_count == 1:
            print("===" + str(sentence_count) + "===")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            if verbosity_level >= REGULAR:
                print("= Prefix of sentence =")
                print(posts)
                print()

        # generate unperturbed and perturbed texts 
        context, unpert_response, pert_responses, losses_in_time = full_text_generation(
            model=model,
            tokenizer=tokenizer,
            strategy_tokenizer=strategy_tokenizer,
            batch=batch,
            device=device,
            max_length=max_length,
            min_length=min_length,
            num_samples=num_samples,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample=sample,
            rsa=rsa,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            emo_w=emo_w,
            str_w=str_w,
            verbosity_level=verbosity_level,
            loss_type=loss_type,
            strategy_prediction_model=strategy_prediction_model,
            strategy_classifier=strategy_classifier,
            nsp_classifier=nsp_classifier,
            gold=gold,
            joint=joint,
            emo_classifier=emo_classifier,
            user_emotional_state=user_emotional_state,
        )

        decode = lambda x: _norm(tokenizer.decode(x))
        dec_sentence = decode(pert_responses[0][0])
        dec_sentence = dec_sentence.strip()
            
        # untokenize unperturbed text 
        if sentence_count %100 == 0 or for_test_run:
            if verbosity_level > REGULAR or for_test_run:
                print(context)
                print("=" * 80)
                print("= Unperturbed generated text =")
                unpert_gen_text = decode(unpert_response.tolist()[0])
                print(unpert_gen_text)
                print()

                for i, pert_res in enumerate(pert_responses):
                    print("= Perturbed response {} =".format(i))
                    pert_res_text = decode(pert_res.tolist()[0])
                    print(pert_res_text)
                    print()                
                
                if page:
                    print('\n= User Emotional State =')
                    print(user_emotional_state)
                
                if emo_classifier is not None:               
                    # og emotion
                    unpert_respon_keys = classifying_emotion(unpert_gen_text, model, tokenizer, emo_classifier, device)
                    # respon strategy
                    respon_keys = classifying_emotion(dec_sentence, model, tokenizer, emo_classifier, device)
                    print('\n= Emotion Prediction Result =')
                    print('response_class:{} unpert_class:{}'.format(respon_keys, unpert_respon_keys))
                
                if strategy_classifier is not None:               
                    # og strategy
                    unpert_respon_keys = classifying_strategy(unpert_gen_text, model, tokenizer, strategy_classifier, device)
                    # respon strategy
                    respon_keys = classifying_strategy(dec_sentence, model, tokenizer, strategy_classifier, device)
                    print('\n= Strategy Prediction Result =')
                    print('gold_class:', tokenizer.decode(batch['strat_id'] + len(tokenizer) - 8))
                    print('response_class:{} unpert_class:{}'.format(respon_keys, unpert_respon_keys))
                    print()
                
                breakpoint()
                
        if not joint:
            dec_sentence = dec_sentence.split(']', 1)[-1].strip()
        if not for_test_run:
            file_pert.write(dec_sentence + '\n')

        if for_test_run:
            loss_record_list=[]
            loss_type_list=[]
            if strategy_loss_record_list:
                loss_record_list.append(strategy_loss_record_list)
                loss_type_list.append('strategy_loss')
            if eng_loss_record_list:
                loss_record_list.append(eng_loss_record_list)
                loss_type_list.append('eng_loss')
            if kl_loss_record_list:
                loss_record_list.append(kl_loss_record_list)
                loss_type_list.append('kl_loss')
            if total_loss_record_list:
                loss_record_list.append(total_loss_record_list)
                loss_type_list.append('total_loss')
            
    # # # === finish time ===
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    file_info.write("begin time: " + begin_time + "\t")
    file_info.write("finish time: " + finish_time + "\n")
    print(begin_time)
    print(finish_time)

    struct_time = time.strptime(begin_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_begin = int(time.mktime(struct_time)) # 轉成時間戳

    struct_time = time.strptime(finish_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_finish = int(time.mktime(struct_time)) # 轉成時間戳
    
    total_time = time_stamp_finish - time_stamp_begin
    
    if page:
        file_info.write('neutral:' + str(neutral_count)+' \n')
        file_info.write('self-contagion:' + str(self_count)+' \n')
        file_info.write('inter-personal:' + str(inter_count)+' \n')
    
    file_info.write('inter-total time(second):' + str(total_time) +' \n')
    file_info.write('strategy prediction accurracy:' + str(round(strat_pred_acc/sentence_count,2)) +' \n')

    print("total time(second): ", total_time)
    print('strategy prediction accurracy: ', round(strat_pred_acc/sentence_count,2))

    file_pert.close()
    file_info.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--inputter_name', type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_checkpoint", '-c', type=str, default=None)
    parser.add_argument("--fp16", type=boolean_string, default=False)
    
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_src_turn", type=int, default=None)
    parser.add_argument("--max_decoder_input_length", type=int, default=256)
    parser.add_argument("--max_knowledge_length", type=int, default=None)
    parser.add_argument('--label_num', type=int, default=None)
    parser.add_argument('--multi_knl', action='store_true', help='allow candidate knowledge items')
    
    parser.add_argument('--only_encode', action='store_true', help='only do encoding')
    parser.add_argument('--only_generate', action='store_true', help='do not conduct evaluations')
    parser.add_argument('--chinese', action='store_true', help='chinese language')
    parser.add_argument('--add_nlg_eval', action='store_true', help='add nlg-eval')
    
    parser.add_argument("--infer_batch_size", type=int, default=1)
    parser.add_argument('--infer_input_file', type=str, nargs='+', required=True)
    
    parser.add_argument("--use_gpu", action='store_true')
    
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=1)
    
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument("--sample", action="store_true")
    
    parser.add_argument("--attribute_type", type=str, default='None')
    parser.add_argument('--joint', action="store_true", help="Include strategy for generation")
    parser.add_argument(
        "--rsa", action="store_true",
        help="Activate Rational Speech Act for generation"
    )
    parser.add_argument(
        "--page", action="store_true",
        help="Activate PAGE for generation"
    )
    parser.add_argument(
        "--gold", action="store_true",
        help="Activate gold_strategy for generation"
    )
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    
    parser.add_argument("--emo_weight", type=float, default=1)
    parser.add_argument("--str_weight", type=float, default=1)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--for_test_run", action="store_true")
    parser.add_argument("--valid", action="store_true")
    
    args = parser.parse_args()
    
    
    run_pplm_example(**vars(args))
