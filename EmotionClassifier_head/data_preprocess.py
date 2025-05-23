# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:16:38 2023

@author: OWNER
"""

import json
import pickle
import argparse
import os
from tqdm import tqdm 
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pplm_classification_head import ClassificationHead


DISCRIMINATOR_MODELS_PARAMS = {
    "Emotion": {
        "path": "../EmotionClassifier_head/output_EDI/EDI_classifier_head_epoch_5.pt",
        "class_size": 33,
        "embed_size": 1024,
        "pretrained_model": "GODEL",
        "class_vocab": {"afraid": 0, "angry": 1, "annoyed": 2, "anticipating": 3, "anxious": 4,
                        "apprehensive": 5, "ashamed": 6, "caring": 7, "confident": 8, "content": 9,
                        "devastated": 10, "disappointed": 11, "disgusted": 12, "embarrassed": 13, "excited": 14,
                        "faithful": 15, "furious": 16, "grateful": 17, "guilty": 18, "hopeful": 19,
                        "impressed": 20, "jealous": 21, "joyful": 22, "lonely": 23, "neutral": 24,
                        "nostalgic": 25, "prepared": 26, "proud": 27, "sad": 28, "sentimental": 29,
                        "surprised": 30, "terrified": 31, "trusting": 32}
    },
}

def get_data(data_path):
    conv_data = []
    data = json.load(open(data_path, 'r'))
    for k, v in data.items():
        conv_item = []
        conv = v[0]
        for i, utt in enumerate(conv):
            utt_item = {}
            speaker = utt['speaker']
            utt_item['id'] = i + 1
            utt_item['speaker'] = speaker
            utterance = utt['utterance']
            utt_item['utterance'] = utterance
            emotion = utt['emotion']
            utt_item['emotion'] = emotion
            if 'expanded emotion cause evidence' in utt:
                evidence = utt['expanded emotion cause evidence']
                utt_item['evidence'] = evidence
            conv_item.append(utt_item)
        conv_data.append(conv_item)
    return conv_data

def emo_encoding(tokenizer, model):
    emo = {'happiness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
    emo_list = [' happiness', ' neutral', ' anger', ' sadness', ' fear', ' surprise', ' disgust']
    # act_list = [' inform', ' question', ' directive', ' promise']
    
    emo_list_ = [tokenizer.encode(e, add_special_tokens=False) for e in emo_list]
    # act_list_ = [tokenizer.encode(e, add_special_tokens=False) for e in act_list]
    
    vocab_weight = model.lm_head.decoder.weight
    
    emo_emb = [vocab_weight[e] for e in emo_list_]
    # act_emb = [vocab_weight[e] for e in act_list_]
    print(emo_emb[0].shape)
    # print(act_emb[0].shape)
    emo_emb = [torch.zeros(1, vocab_weight.shape[1])] + emo_emb
    # act_emb = [torch.zeros(1, vocab_weight.shape[1])] + act_emb
    emo_emb = torch.cat(emo_emb, dim=0)
    # act_emb = torch.cat(act_emb, dim=0)
    pickle.dump(emo_emb, open('emotion_embeddings.pkl', 'wb'))
    # pickle.dump(act_emb, open('act_embeddings.pkl', 'wb'))

train_data = get_data('dailydialog_train.json')
dev_data = get_data('dailydialog_valid.json')
test_data = get_data('dailydialog_test.json')

pickle.dump(train_data, open('dailydialog_train.pkl', 'wb'))
pickle.dump(dev_data, open('dailydialog_dev.pkl', 'wb'))
pickle.dump(test_data, open('dailydialog_test.pkl', 'wb'))

def get_classifier(
        name: Optional[str],
        device: str
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-src_file', type=str, required=False) # preficted result file
    parser.add_argument('-pred_file', type=str, required=True) # preficted result file
    parser.add_argument('-label_file', type=str, required=False) # preficted result file

    opt = parser.parse_args()
        
    # predict
    preds = []
    preds_line = []
    preds_line_tokenize = []
    with open(opt.pred_file, 'r', encoding='utf-8') as fpred:
        for line in fpred:
            line = line.replace('[SEP]','').strip()
            line = line.replace('<|endoftext|>','').strip()
            preds_line.append(line)
        preds.append(preds_line)                 
                       
    print(opt.pred_file)
    
    # set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    
    finetune_generation_model = '../DialoGPT/GODEL-large_ftED/ckpt/checkpoint-epoch-25160'
    model = AutoModelForSeq2SeqLM.from_pretrained(finetune_generation_model) 
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
    
    intent_classifier = get_classifier('Emotion', device)      
    
    count = 0
    write_count = 0
            
    file_dir = opt.pred_file.split('/')[-1]
    os.makedirs(os.path.dirname('labelling_EDI_ml/{}'.format(file_dir)), exist_ok=True)

    with open('labelling_EDI_ml/{}'.format(file_dir), 'w+', encoding='utf-8') as f:
        for response in tqdm(preds_line , mininterval=2, desc='  - (Emotion Labelling) -  ', leave=False):
            count += 1
            
            respon_keys = classifying_intent(response, model, tokenizer, intent_classifier, device)
            
            if count <=5:
                print('response:',response)
                print(respon_keys)
            
            if not respon_keys:
                print('No intent classfied.')
                print('line: ',write_count)
                print(response)
                print(respon_keys)
                respon_keys.add('questioning')
                #break
            
            for i, keys in enumerate(respon_keys):
                # write into files
                if (i+1) == len(respon_keys):
                    f.write(f"{keys}\n")
                    write_count += 1
                else:
                    f.write(f"{keys},")

        print(count)
        print(write_count)

if __name__ == '__main__':
    main()