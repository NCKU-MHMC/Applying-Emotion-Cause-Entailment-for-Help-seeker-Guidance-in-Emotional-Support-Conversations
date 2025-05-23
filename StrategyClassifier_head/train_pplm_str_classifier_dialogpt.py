#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import argparse
import csv
import sys
import json
import numpy as np
import os
import time
import torch
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from torch.distributed import get_rank

from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy

from tqdm import tqdm
from transformers import (AutoTokenizer, AutoConfig)

from pplm_classification_head import ClassificationHead

sys.path.append("..") 
sys.path.insert(0, '../GenerationModel')
from GenerationModel.models import models

torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "What's special with the service ?"
example_sentence2 = "This is disgusting! I hate it!"
example_sentence3 = "That just isn't good enough . Cet me your general manager . I want to speak to him now ."
max_length_seq = 500


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size=None,
            config_name=None,
            checkpoint=None,
            classifier_head=None,
            local_rank=-1,
            cached_mode=False,
            device='cpu'
    ):
        super(Discriminator, self).__init__()
        
        if not os.path.exists(f'../GenerationModel/CONFIG/{config_name}.json'):
            raise ValueError
        
        with open(f'../GenerationModel/CONFIG/{config_name}.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'model_name' not in config or 'pretrained_model_path' not in config:
            raise ValueError
        
        pretrained_model_path =  os.path.join('../GenerationModel', os.path.basename(config['pretrained_model_path']))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        
        Model = models[config['model_name']]
        self.encoder = Model.from_pretrained(pretrained_model_path)
        if config.get('custom_config_path', None) is not None:
            self.encoder = Model(AutoConfig.from_pretrained(config['custom_config_path']))
            
        self.embed_size = self.encoder.config.hidden_size
        
        if 'gradient_checkpointing' in config:
            setattr(self.encoder.config, 'gradient_checkpointing', config['gradient_checkpointing'])
        
        if 'expanded_vocab' in config:
            self.tokenizer.add_tokens(config['expanded_vocab'], special_tokens=True)
        self.encoder.tie_tokenizer(self.tokenizer)
        
        if checkpoint is not None:
            if local_rank == -1 or get_rank() == 0:
                print('loading finetuned model from %s' % checkpoint)
            self.encoder.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        
        if classifier_head:
            self.classifier_head = classifier_head
        else:
            if not class_size:
                raise ValueError("must specify class_size")
            self.classifier_head = ClassificationHead(
                class_size=class_size,
                embed_size=self.embed_size
            )
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x, attention_mask):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            hidden, _ = self.encoder.transformer(x, return_dict=False)
        else:
            # for bert
            output = self.encoder(input_ids=x, decoder_input_ids=x, output_hidden_states=True)
            hidden = output['decoder_hidden_states'][-1]
            
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden
    
    def representation(self, x):
        
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            hidden, _ = self.encoder.transformer(x, return_dict=False)
        else:
            # for bert
            hidden, _ = self.encoder(x, return_dict=False)

        last_hidden = hidden[:,-1,:]

        return last_hidden

    def forward(self, x, attention_mask):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device), attention_mask)

        logits = self.classifier_head(avg_hidden)
        # probs = F.log_softmax(logits, dim=-1)

        return logits

    def predict(self, input_sentence):
        input_t = self.tokenizer.encode(input_sentence) 
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        if self.cached_mode:
            input_t = self.avg_representation(input_t)

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        # prob = [math.exp(log_prob) for log_prob in log_probs]
        return log_probs


class Dataset(data.Dataset):
    def __init__(self, X, y, attention_mask):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        data['attention_mask']= self.attention_mask[index]
        
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]
            # padded_sequences[i, -end:] = seq[:end] # left pad

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)
    attention_mask_batch, _ = pad_sequences(item_info["attention_mask"])
    
    return x_batch, y_batch, attention_mask_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def train_epoch(data_loader, discriminator, optimizer,
                epoch=0, log_interval=10, device='cpu'):
    samples_so_far = 0
    discriminator.train_custom()
    loss_function = torch.nn.BCEWithLogitsLoss()
    for batch_idx, (input_t, target_t, attention_mask) in enumerate(tqdm(data_loader)):
        input_t, target_t, attention_mask_t = input_t.to(device), target_t.to(device), attention_mask.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t, attention_mask_t)

        loss = loss_function(output_t.to(torch.float32), target_t.to(torch.float32))
        loss.backward(retain_graph=True)
        optimizer.step()
        
        samples_so_far += len(input_t)

        # if batch_idx % log_interval == 0:
        #     print(
        #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             epoch + 1,
        #             samples_so_far, len(data_loader.dataset),
        #             100 * samples_so_far / len(data_loader.dataset), loss.item()
        #         )
        #     )


def evaluate_performance(data_loader, discriminator, device='cpu'):
    discriminator.eval()
    test_loss = 0
    correct = 0
    false = 0
    score = 0
    accuracy = 0
    loss_function = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for input_t, target_t, attention_mask in data_loader:
            input_t, target_t, attention_mask = input_t.to(device), target_t.to(device), attention_mask.to(device)
            output_t = discriminator(input_t, attention_mask)
            # sum up batch loss
            test_loss += loss_function(output_t.to(torch.float32), target_t.to(torch.float32))
            # test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            
            # get the index of the max log-probability
            torch.set_printoptions(profile="full")
            # pred_t = output_t.argmax(dim=1, keepdim=True)

            correct += output_t.eq(target_t.view_as(output_t)).sum().item()/output_t.size(1)
            false += output_t.ne(target_t.view_as(output_t)).sum().item()/output_t.size(1)
           
            # f1 = F1Score(task="multilabel", num_classes=28)
            acc = MultilabelAccuracy(num_labels=output_t.size(1), average='weighted')
            f1 = MultilabelF1Score(num_labels=output_t.size(1), average='weighted')
            accuracy += acc(output_t.cpu(), target_t.cpu())
            score += f1(output_t.cpu(), target_t.cpu())

    test_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader)
    score /= len(data_loader)
    print(
        "Performance on valid set: "
        "Average loss: {:.4f}, Accuracy: {:.2f}% F1:{:.2f}".format(
            test_loss,
            100. * accuracy,
            score
        )
    )

    return test_loss, accuracy, score


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    feature = model.tokenizer(input_sentence)
    input_t = feature['input_ids']
    attetion_mask = feature['attention_mask']

    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    attetion_mask_t = torch.tensor([attetion_mask], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t, attetion_mask_t).data.cpu().numpy().flatten().tolist()
    
    print("\nInput sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, log_prob) for c, log_prob in
        zip(classes, log_probs)
    ))


def get_cached_data_loader(dataset, batch_size, discriminator,
                           shuffle=False, device='cpu'):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader


def get_idx2class(dataset_fp):
    classes = set()
    with open(dataset_fp,'r', newline='', encoding= 'unicode_escape') as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in tqdm(csv_reader, ascii=True):
            if row:
                classes.add(row[0])

    return sorted(classes)


def get_generic_dataset(dataset_fp, tokenizer, device,
                        idx2class=None, add_eos_token=False):
    if not idx2class:
        idx2class = get_idx2class(dataset_fp)
    class2idx = {c: i for i, c in enumerate(idx2class)}

    x = []
    y = []
    attention_mask = []
    with open(dataset_fp,'r', newline='', encoding= 'unicode_escape') as f:
        csv_reader = csv.reader(f, delimiter=",")
        
        for i, row in enumerate(tqdm(csv_reader, ascii=True)):
            if row:
                label = row[0]
                text = row[1]
                            
                feature = tokenizer(text)
                seq = feature['input_ids']
                atten = torch.tensor(feature['attention_mask'], device=device)
                
                if (len(seq) <= max_length_seq):
                    if add_eos_token:
                        seq = seq + tokenizer.eos_token_id
                    
                    seq = torch.tensor(
                        seq,
                        device=device,
                        dtype=torch.long
                    )

                else:
                    print(
                        "Line {} is longer than maximum length {}".format(
                            i, max_length_seq
                        ))
                    continue

                x.append(seq)
                
                label_vector = []
                for i, label_m in enumerate(class2idx):
                    if label_m in label.split(','):
                        label_vector.append(1.0)
                    else:
                        label_vector.append(0.0)
                    
                label = label_vector

                y.append(label)
                attention_mask.append(atten)
        
    return Dataset(x, y, attention_mask)


def train_discriminator(
        dataset,
        dataset_fp=None,
        dataset_fp_train=None,
        dataset_fp_valid=None,
        dataset_fp_test=None,
        config_name=None,
        load_checkpoint=None,
        epochs=10,
        learning_rate=0.0001,
        batch_size=64,
        log_interval=10,
        save_model=False,
        cached=False,
        no_cuda=False,
        output_fp='.'
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    # add_eos_token = pretrained_model.startswith("gpt2")
    add_eos_token = False

    if save_model:
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)
    classifier_head_meta_fp = os.path.join(
        output_fp, "{}_classifier_head_meta.json".format(dataset)
    )
    classifier_head_fp_pattern = os.path.join(
        output_fp, "{}_classifier_head_epoch".format(dataset) + "_{}.pt"
    )

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()
       
    idx2class = get_idx2class(dataset_fp_test)
    
    idx2class = {"Question", "Self-disclosure", "Affirmation and Reassurance", "Providing Suggestions",
               "Reflection of feelings", "Information", "Restatement or Paraphrasing", "Others"}
    
    discriminator = Discriminator(
        class_size=len(idx2class),
        config_name=config_name,
        checkpoint=load_checkpoint,
        cached_mode=cached,
        device=device
    ).to(device)
    
    train_dataset = get_generic_dataset(
        dataset_fp_train, discriminator.tokenizer, device, idx2class,
        add_eos_token=add_eos_token
    )
    valid_dataset = get_generic_dataset(
        dataset_fp_valid, discriminator.tokenizer, device, idx2class,
        add_eos_token=add_eos_token
    )   
    test_dataset = get_generic_dataset(
        dataset_fp_test, discriminator.tokenizer, device, idx2class,
        add_eos_token=add_eos_token
    )      
    
    discriminator_meta = {
        "class_size": len(idx2class),
        "embed_size": discriminator.embed_size,
        "config_name": config_name,
        "checkpoint": load_checkpoint,
        "class_vocab": {c: i for i, c in enumerate(idx2class)},
    }

    end = time.time()
    print("Preprocessed {} data points".format(
        len(train_dataset) + len(test_dataset))
    )
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,
            shuffle=True, device=device
        )
        valid_loader = get_cached_data_loader(
            valid_dataset, batch_size, discriminator, device=device
        )
        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator, device=device
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

    if save_model:
        with open(classifier_head_meta_fp, "w") as meta_file:
            json.dump(discriminator_meta, meta_file)

    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    valid_losses = []
    valid_accuracies = []
    valid_f1s = []

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )
        valid_loss, valid_accuracy, valid_f1 = evaluate_performance(
            data_loader=valid_loader,
            discriminator=discriminator,
            device=device
        )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        valid_f1s.append(valid_f1)

        print("\nExample prediction")
        predict(example_sentence, discriminator, idx2class,
                cached=cached, device=device)
        predict(example_sentence2, discriminator, idx2class,
                cached=cached, device=device)
        
        if save_model:
            # torch.save(discriminator.state_dict(),
            #           "{}_discriminator_{}.pt".format(
            #               args.dataset, epoch + 1
            #               ))
            torch.save(discriminator.get_classifier().state_dict(),
                       classifier_head_fp_pattern.format(epoch + 1))

    min_loss = float("inf")
    min_loss_epoch = 0
    max_acc = 0.0
    max_acc_epoch = 0
    max_f1 = 0.0
    max_f1_epoch = 0
    print("Valid performance per epoch")
    print("epoch\tloss\tacc")
    for e, (loss, acc, f1) in enumerate(zip(valid_losses, valid_accuracies, valid_f1s)):
        print("{}\t{}\t{}".format(e + 1, loss, acc))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = e + 1
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = e + 1
        if f1 > max_f1:
            max_f1 = f1
            max_f1_epoch = e + 1
    print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))
    print("Max f1: {} - Epoch: {}".format(max_f1, max_f1_epoch))
    
    print("\nTest performance")
    exp_discriminator, _ = load_discriminator(classifier_head_fp_pattern.format(min_loss_epoch),
                                            classifier_head_meta_fp, device=device)
    exp_discriminator = exp_discriminator.to(device)
    
    test_loss, test_accuracy, test_f1 = evaluate_performance(
        data_loader=test_loader,
        discriminator=exp_discriminator,
        device=device
    )
    print("Test loss: {} - Epoch: {}".format(test_loss, min_loss_epoch))
    print("Test acc: {} - Epoch: {}".format(test_accuracy, min_loss_epoch))
    print("Test f1: {} - Epoch: {}".format(test_f1, min_loss_epoch))

    print()
    
    predict(example_sentence2, exp_discriminator, idx2class,
            cached=cached, device=device)

    return discriminator, discriminator_meta


def load_classifier_head(weights_path, meta_path, device='cpu'):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size']
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(weights_path, map_location=device))
    classifier_head.eval()
    return classifier_head, meta_params


def load_discriminator(weights_path, meta_path, device='cpu'):
    classifier_head, meta_param = load_classifier_head(
        weights_path, meta_path, device
    )
    discriminator = Discriminator(
        config_name=meta_param['config_name'],
        checkpoint=meta_param['checkpoint'],
        classifier_head=classifier_head,
        cached_mode=False,
        device=device
    )
    return discriminator, meta_param


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="ESC")
    parser.add_argument("--dataset_fp", type=str, default="",
                        help="File path of the dataset to use. "
                             "Needed only in case of generic datadset")
    parser.add_argument("--dataset_fp_train", type=str, default="",
                    help="File path of the dataset to use. ")
    parser.add_argument("--dataset_fp_valid", type=str, default="",
                    help="File path of the dataset to use. ")
    parser.add_argument("--dataset_fp_test", type=str, default="",
                    help="File path of the dataset to use. ")
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument("--load_checkpoint", '-c', type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learnign rate")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save_model", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--cached", action="store_true",
                        help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true",
                        help="use to turn off cuda")
    parser.add_argument("--output_fp", default=".",
                        help="path to save the output to")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))
