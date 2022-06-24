import os.path
import numpy as np
import nltk

# C4数据集构建
from transformers import BertTokenizer
import json
import torch

import random


def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['[SEP]']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_next_sentence(sentence, next_sentence, sample_sentence):
    if random.random() < 0.5:
        is_next = 1
    else:
        next_sentence = sample_sentence[0]
        is_next = 0
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, sample_sentence, max_len):
    nsp_data_from_paragraph = []
    tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[0], paragraph[1], sample_sentence)
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, tokenizer):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '[MASK]'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.choice(list(tokenizer.vocab))
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, tokenizer):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEQ]']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, tokenizer)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return tokenizer.convert_tokens_to_ids(mlm_input_tokens), pred_positions, tokenizer.convert_tokens_to_ids(
        mlm_pred_labels)


def _pad_bert_inputs(examples, max_len, vocab):
    token_ids, pred_positions, mlm_pred_label_ids, segments, is_next = examples[0]
    if len(token_ids) > max_len:
        max_len = len(token_ids)
    all_token_ids = torch.tensor(token_ids + [vocab['[PAD]']] * (max_len - len(token_ids)), dtype=torch.long)
    all_segments = torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long)
    attention_mask = torch.tensor([1] * len(token_ids) + [0] * (len(all_token_ids) - len(token_ids)))
    mlm_label = [-1] * max_len
    for i, pred_position in enumerate(pred_positions):
        mlm_label[pred_position] = mlm_pred_label_ids[i]
    all_mlm_labels = torch.tensor(mlm_label, dtype=torch.long)
    nsp_labels = torch.tensor(is_next, dtype=torch.long)
    valid = len(token_ids)
    return all_token_ids, all_segments, attention_mask, all_mlm_labels, nsp_labels, valid


class Uncase_C4(torch.utils.data.Dataset):
    '''
        self.data : json中的所有句子[所有句子]
        self.is_end_start ：[所有句子]，元素为0/1，当为0时，表示该句子在是段落的最后一句话，当元素为1时，表示句子为段落的中间部分
    '''
    
    def __init__(self, root='D:\study\competiton\dataset\c4\c4-train.00000-of-01024.json', max_len=512):
        '''
            将json文件一次全部读取完，并返回data和is_end_start
        '''
        file = open(root, 'r', encoding='utf-8')
        self.data = []
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.is_end_start = []
        for line in file.readlines():
            dic = json.loads(line)
            text = dic['text'].lower()
            sentences = nltk.sent_tokenize(text)
            self.is_end_start.extend([True] * (len(sentences) - 1) + [False])
            self.data.extend(sentences)
    
    def convert_paragraph_token(self, paragraph):
        '''
            将句子划分成wordpice
            paragraph : [2,]或[1,],
        '''
        sentence_token = []
        for sentence in paragraph:
            worpice = self.tokenizer.tokenize(sentence)
            sentence_token.append(worpice)
        return sentence_token
    
    def conver_new(self, convert: list):
        '''
            将多余的数据删除
            example: convert[1,2,3]表示第1,2,3个数据删除
        '''
        for i in range(len(convert) - 1, -1, -1):
            item = convert[i]
            del self.data[item]
            if self.is_end_start[item]:
                self.is_end_start[item - 1] = False
                del self.is_end_start[item]
            else:
                if self.is_end_start[item - 1]:
                    self.is_end_start[item - 1] = False
                    del self.is_end_start[item]
                else:
                    del self.is_end_start[item]
    
    def create_pair(self, idx):
        '''
            根据索引制造样本对
            制造原则为：当id为段落中的句子时，返回的样本对为该句子和下一个句子，否则返回当前句子和上一个句子
        '''
        if self.is_end_start[idx]:
            sentence_token = self.convert_paragraph_token([self.data[idx], self.data[idx + 1]])
        else:
            sentence_token = self.convert_paragraph_token([self.data[idx], self.data[idx - 1]])
        
        # 随机生成摸个句子
        sample_sentence = self.convert_paragraph_token([random.choice(self.data)])
        # 制造nsp标签
        sample = _get_nsp_data_from_paragraph(sentence_token, sample_sentence, self.max_len)
        
        self.vocab = self.tokenizer.vocab
        sample_tokenid = []
        len_id = 0
        for tokens, segments, is_next in sample:
            # 得到带有遮挡数据的标签
            try:
                sample_tokenid = [(_get_mlm_data_from_tokens(tokens, self.tokenizer) + (segments, is_next))]
                len_id = len(sample_tokenid[0][0])
            except KeyError as e:
                sample_tokenid = [], [], [], [], []
                len_id = 1000
                print('{} in  have key:{} that cannot convert to ids'.format(tokens, e))
        
        return sample_tokenid, len_id
        
        # 将数据进行padding
        # return _pad_bert_inputs(sample_tokenid, self.max_len, self.vocab)
    
    def __getitem__(self, idx):
        sample_tokenid, valid = self.create_pair(idx)
        return sample_tokenid, valid
    
    def __len__(self):
        return len(self.data)


def coofn(sample):
    max_len = 512
    
    token_ids, type_ids, attentionmasks, mlmlabels, nsp_labels = [], [], [], [], []
    max_len_id = 0
    flag = [True] * len(sample)
    for i, (_, valid_len) in enumerate(sample):
        if valid_len > max_len_id:
            if valid_len > max_len:
                flag[i] = False
            else:
                max_len_id = valid_len
    for i in range(len(flag) - 1, -1, -1):
        if flag[i] == False:
            del sample[i]
    for t_list, _ in sample:
        id, position, mlmlabel, type_id, nsplabel = t_list[0]
        token_ids.append(torch.tensor(id + [0] * (max_len_id - len(id)), dtype=torch.long))
        type_ids.append(torch.tensor(type_id + [0] * (max_len_id - len(type_id)), dtype=torch.long))
        attentionmasks.append(torch.tensor([1] * len(id) + [0] * (max_len_id - len(id))))
        tmp = [-1] * max_len_id
        for i, pred_position in enumerate(position):
            tmp[pred_position] = mlmlabel[i]
        mlmlabels.append(torch.tensor(tmp, dtype=torch.long))
        nsp_labels.append(torch.tensor(nsplabel))
    token_ids = torch.stack(token_ids, 0)
    type_ids = torch.stack(type_ids, 0)
    attentionmasks = torch.stack(attentionmasks, 0)
    mlmlabels = torch.stack(mlmlabels, 0)
    nsp_labels = torch.stack(nsp_labels, 0)
    return token_ids, type_ids, attentionmasks, mlmlabels, nsp_labels
