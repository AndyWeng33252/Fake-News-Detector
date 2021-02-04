# models.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import collections
import random
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from utils import*
import math
import csv
import copy
from torch.nn.utils.rnn import pad_sequence

#####################
# MODELS FOR BERT   #
#####################


def print_evaluation(dev, model):
    """
    Runs the classifier on the given text
    :param text:
    :param lm:
    :return:
    """
    # predictions = []
    # if output_file is not None:
    #     with open(output_file,'w') as f:
    #         json.dump()
    num_correct = 0
    for ex in dev:
        current = model.predict(ex.sent)
        # predictions.append(current)
        if current == ex.label:
            num_correct += 1

    num_total = len(dev)
    data = {'correct': num_correct, 'total': num_total, 'accuracy': float(num_correct)/num_total * 100.0}
    print("=====Results=====")
    print(data)
    return float(num_correct)/num_total * 100.0


class BERT(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """

    def __init__(self, dropout, num_class):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param alphabet_size: size of input (integer)
        :param input_size: size of input(integer)
        :param hidden_size: size of hidden layer(integer), which should be the number of classes
        """
        super(BERT, self).__init__()
        self.BERT = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=dropout)
        self.W = nn.Linear(self.BERT.config.hidden_size, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, attention_mask):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        last_hidden, pooled_output = self.BERT(input_ids=input_id, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        output = self.W(output)
        return self.softmax(output)


def tokenize(sentence, tokenizer, max_len):
    token_ids_list = []
    attention_mask_list = []
    encoding = tokenizer.encode_plus(sentence,
                                     max_length=max_len,
                                     add_special_tokens=True,
                                     pad_to_max_length=True,
                                     return_attention_mask=True,
                                     return_token_type_ids=False)
    token_ids_list += encoding['input_ids']
    attention_mask_list += encoding['attention_mask']

    return token_ids_list, attention_mask_list


def batch_bert(train_exs, batch_size, tokenizer, max_len):
    temp_batch_id = []
    temp_batch_mask = []
    batch_id_list = []
    batch_mask_list = []
    temp_labels = []
    labels = []
    counter = 1

    for examples in train_exs:
        temp_labels.append(examples.label)
        input_ids, attention_mask = tokenize(examples.words, tokenizer, max_len)
        temp_batch_id.append(input_ids)
        temp_batch_mask.append(attention_mask)
        if counter == batch_size:
            labels.append(temp_labels)
            temp_labels = []
            batch_id_list.append(temp_batch_id)
            batch_mask_list.append(temp_batch_mask)
            temp_batch_id = []
            temp_batch_mask = []
            counter = 1
        else:
            counter += 1

    return batch_id_list, batch_mask_list, labels


def train_bert(args, train_exs, dev_exs, max_len):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # num_class = 4
    num_class = 6
    num_epochs = 10
    batch_size = 4
    dropout = 0.1
    x = 0
    cross_entropy_loss = nn.CrossEntropyLoss()
    model = BERT(dropout, num_class)
    initial_learning_rate = 4e-6
    best_model = model
    time_not_better = 0
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    model.cuda()
    for epoch in range(0, num_epochs):
        model.train()
        random.shuffle(train_exs)
        batch_id_list, batch_mask_list, labels = batch_bert(train_exs, batch_size, tokenizer, max_len)
        batch_iteration = 0
        total_loss = 0.0
        for one_batch in batch_id_list:
            # Zero out the gradients from the RNN object. *THIS IS VERY IMPORSTANT TO DO BEFORE CALLING BACKWARD()*
            model.zero_grad()
            output = model.forward(torch.tensor(one_batch).long().cuda(), torch.tensor(batch_mask_list[batch_iteration]).long().cuda())
            loss = cross_entropy_loss(output, torch.tensor(labels[batch_iteration]).long().cuda())
            print(loss)
            # Computes the gradient and takes the optimizer step
            total_loss += loss
            loss.backward()
            optimizer.step()
            batch_iteration += 1
        print("average:", total_loss / (len(batch_id_list)))
        model.eval()
        current = print_evaluation(dev_exs, FakeNewsClassifier(model, tokenizer))
        if current > x:
            x = current
            best_model = copy.deepcopy(model)
            time_not_better = 0
            print("***************")
            print(x)
        else:
            time_not_better += 1
            if time_not_better == 3:
                return FakeNewsClassifier(best_model, tokenizer)

    return FakeNewsClassifier(best_model, tokenizer)


class FakeNewsClassifier():
    def __init__(self, bert, tokenizer):
        self.bert = bert
        self.tokenizer = tokenizer

    def predict(self, sentence):
        input_ids, attention_mask = tokenize(sentence, self.tokenizer, 512)
        output = self.bert.forward(torch.tensor([input_ids]).cuda(), torch.tensor([attention_mask]).cuda())
        return torch.argmax(output)


class LSTM(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """

    def __init__(self,  alphabet_size, input_size, hidden_size, dropout, num_class):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param alphabet_size: size of input (integer)
        :param input_size: size of input(integer)
        :param hidden_size: size of hidden layer(integer), which should be the number of classes
        """
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(alphabet_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        self.W = nn.Linear(hidden_size, num_class)
        self.V = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        # print(x.shape)
        embed_x = self.embed(x)
        # print(embed_x.shape)
        output, (hidden_state, cell_state) = self.LSTM(embed_x)

        # print(hidden_state.shape)
        # hidden_state = self.W(hidden_state)
        hidden_state = hidden_state.transpose(0, 1).contiguous()
        hidden_state = torch.mean(hidden_state, dim=1)
        # print(hidden_state.shape)
        # input()
        # print(output.shape)
        # print('h:', hidden_state.shape)
        # print('c:', cell_state.shape)
        # input()
        probs = self.log_softmax(self.W(hidden_state.squeeze()))
        # print(hidden_state.shape)
        # output = self.V(output)
        # print(output.shape)

        # return output, hidden_state, cell_state
        return probs


class RNNClassifier():
    def __init__(self, rnn, vocab_index):
        self.rnn = rnn
        self.vocab_index = vocab_index

    def predict(self, context):
        # example = sentence_to_token(context, self.vocab_index)
        # output, hidden_state, cell_state = self.rnn.forward(torch.tensor(example).unsqueeze(0))
        # hidden_state = hidden_state.squeeze(0)
        # return torch.argmax(hidden_state)
        current_index_list = sentence_to_token(context, self.vocab_index)
        # print(torch.tensor(current_index_list).shape)
        probs = self.rnn.forward(torch.tensor(current_index_list).unsqueeze(0).cuda())
        # print(probs)
        prediction = torch.argmax(probs, dim=-1)
        return prediction


def sentence_to_token(sentence, indexer):
    token = indexer.tokenize(sentence)
    return indexer.convert_tokens_to_ids(token)


def batch(train_exs, batch_size, vocab_indexer):
    temp_batch = []
    batch_list = []
    temp_labels = []
    labels = []
    counter = 1

    for examples in train_exs:
        temp_labels.append(examples.label)
        temp_batch.append(torch.tensor(sentence_to_token(examples.sent, vocab_indexer)))
        if counter == batch_size:
            temp_batch = pad_sequence(temp_batch, batch_first=True, padding_value=0)
            labels.append(temp_labels)
            temp_labels = []
            batch_list.append(temp_batch)
            temp_batch = []
            counter = 1
        else:
            counter += 1

    return batch_list, labels


def train_lstm_classifier(args, train_exs, dev_exs):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :return: an RNNClassifier instance trained on the given data
    """
    num_epochs = 10
    hidden_size = 128
    input_size = 128
    batch_size = 16
    # num_class = 4
    num_class = 6
    x = 0
    time_not_better = 0
    indexer = BertTokenizer.from_pretrained('bert-base-uncased')
    # cross_entropy_loss = nn.CrossEntropyLoss()
    nll_loss = nn.NLLLoss()
    model = LSTM(len(indexer), input_size, hidden_size, 0.1, num_class)
    best_model = model
    initial_learning_rate = .01
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    batch_list, labels = batch(train_exs, batch_size, indexer)
    model.cuda()
    for epoch in range(0, num_epochs):
        model.train()
        c = list(zip(batch_list, labels))
        random.shuffle(c)
        batch_list, labels = zip(*c)
        curr_batch_iteration = 0
        total_loss = 0.0
        for one_batch in batch_list:
            # print("hi")
            # Zero out the gradients from the RNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            model.zero_grad()
            # output, hidden_state, cell_state = model.forward(torch.tensor(one_batch).long())
            # hidden_state = hidden_state.squeeze(0)
            # loss = cross_entropy_loss(hidden_state, torch.tensor(labels[curr_batch_iteration]).long())
            probs = model.forward(one_batch.cuda())
            # print(labels[curr_batch_iteration])
            transfered_labels = torch.tensor(labels[curr_batch_iteration]).long().cuda()
            # print(transfered_labels)
            # print(probs.shape)
            loss = nll_loss(probs, transfered_labels)
            print(loss)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
            curr_batch_iteration += 1
        print("average:", total_loss / (len(batch_list)))
        model.eval()
        current = print_evaluation(dev_exs, RNNClassifier(model, indexer))
        if current > x:
            x = current
            best_model = copy.deepcopy(model)
            time_not_better = 0
            print("***************")
            print(x)
        else:
            time_not_better += 1
            if time_not_better == 3:
                return RNNClassifier(best_model, indexer)
        # print("Total loss on epoch %i: %f" % (epoch, total_loss / len(batch_list)))

    return RNNClassifier(best_model, indexer)
