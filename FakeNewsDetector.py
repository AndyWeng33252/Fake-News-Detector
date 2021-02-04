# lm_classifier.py
import torch
import argparse
import csv
import time
from fake_news_data import *
from Model import*


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='FakeNewsDetector.py')
    parser.add_argument('--model', type=str, default='BERT', help='model to run (BERT or LSTM)')
    parser.add_argument('--train', type=str, default='train.csv', help='path to train examples')
    parser.add_argument('--dev', type=str, default='dev.csv', help='path to dev examples')
    parser.add_argument('--output', type=str, default='output.csv', help='path to output file')
    args = parser.parse_args()
    return args


def print_evaluation(dev, model, output_file=None):
    """
    Runs the classifier on the given text
    :param text:
    :param lm:
    :return:
    """
    if output_file is not None:
        fout = open(output_file, 'w')
        csv_writer = csv.writer(fout, delimiter='\t')

    num_correct = 0
    for ex in dev:
        prediction = model.predict(ex.sent)
        if output_file is not None:
            if prediction == ex.label:
                csv_writer.writerow([ex.sent, int(prediction.cpu().numpy()), ex.label, "YES"])
            else:
                csv_writer.writerow([ex.sent, int(prediction.cpu().numpy()), ex.label, "NO"])
        if prediction == ex.label:
            num_correct += 1

    num_total = len(dev)
    data = {'correct': num_correct, 'total': num_total, 'accuracy': float(num_correct)/num_total * 100.0}

    print("=====Results=====")
    print(data)


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()

    # dev_exs = read_fakenews_examples(args.dev)
    dev_exs = read_fakenews_examples_csv(args.dev)
    system_to_run = args.model
    # Train our model
    if system_to_run == "BERT":
        BERT_MAX_LEN = 512
        # train_exs = read_fakenews_examples(args.train, restrict=True, max_len=BERT_MAX_LEN)
        train_exs = read_fakenews_examples_csv(args.train, restrict=True, max_len=BERT_MAX_LEN)
        model = train_bert(args, train_exs, dev_exs, BERT_MAX_LEN)

    elif system_to_run == "LSTM":
        LSTM_MAX_LEN = 2048
        # train_exs = read_fakenews_examples(args.train, restrict=True, max_len=LSTM_MAX_LEN)
        train_exs = read_fakenews_examples_csv(args.train, restrict=True, max_len=LSTM_MAX_LEN)
        model = train_lstm_classifier(args, train_exs, dev_exs)
    else:
        raise Exception("Pass in either BERT or LSTM to run the appropriate system")
    print_evaluation(dev_exs, model, output_file=args.output)
