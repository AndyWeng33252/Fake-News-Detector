import csv
class FakeNewsExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label, sent):
        self.words = words
        self.label = label
        self.sent = sent

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()


def read_fakenews_examples(infile: str, restrict=False, max_len=0) -> [FakeNewsExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    NOTE: Compared to Assignment 1, we lowercase the data for you. This is because the GloVe embeddings don't
    distinguish case and so can only be used with lowercasing.

    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """
    f = open(infile)
    exs = []
    for line in f:
        if len(line.lower().strip()) > 0:
            fields = line.lower().split("\t")
            if len(fields) != 2:
                fields = line.split()
                if "1" in fields[0]:
                    label = 1
                elif "2" in fields[0]:
                    label = 2
                elif "3" in fields[0]:
                    label = 3
                else:
                    label = 4
                sent = " ".join(fields[1:])
            else:
                # Slightly more robust to reading bad output than int(fields[0])
                if "1" in fields[0]:
                    label = 1
                elif "2" in fields[0]:
                    label = 2
                elif "3" in fields[0]:
                    label = 3
                else:
                    label = 4
                sent = fields[1]

            tokenized_cleaned_sent = list(filter(lambda x: x != '', sent.rstrip().split(" ")))

            if restrict:
                if len(tokenized_cleaned_sent) <= max_len:
                    exs.append(FakeNewsExample(tokenized_cleaned_sent, label, sent))
            else:
                exs.append(FakeNewsExample(tokenized_cleaned_sent, label, sent))
    f.close()
    return exs


def read_fakenews_examples_csv(infile: str, restrict=False, max_len=0) -> [FakeNewsExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    NOTE: Compared to Assignment 1, we lowercase the data for you. This is because the GloVe embeddings don't
    distinguish case and so can only be used with lowercasing.

    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """
    f = open(infile)
    exs = []
    csv_reader = csv.reader(f, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            sent = row[1].lower()
            label = int(row[2])
            line_count += 1
            tokenized_cleaned_sent = list(filter(lambda x: x != '', sent.rstrip().split(" ")))
            if restrict:
                if len(tokenized_cleaned_sent) <= max_len:
                    exs.append(FakeNewsExample(tokenized_cleaned_sent, label, sent))
            else:
                exs.append(FakeNewsExample(tokenized_cleaned_sent, label, sent))
    f.close()
    return exs
