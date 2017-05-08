import pandas as pd
import numpy as np
import random
import pickle
from keras.utils import np_utils

def getTokens(input):
    # tokensBySlash = str(input.encode('utf-8')).split('/')
    # print(tokensBySlash)
    # tokensBySlash = str(input).split('/')
        allTokens = []
    # for i in range(0, len(tokensBySlash)):
        token = str(input)
        # token = tokensBySlash[i]
        tokens = []
        token = token.replace('.', '/')
        token = token.replace('=', '/')
        token = token.replace('&', '/')
        token = token.replace('?', '/')
        token = token.replace('-', '/')
        token = token.replace('@', '/')
        token = token.replace(':', '/')
        tokens = token.split('/')
        allTokens = allTokens + tokens
        # allTokens = list(set(allTokens))  # remove redundant tokens
        return allTokens


def load_data_and_labels(path):
    # allurls = '/Users/zcw/Documents/python/DetectMaliciousURL/data/data.csv'	#path to our all urls file
    allurlscsv = pd.read_csv(path, ',', error_bad_lines=False)  # reading file
    allurlsdata = pd.DataFrame(allurlscsv)  # converting to a dataframe
    allurlsdata = np.array(allurlsdata)  # converting it into an array
    random.shuffle(allurlsdata)  # shuffling
    y = [d[1] for d in allurlsdata]  # all labels
    x = [d[0] for d in allurlsdata]  # all urls corresponding to a label (either good or bad)
    for i in range(0,len(y)):
        if y[i] =='bad':
            y[i]=0
        else:
            y[i]=1
    label = np_utils.to_categorical(y, 2)
    return (x, label)

def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    sentences = [getTokens(sentence) for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
            [len(sentence) for sentence in sentences])
    i=0
    all_vector=[]
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
        all_vector.append(sentence)
    return (all_vector, max_sentence_length)


def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f)


def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generate a batch iterator for a dataset
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx: end_idx]


if __name__ == "__main__":
    test = "rapiseebrains.com/?a=401336&c=cpc&s=050217"
    ans = getTokens(test)
    print ans, len(ans)
