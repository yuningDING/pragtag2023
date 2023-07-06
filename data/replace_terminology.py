from wordfreq import word_frequency
import os
from tqdm import tqdm
import csv
import string
from nltk.corpus import wordnet as wn
#from nltk.stem import PorterStemmer
#nltk.download("punkt")


def get_all_tokens_and_corpus_freq(directory):
    token_list = []
    tokens = set()
    #ps = PorterStemmer()
    for f in tqdm(list(os.listdir(directory))):
        tokens_in_text = open(directory + '/' + f, 'r', encoding='utf-8').read().strip().split()
        for t in tokens_in_text:
            t = t.lower().strip(string.punctuation)
            tokens.add(t)
            token_list.append(t)
    frequency = {word: token_list.count(word) for word in tokens}
    return tokens, dict(sorted(frequency.items(), key=lambda x:x[1], reverse=True))


def get_common_token_freq(token_list):
    token_freq = {}
    for t in token_list:
        token_freq[t] = word_frequency(t, 'en')
    return dict(sorted(token_freq.items(), key=lambda x:x[1], reverse=True))


# find the frequency of the last adjective
def get_threshold(token_freq):
    last_adj = ''
    for t in token_freq.keys():
        synset = wn.synsets(t)
        if len(synset) == 0:
            continue
        pos = synset[0].pos()
        if pos == 'v':
            last_adj = t
    print("Adjective with the lowest frequency found: " + last_adj + " : " + str(token_freq.get(last_adj)))
    return token_freq.get(last_adj)


if __name__ == '__main__':
    token_set, corpus_freq = get_all_tokens_and_corpus_freq("all")
    # with open('corpus_token_frequency.csv', 'w', encoding='utf-8', newline='') as f:
    #     w = csv.writer(f)
    #     w.writerows(corpus_freq.items())
    # threshold_freq = get_threshold(corpus_freq)

    token_freq = get_common_token_freq(token_set)
    # with open('common_token_frequency.csv', 'w', encoding='utf-8', newline='') as f:
    #     w = csv.writer(f)
    #     w.writerows(token_freq.items())
    # threshold_freq = get_threshold(token_freq)

    threshold_freq = 1.05e-06

    keep_words = []
    replace_words = []
    for t in token_set:
        if corpus_freq.get(t)>1 or token_freq.get(t)>threshold_freq:
            keep_words.append(t)
        else:
            replace_words.append(t)

    with open("keep_words.txt", 'w', encoding='utf-8') as output:
        for w in keep_words:
            output.write(str(w) + '\n')
    with open("replace_words.txt", 'w', encoding='utf-8') as output:
        for w in replace_words:
            output.write(str(w) + '\n')

