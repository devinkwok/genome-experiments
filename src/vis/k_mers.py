import pickle
import operator

import numpy as np
import matplotlib.pyplot as plt

import util


class TrieNode:

    def __init__(self, parent, char, count=0):
        self.count = count
        self.parent = parent
        if parent is None:
            self.substr = char
        else:
            self.substr = parent.substr + char
        self.children = {}

    def add_child(self, char):
        if char in self.children:
            self.children[char].count += 1
        else:
            self.children[char] = TrieNode(self, char, count=1)
        return self.children[char]  # returns the added child node
    
    # puts all nodes depth levels further down into node_dict, this mutates node_dict!
    def collect_nodes(self, node_dict, depth):
        if depth == 0:
            node_dict[self.substr] = self.count
        else:
            for _, child in self.children.items():
                child.collect_nodes(node_dict, depth - 1)

    def __repr__(self):
        str_rep = ' (' + str(self.count) + ' ' + self.substr
        for _, child in self.children.items():
            str_rep += str(child) + ','
        return str_rep + ')'


# prefix tree
class Trie:
    
    def __init__(self, max_depth):
        self.max_depth = max_depth  # max length of n-gram to track
        self.endpoints = []  # track where to append next character
        self.longest_index = 0  # index of the end_node to point back to next single char string
        self.root_node = TrieNode(None, '')

    def append_char(self, char):
        # either add new endpoint
        if len(self.endpoints) < self.max_depth:
            self.endpoints.append(self.root_node)
        else:  # or if max_depth reached, reset the longest endpoint
            self.endpoints[self.longest_index] = self.root_node
            self.longest_index += 1
            self.longest_index %= self.max_depth
        # add next char to all endpoints
        for i, node in enumerate(self.endpoints):
            child = node.add_child(char)
            self.endpoints[i] = child
    
    # return an array of n-grams sorted by most frequent occurrences
    # remove n-grams with less than truncate_freq occurrences
    def ordered_sum(self, ngram_len, truncate_freq=0):
        ngrams = {}
        # go through tree and collect all nodes at ngram_len depth
        self.root_node.collect_nodes(ngrams, ngram_len)
        # sort by number of times the n-gram appears
        sorted_ngrams = sorted(ngrams.items(), key=operator.itemgetter(1), reverse=True)
        # linear search because I'm lazy
        for i, (ngram, freq) in enumerate(sorted_ngrams):
            if freq < truncate_freq:
                return sorted_ngrams[0:i]
        return sorted_ngrams

    def __repr__(self):
        return str(self.root_node)

def test_Trie(char_seq, n):
    trie = Trie(n)
    for char in char_seq:
        trie.append_char(char)
    print(trie)
    return trie



VALID_BASES = {b'A', b'T', b'C', b'G'}

def seq_k_mer(filename, max_depth):
    seq = util.read_seq(filename, print_progress=1000)
    k_mer_trie = Trie(max_depth)
    for base in seq:
        if base in VALID_BASES:
            k_mer_trie.append_char(base.decode('ascii'))

    out_filename =  util.output_path('ngrams_', filename, '.pickle')
    with open(out_filename, mode='wb') as file:
        pickle.dump(k_mer_trie, file)
    return k_mer_trie

def plot_ngrams(trie):
    for i in range(1, trie.max_depth):
        ngram = np.array(trie.ordered_sum(i))
        labels = ngram[::-1,0]
        freqs = np.array(ngram[::-1,1], dtype='int32')
        n_total = np.sum(freqs)
        n_unique = len(labels)

        rand_expectation = 1 / 4 ** i  # expected frequencies if ngrams are perfectly random
        semirand_expectation = rand_expectation
        if n_unique > 0:  # expected frequencies if observed ngrams are perfectly random
            semirand_expectation = 1 / n_unique
        print(n_total, 4**i, n_unique, rand_expectation, semirand_expectation)
        if n_unique <= 256:
            fig = plt.figure(figsize=(6, n_unique*0.2 + 2))
            axes = fig.add_axes([0.15,0.1,0.8,0.8])
            axes.barh(labels, (freqs / n_total))
            axes.axvline(x=rand_expectation, ymin=0, ymax=1, color='black')
            axes.axvline(x=semirand_expectation, ymin=0, ymax=1, color='red')
        else:
            plt.plot(freqs, range(n_unique))
            plt.axvline(x=rand_expectation, ymin=0, ymax=1, color='black')
            plt.axvline(x=semirand_expectation, ymin=0, ymax=1, color='red')

        plt.savefig(util.output_path('ngrams_', filename, str(i) + '.png'))
        plt.clf()


filename = 'data/ref_genome/chr22_excerpt.fa'
# filename = 'data/ref_genome/chr22.fa'
trie = seq_k_mer(filename, 10)

with open(util.output_path('ngrams_', filename, '.pickle'), 'rb') as file:
    trie = pickle.load(file)
plot_ngrams(trie)
