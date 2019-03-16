import os
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import os
from random import uniform
import time
import random

def read_sequences(file):
    #Open the file
    f = open ( file , 'r')
    #Read all the lines
    lines = f.read().splitlines()
    #Initialize array to store sequences
    pos_sequences = []
    for line in lines:
        n = len(line)
        pos_sequences.append([line,1, line.count('A')/n, line.count('T')/n, line.count('G')/n, line.count('C')/n ])
    return pos_sequences

def reverse_complement(sequences):
    conversion = {"A":"T","T":"A","C":"G","G":"C"}
    rev_sequences = []
    for pair in sequences:
        rev_sequence = ''
        for base in pair[0]:
            rev_sequence = conversion[base] + rev_sequence
        n = len(rev_sequence)
        rev_sequences.append([rev_sequence, pair[1], rev_sequence.count('A')/n, rev_sequence.count('T')/n, rev_sequence.count('G')/n, rev_sequence.count('C')/n ])
    sequences = sequences + rev_sequences
    return sequences
def read_neg_sequences(file):
    #Open the file
    f = open ( file , 'r')
    #Read all the lines
    lines = f.read().splitlines()
    neg_sequences = []
    sequence = ''
    for line in lines[1:]:
        if line[0] == '>':
            sub_seq = []
            for i in range(1000-17):
                sub_seq.append(sequence[i:i+17])
            neg_sequences.extend(sub_seq)
            sequence = ''
        else:
            sequence = sequence + line
    return neg_sequences
def Diff(li1, li2):
    return list(set(li1) - set(li2))


def convert_to_numeric(sequences):
    base_2_vec = {"A":[1,0,0,0],"T":[0,1,0,0], "C":[0,0,1,0], "G":[0,0,0,1]}
    numeric_input = []
    for pair in sequences:
        input = []
        for j in pair[0]:
            input.extend(base_2_vec[j]) #= [base_2_vec[j] for j in pair[0]]
        numeric_input.append([np.array([input]).T, float(pair[1])])
    return numeric_input
