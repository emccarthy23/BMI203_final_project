from .io import read_sequences, reverse_complement, read_neg_sequences, Diff, convert_to_numeric
import neural_net
import numpy as np
import random
import matplotlib.pyplot as plt
import time

#Not testing write_alignment
def test_read_sequences():
    pairs = io.read_sequences('data/rap1-lieb-positives.txt')
    assert pairs[0][0] == 'ACATCCGTGCACCTCCG'


def test_reverse_complement():
    pairs = io.read_sequences('data/rap1-lieb-positives.txt')
    pairs = io.reverse_complement(pairs)
    assert pairs[137][0] == 'CGGAGGTGCACGGATGT'


def test_neural_net_seq_output():
    pos_pairs = io.read_sequences('data/rap1-lieb-positives.txt')
    pos_pairs = io.reverse_complement(pairs)
    pos_input = io.convert_to_numeric(pos_pairs)
    neg_pairs = np.load("neg_sequences.npy")
    neg_input = io.convert_to_numeric(neg_pairs)
    training_input = pos_input + neg_input

    net = neuralnet.Network([68,34,1])
    net.SGD(training_input,300,10,1)
    output = net.output(neg_pairs)
    assert output[0][0] == "CCGCCCATGTCTACCAG"

def test_8_3_8():
    test_data = [[np.array([[1,0,0,0,0,0,0,0]]).T,np.array([[1,0,0,0,0,0,0,0]]).T],
                 [np.array([[0,1,0,0,0,0,0,0]]).T,np.array([[0,1,0,0,0,0,0,0]]).T],
                [np.array([[0,0,1,0,0,0,0,0]]).T,np.array([[0,0,1,0,0,0,0,0]]).T],
                [np.array([[0,0,0,1,0,0,0,0]]).T,np.array([[0,0,0,1,0,0,0,0]]).T],
                [np.array([[0,0,0,0,1,0,0,0]]).T,np.array([[0,0,0,0,1,0,0,0]]).T],
                [np.array([[0,0,0,0,0,1,0,0]]).T,np.array([[0,0,0,0,0,1,0,0]]).T],
                [np.array([[0,0,0,0,0,0,1,0]]).T,np.array([[0,0,0,0,0,0,1,0]]).T],
                [np.array([[0,0,0,0,0,0,0,1]]).T,np.array([[0,0,0,0,0,0,0,1]]).T]]
    unsorted_test = test_data
    net_test.SGD(test_data,300,1,1)
    output = net_test.evaluate_8_3_8(unsorted_test)
    assert output[0][0] ==[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
