import sys
sys.path.append('/home/demolwang/demolwang/math_word_problem/critical-based/seq2seq_v2/src')
from model import EncoderRNN
import torch
from torch.autograd import Variable
import torch.nn as nn
import unittest


class TestEncoderRNN(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.vocab_size = 100
        self.input_var = Variable(torch.randperm(self.vocab_size).view(20, 5))
        self.lengths = [5] * 20

    def test_input_dropout_WITH_PROB_ZERO(self):
        rnn = EncoderRNN(self.vocab_size, None, 50, 16, input_dropout_p=0, n_layers=3,\
                         bidirectional=True, rnn_cell_name='lstm')
        print rnn
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        output1, _ = rnn(self.input_var, self.lengths)
        if isinstance(_, tuple):
            #print 'outputs', [elem.size() for elem in output1]
            print 'outputs', output1.size()
            print 'hidden', [elem.size() for elem in _]
        else:
            print 'outputs', output1.size()
            print 'hidden', _.size()
        output2, _ = rnn(self.input_var, self.lengths)
        self.assertTrue(torch.equal(output1.data, output2.data))


