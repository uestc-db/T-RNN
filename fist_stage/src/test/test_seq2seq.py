import sys
sys.path.append('/home/demolwang/demolwang/math_word_problem/critical-based/seq2seq_v2/src')
from model import EncoderRNN, DecoderRNN_1, Seq2seq
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb


embed_model = nn.Embedding(1000, 100)

encode_model = EncoderRNN(1000, embed_model, 100, 128, 0, 0, 4, True, None, 'lstm', True)

decode_model = DecoderRNN_1(1000, 10, embed_model, 100, 256, 3, None, 'gru', 1, 0, 0, 0)

seq2seq = Seq2seq(encode_model, decode_model)

input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
target = Variable(torch.LongTensor([[4,3,2], [11,3,4]]))

lengths = [4,4]

dol, dh, ssl = seq2seq(input, lengths, target, 0, 3)
pdb.set_trace()
pass



