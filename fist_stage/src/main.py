from config import *
import pdb
import os
import logging
import torch
from torch.autograd import Variable
import torch.nn as nn

from utils import DataLoader
from train import SupervisedTrainer
from model import EncoderRNN, DecoderRNN_1, DecoderRNN_2, DecoderRNN_3, Seq2seq
from utils import NLLLoss, Optimizer, Checkpoint, Evaluator

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id) 


def step_one():

    if args.mode == 0:
        encoder_cell = 'lstm'
        decoder_cell = 'lstm'
    elif args.mode == 1:
        encoder_cell = 'gru'
        decoder_cell = 'gru'
    elif args.mode == 2:
        encoder_cell = 'gru'
        decoder_cell = 'lstm'
    else:
        encoder_cell = 'lstm'
        decoder_cell = 'gru'

    data_loader = DataLoader()
    embed_model = nn.Embedding(data_loader.vocab_len, 128)
    embed_model.weight.data.copy_(torch.from_numpy(data_loader.word2vec.emb_vectors))
    encode_model = EncoderRNN(vocab_size = data_loader.vocab_len,
                              embed_model = embed_model,
                              emb_size = 128,
                              hidden_size = args.encoder_hidden_size,
                              input_dropout_p = float(args.input_dropout),
                              dropout_p = float(args.dropout_p),
                              n_layers = int(args.layer),
                              bidirectional = True,
                              rnn_cell = None,
                              rnn_cell_name = encoder_cell,
                              variable_lengths = True)
    decode_model = DecoderRNN_3(vocab_size = data_loader.vocab_len,
                                class_size = data_loader.classes_len,
                                embed_model = embed_model,
                                emb_size = 128,
                                hidden_size = args.decoder_hidden_size,
                                n_layers = int(args.layer),
                                rnn_cell = None,
                                rnn_cell_name=decoder_cell,
                                sos_id = data_loader.vocab_dict['END_token'],
                                eos_id = data_loader.vocab_dict['END_token'],
                                input_dropout_p = args.input_dropout,
                                dropout_p = args.dropout_p)
    seq2seq = Seq2seq(encode_model, decode_model)

    if args.cuda_use:
        seq2seq = seq2seq.cuda()

    weight = torch.ones(data_loader.classes_len)
    pad = data_loader.decode_classes_dict['PAD_token']
    loss = NLLLoss(weight, pad)

    st = SupervisedTrainer(vocab_dict = data_loader.vocab_dict,
                           vocab_list = data_loader.vocab_list,
                           decode_classes_dict = data_loader.decode_classes_dict,
                           decode_classes_list = data_loader.decode_classes_list,
                           cuda_use = args.cuda_use,
                           loss = loss,
                           print_every = 10,
                           teacher_schedule = False,
                           checkpoint_dir_name = args.checkpoint_dir_name)


    print 'start training'
    st.train(model = seq2seq, 
             data_loader = data_loader,
             batch_size = 64,
             n_epoch = 100,
             template_flag = True,
             resume = args.resume,
             optimizer = None,
             mode = args.mode,
             teacher_forcing_ratio=args.teacher_forcing_ratio)

def step_one_test():

    data_loader = DataLoader()
    #print '---', data_loader.math23k_test_list

    #Checkpoint.CHECKPOINT_DIR_NAME = "0120_0030"
    Checkpoint.CHECKPOINT_DIR_NAME = args.checkpoint_dir_name
    checkpoint_path = os.path.join("./experiment", Checkpoint.CHECKPOINT_DIR_NAME, "best_0.69")
    checkpoint = Checkpoint.load(checkpoint_path)

    seq2seq = checkpoint.model
    if args.cuda_use:
        seq2seq = seq2seq.cuda()

    seq2seq.eval()
    evaluator = Evaluator(vocab_dict = data_loader.vocab_dict,
                          vocab_list = data_loader.vocab_list,
                          decode_classes_dict = data_loader.decode_classes_dict,
                          decode_classes_list = data_loader.decode_classes_list,
                          loss = NLLLoss(),
                          cuda_use = args.cuda_use)
    test_temp_acc, test_ans_acc = evaluator.evaluate(model = seq2seq,
                                                     data_loader = data_loader,
                                                     data_list = data_loader.math23k_test_list,
                                                     template_flag = True,
                                                     batch_size = 64,
                                                     evaluate_type = 0,
                                                     use_rule = False,
                                                     mode = args.mode)
    print test_temp_acc, test_ans_acc

def step_three():

    data_loader = DataLoader()

    Checkpoint.CHECKPOINT_DIR_NAME = args.checkpoint_dir_name
    checkpoint_path = os.path.join("./experiment", Checkpoint.CHECKPOINT_DIR_NAME, "best")
    checkpoint = Checkpoint.load(checkpoint_path)

    seq2seq = checkpoint.model
    if args.cuda_use:
        seq2seq = seq2seq.cuda()

    seq2seq.eval()
    evaluator = Evaluator(vocab_dict = data_loader.vocab_dict,
                          vocab_list = data_loader.vocab_list,
                          decode_classes_dict = data_loader.decode_classes_dict,
                          decode_classes_list = data_loader.decode_classes_list,
                          loss = NLLLoss(),
                          cuda_use = args.cuda_use)
    test_temp_acc, test_ans_acc = evaluator.evaluate(model = seq2seq,
                                                     data_loader = data_loader,
                                                     data_list = data_loader.math57k_data_list,
                                                     template_flag = False,
                                                     batch_size = 64,
                                                     evaluate_type = 0,
                                                     use_rule = True,
                                                     mode = args.mode)
    print test_temp_acc, test_ans_acc

if __name__ == "__main__":
    if args.run_flag == 'test_23k':
        step_one_test()
    elif args.run_flag == 'test_57k':
        step_three()
    elif args.run_flag == 'train_23k':
        step_one()
    else:
        print 'emmmm..................'
