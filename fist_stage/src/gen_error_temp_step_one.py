from config import *
import numpy as np
import pdb
import os
import logging
import torch
from torch.autograd import Variable
import torch.nn as nn

from utils import DataLoader
from train import SupervisedTrainer
from model import EncoderRNN, DecoderRNN_1, Seq2seq
from utils import NLLLoss, Optimizer, Checkpoint, Evaluator

args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def gen_math23k_error():
    data_loader = DataLoader()
    Checkpoint.CHECKPOINT_DIR_NAME = args.checkpoint_dir_name
    checkpoint_path = os.path.join("./experiment", Checkpoint.CHECKPOINT_DIR_NAME, args.load_name)
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

    evaluator.gen_rl_data(model = seq2seq,
                          data_loader = data_loader,
                          data_list = data_loader.math23k_train_list,
                          template_flag = True,
                          batch_size = 16,
                          evaluate_type = 0,
                          use_rule = False,
                          mode = args.mode,
                          filename = args.load_name)

def gen_math57k_error():
    data_loader = DataLoader()
    Checkpoint.CHECKPOINT_DIR_NAME = args.checkpoint_dir_name
    checkpoint_path = os.path.join("./experiment", Checkpoint.CHECKPOINT_DIR_NAME, args.load_name)
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

    evaluator.gen_rl_data(model = seq2seq,
                          data_loader = data_loader,
                          data_list = data_loader.math57k_data_list,
                          template_flag = False,
                          batch_size = 16,
                          evaluate_type = 0,
                          use_rule = True,
                          mode = args.mode,
                          filename = args.load_name)


def gen_best_23_error():
    data_loader = DataLoader()
    Checkpoint.CHECKPOINT_DIR_NAME = args.checkpoint_dir_name
    checkpoint_path = os.path.join("./experiment", Checkpoint.CHECKPOINT_DIR_NAME, 'best')
    checkpoint = Checkpoint.load(checkpoint_path)

    seq2seq = checkpoint.model
    if args.cuda_use:
        seq2seq = seq2seq.cuda()

    seq2seq.eval()

    emb_model = seq2seq.encoder.embedding
    emb_np = emb_model.weight.cpu().data.numpy()
    np.save("./data/rl_train_data/emb.npy", emb_np)

    evaluator = Evaluator(vocab_dict = data_loader.vocab_dict,
                          vocab_list = data_loader.vocab_list,
                          decode_classes_dict = data_loader.decode_classes_dict,
                          decode_classes_list = data_loader.decode_classes_list,
                          loss = NLLLoss(),
                          cuda_use = args.cuda_use)

    evaluator.gen_rl_data(model = seq2seq,
                          data_loader = data_loader,
                          data_list = data_loader.math23k_train_list,
                          template_flag = False,
                          batch_size = 16,
                          evaluate_type = 0,
                          use_rule = False,
                          mode = args.mode,
                          filename = args.load_name)



if __name__=="__main__":
    if args.load_name == 'best':
        print 'unlabel57k'
        gen_math57k_error() 
    elif args.load_name == 'best_23':
        print 'best 23k'
        gen_best_23_error()
    else:
        print 'math23k'
        gen_math23k_error()

