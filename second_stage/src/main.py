import json                                                                                                                                                                                                                                                                   
import numpy as np
import time
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import logging
import random
import pickle
import sys

import os

from utils import *
from model import *
from data_loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def main():

    dataset = read_data_json("./data/math23k_final.json")
    emb_vectors = np.load('./data/emb_100.npy')
    #dict_keys(['text', 'ans', 'mid_template', 'num_position', 'post_template', 'num_list', \
    #'template_text', 'expression', 'numtemp_order', 'index', 'gd_tree_list'])
    count = 0
    max_l = 0
    norm_templates = read_data_json("./data/post_dup_templates_num.json")
    for key, elem in dataset.items():
        #print (elem['post_template'])
        #print (norm_templates[key])
        #print ()
        elem['post_template'] = norm_templates[key]
        elem['gd_tree_list'] = form_gdtree(elem)
        if len(elem['gd_tree_list']):
            #print (elem.keys())
            #print (elem['text'])
            #print (elem['mid_template'])
            #print (elem['post_template'])
            #print (elem['post_template'][2:])
            l = len(elem['post_template'])
            if max_l < l:
                max_l = l
            count += 1
    print (max_l)
        #print (elem['gd_tree_list'])
    print (count)

    data_loader = DataLoader(dataset) 
    print ('loading finished')

    if os.path.isfile("./ablation_recursive-Copy1.log"):
        os.remove("./ablation_recursive-Copy1.log")

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='mylog.log', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)


    params = {"batch_size": 32,
              "start_epoch" : 1,
              "n_epoch": 100,
              "rnn_classes":5,
              "save_file": "model_xxx_att.pkl"
             }
    encode_params = {
        "emb_size":  100,
        "hidden_size": 160,
        "input_dropout_p": 0.2,
        "dropout_p": 0.5,
        "n_layers": 2,
        "bidirectional": True,
        "rnn_cell": None,
        "rnn_cell_name": 'lstm',
        "variable_lengths_flag": True
    }

    #dataset = read_data_json("/home/wanglei/aaai_2019/pointer_math_dqn/dataset/source_2/math23k_final.json")
    #emb_vectors = np.load('/home/wanglei/aaai_2019/parsing_for_mwp/data/source_2/emb_100.npy')
    #data_loader = DataLoader(dataset) 
    #print ('loading finished')


    recu_nn = RecursiveNN(data_loader.vocab_len, encode_params['emb_size'], params["rnn_classes"])
    recu_nn = recu_nn.cuda()
    self_att_recu_tree = Self_ATT_RTree(data_loader, encode_params, recu_nn)
    self_att_recu_tree = self_att_recu_tree.cuda()
    #for name, params in self_att_recu_tree.named_children():
    #    print (name, params)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self_att_recu_tree.parameters()), \
                                lr=0.01, momentum=0.9, dampening=0.0)

    trainer = Trainer(data_loader, params)
    trainer.train(self_att_recu_tree, optimizer)


main()
