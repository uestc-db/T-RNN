import json                                                                                                                                                                                                                                                                   
import numpy as np
import time

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

from utils import *

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, input_dropout_p, dropout_p, \
                          n_layers, rnn_cell_name):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn_cell_name = rnn_cell_name
        if rnn_cell_name.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_name.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_name))
        self.dropout_p = dropout_p
 
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, embed_model, emb_size=100, hidden_size=128, \
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
                 rnn_cell=None, rnn_cell_name='gru', variable_lengths_flag=True):
        super(EncoderRNN, self).__init__(vocab_size, emb_size, hidden_size,
              input_dropout_p, dropout_p, n_layers, rnn_cell_name)
        self.variable_lengths_flag = variable_lengths_flag
        self.bidirectional = bidirectional
        self.embedding = embed_model
        if rnn_cell == None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            self.rnn = rnn_cell
 
    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        #pdb.set_trace()
        if self.variable_lengths_flag:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths_flag:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

class Attention_1(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attention_1, self).__init__()
        self.linear_out = nn.Linear(input_size, output_size)
        #self.mask = Parameter(torch.ones(1), requires_grad=False)
         
    #def set_mask(self, batch_size, input_length, num_pos):
    #    self.mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
    #    for mask_i in range(batch_size):
    #        self.mask[mask_i][num_pos[mask_i]] = 1 
    def init_mask(self, size_0, size_1, input_length):
        mask = Parameter(torch.ones(1), requires_grad=False)
        mask = mask.repeat(size_1).unsqueeze(0).repeat(size_0, 1)
        #for i in range(input_length)
        input_index = list(range(input_length))
        for i in range(size_0):
            mask[i][input_index] = 0
        #print (mask)
        mask = mask.byte()
        mask = mask.cuda()
        return mask
            
        
         
    def _forward(self, output, context, input_lengths, mask):
        '''
        output: len x hidden_size
        context: num_len x hidden_size
        input_lengths: torch scalar
        '''
        #print (output.size()) torch.Size([5, 256])
        #print (.size()) torch.Size([80, 256])
        #print (input_lengths)
        attn = torch.matmul(output, context.transpose(1,0))
        #print (attn.size()) 0 x 1
        attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn, dim=1)
        #print (attn)
        mix = torch.matmul(attn, context)
        #print ("mix:", mix)
        #print ("output:", output)
        combined = torch.cat((mix, output), dim=1)
        #print ("combined:",combined)
        output = F.tanh(self.linear_out(combined))
        
        #print ("output:",output)
        #print ("------------")
        #print ()
        return output, attn
        
        
        
    
    def forward(self, output, context, num_pos, input_lengths):
        '''
        output: decoder,  (batch, 1, hiddem_dim2)
        context: from encoder, (batch, n, hidden_dim1)
        actually, dim2 == dim1, otherwise cannot do matrix multiplication 
         
        '''
        batch_size = output.size(0)
        hidden_size = output.size(2)                                                                                                                                                                                                                                          
        input_size = context.size(1)
        #print ('att:', hidden_size, input_size)
        #print ("context", context.size())
        
        attn_list = []
        mask_list = []
        output_list = []
        for b_i in range(batch_size):
            per_output = output[b_i]
            per_num_pos = num_pos[b_i]
            current_output = per_output[per_num_pos]
            per_mask = self.init_mask(len(per_num_pos), input_size, input_lengths[b_i])
            mask_list.append(per_mask)
            #print ("current_context:", current_context.size())
            per_output, per_attn = self._forward(current_output, context[b_i], input_lengths[b_i], per_mask)
            #for p_j in range(len(per_num_pos)):
            #    current_context = per_context[per_num_pos[p_j]]
            #    print ("c_context:", current_context.size())
            output_list.append(per_output)
            attn_list.append(per_attn)
            
            
        
        #self.set_mask(batch_size, input_size, num_pos)
        # (b, o, dim) * (b, dim, i) -> (b, o, i)
        '''
        attn = torch.bmm(output, context.transpose(1,2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
         
        # (b, o, i) * (b, i, dim) -> (b, o, dim)
        mix = torch.bmm(attn, context)
         
        combined = torch.cat((mix, output), dim=2)
         
        #output = F.tanh(self.linear_out(combined.view(-1, 2*hidden_size)))\
                            .view(batch_size, -1, hidden_size)
         
        # output: (b, o, dim)
        # attn  : (b, o, i)
        #return output, attn
        '''
        return output_list, attn_list

class RecursiveNN(nn.Module):
    def __init__(self, vocabSize, embedSize=100, numClasses=5):
        super(RecursiveNN, self).__init__()
        #self.embedding = nn.Embedding(int(vocabSize), embedSize)
        #self.self-att -> embedding 
        #self.att 
        #self.self_att_model = self_att_model
        self.W = nn.Linear(2*embedSize, embedSize, bias=True)
        self.projection = nn.Linear(embedSize, numClasses, bias=True)
        self.activation = F.relu
        self.nodeProbList = []
        self.labelList = []
        self.classes = ['+','-','*','/','^']
    
    def leaf_emb(self, node, num_embed, look_up):
        if node.is_leaf:
            node.node_emb = num_embed[look_up.index(node.root_value)]
        else:
            self.leaf_emb(node.left_tree, num_embed, look_up)
            self.leaf_emb(node.right_tree, num_embed, look_up)
    
    def traverse(self, node):
        if node.is_leaf:
            currentNode = node.node_emb.unsqueeze(0)
        else:
            #currentNode = self.activation(self.W(torch.cat((self.traverse(node.left_tree),self.traverse(node.right_tree)),1)))
            left_vector = self.traverse(node.left_tree)#.unsqueeze(0)
            right_vector = self.traverse(node.right_tree)#.unsqueeze(0)
            #print (left_vector)
            combined_v = torch.cat((left_vector, right_vector),1)
            currentNode = self.activation(self.W(combined_v))
            node.node_emb = currentNode.squeeze(0)
            assert node.is_leaf==False, "error is leaf"
            #self.nodeProbList.append(self.projection(currentNode))
            proj_probs = self.projection(currentNode)
            self.nodeProbList.append(proj_probs)
            #node.numclass_probs = proj_probs 
            self.labelList.append(self.classes.index(node.root_value))
        return currentNode
    
    def forward(self, tree_node, num_embed, look_up):
        
        self.nodeProbList = []
        self.labelList = []
        self.leaf_emb(tree_node, num_embed, look_up)
        self.traverse(tree_node)
        self.labelList = torch.LongTensor(self.labelList)
        
        self.labelList = self.labelList.cuda()
        #print (torch.cat(self.nodeProbList).size())
        #print (self.labelList)
        return torch.cat(self.nodeProbList)#, tree_node
    
    def getLoss_train(self, tree_node, num_embed, look_up):
        nodes = self.forward(tree_node, num_embed, look_up)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labelList)
        #print (self.labelList.size())
        #print (predictions.size())
        #print ()
        acc_elemwise, acc_t = self.compute_acc_elemwise(predictions, self.labelList)
        acc_integrate = self.compute_acc_integrate(predictions, self.labelList)
        return predictions, loss, acc_elemwise, acc_t, acc_integrate#, tree_node

    def test_forward(self, tree_node, num_embed, look_up):
        nodes = self.forward(tree_node, num_embed, look_up)
        predictions = nodes.max(dim=1)[1]
        acc_elemwise, acc_t = self.compute_acc_elemwise(predictions, self.labelList)
        acc_integrate = self.compute_acc_integrate(predictions, self.labelList)
        return predictions, acc_elemwise, acc_t, acc_integrate#, tree_node

    def predict_traverse(self, node):
        if node.is_leaf:
            currentNode = node.node_emb.unsqueeze(0)
        else:
            #currentNode = self.activation(self.W(torch.cat((self.traverse(node.left_tree),self.traverse(node.right_tree)),1)))
            left_vector = self.predict_traverse(node.left_tree)#.unsqueeze(0)
            right_vector = self.predict_traverse(node.right_tree)#.unsqueeze(0)
            #print (left_vector)
            combined_v = torch.cat((left_vector, right_vector),1)
            currentNode = self.activation(self.W(combined_v))
            node.node_emb = currentNode.squeeze(0)
            assert node.is_leaf==False, "error is leaf"
            #self.nodeProbList.append(self.projection(currentNode))
            proj_probs = self.projection(currentNode)
            node_id = proj_probs.max(dim=1)[1]
            node_marker = self.classes[node_id]
            node.root_value = node_marker
            #print ('_++_', node_marker)
            #print ("_++_", proj_probs.size())
            #self.nodeProbList.append(proj_probs)
            #node.numclass_probs = proj_probs 
            #self.labelList.append(self.classes.index(node.root_value))
        return currentNode
    
    def predict(self, tree_node, num_embed, look_up):#, num_list, gold_ans):
        self.leaf_emb(tree_node, num_embed, look_up)
        self.predict_traverse(tree_node)
        post_equ = post_order(tree_node)
        #print ('tst:', post_equ)
        #pred_ans = post_solver(post_equ)
        
        return tree_node, post_equ#, pred_ans

    def compute_acc_elemwise(self, pred_tensor, label_tensor):
        return torch.sum((pred_tensor == label_tensor).int()).item() , len(pred_tensor)
        
    def compute_acc_integrate(self, pred_tensor, label_tensor):
        return 1 if torch.equal(pred_tensor, label_tensor) else 0
    
    def evaluate(self, trees):
        n = nAll = correctRoot = correctAll = 0.0
        return correctRoot / n, correctAll/nAll
    
    def forward_one_layer(self, left_node, right_node):
        left_vector = left_node.node_emb.unsqueeze(0)
        right_vector = right_node.node_emb.unsqueeze(0)
        combined_v = torch.cat((left_vector, right_vector),1)
        currentNode = self.activation(self.W(combined_v))
        root_node = BinaryTree()
        root_node.is_leaf = False
        root_node.node_emb = currentNode.squeeze(0)
        proj_probs = self.projection(currentNode)
        #print ('recur:', proj_probs)
        #print ('r_m',proj_probs.max(1)[1][0].item())
        pred_idx = proj_probs.max(1)[1][0].item()
        root_node.root_value = self.classes[pred_idx]
        root_node.left_tree = left_node
        root_node.right_tree = right_node
        return root_node, proj_probs

class Self_ATT_RTree(nn.Module):
    def __init__(self, data_loader, encode_params, RecursiveNN):
        super(Self_ATT_RTree, self).__init__()
        self.data_loader = data_loader
        self.encode_params = encode_params
        self.embed_model =  nn.Embedding(data_loader.vocab_len, encode_params['emb_size'])
        self.embed_model = self.embed_model.cuda()
        self.encoder = EncoderRNN(vocab_size = data_loader.vocab_len,
                              embed_model = self.embed_model,
                              emb_size = encode_params['emb_size'],
                              hidden_size = encode_params['hidden_size'],
                              input_dropout_p = encode_params['input_dropout_p'],
                              dropout_p = encode_params['dropout_p'],
                              n_layers = encode_params['n_layers'],
                              bidirectional = encode_params['bidirectional'],
                              rnn_cell = encode_params['rnn_cell'],
                              rnn_cell_name = encode_params['rnn_cell_name'],
                              variable_lengths_flag = encode_params['variable_lengths_flag'])
        if encode_params['bidirectional'] == True:
            self.self_attention = Attention_1(encode_params['hidden_size']*4, encode_params['emb_size'])
        else:
            self.self_attention = Attention_1(encode_params['hidden_size']*2, encode_params['emb_size'])
        
        if encode_params['bidirectional'] == True:
            decoder_hidden_size = encode_params['hidden_size']*2
        
        self.recur_nn = RecursiveNN
        
        self._prepare_for_recur()
        self._prepare_for_pointer()
        
    def _prepare_for_recur(self):
        self.fixed_num_symbol = ['1', 'PI']
        self.fixed_num_idx = [self.data_loader.vocab_dict[elem] for elem in self.fixed_num_symbol]
        self.fixed_num = torch.LongTensor(self.fixed_num_idx)

        self.fixed_num = self.fixed_num.cuda()
        
        self.fixed_num_emb = self.embed_model(self.fixed_num)

    def _prepare_for_pointer(self):
        self.fixed_p_num_symbol = ['EOS','1', 'PI']
        self.fixed_p_num_idx = [self.data_loader.vocab_dict[elem] for elem in self.fixed_p_num_symbol]
        self.fixed_p_num = torch.LongTensor(self.fixed_p_num_idx)

        self.fixed_p_num = self.fixed_p_num.cuda()
        
        self.fixed_p_num_emb = self.embed_model(self.fixed_p_num)

    def forward(self, input_tensor, input_lengths, num_pos, b_gd_tree):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)
        en_output_list, en_attn_list = self.self_attention(encoder_outputs, encoder_outputs, num_pos, input_lengths)
        
        batch_size = len(en_output_list)
        
        batch_predictions = []
        batch_acc_e_list = []
        batch_acc_e_t_list = []
        batch_acc_i_list = []
        batch_loss_l = torch.FloatTensor([0])[0].cuda()
        batch_count = 0
        for b_i in range(batch_size):
            #en_output = en_output_list[b_i]
            en_output = torch.cat([self.fixed_num_emb, en_output_list[b_i]], dim=0)
            #print (num_pos)
            look_up = self.fixed_num_symbol + ['temp_'+str(temp_i) for temp_i in range(len(num_pos[b_i]))]
            #print (b_gd_tree[b_i])
            if len(b_gd_tree[b_i]) == 0:
                continue
            gd_tree_node = b_gd_tree[b_i][-1]
            
            #print (en_output)
            #print (look_up.index('temp_0'))
            #print (en_output[look_up.index("temp_0")])
            #print ()
            
            #self.recur_nn(gd_tree_node, en_output, look_up)
            p, l, acc_e, acc_e_t, acc_i = self.recur_nn.getLoss_train(gd_tree_node, en_output, look_up)
            #print (p)tensor([ 0,  2])
            #print (l)tensor(1.5615)
            #print (l)
            #print (post_order(gd_tree_node))
            #get_info_teacher_pointer(gd_tree_node)
            
            batch_predictions.append(p)
            batch_acc_e_list.append(acc_e)
            batch_acc_e_t_list.append(acc_e_t)
            batch_acc_i_list.append(acc_i)
            #batch_loss_l.append(l)
            #batch_loss_l += l
            batch_loss_l = torch.sum(torch.cat([ batch_loss_l.unsqueeze(0), l.unsqueeze(0)], 0))
            batch_count += 1
            
            #print (post_order(gd_tree_final))
            #print ("hhhhhh:", en_output)
        #print (batch_loss_l)
        #torch.cat(batch_loss_l)
        #print (torch.sum(torch.cat(batch_loss_l)))
        #print ()
        return batch_predictions, batch_loss_l, batch_count, batch_acc_e_list, batch_acc_e_t_list, batch_acc_i_list

    def test_forward_recur(self, input_tensor, input_lengths, num_pos, b_gd_tree):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)
        en_output_list, en_attn_list = self.self_attention(encoder_outputs, encoder_outputs, num_pos, input_lengths)
        
        batch_size = len(en_output_list)
        
        batch_predictions = []
        batch_acc_e_list = []
        batch_acc_e_t_list = []
        batch_acc_i_list = []
        batch_count = 0
        for b_i in range(batch_size):
            #en_output = en_output_list[b_i]
            en_output = torch.cat([self.fixed_num_emb, en_output_list[b_i]], dim=0)
            #print (num_pos)
            look_up = self.fixed_num_symbol + ['temp_'+str(temp_i) for temp_i in range(len(num_pos[b_i]))]
            #print (b_gd_tree[b_i])
            if len(b_gd_tree[b_i]) == 0:
                continue
            gd_tree_node = b_gd_tree[b_i][-1]
            
            #print (en_output)
            #print (look_up.index('temp_0'))
            #print (en_output[look_up.index("temp_0")])
            #print ()
            
            #self.recur_nn(gd_tree_node, en_output, look_up)
            p, acc_e, acc_e_t, acc_i = self.recur_nn.test_forward(gd_tree_node, en_output, look_up)
            #print (p)tensor([ 0,  2])
            #print (l)tensor(1.5615)
            #print (l)
            #print (post_order(gd_tree_node))
            #get_info_teacher_pointer(gd_tree_node)
            
            batch_predictions.append(p)
            batch_acc_e_list.append(acc_e)
            batch_acc_e_t_list.append(acc_e_t)
            batch_acc_i_list.append(acc_i)
            #batch_loss_l.append(l)
            #batch_loss_l += l
            batch_count += 1
            
            #print (post_order(gd_tree_final))
            #print ("hhhhhh:", en_output)
        #print (batch_loss_l)
        #torch.cat(batch_loss_l)
        #print (torch.sum(torch.cat(batch_loss_l)))
        #print ()
        return batch_predictions, batch_count, batch_acc_e_list, batch_acc_e_t_list, batch_acc_i_list

    def predict_forward_recur(self, input_tensor, input_lengths, num_pos, batch_seq_tree, batch_flags):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)
        en_output_list, en_attn_list = self.self_attention(encoder_outputs, encoder_outputs, num_pos, input_lengths)
        #print ('-x-x-x-',en_attn_list[0])
        batch_size = len(en_output_list)
        batch_pred_tree_node = []
        batch_pred_post_equ = []
        #batch_pred_ans = []
        
        for b_i in range(batch_size):
            
            flag = batch_flags[b_i]
            #num_list = batch_num_list[b_i]
            if flag == 1:
                
            
                en_output = torch.cat([self.fixed_num_emb, en_output_list[b_i]], dim=0)

                look_up = self.fixed_num_symbol + ['temp_'+str(temp_i) for temp_i in range(len(num_pos[b_i]))]

                seq_node = batch_seq_tree[b_i]
                #num_list = batch_num_list[i]
                #gold_ans = batch_solution[i]

                tree_node, post_equ = self.recur_nn.predict(seq_node, en_output, look_up)#, num_list, gold_ans)
                #p, acc_e, acc_e_t, acc_i = self.recur_nn.test_forward(gd_tree_node, en_output, look_up)

                batch_pred_tree_node.append(tree_node)
                batch_pred_post_equ.append(post_equ)
    
            else:
                batch_pred_tree_node.append(None)
                batch_pred_post_equ.append([])
               
        return batch_pred_tree_node, batch_pred_post_equ,en_attn_list
