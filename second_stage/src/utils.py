import json                                                                                                                                                                                                                                                                   
import numpy as np
import time

import random
import pickle
import sys

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def split_by_feilong_23k(data_dict):
    t_path = "./data/id_ans_test"
    v_path = "./data/valid_ids.json"
    valid_ids = read_data_json(v_path)
    test_ids = []
    with open(t_path, 'r') as f:
        for line in f:
            test_id = line.strip().split('\t')[0]
            test_ids.append(test_id)
    train_list = []
    test_list = []
    valid_list = []
    for key, value in data_dict.items():
        if key in test_ids:
            test_list.append((key, value))
        elif key in valid_ids:
            valid_list.append((key, value))
        else:
            train_list.append((key, value))
    #print len(train_list), len(valid_list), len(test_list)
    return train_list, valid_list, test_list

def string_2_idx_sen(sen,  vocab_dict):                                                                                                                                                                                                                                       
    return [vocab_dict[word] for word in sen]
 
def pad_sen(sen_idx_list, max_len=115, pad_idx=1):                                                                                                                                                                                                                            
    return sen_idx_list + [pad_idx]*(max_len-len(sen_idx_list))

def encoder_hidden_process(encoder_hidden, bidirectional):
    if encoder_hidden is None:
        return None
    if isinstance(encoder_hidden, tuple):
        encoder_hidden = tuple([_cat_directions(h, bidirectional) for h in encoder_hidden])
    else:
        encoder_hidden = _cat_directions(encoder_hidden, bidirectional)
    return encoder_hidden
 
def _cat_directions(h, bidirectional):
    if bidirectional:
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

def post_solver(post_equ):
    stack = [] 
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        if elem not in op_list:
            op_v = elem
            if '%' in op_v:
                op_v = float(op_v[:-1])/100.0
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            op_v_2 = stack.pop()
            op_v_2 = float(op_v_2)
            if elem == '+':
                stack.append(str(op_v_2+op_v_1))
            elif elem == '-':
                stack.append(str(op_v_2-op_v_1))
            elif elem == '*':
                stack.append(str(op_v_2*op_v_1))
            elif elem == '/':
                stack.append(str(op_v_2/op_v_1))
            else:
                stack.append(str(op_v_2**op_v_1))
    return stack.pop()


class BinaryTree():
    def __init__(self):
        self._init_node_info()
    
    def _init_node_info(self):
        self.root_value = None
        self.left_tree = None
        self.right_tree = None
        #self.height = 0
        self.is_leaf = True
        self.pre_order_list = []
        self.mid_order_list = []
        self.post_order_list = []
        self.node_emb = None
    
def _pre_order(root, order_list):
    if root == None:
        return
    #print ('pre\t', root.value,)
    order_list.append(root.root_value)
    _pre_order(root.left_tree, order_list)
    _pre_order(root.right_tree, order_list)
    
def pre_order(root):
    order_list = []
    _pre_order(root, order_list)
    #print ("post:\t", order_list) 
    return order_list
        
def _mid_order(root, order_list):
    if root == None:
        return
    _mid_order(root.left_tree, order_list)
    #print ('mid\t', root.root_value,)
    order_list.append(root.root_value)
    _mid_order(root.right_tree, order_list)

def mid_order(root):
    order_list = []
    _mid_order(root, order_list)
    #print ("post:\t", order_list) 
    return order_list
    
def _post_order(root, order_list):
    if root == None:
        return
    _post_order(root.left_tree, order_list)
    _post_order(root.right_tree, order_list)
    #print (root.root_value, ' ->\t', root.node_emb,)
    order_list.append(root.root_value)

def post_order(root):
    order_list = []
    #print ("post:")
    _post_order(root, order_list)
    #print ("post:\t", order_list)  
    #print ()
    return order_list
    
def construct_tree(post_equ):
    stack = []
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        node = BinaryTree()
        node.root_value = elem
        if elem in op_list:
            node.right_tree = stack.pop()
            node.left_tree = stack.pop()
            
            node.is_leaf = False
            
            stack.append(node)
        else:
            stack.append(node)
    return stack.pop()

def form_gdtree(elem):
    post_equ = elem['post_template']
    tree = construct_tree(post_equ)
    gd_list = []
    def _form_detail(root):
        if root == None:
            return
        _form_detail(root.left_tree)
        _form_detail(root.right_tree)
        #if root.left_tree != None and root.right_tree != None:
        if root.is_leaf == False:
            gd_list.append(root)
    _form_detail(tree)
    return gd_list[:]
    #print ('+++++++++')
    #print (gd_list)
    #print (post_equ)
    #for elem in gd_list:
        #print (post_order(elem))
        #print (elem.root_value)
    #print ('---------')
    #print ()

def construct_tree_opblank(post_equ):
    stack = []
    op_list = ['<OP>', '^']
    for elem in post_equ:
        node = BinaryTree()
        node.root_value = elem
        if elem in op_list:
            node.right_tree = stack.pop()
            node.left_tree = stack.pop()
            right_len = post_order(node.right_tree)
            left_len = post_order(node.left_tree)
            if left_len <= right_len:
                node.temp_value = node.left_tree.temp_value
            else:
                node.temp_value = node.right_tree.temp_value
            node.is_leaf = False
            
            stack.append(node)
        else:
            node.temp_value = node.root_value
            stack.append(node)
    return stack.pop()
     
def form_seqtree(seq_tree):
    #post_equ = elem[u'mask_post_equ_list'][2:]
    #tree = construct_tree(post_equ)
    seq_list = []
    def _form_detail(root):
        if root == None:
            return
        _form_detail(root.left_tree)
        _form_detail(root.right_tree)
        #if root.left_tree != None and root.right_tree != None:
        if root.is_leaf == False:
            seq_list.append(root)
    _form_detail(seq_tree)
    return seq_list[:]
