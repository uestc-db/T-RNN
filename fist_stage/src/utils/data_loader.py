#encoding:utf-8
from data_tools import *
from equ_tools import *
from chinese import convertChineseDigitsToArabic
import numpy as np
import json
import pdb
import jieba

class DataMath23k():
    def __init__(self, flag=False):
        filename = "./data/math_json_wy.json"
        self._data_list = read_data_json(filename)
        if flag == False:
            self._data_dict = extract_number_and_align_all(self._data_list)
            self.convert_style()
            write_data_json(self.data_dict, "./data/math23k_0108.json")
        else:
            self.data_dict = read_data_json("./data/math23k_0108.json")

    def process_ans(self, ans):
        try:
            float(ans)
            return ans
        except:
            if '%' in ans:
                ans = ans[:-1]
                ans = float(ans)/100
                return str(ans)
            else:
                s_1 = ans.find('(')
                if s_1 >0:
                    num_pre = ans[:s_1]
                else:
                    num_pre = 0 
                s_1 = ans.find('(', s_1+1)
                e_1 = ans.find(')', s_1+1)
                num_1 = ans[s_1+1:e_1]
                s_2 = ans.find('(', e_1)
                e_2 = ans.find(')', s_2)
                num_2 = ans[s_2+1:e_2]
                ans = float(num_pre)+float(num_1)/float(num_2)
                return str(ans)

    def process_single_num(self, ans):
        try:
            float(ans)
            return ans
        except:
            if '%' in ans:
                ans = ans[:-1]
                ans = float(ans)/100
                return str(ans)
            else:
                s_1 = ans.find('(')
                if s_1 >0:
                    num_pre = ans[:s_1]
                else:
                    num_pre = 0 
                e_1 = ans.find('/', s_1+1)
                num_1 = ans[s_1+1:e_1]
                e_2 = ans.find(')', e_1)
                num_2 = ans[e_1+1:e_2]
                ans = float(num_pre)+float(num_1)/float(num_2)
                return str(ans)

    def process_num_list(self, num_list):
        new_num_list = []
        for num in num_list:
            num =self.process_single_num(num)
            new_num_list.append(num)
        return new_num_list

    def check_ans(self, num_list, template, ans):
        alphabets = 'abcdefghijklmnopqrstuvwxyz'
        new_equ = []
        for elem in template:
            if 'temp' in elem:
                new_equ.append(str(num_list[alphabets.find(elem[-1])]))
            else:
                new_equ.append(elem)
        return equ_api_1(new_equ, ans)
        
        
    def convert_style(self):
        '''
        idx: temp_text_str, temp_list, num_list, ans
        '''
        data_dict = {}
        for key, elem in self._data_dict.items():
            elem_tuple = elem['tuple']
            data_dict[key] = {'text':'','target_template':[],'gen_template':[],'num_list':[],'ans':0}
            data_dict[key]['text'] = ' '.join(elem_tuple[0]) # str
            data_dict[key]['target_template'] = elem_tuple[1] # list x= temp_a ,,,
            data_dict[key]['num_list'] = self.process_num_list(elem_tuple[2]) # [float, float]
            data_dict[key]['ans'] = self.process_ans(elem['solution'][0]) # float 
        self.data_dict = data_dict


class DataUnlabel57k():
    def __init__(self, flag=False):
        filename = "./data/ex_data/external_data.json"
        self._data_dict = read_data_json(filename)
        if flag == False:
            self.convert_style()
            write_data_json(self.data_dict, "./data/math57k_0108.json")
        else:
            self.data_dict = read_data_json("./data/math57k_0108.json")

    def process_norm_line(self, line):
        alphabets = 'abcdefghijklmnopqrstuvwxyz'
        line_list = list(jieba.cut(line))
        filter_line_list = filter(lambda x: x!='[' and x!=']', line_list)
        new_line_list = []
        for elem in filter_line_list:
            if 'tmp' in elem:
                new_line_list.append('temp_'+alphabets[int(elem[3:])])
            else:
                new_line_list.append(elem)
        return ' '.join(new_line_list)

    def process_frac(self, num_elem):
        start = num_elem.find('\\frac')
        elem = ''
        if start > 0 :
            num_pre = num_elem[:start]
        else:
            num_pre = ''
        if start != -1:
            s_1 = num_elem.find('{', start+1)
            e_1 = num_elem.find('}', s_1+1)
            s_2 = num_elem.find('{', e_1)
            e_2 = num_elem.find('}', s_2)
            elem = '('+num_elem[s_1+1:e_1]+'/'+ num_elem[s_2+1:e_2]+')' + num_elem[e_2+1:]
            elem = num_pre+elem
            return elem
        else:
            return num_elem

    def is_chinese_filter(self, num_elem):
        chinese_nums = [u'一',u'二',u'三',u'四',u'五',u'六',u'七',u'八',u'九',u'十']
        chinese_dict = dict([(num, idx+1) for idx, num in enumerate(chinese_nums)])
        flag = False
        for chinese_num in chinese_nums:
            if chinese_num in num_elem:
                flag = True
                break
        return flag
        

    def chinese_num(self, num_elem):
        chinese_nums = [u'一',u'二',u'三',u'四',u'五',u'六',u'七',u'八',u'九',u'十']
        chinese_dict = dict([(num, idx+1) for idx, num in enumerate(chinese_nums)])
        flag = False
        for chinese_num in chinese_nums:
            if chinese_num in num_elem:
                flag = True
                break
        elem = ''
        if flag == True:
            if u'折' in num_elem:
                num_elem_ = num_elem[:-1]
                for per in num_elem_:
                    elem += str(chinese_dict[per])
                elem = str(float(elem)/(10**len(elem)))
                return elem          
                #print num_elem.encode('utf-8')
                #print num_elem.encode('utf-8'),
                #print elem
            else:
                #print num_elem.encode('utf-8'),
                for per in num_elem:
                    if per in chinese_dict:
                        elem += per
                #print elem.encode('utf-8'),
                num = convertChineseDigitsToArabic(elem)
                #print num
                return  str(num)
        else:
            return num_elem

    def split_num_and_unit(self, num_unit):
        num = ''
        unit = ''
        for idx in range(len(num_unit)):
            char = num_unit[idx]
            if char.isdigit() or char in ['.', '/', '(', ')', '%']:
                num += char
            else:
                unit += char
        return num, unit
        
    def process_single_num(self, ans):
        if ans[-1] == '/':
            ans = ans[:-1]
        try:
            float(ans)
            return ans
        except:
            if '%' in ans:
                ans = ans[:-1]
                ans = float(ans)/100
                return ans
            elif '/' not in ans:
                return ans[1:-1]
            else:
                s_1 = ans.find('(')
                if s_1 >0:
                    num_pre = ans[:s_1]
                else:
                    num_pre = 0 
                e_1 = ans.find('/', s_1+1)
                num_1 = ans[s_1+1:e_1]
                e_2 = ans.find(')', e_1)
                num_2 = ans[e_1+1:e_2]
                ans = float(num_pre)+float(num_1)/float(num_2)
                return str(ans)


    def process_num_list(self, num_list):
        pure_num_list = []
        #print ' '.join([elem.encode('utf-8') for elem in num_list])
        for elem in num_list:
            elem = elem[elem.find('[')+1:elem.find(']')]
            elem = self.process_frac(elem)
            #if self.is_chinese_filter(elem):
            #    print elem.encode('utf-8')
            elem = self.chinese_num(elem)
            num, unit = self.split_num_and_unit(elem)
            if num == '' :
                continue
            num = self.process_single_num(num)
            pure_num_list.append(str(num))
            #print elem.encode('utf-8'),
        #convertChineseDigitsToArabic
        #print pure_num_list
        #print 
        return pure_num_list 

    def process_ans(self, ans): 
        ans = self.process_frac(ans)
        ans = self.chinese_num(ans)
        ans = self.process_single_num(ans)
        return str(ans)

    def convert_style(self):
        data_dict = {}
        count = 0
        for key, elem in self._data_dict.items():
            #print elem.keys()
            #for k, v in elem.items():
            #    print k , v.encode('utf-8')
            #break
            #print elem['text'].strip().encode('utf-8')
            line = elem['normltext']
            num_line = elem['formal-procedure'].strip()
            num_list = num_line[6:].strip().split(', ')
            if len(num_list) > 10 or len(num_list) == 0:
                continue
            ans = elem['ans'].strip()
            line = self.process_norm_line(line)
            #print ' '.join([elem.encode('utf-8') for elem in num_list])
            num_list = self.process_num_list(num_list)
            ans = self.process_ans(ans)
            #print num_list
            idx = str(count)
            data_dict[idx] = {'text':'','target_template':[],'gen_template':[],'num_list':[],'ans':''}
            data_dict[idx]['text'] = line
            data_dict[idx]['num_list'] = num_list
            data_dict[idx]['ans'] = ans
            count += 1
        self.data_dict = data_dict

class Word2vec:
    def __init__(self, data_23k, data_57k, flag = False):
        self.data_23k = data_23k
        self.data_57k = data_57k
        if flag == False:
            self.train_word2vec()
        else:
            self.emb_vectors = np.load('./data/emb.npy')

    def train_word2vec(self):
        new_data ={}
        sentences = []
        for k, v in self.data_23k.data_dict.items():
            sentence = v['text'].strip().split(' ')
            sentences.append(sentence)
            for elem in sentence:
                new_data[elem] = new_data.get(elem, 0) + 1

        for k, v in self.data_57k.data_dict.items():
            sentence = v['text'].strip().split(' ')
            sentences.append(sentence)
            for elem in sentence:
                new_data[elem] = new_data.get(elem, 0) + 1

        from gensim.models import word2vec
        model = word2vec.Word2Vec(sentences, size=128, min_count=1)
        token_list = ['PAD_token', 'UNK_token', 'END_token']

        emb_vectors = []
        emb_vectors.append(np.zeros((128)))
        emb_vectors.append(np.random.rand((128))/1000.0)
        emb_vectors.append(np.random.rand((128))/1000.0)

        ext_list = ['PAD_token', 'END_token']+[u'+', u'*', u'-', u'/', u'1', u'PI', u'temp_m', u'temp_l', u'temp_o', u'temp_n', u'temp_i', u'temp_h', u'temp_k', u'temp_j', u'temp_e', u'temp_d', u'temp_g', u'temp_f', u'temp_a', u'temp_c', u'temp_b', u'^']

        for k, v in new_data.items(): 
            token_list.append(k)
            emb_vectors.append(np.array(model.wv[k]))

        for elem in ext_list:
            if elem not in token_list:
                token_list.append(elem)
                emb_vectors.append(np.random.rand((128))/1000.0)
        emb_vectors = np.array(emb_vectors)

        np.save("./data/emb.npy", emb_vectors)
        with open("./data/token_list.json", 'w') as f:
            json.dump(token_list, f)
        self.emb_vectors = emb_vectors

class DataLoader():
    def __init__(self, args=None):
        '''
        seq2seq model data
        '''
        self.args = args
        print 'loading 23k, 57k, word2vec'
        self.data_23k = DataMath23k(True)
        self.data_57k = DataUnlabel57k(True)
        self.word2vec = Word2vec(self.data_23k, self.data_57k, True)
        self.vocab_list = read_data_json("./data/token_list.json")
        self.vocab_dict = dict([(elem, idx) for idx, elem in enumerate(self.vocab_list)])
        self.vocab_len = len(self.vocab_list)
        print 'load 23k, 57k, word2vec sucessfully'

        self.decode_classes_list =  ['PAD_token', 'END_token']+[u'^', u'1', u'PI', u'temp_m', u'temp_l', u'temp_o', u'temp_n', u'temp_i', u'temp_h', u'temp_k', u'temp_j', u'temp_e', u'temp_d', u'temp_g', u'temp_f', u'temp_a', u'temp_c', u'temp_b']
        self.decode_classes_dict = \
              dict([(elem, idx) for idx, elem in enumerate(self.decode_classes_list)])
        self.classes_len = len(self.decode_classes_list)

        #train_list, test_list = split_train_test(self.data_23k.data_dict, 10)#args.random_seed)
        train_list, valid_list, test_list = split_by_feilong_23k(self.data_23k.data_dict)
        self.math23k_train_list = train_list
        self.math23k_valid_list = valid_list
        self.math23k_test_list = test_list
        self.math57k_data_list = self.data_57k.data_dict.items()
        #self.templates = read_data_json("./data/norm_templates.json")
        #self.templates = read_data_json("./data/op_norm_templates.json")
        self.templates = read_data_json("./data/op_post_dup_templates.json")
        ### [(u'11542', {u'num_list:,[str,str] text:' ', target_template, gen_tempalte, ans}, ...]

    #def split_and_write(self, data_dict, ttype, random_seed): 



    def prepare_rl_data():
        '''
        rl model data
        rl 数据得另外起一个类
        注意，rl需要的数据有，emb, 每个句子的hidden vectors, 
        '''
        pass

    def inverse_temp_to_num(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                new_equ_list.append(num_list[index])
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def check_(self, equ, num_list, t_ans):
        equ_list = self.inverse_temp_to_num(equ, num_list)
        ans = post_solver(equ_list)
        print t_ans, '--', ans, abs(float(t_ans) - float(ans)) < 1e-5  
        
        
    def _data_batch_preprocess(self, data_batch, template_flag):
        batch_encode_idx = []
        batch_decode_idx = []
        batch_encode_len = []
        batch_decode_len = []

        batch_idxs = []
        batch_text = []
        batch_num_list = []
        batch_solution = []

        for elem in data_batch:
            idx = elem[0]
            encode_sen = elem[1]['text']
            encode_sen_idx = string_2_idx_sen(encode_sen.strip().split(' '), self.vocab_dict)
            batch_encode_idx.append(encode_sen_idx)
            batch_encode_len.append(len(encode_sen_idx))

            if template_flag == True:
                decode_sen = elem[1]['target_template']
                decode_sen = self.templates[idx]
                #decode_sen = postfix_equation(decode_sen[2:])
                #pdb.set_trace()
                #self.check_(decode_sen, elem[1]['num_list'], elem[1]['ans'])
                #print decode_sen
                decode_sen.append('END_token')
                decode_sen_idx = string_2_idx_sen(decode_sen, self.vocab_dict)
                batch_decode_idx.append(decode_sen_idx)
                batch_decode_len.append(len(decode_sen_idx))

            batch_idxs.append(idx)
            batch_text.append(encode_sen)
            batch_num_list.append(elem[1]['num_list'])
            batch_solution.append(elem[1]['ans'])

        #print batch_encode_len
        max_encode_len =  max(batch_encode_len)
        batch_encode_pad_idx = []

        if template_flag == True:
            max_decode_len =  max(batch_decode_len)
            batch_decode_pad_idx = []

        for i in range(len(data_batch)):
            encode_sen_idx = batch_encode_idx[i]
            encode_sen_pad_idx = pad_sen(\
                               encode_sen_idx, max_encode_len, self.vocab_dict['PAD_token'])
            batch_encode_pad_idx.append(encode_sen_pad_idx)

            if template_flag:
                decode_sen_idx = batch_decode_idx[i]
                decode_sen_pad_idx = pad_sen(\
                              decode_sen_idx, max_decode_len, self.vocab_dict['PAD_token'])
                              #decode_sen_idx, max_decode_len, self.decode_classes_dict['PAD_token'])
                batch_decode_pad_idx.append(decode_sen_pad_idx)

        batch_data_dict = dict()
        batch_data_dict['batch_encode_idx'] = batch_encode_idx
        batch_data_dict['batch_encode_len'] = batch_encode_len
        batch_data_dict['batch_encode_pad_idx'] = batch_encode_pad_idx

        batch_data_dict['batch_index'] = batch_idxs
        batch_data_dict['batch_text'] = batch_text
        batch_data_dict['batch_num_list'] = batch_num_list
        batch_data_dict['batch_solution'] = batch_solution

        if template_flag:
            batch_data_dict['batch_decode_idx'] = batch_decode_idx
            batch_data_dict['batch_decode_len'] = batch_decode_len
            batch_data_dict['batch_decode_pad_idx'] = batch_decode_pad_idx

        if len(data_batch) != 1:
            new_batch_data_dict = self._sorted_batch(batch_data_dict)
        else:
            new_batch_data_dict = batch_data_dict
        return new_batch_data_dict

    def _sorted_batch(self, batch_data_dict):
        new_batch_data_dict = dict()
        batch_encode_len = np.array(batch_data_dict['batch_encode_len'])
        sort_idx = np.argsort(-batch_encode_len)
        for key, value in batch_data_dict.items():
            new_batch_data_dict[key] = np.array(value)[sort_idx]
        return new_batch_data_dict


    def get_batch(self, data_list, batch_size, template_flag = False, verbose=0):
        #print data_list
        batch_num = len(data_list)/batch_size+1
        for idx in range(batch_num):
            batch_start = idx*batch_size
            batch_end = min((idx+1)*batch_size, len(data_list))
            #print batch_start, batch_end, len(data_list)
            batch_data_dict = self._data_batch_preprocess(data_list[batch_start: batch_end],\
                                                               template_flag)
            yield batch_data_dict
        
def test():
    data_loader = DataLoader()
    gen_data = data_loader.get_batch(data_loader.math23k_train_list, 32, True)
    for elem in gen_data:
        pass
        #print len(elem)
    #for elem in gen_data:
    #    pass
        #print elem.keys()
    #data_23k = DataMath23k(False) 
    #data_57k = DataUnlabel57k(False)
    #word2vec = Word2vec(data_23k, data_57k, False)
    #print word2vec.emb_vectors.shape

def test_57k():
    data_57k = DataUnlabel57k(False)

def test_w2v():
    data_23k = DataMath23k(False) 
    data_57k = DataUnlabel57k(False)
    word2vec = Word2vec(data_23k, data_57k, False)
#test()
#test_57k()
#test_w2v()
