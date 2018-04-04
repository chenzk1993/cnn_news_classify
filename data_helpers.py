# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 08:31:04 2017

@author: E601
"""

# encoding: UTF-8

import numpy as np
import re
import os
from jieba import cut
#import time
import pickle
def seperate_line(line):
    #将每个汉字都用空格分隔，最后变成字符串
    return ''.join([word + ' ' for word in line])

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #将任何非中文字符都用空格代替
    string = re.sub(u"[^\u4e00-\u9fff]", " ", string)
    #将所有多于两个的空白字符都用空格代替
    string = re.sub(r"\s{2,}", " ", string)
    #移除句子开头和结尾的空格
    return string.strip()
#进行分词
def cut_word(inputList):
    outList=[]
    for sentence in inputList:
        sentence=sentence.replace(' ','')  #字与字间的空格删除
        sentence=list(cut(sentence))   #将句子切分
        outList.append(sentence)
    return outList  #返回一个二维列表，每个元素是一个词
        
        
def read_and_clean_zh_file(input_file, output_cleaned_file = None):
    lines = list(open(input_file, "r",encoding='utf-8').readlines())
    lines = [clean_str(seperate_line(line)) for line in lines]  #lines为所有句子的列表
    lines=cut_word(lines)
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w',encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
    return lines  #返回一个二维列表，每个元素是一个词
def load_data_and_labels(input_text_file, input_label_file, num_labels):
    x_text = read_and_clean_zh_file(input_text_file)
    #如果输入文件标签名路径不存在，则y返回None
    if not os.path.exists(input_label_file):
       y = None
    else:
        y = map(int, list(open(input_label_file, "r",encoding='utf-8').readlines()))
    return (x_text, y)

def load_data_files(sports_file, amusement_file,home_file,estate_file,
                     education_file,fashion_file,politics_file,game_file,
                     technology_file,finance_file):
    """
    将词和标签,组成一个向量,维度是最长的那句话中词的个数,深度是10，分别表示10种新闻
    """
    # Load data from files
    sports_examples = read_and_clean_zh_file(sports_file)
    amusement_examples = read_and_clean_zh_file(amusement_file)
    home_examples = read_and_clean_zh_file(home_file)
    estate_examples = read_and_clean_zh_file(estate_file)
    education_examples = read_and_clean_zh_file(education_file)
    fashion_examples = read_and_clean_zh_file(fashion_file)
    politics_examples = read_and_clean_zh_file(politics_file)
    game_examples = read_and_clean_zh_file(game_file)
    technology_examples = read_and_clean_zh_file(technology_file)
    finance_examples = read_and_clean_zh_file(finance_file)
    #合并数据
    x_text = np.concatenate([sports_examples, amusement_examples,home_examples,
                             estate_examples,education_examples,fashion_examples,
                             politics_examples,game_examples,technology_examples,
                             finance_examples], axis=0)     #将数据垂直连接
    # Generate labels
    #表示有多少句话，就有多少个[0,1]组成的集合
    sports_labels = [[1,0,0,0,0,0,0,0,0,0] for _ in sports_examples]
    amusement_labels = [[0,1,0,0,0,0,0,0,0,0] for _ in amusement_examples]
    home_labels = [[0,0,1,0,0,0,0,0,0,0] for _ in home_examples]
    estate_labels = [[0,0,0,1,0,0,0,0,0,0] for _ in estate_examples]
    education_labels = [[0,0,0,0,1,0,0,0,0,0] for _ in education_examples]
    fashion_labels = [[0,0,0,0,0,1,0,0,0,0] for _ in fashion_examples]
    politics_labels = [[0,0,0,0,0,0,1,0,0,0] for _ in politics_examples]
    game_labels = [[0,0,0,0,0,0,0,1,0,0] for _ in game_examples]
    technology_labels = [[0,0,0,0,0,0,0,0,1,0] for _ in technology_examples]
    finance_labels = [[0,0,0,0,0,0,0,0,0,1] for _ in finance_examples]
    y = np.concatenate([sports_labels, amusement_labels,home_labels,estate_labels,
                        education_labels,fashion_labels,politics_labels,
                        game_labels,technology_labels,finance_labels], axis=0)     #将数据垂直连接
    return [x_text, y]
#padding_token表示当句子长度不到最大长度时，用来补齐的内容,padding_sentence_length为指定的句子最大
#长度
def padding_sentences(input_sentences, padding_token, max_seq_length = None):
    #input_sentences为每一句话构成的列表而构成的列表
    padding_sentences=[]
    if max_seq_length is not None:
       max_sentence_length = max_seq_length
    else:
       max_sentence_length = max([len(sentence) for sentence in input_sentences]) 
    for sentence in input_sentences:
        #如果句子长度大于给定的长度最大值，则取前max_sentence_length个字符
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
        #如果句子长度小于给定的长度最大值，则用给定的字符将原句填充到最大长度
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
        padding_sentences.append(sentence)
    return padding_sentences, max_sentence_length

def batch_iter(data, batch_size, shuffle=True):
    '''
    为数据集产生一个批迭代器
    '''
    #每次只输出shuffled_data[start_index:end_index]这么多
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1  #每一个epoch有多少个batch_size
    #batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
    #iteration：1个iteration等于使用batchsize个样本训练一次；
    #epoch：1个epoch等于使用训练集中的全部样本训练一次；
    #练集有1000个样本，batchsize=10，那么：
    #训练完整个样本集需要：100次iteration，1次epoch。
    #如果混排标志为True，则混排数据，否则不混排。
    if shuffle:
	        #产生随机数据的索引
       shuffle_indices = np.random.permutation(np.arange(data_size))
       shuffled_data = data[shuffle_indices]
    else:
       shuffled_data = data            
    for batch_num in range(num_batches_per_epoch):
        start_idx = batch_num * batch_size    # 当前batch的索引开始
        end_idx = min((batch_num + 1) * batch_size, data_size)   #判断下一个batch是不是超过最后一个数据了
        yield shuffled_data[start_idx : end_idx]

def mkdir_if_not_exist(dirpath):
    #如果一个文件路径不存在，则创建它
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
def saveDict(input_dict, output_file):
    #使用pickle.dump写，文件必须以'wb'方式打开
    with open(output_file, 'wb') as f:
#python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象
#信息保存到文件中去。pickle.dump(obj, file, protocol)将对象obj保存到文件file中去。protocol为序列化使用
#的协议版本。0：ASCII协议;1：老式的二进制协议；2:新二进制协议，较以前的更高效
         pickle.dump(input_dict, f) 
    #注意当使用txt打开f时，会乱码，但用pickle.load加载文件时并不会乱码

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
         #pickle.load(file)从file中读取一个字符串，并将它重构为原来的python对象。
         #使用pickle.load时，文件必须以'rb'方式打开
         output_dict = pickle.load(f)
    return output_dict
