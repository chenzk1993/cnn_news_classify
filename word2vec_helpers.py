# -*- coding: utf-8 -*-
'''
python word2vec_helpers.py input_file output_model_file output_vector_file

'''

# import modules & set up logging
import os
import sys
import logging
import multiprocessing
import time
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def embedding_sentences(sentences, embedding_size = 200,ext_emb_path = None):
    embedding_vocabulary={}
    assert os.path.isfile(ext_emb_path)
    # Load pretrained embeddings from file
    for line in open(ext_emb_path, 'r', encoding='utf-8'):
        line=line.rstrip().split(' ')
        embedding_vocabulary[line[0]]=[]
        for i in line[1:]:
            embedding_vocabulary[line[0]].append(float(i))
    all_vectors = []
    #产生正态随机数
    embeddingUnknown=list(np.random.normal(loc=0, scale=0.2, size=embedding_size))
    #输出词的个数，此模型中总共存错了8561个词的词向量
    #print(len(embedding_vocabulary))
    for sentence in sentences:
        current_vector = []
        for word in sentence:
            #如果某词语在训练好的词向量的词汇表中，则该词语的词向量就为训练好的词向量
            if word in embedding_vocabulary.keys():
                current_vector.append(embedding_vocabulary[word])
            #如果存在不在已训练过的词汇表中的词，则它的词向量用全零表示
            else:
                current_vector.append(embeddingUnknown)
        all_vectors.append(current_vector)
    #all_vectors为所有句子构成的列表，而现在每句话是由每个词的词向量所构成的二维列表。
    return all_vectors


def generate_word2vec_files(input_file, output_model_file, output_vector_file, size = 128, window = 5, min_count = 5):
    start_time = time.time()

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model = Word2Vec(LineSentence(input_file), size = size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

def run_main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    input_file, output_model_file, output_vector_file = sys.argv[1:4]

    generate_word2vec_files(input_file, output_model_file, output_vector_file) 

def test():
    vectors = embedding_sentences([['first', 'sentence'], ['second', 'sentence']], embedding_size = 4, min_count = 1)
    print(vectors)
