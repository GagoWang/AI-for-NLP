import os
import jieba
from gensim.models import Word2Vec
from collections import defaultdict
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
import time
import csv
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import json
import pickle

file_name = 'person_words.csv'
sentence_file_name = 'news_cut_sentence.txt'


def sentence_to_vec(sentence_list, embedding_size, all_words_frequency, model, a=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        cut_sentence = list(jieba.cut(sentence))
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = len(cut_sentence)
        for word in cut_sentence:
            if word not in all_words_frequency: continue
            a_value = a / (a + get_word_frequency(word, all_words_frequency))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, model[word]))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs


def get_word_frequency(word, all_words_frequency):
    if word in all_words_frequency:
        return all_words_frequency[word]
    else:
        return 1.0


def has_say(words, say_words):
    index = {}
    for i in range(len(words)):
        if words[i] in say_words:
            index[i] = words[i]
    return index


def get_child_node(parent, arcs):
    index_list = [parent]
    # print(index_list)
    for i in range(len(arcs)):
        if arcs[i].head == parent:
            # print('arcs[i].head :{}'.format(i))
            # index_list.append(i)
            index_list += get_child_node(i + 1, arcs)
    return index_list


def write_file(content):
    with open(file_name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(content)


def tokenize(string):
    return ''.join(re.findall('[\w|\d]+', string))


def get_sentence():
    with open('../train/news_cut_sentences.txt', 'r', encoding='utf-8') as f:
        return f.read().split('\n')


if __name__ == '__main__':
    model = Word2Vec.load('../word2vecModel/wiki_news_corpus_model_1_100')  # wiki+新闻语料结合

    say_keywords = set()
    with open('../train/say_words.txt', 'r') as f:
        say_keywords = set(f.read().split())
    with open('../train/word_frequency.txt', 'r') as f:
        all_words_frequency = json.loads(f.read())
    print(say_keywords)

    people_word = []
    sentences = get_sentence()
    time1 = time.time()
    LTP_DATA_DIR = '/Users/Gago/Workplace/Project/NLP/ltp_data_v3.4.0'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 词性标注模型路径，模型名称为`cws.model`
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    time2 = time.time()
    print('load ltp model:{}'.format(time2-time1))
    sentence_vectors = sentence_to_vec(sentences, 100, all_words_frequency, model)
    len_sentences = len(sentence_vectors)
    print(len_sentences)
    with open('../train/sentence_vector.txt', 'wb') as file:
        pickle.dump(sentence_vectors, file)
    with open('../train/sentence_vector.txt', 'rb') as fp:
        sentence_vectors = pickle.load(fp)
    #print(sentence_vectors)
    print(len(sentence_vectors))
    time3 = time.time()
    print('sentence to vec:{}'.format(time3-time2))
    #for i in range(len(sentence_vectors[:100])):
    #    if i % 2 == 0:
    #        sim = cosine_similarity([sentence_vectors[i]], [sentence_vectors[i + 1]])
    #        print(sim)

    last_index = -1
    for s_n in range(len(sentences)):
        print(s_n)
        sentence = sentences[s_n]
        if s_n >0 and last_index==s_n-1:
            sim = cosine_similarity([sentence_vectors[s_n-1]], [sentence_vectors[s_n]])
            if sim>0.95 and people_word:
                people_word[-1][2] = people_word[-1][2] + sentence
                last_index = s_n
                continue
        # words = list(jieba.cut(sentence))
        words = segmentor.segment(sentence)  # 分词

        postags = postagger.postag(words)  # 词性标注

        netags = recognizer.recognize(words, postags)  # 命名实体识别

        arcs = parser.parse(words, postags)  # 句法分析

        sentence_word = has_say(words, say_keywords)
        #print(sentence_word)
        if not sentence_word:
            continue
        #print(s_n)
        #print(sentence)
        i = sorted(sentence_word, key=lambda x: arcs[x].head)[0]

        arcs_head = arcs[i].head

        # child_node = get_child_node(i + 1, arcs)

        sbv = ''
        vob = ''
        for n in range(len(arcs)):
            # print('arcs[n].head:{}'.format(arcs[n].head))
            if i + 1 == arcs[n].head:
                # child_node.append(n)
                if arcs[n].relation == 'SBV':
                    sbv_list = get_child_node(n + 1, arcs)
                    sbv_list = sorted([x - 1 for x in sbv_list], key=lambda x: x)
                    sbv = ''.join(words[sbv_list[0]:sbv_list[-1] + 1])
                # if arcs[n].relation == 'VOB':
                #     vob_list = get_child_node(n + 1, arcs)
                #     vob_list = sorted([x - 1 for x in vob_list], key=lambda x: x)
                #     vob = ''.join(words[vob_list[0]:vob_list[-1] + 1])
                #     print('vob_list:{}'.format(vob_list))
                #     print('vob:{}'.format(vob))
            vob = sentence
        if sbv and vob:
            people_word.append([sbv, sentence_word[i], vob])
            last_index = s_n
    time4 = time.time()
    print('word extract:{}'.format(time4-time3))
    #print('people_word:{}'.format(people_word))
    #print('\t'.join(postags))
    #print('\t'.join(netags))
    #print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

    segmentor.release()  # 释放模型
    postagger.release()
    recognizer.release()
    parser.release()
    write_file(people_word)
