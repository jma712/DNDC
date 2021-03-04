# coding=utf-8
'''
utils for data loading/ simulation
created on 2019-12-12
'''

import json
import os
import re
import csv
import numpy as np
import scipy.io as scio
import random
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import sparse
from scipy.sparse import csc_matrix
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from datetime import datetime
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib
import matplotlib.pyplot as plt

random.seed(111)
np.random.seed(111)


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def loadData_peer(path):
    trn_rate = 0.6
    tst_rate = 0.2

    files = os.listdir(path + 'reviews/')
    authors_all = {}  # author: paper num
    authors_time = {}  # author: first time

    time_authors = {}  # time: set{authors}

    edge_time = {}  # (author1, author2): time

    paper_time = {}  # paper_rawID: time
    paper_accept = {}  # paper_rawID: bool
    author_paper = {}  # author: paper_rawID
    author_time_accept = {}  # author: {time: acceptNum}
    time_author_paper = {}  # time: {authorname: {paperRawID}}; new, not accumulated

    file_num = 0
    author_num_all = 0

    # [time_start, time_end)
    time_select = [1080, 1085, 1090, 1095, 1100, 1105, 1110, 1115, 1120, 1125, 1130, 1135, 1140, 1145, 1150, 1155, 1160]

    time_start = time_select[0]
    time_end = time_select[-1]

    for file in files:  # all files (papers) in the fold
        filepath = path + 'reviews/' + file
        with open(filepath) as json_file:
            data = json.load(json_file)
            authors = data['authors']

            # French
            for a_i in range(len(authors)):
                authors[a_i] = authors[a_i].encode('ascii', 'ignore')
            # year = str(data['DATE_OF_SUBMISSION'][-4:])
            # month = str(data['DATE_OF_SUBMISSION'][-8:-5])
            # time = year + ' ' + month

            time = data['DATE_OF_SUBMISSION']
            time = datetime.strptime(time, '%d-%b-%Y')

            accept = data['accepted']

            if file not in paper_time:
                paper_time[file] = time

            for author in authors:
                # time_author_paper
                if time in time_author_paper:
                    if author in time_author_paper[time]:
                        time_author_paper[time][author].add(file)
                    else:
                        time_author_paper[time][author] = set([file])
                else:
                    time_author_paper[time] = {author: set([file])}

                # authors_all, authors_time
                if author in authors_all:
                    authors_all[author] += 1
                else:
                    authors_all[author] = 1
                    authors_time[author] = time

                # time_authors
                if time in time_authors:
                    time_authors[time].add(author)
                else:
                    time_authors[time] = set([author])

                # author_time_accept
                if accept:
                    if author in author_time_accept:
                        if time in author_time_accept[author]:
                            author_time_accept[author][time] += 1
                        else:
                            author_time_accept[author][time] = 1
                    else:
                        author_time_accept[author] = {time: 1}

            author_cur_l = list(authors)
            author_cur_l.sort()  # sort by name

            for i in range(len(author_cur_l)):
                for j in range(i + 1, len(author_cur_l)):
                    if (author_cur_l[i], author_cur_l[j]) not in edge_time:
                        edge_time[(author_cur_l[i], author_cur_l[j])] = time

            file_num += 1

    print(file_num)

    time_list = list(set(time_authors.keys()))
    time_list.sort()
    id2time = {i: time_list[i] for i in range(len(time_list))}
    time2id = {time_list[i]: i for i in range(len(time_list))}

    time_authors = {time2id[t]: time_authors[t] for t in time_authors}
    time_edges = {time2id[t]: set() for t in time2id}
    for edge in edge_time:
        time_edges[time2id[edge_time[edge]]].add(edge)

    # union
    time_authors_keys = list(time_authors.keys())
    time_authors_keys.sort()
    for t in time_authors_keys:
        # if t == len(id2time) - 1:
        #    break
        if t >= max(time_select):
            break
        time_authors[t + 1] = time_authors[t].union(time_authors[t + 1])
        time_edges[t + 1] = time_edges[t].union(time_edges[t + 1])

        # time_author_paper
        t_str = id2time[t]
        t_next_str = id2time[t + 1]
        author_t_set = set(time_author_paper[t_str])
        author_t1_set = set(time_author_paper[t_next_str])
        for author in author_t_set.union(author_t1_set):
            if author in author_t_set and author in author_t1_set:
                time_author_paper[t_next_str][author] = \
                    time_author_paper[t_next_str][author].union(time_author_paper[t_str][author])
            elif author in author_t_set and author not in author_t1_set:
                time_author_paper[t_next_str][author] = time_author_paper[t_str][author]

    # statistics
    time_authors_num = {t: len(time_authors[t]) for t in time_authors}
    time_edge_num = {t: len(time_edges[t]) for t in time_edges}

    time_authors_num_select = {t: time_authors_num[t] for t in time_select}
    time_edge_num_select = {t: time_edge_num[t] for t in time_select}
    print("time_node_num: ", time_authors_num_select)
    print("time_edge_num: ", time_edge_num_select)

    paperNum_distri = Counter(authors_all.values())
    print(paperNum_distri)

    # extract info in the time range [time_start, time_end)
    # info includes node, edge, treatment, outcome
    author_time_select = {a: authors_time[a] for a in time_authors[time_end]}  # name: first time
    author_select = sorted(author_time_select.items(), key=lambda kv: kv[1])  # sort by first time
    author2id_select = {author_select[i][0]: i for i in
                        range(len(author_select))}  # name: id; sorted by increasing order of first appearing time

    # network_list
    n = len(author2id_select)  # author num (last selected time step)
    network_list = []
    for ti in range(len(time_select)):
        t = time_select[ti]
        network_t = np.zeros((n, n), dtype=float)
        for (a1, a2) in time_edges[t]:
            if a1 not in author2id_select:
                print(author_time_select[a2])
            a1_id = author2id_select[a1]
            a2_id = author2id_select[a2]
            network_t[a1_id][a2_id] = network_t[a2_id][a1_id] = 1
        network_list.append(network_t)

    time_author_paper_select = {time: time_author_paper[time] for time in time_author_paper if
                                time2id[time] in time_select}
    feature_list, buzzy_dims_new, author_treat_list = extract_feature_peer(path, author2id_select, time_author_paper_select, time_select,
                                                        id2time)

    # sparse
    feature_list_sparse = [csc_matrix(feat) for feat in feature_list]
    network_list_sparse = [csc_matrix(net) for net in network_list]

    # trn, val, tst
    trn_id_list = random.sample(range(n), int(n * trn_rate))
    not_trn = list(set(range(n)) - set(trn_id_list))
    tst_id_list = random.sample(not_trn, int(n * tst_rate))
    val_id_list = list(set(not_trn) - set(tst_id_list))
    trn_id_list.sort()
    val_id_list.sort()
    tst_id_list.sort()

    # save
    scio.savemat('../dataset/simulate_new/PeerRead/' + 'peerRead' + '.mat', {
        'X': feature_list_sparse, 'A': network_list_sparse,
        'buzzy_dims': buzzy_dims_new,
        'orin_treat': author_treat_list,
        'trn_idx': trn_id_list, 'val_idx': val_id_list, 'tst_idx': tst_id_list
    })

    return feature_list, network_list


def extract_feature_peer(path, author2id, time_author_paper, time_selected, id2time):
    '''
    For selected authors, extract their features in selected time period
    :param path:
    :param author2id: {name (str): ID (0-num)}
    :param time_author_paper: {time: {name (str): set{paperRawID}}}  [new papers, not accumulated]
    :param paper_time: {paper: time (xxxx-xx-xx)}
    :return:
    '''
    sample_obs_rate = 0.5  # observed text feature
    # max_feature = 500  # top words
    C = 200
    num_topics = 50
    num_reduce_topics = 25
    buzzy_word_set = set(["deep", "neural", "network", "model"])
    # nltk.download()

    files = os.listdir(path + 'parsed_pdfs/')
    corpus = []

    filename2id = {}  # rawid: id (0~num-1)
    fileNum = 0

    for file in files:  # all files (papers) in the fold
        filepath = path + 'parsed_pdfs/' + file
        # if file not in paper_all_set:
        #    continue
        file = file.replace('.pdf', '')
        filename2id[file] = fileNum
        fileNum += 1
        with open(filepath) as json_file:
            data = json.load(json_file)

            title = data['metadata']['title']
            abstract = data['metadata']['abstractText']

            if title is not None:
                title = title.encode('ascii', 'ignore')
            else:
                with open(path + 'reviews/' + file.replace(".pdf", "")) as review_file:
                    rw_data = json.load(review_file)
                    title = rw_data['title'].encode('ascii', 'ignore')

            if abstract is not None:
                abstract = abstract.encode('ascii', 'ignore')
            else:
                with open(path + 'reviews/' + file.replace(".pdf", "")) as review_file:
                    rw_data = json.load(review_file)
                    abstract = rw_data['abstract'].encode('ascii', 'ignore')

            title = title.decode("utf-8")
            abstract = abstract.decode("utf-8")
            text = title + '. ' + abstract
            corpus.append(text)

    corpus = text_clean(corpus)  # clean the text

    # observed features
    last_time = max(time_author_paper)
    author_paper_obs = {}
    author_paper_to_id = {}
    corpus_sampled = []  # each element corresponds to a (author, paper)
    id_sample = 0
    for author in time_author_paper[last_time]:
        # randomly sample a piece of text
        for paper in time_author_paper[last_time][author]:
            paper_id = filename2id[paper]
            text = corpus[paper_id]

            tokenized_text = word_tokenize(text)
            samples = random.sample(tokenized_text, int(sample_obs_rate * len(tokenized_text)))
            text_sample = ""  # sampled text for this author, this paper
            for smp in samples:
                text_sample += (" " + smp)
            if author in author_paper_obs:
                author_paper_obs[author][paper] = text_sample
            else:
                author_paper_obs[author] = {paper: text_sample}

            # by sent tokenization
            '''
            tokenized_text=sent_tokenize(text)
            num_sent = len(tokenized_text)
            samples = random.sample(range(num_sent), int(sample_obs_rate * num_sent))

            text_sample = ""      # sampled text for this author, this paper
            for smp in samples:
                text_sample += (" " + tokenized_text[smp])
            if author in author_paper_obs:
                author_paper_obs[author][paper] = text_sample
            else:
                author_paper_obs[author] = {paper: text_sample}
            '''
            corpus_sampled.append(text_sample)
            author_paper_to_id[(author, paper)] = id_sample
            id_sample += 1

    # =============  features by counting words ====================
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=LemmaTokenizer())

    text_feat = cv.fit_transform(corpus_sampled)  # paper num x word num
    text_feat = text_feat.toarray()

    feature_name = cv.get_feature_names()  # dictionary
    print('word num:', len(feature_name))

    ps = PorterStemmer()
    buzzy_word_set_stem = set([ps.stem(bw) for bw in buzzy_word_set])

    buzzy_dims = np.array([wi for wi in range(len(feature_name)) if feature_name[wi] in buzzy_word_set_stem],
                          dtype=np.int32)

    buzzy_words_new = [feature_name[b_dim] for b_dim in buzzy_dims]
    print("buzzy: ", buzzy_words_new)

    # get the most freq k words of each topic
    lda = LatentDirichletAllocation(n_components=num_topics).fit(text_feat)
    topics = lda.components_  # topic x feat

    # calculate the topic k words in each topic
    topics_k_dims = np.argsort(topics, axis=1)[:, -num_reduce_topics:]  # topic x dim_x_reduce

    # then we get a union of all those top 100 words
    unique_k_dims = set(np.unique(topics_k_dims))
    unique_k_dims = list(set(buzzy_dims).union(unique_k_dims))

    unique_k_dims.sort()
    unique_k_dims = np.array(unique_k_dims)  # buzzy word + top word

    buzzy_dims_new = [i for i in range(len(unique_k_dims)) if unique_k_dims[i] in buzzy_dims]

    # reduce the dimensions by extract the selected words
    text_feat = text_feat[:, unique_k_dims]

    max_feature = text_feat.shape[1]
    print('feature dim: ', text_feat.shape[1])

    author_feat_list = []  # list of size T, each elem is a matrix of size n_t x feat_dim

    time_author_paper_add = {}
    for i in range(len(time_selected) - 1, -1, -1):  # t-1~1
        t = time_selected[i]
        t_str = id2time[t]
        if i == 0:
            time_author_paper_add[t_str] = time_author_paper[t_str]
            break
        t_last = time_selected[i - 1]
        t_last_str = id2time[t_last]
        time_author_paper_add[t_str] = {}
        for a in time_author_paper[t_str]:
            if a in time_author_paper[t_last_str]:
                time_author_paper_add[t_str][a] = time_author_paper[t_str][a] - time_author_paper[t_last_str][a]
            else:
                time_author_paper_add[t_str][a] = time_author_paper[t_str][a]

    for t in time_selected:
        t_str = id2time[t]
        author_feat = np.zeros((len(author2id), max_feature))  # author_num x feat
        for a in time_author_paper_add[t_str]:
            a_id = author2id[a]
            for paper in time_author_paper_add[t_str][a]:
                text_id = author_paper_to_id[(a, paper)]
                author_feat[a_id] += text_feat[text_id]
        # author_feat = row_norm(author_feat)       # normalize features
        author_feat_list.append(author_feat)

    # ============= treatment: buzzy word =============
    paper_buzzy_num = get_buzzy_word(corpus, buzzy_word_set)  # |paper|, num of buzzy words
    print("paper_buzzy_num: ", len(paper_buzzy_num) - paper_buzzy_num.count(0), paper_buzzy_num.count(0))

    treat_rate_list = []
    author_treat_list = []
    for t in time_selected:
        author_treat = np.zeros(len(author2id))
        t_str = id2time[t]
        author_buzzy_num = np.zeros(len(author2id))
        for author in time_author_paper[t_str]:
            a_id = author2id[author]
            for paper in time_author_paper[t_str][author]:
                #print(paper)
                paper_id = filename2id[paper]
                author_buzzy_num[a_id] += paper_buzzy_num[paper_id]
            if author_buzzy_num[a_id] > 0:  # treated condition
                author_treat[a_id] = 1.0
        author_treat_list.append(author_treat)

        # statistics
        values, counts = np.unique(author_treat, return_counts=True)
        treated_rate = counts[1] / len(author_treat)
        treat_rate_list.append(treated_rate)
        print("t= ", t, '; ', values, '; ', counts, " treated rate: ", treated_rate)
    print('averaged treated rate (real): ', sum(treat_rate_list)/len(treat_rate_list))

    # ============= potential outcome =============
    '''
    author_num = author_feat_list[0].shape[0]
    eta = np.random.normal(0, 1, author_num).reshape((-1, 1))  # sample noise from Gaussian

    W_x = np.random.random((max_feature, 1))
    W_c = np.random.random((1, 1)) * C
    Bias_1 = np.random.random(1)
    Bias_0 = np.random.random(1)

    author_y1_list = []
    author_y0_list = []
    for ti in range(len(time_selected)):
        a_feature = author_feat_list[ti]
        Y1 = np.matmul(a_feature, W_x) + np.matmul(np.ones((author_num, 1)), W_c) + Bias_1 + eta
        Y0 = np.matmul(a_feature, W_x) + np.matmul(np.zeros((author_num, 1)), W_c) + Bias_0 + eta
        Y1 = Y1.reshape(-1)
        Y0 = Y0.reshape(-1)
        author_y1_list.append(Y1)
        author_y0_list.append(Y0)

    # statistics
    for ti in range(len(time_selected)):
        X = author_feat_list[ti]
        Y1 = author_y1_list[ti]
        Y0 = author_y0_list[ti]
        ATE = np.mean(Y1 - Y0)
        T = author_treat_list[ti]
        # print('ATE is %.3f' % (np.mean(Y1 - Y0)))
        print("t=", ti, " exp_id=", 0, ' x sum=',np.sum(X), " x mean= ", np.mean(X), " x std", np.std(X), "T:",
              float(np.count_nonzero(T)) / T.size, " ATE:", np.mean(Y1 - Y0), np.std(Y1 - Y0))
    '''
    return author_feat_list, buzzy_dims_new, author_treat_list


def text_clean(corpus_sampled):
    r1 = u'[0-9!?"#$%&\'()*+,./:;<=>@\'\\\/[\\]^_{|}~]+'
    ps = PorterStemmer()
    corpus_clean = []

    for sentence in corpus_sampled:
        sentence = re.sub(r1, ' ', sentence)  # remove numbers, symbols
        sentence = re.sub('abstract', ' ', sentence)
        sentence = re.sub('introduction', ' ', sentence)
        sentence = sentence.lower()  # lowercase

        # stemming and lemmatization
        word_tokens = word_tokenize(sentence)
        word_stem = [ps.stem(w) for w in word_tokens]

        sentence = ' '.join(word_stem)

        corpus_clean.append(sentence)

    return corpus_clean


def get_buzzy_word(corpus, buzzy_word_set):
    paper_buzzy_num = []
    for i in range(len(corpus)):
        text = corpus[i]
        tokens_set = set(word_tokenize(text))
        buzzy_num = len(buzzy_word_set.intersection(tokens_set))
        paper_buzzy_num.append(buzzy_num)
    return paper_buzzy_num


def loadData_peer2(path):
    '''
    the data loader for "peerRead"
    :param path:
    :return: feature, network
    '''
    files = os.listdir(path)

    authors_all = {}
    authors_time = {}

    Months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    Days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', ]

    time2id = {}

    time = 0
    for year in Years:
        for mon in Months:
            time2id[year + ' ' + mon] = time
            time += 1

    id2time = {time2id[t]: t for t in time2id}

    time_edges = {time2id[t]: 0 for t in time2id}

    # year2id = {Years[i]:i for i in range(len(Years))}
    # s = []
    time_authors = {t: set() for t in id2time}

    file_num = 0
    author_num_all = 0

    for file in files:  # all files in the fold
        filepath = path + file
        with open(filepath) as json_file:
            data = json.load(json_file)
            authors = data['authors']
            year = str(data['DATE_OF_SUBMISSION'][-4:])
            month = str(data['DATE_OF_SUBMISSION'][-8:-5])
            time = year + ' ' + month

            for author in authors:
                author = author.encode('ascii', 'ignore')
                if author in authors_all:
                    authors_all[author] += 1
                    authors_time[author].add(time)
                else:
                    authors_all[author] = 1
                    authors_time[author] = set([time])

                time_authors[time2id[time]].add(author)

            author_num = len(authors)
            time_edges[time2id[time]] += author_num * (author_num - 1) / 2

            author_num_all += author_num
            file_num += 1

    print(author_num_all, file_num)

    for t in time_authors:
        if id2time[t] == '2017 Dec':
            continue
        time_authors[t + 1] = time_authors[t].union(time_authors[t + 1])
        time_edges[t + 1] = time_edges[t] + time_edges[t + 1]
    year_authors_num = {t: len(time_authors[t]) for t in time_authors}

    print(authors_all)
    return


# randomly add edges,  add num = ratio * edgenum
def add_edges(A_dense, ratio=0.001):
    n = len(A_dense)
    num_edges = np.count_nonzero(A_dense == 1) / 2
    half_A_dense = A_dense + np.tril(np.ones((n, n)))  # keep the part (s, t), s < t
    index_zero = np.where(half_A_dense == 0)

    n_zero = len(index_zero[0])  # num of zero elem in half_A_dense
    add_num = int(ratio * num_edges)
    index_add = np.random.choice(n_zero, add_num)

    A_dense[(index_zero[0][index_add], index_zero[1][index_add])] = 1
    A_dense[(index_zero[1][index_add], index_zero[0][index_add])] = 1

    # check new number of edges
    num_edges_now = np.count_nonzero(A_dense == 1) / 2
    edge_add = num_edges_now - num_edges
    return A_dense

# randomly remove edges, remove num = ratio * edgenum
def remove_edges(A_dense, ratio=0.001):
    n = len(A_dense)
    num_edges = np.count_nonzero(A_dense == 1) / 2

    half_A_dense = A_dense * (1 - np.tril(np.ones((n, n))))  # keep the part (s, t), s < t
    index_nonzero = np.where(half_A_dense != 0)

    # random sample edges to remove
    n_nonzero = len(index_nonzero[0])
    rm_num = int(ratio * num_edges)
    index_rm = np.random.choice(n_nonzero, rm_num)

    A_dense[(index_nonzero[0][index_rm], index_nonzero[1][index_rm])] = 0
    A_dense[(index_nonzero[1][index_rm], index_nonzero[0][index_rm])] = 0

    num_edges_now = np.count_nonzero(A_dense == 1) / 2
    edge_remove = num_edges_now - num_edges

    return A_dense


# perturb the degree of the graph
def perturbGraph(A_dense):
    n = A_dense.shape[0]
    alpha = 0.01
    num_edges = np.sum(A_dense, axis=1)
    top_num = int(alpha * n)
    top_deg_index = num_edges.argsort()[-top_num:][::-1]
    least_deg_index = num_edges.argsort()[:top_num]

    perturb_matrix = np.eye(n)
    tmp = perturb_matrix[top_deg_index].copy()
    perturb_matrix[top_deg_index] = perturb_matrix[least_deg_index]
    perturb_matrix[least_deg_index] = tmp

    A_dense_pert = np.matmul(np.matmul(perturb_matrix, A_dense), perturb_matrix.T)
    return A_dense_pert

# no good network
def loadData_flickr3(path):
    data = scio.loadmat(path)

    X = data['Attributes']  # csc_matrix: 7575 x 12047
    n = X.shape[0]
    X_dense = np.array(X.todense())

    A = data['Network']  # csc_matrix: 7575 x 7575
    A_dense = np.array(A.todense())

    # get the most freq k words of each topic
    num_topics = 50
    num_reduce_topics = 25
    lda = LatentDirichletAllocation(n_components=num_topics).fit(X_dense)
    topics = lda.components_  # topic x feat

    # calculate the topic k words in each topic
    topics_k_dims = np.argsort(topics, axis=1)[:, -num_reduce_topics:]  # topic x dim_x_reduce

    # then we get a union of all those top 100 words
    unique_k_dims = np.unique(topics_k_dims)

    X_dense = X_dense[:, unique_k_dims]  # original feature

    edge_num = np.count_nonzero(A_dense == 1)/2
    print('original edge number: ', edge_num)

    if is_symmetric(A_dense):
        print("Symmetric")
    degree_node = np.zeros(n)
    for i in range(n):
        degree_node[i] = len(np.nonzero(A_dense[i])[0])
    ave_degree = np.average(degree_node)
    degree_node.sort()

    print('degree distri:', degree_node, ' average degree:', ave_degree)

    # ================   generate dynamic graph  ===============================
    X_dense_list = [X_dense.copy()]
    A_dense_list = [A_dense.copy()]

    MX = np.mean(row_norm(X_dense))
    VX = np.mean(np.std(row_norm(X_dense), axis=0))
    print("time: ", 0, MX, VX)

    time_step = 25

    # randomly inject/remove 0.1% new edges and change 0.1% attribute values
    rate_new_edge = 0.001
    rate_update_feature = 0.0001
    sigma = 2

    A_dense = perturbGraph(A_dense)

    for t in range(time_step - 1):
        # 1. Network: randomly inject/remove 0.1% edges
        if t % 2 == 1:
            A_dense = add_edges(A_dense, ratio=rate_new_edge)
        else:
            A_dense = remove_edges(A_dense, ratio=rate_new_edge)

        A_dense_list.append(A_dense.copy())
        edge_num = np.count_nonzero(A_dense == 1) / 2

        # 2. Features: perturbate 0.1% node features (based on the noises sampled from N(0, 0.01^2)
        # change 0.1% attribute values
        # feat_space = X_dense.shape[0] * X_dense.shape[1]
        # num_new_feat = int(feat_space * rate_update_feature)
        # index = random.sample(range(feat_space), num_new_feat)
        # update_info = np.random.normal(0, sigma, num_new_feat)
        # update_info = update_info.astype(int)  # word num, so must be int
        #
        # # print("update: ",update_info)
        # noise = np.zeros((X_dense.shape[0], X_dense.shape[1]))
        #
        # idx_row = [int(idx / X_dense.shape[1]) for idx in index]
        # idx_col = [idx - int(idx / X_dense.shape[1]) * X_dense.shape[1] for idx in index]
        # index_newfeat = (idx_row, idx_col)
        #
        # noise[index_newfeat] = update_info
        # X_dense = X_dense + noise
        #
        # X_dense[np.where(X_dense < 0)] = 0  # make sure >= 0
        X_dense_list.append(X_dense.copy())

        MX = np.mean(row_norm(X_dense))
        VX = np.mean(np.std(row_norm(X_dense), axis=0))
        print("time: ", t, edge_num, MX, VX)

    return X_dense_list, A_dense_list


def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def initialization(X, A_dense, row_norm_AZ):
    lda = LatentDirichletAllocation(n_components=50)
    lda.fit(X)
    Z = lda.transform(X)  # node x n_topics

    AZ = np.matmul(A_dense, Z)  # neighbor propagation
    if row_norm_AZ:
        AZ = row_norm(AZ)

    return Z, AZ


def row_norm(a):
    # for normalize the AZ matrix, which can be too huge
    row_sums = a.sum(axis=1)
    row_sums[np.where(row_sums == 0)] = 1  # if sum = 0
    norm_a = a.astype(float) / row_sums[:, np.newaxis]
    return norm_a


def statics(features, network):
    return


def rbf_kernel(x, sigma=2.0):
    return np.exp(-np.sum(np.abs(x), axis=-1) / sigma)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
