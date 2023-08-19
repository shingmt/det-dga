import logging
import re
from collections import defaultdict, Counter

import numpy
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
import math
from collections import Counter
from keras.utils import pad_sequences

import warnings
warnings.simplefilter("ignore")

#sl = 'google'
#sl = 'xkjhqo5cnxizd7fezeiguontwpn'
# lay du lieu

#domains_data = open('dulieu1.txt','r')
#domains = [x.rstrip() for x in domains_data]

#1(128) so luong ki tu : độ dài trung bình của SL trong bộ dữ liệu lành tính là 8,97 ký tự, độ dài SL trung bình của SL dựa trên DGA là 14.75 ký tự.
def Domain_length(sl):
    MIN = 1
    MAX = 57
    len_sl = len(sl)
    return ((len_sl - MIN)/(MAX - MIN))
#print(Domain_length(sl))

#2(129) ti le nguyen am - phu am
def Vowel_Consonant_ratio(sl):
    V = 0
    C = 0
    vowels = {'a' , 'e' , 'o' , 'i' , 'u'} #5 nguyen am
    consonants = {'b' , 'c' , 'd' , 'f' , 'g' , 'h' , 'j' , 'k' , 'l' , 'm' , 'n' , 'p' , 'q' , 'r' , 's' , 't' , 'v' , 'w' , 'x' , 'y' , 'z' } #21 phu am
    for kitu in sl:
        if kitu in vowels:
            V = V + 1
        if kitu in consonants:
            C = C + 1
    if C==0 :
        R = V
    else :
        R = V/C
    # >= 0,5: doc hai, < 0,5: lanh tinh
    return R

#print(Vowel_Consonant_ratio(sl))

#3(130) so luong phu am lien tiep toi da
def Maximum_consecutive_consonant_count(sl):
    MIN = 0
    MAX = 25
    count = 0
    max = 0
    x = 0
    consonants = {'b' , 'c' , 'd' , 'f' , 'g' , 'h' , 'j' , 'k' , 'l' , 'm' , 'n' , 'p' , 'q' , 'r' , 's' , 't' , 'v' , 'w' , 'x' , 'y' , 'z' } #21 phu am
    for i in range(len(sl)):
        if sl[i] in consonants:
            count += 1
            if count > max:
                max = count
        else:
            count = 0
    return max

#print(Maximum_consecutive_consonant_count(sl))
#4(131) ti le giua so va chu
def Number_count(sl):
    num = 0
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    for i in sl:
        if i in numbers:
            num += 1
    return (num/len(sl))

#for i in domains:
#    print(i)
#    print(Number_count(i))

#5(132) so luong gach noi
def Hyphen_count(sl):
    x = 0
    hyp = '-'
    if hyp in sl:
        x = 1
    return x
#print(Hyphen_count(sl))

#6(133) phu am lap di lap lai
def Repeated_consonants(sl):
    MIN = 0
    MAX = 9
    cons = []
    count = {}
    consonants = {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y','z'}  # 21 phu am
    for i in sl:
        if i in consonants:
            cons += i
    for i in cons:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    #print(count)
    max = 0
    for i in count:
        if count[i] > max:
            max = count[i]
    #print(max)
    return max
#Repeated_consonants(sl)

#7(134) nguyen am lap di lap lai
def Repeated_vowels(sl):
    MIN = 0
    MAX = 9
    cons = []
    count = {}
    vowels = {'a' , 'e' , 'o' , 'i' , 'u'} #5 nguyen am
    for i in sl:
        if i in vowels:
            cons += i
    for i in cons:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    #print(count)
    max = 0
    for i in count:
        if count[i] > max:
            max = count[i]

    #print(max)
    return max

#8(135) Letter score
def Letter_score(sl):
    s = 0
    n = len(sl)
    score = {'a':0.08,'b': 0.02,'c':0.03,'d':0.04,'e':0.13,'f':0.02,'g':0.02,'h':0.06,'i':0.07,'k':0.01,'l':0.04,'m':0.02,'n':0.07,'o':0.08,'p':0.02,'q':0.00,'r':0.06,'s':0.06,'t':0.09,'u':0.03,'v':0.01,'w':0.02,'x':0.00,'y':0.02,'z':0.00}
    for i in sl:
        for j in score:
            if i == j:
                s += score[j]
    x = s/n
    return x
#print(Letter_score(sl))

#9(136) entropy: mức độ ngẫu nhiên của thành phần ký tự trong nhãn miền.
def entropy(sl):
    p, lns = Counter(sl), float(len(sl))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())
#print(entropy(sl))

#10,11,12(137,138,139) so luong tu co trong tu dien
def dictionary_w(sl):

    dictionary = open('dga/words.txt', 'r')
    words = [x.rstrip() for x in dictionary]
    count_word = 0
    aver_dic_len = 0
    max_dic_len = 0
    dic_word = []
    dic_len = []

    for i in words:
        if i in sl:
            dic_word.append(i)
            count_word += 1
            dic_len.append(len(i))
    #print(dic_word, count_word, dic_len)
    if count_word != 0:
        aver_dic_len = sum(dic_len) / count_word
        max_dic_len = max(dic_len)
    #print(aver_dic_len, max_dic_len)

    return ([count_word,aver_dic_len,max_dic_len])

#13 n-gram
def __stats_over_n_grams(npa):
    """
    Calculates statistical features over ngrams decoded in numpy arrays
    stddev, median, mean, min, max, quartils, alphabetsize (length of the ngram)
    :param npa:
    :return:
    """
# std: do lech chuan, median: trung vi, mean: trung binh cong, percentile: tinh toan phan vi thu n
    if npa.size > 0:
        stats = [npa.std(), numpy.median(npa), npa.mean(), numpy.min(npa), numpy.max(npa), numpy.percentile(npa, 25),
             numpy.percentile(npa, 75)]
    else:
        stats = [-1, -1, -1, -1, -1, -1, -1]

    return stats

def n_grams(sl):
    """
    Calculates various statistical features over the 1-,2- and 3-grams of the suffix and dot free domain
    """
    domain = sl
    global __unigram
    feature = []

    for i in range(1,4):
        ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(i, i))
        counts = ngram_vectorizer.build_analyzer()(domain)
        npa = numpy.array(list(Counter(counts).values()), dtype=int)
        if i == 1:
            __unigram = npa

        feature += __stats_over_n_grams(npa)

    return feature

#print(n_grams())

#data = ['google','xkjhqo5cnxizd7fezeiguontwpn','']

### main
def morphol_features(data):
    X = []
    for i in data:
        if len(i) == 0:
            f = [0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            f = [Domain_length(i), Vowel_Consonant_ratio(i), Maximum_consecutive_consonant_count(i), Number_count(i), Hyphen_count(i), Repeated_consonants(i), Repeated_vowels(i), Letter_score(i), entropy(i)]
            f.extend(dictionary_w(i))
            #f.extend(n_grams(i))
        X.append(f)
    maxlen = numpy.max([len(x) for x in X])  # max do dai cua domain
    X = pad_sequences(X, maxlen=maxlen)
    return (numpy.array(X))
#print(morphol_features(data))
#sl = 'google'
#print(n_grams(sl))