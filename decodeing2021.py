# -*- coding: utf-8 -*-
"""

city： BJ SH
dataround: 202010
category: 'LIQM','YD','BEER','WATER','CSD','SNACK','EDOIL','SP','NUTS','WINE','NOOD','ICECR'
ITEMCODE	Barcode
59957243	6901721494267 停产

coffee -
三佳利，三嘉利，三嘉利亚，Sangaria



"""

import os
import pandas as pd
import jieba
import re


# store_list = pd.read_excel(r'C:\Users\chas0004\Desktop\HUACHUANG\简称对应信息表-20210615.xlsx',
#                    dtype=str)
# store_list = store_list[['局点名']]
# store_list.columns=['storename']
# chain_lists=pd.ExcelFile(r'C:\Users\chas0004\Desktop\TTR\Mega panel\CHAIN\Chain List (BB_COS_MT) 0323.xlsx')
# sheet_names=chain_lists.sheet_names
import locale
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                                       
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): 
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

def chain_identification(s):
    print(s)
    ## remove space in store name
    s = re.sub(" ", "", str(s))
    ## segments the store name into separated words
    s_split = list(jieba.cut(s, cut_all = False))
    s_chain_name = [x for x in chain_list if str(x) in s_split]
    return s_chain_name

def string_clean(l):
    l = str(l).replace("[", "")
    l = l.replace("]", "")
    l = l.replace("', '", "/")
    l = l.replace("@", "")
    l = l.replace("$",'')
    l = l.replace("*",'')
    l = l.replace("/",'')
    l = l.replace(" ",'')
    l = l.replace("（",'')
    l = l.replace("）",'')
    l = l.replace(".",'')
    l = l.replace("^",'')
    l = l.replace("_",'')
    l = l.replace("?",'')
    l = l.replace(" ",'')
    return l
#path2chain = r'C:\Users\waya0001\Downloads\TTR\workfile'



def chain(sheet_names,chain_lists,store_list):
    for i in sheet_names:
        chain_list = pd.read_excel(chain_lists,i)
        chain_list = list(chain_list['零售商'].str.upper())
        ## add chain names to word pool for Jieba
    for i in chain_list:
        jieba.add_word(i,freq=10,tag='nz')
        
        store_list['chain_name_'+str(i)] = store_list['storename'].str.upper().apply(chain_identification)
        store_list['chain_name_'+str(i)] = store_list['chain_name_'+str(i)].apply(string_clean)
        return store_list
#store_list.to_excel(r'C:\Users\chas0004\Desktop\HUACHUANG\check_chain2.xlsx' , index=False, na_rep='',encoding='GB18030')


#%%
import numpy
from sklearn.preprocessing import StandardScaler


import os
#from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
import csv

import jieba
import sys
import os
#from config_ch import *
import chardet
import numpy as np
import pandas as pd
import xlrd
import copy
import glob
import jieba.posseg
import jieba.analyse
import io
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import Levenshtein



from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

os.getcwd()
## CHANGE
p = r'C:\Users\base1001\Documents\Python Scripts'
os.chdir(p)

pd.set_option('display.max_rows', 310)
pd.set_option('display.max_columns', 310)
pd.set_option('display.width', 500)

from round1_func import *
#from nlp_func1 import *

import locale
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                                       
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): 
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def cut_word(word):
    #cw = jieba.cut_for_search(word)
    cw = jieba.cut(word)
    return list(cw)

def string_clean(l):
    l = str(l).replace("[", "")
    l = l.replace("]", "")
    l = l.replace("', '", "/")
    l = l.replace("@", "")
    l = l.replace("(", "")
    l = l.replace(")", "")
    l = l.replace("#", "")
    l = l.replace('【领券满99减20】', "")
    l = l.replace('【领券满39减8】', "")
    l = l.replace('【领券满89减8】', "")
    l = l.replace('【领券89减20】', "")
    l = l.replace('【闪购福利】', "")
    l = l.replace('【泡面】', "")
    l = l.replace('【碗】', "")
    l = l.replace('【整箱】', "")
    l = l.replace('【整包】', "")
    l = l.replace('【大包】', "")
    l = l.replace('【单包】', "")
    l = l.replace('【五连包】', "")
    l = l.replace('【网红推荐】', "")
    l = l.replace('【网红爆款】', "")
    l = l.replace('【抖音爆款】', "")
    l = l.replace('抖音', "")
    l = l.replace('【网红】', "")
    l = l.replace("【", "")
    l = l.replace("】", "")
    l = l.replace("亚州", "亚洲")
    l = l.replace("$",'')
    l = l.replace("*",'')
    l = l.replace("/",'')
    l = l.replace("@@",'')
    l = l.replace(" ",'')
    l = l.replace("（",'')
    l = l.replace("）",'')
    l = l.replace(".",'')
    l = l.replace("^",'')
    l = l.replace("_",'')
    l = l.replace("?",'')
    l = l.replace("4合1",'四合一')
    l = l.replace("3合1",'三合一')
    l = l.replace("2合1",'二合一')
    l = l.replace("1+2",'雀巢一加二')
    l = l.replace("3点1刻",'三点一刻')
    l = l.replace("3点一刻",'三点一刻')
    l = l.replace("Coca-Cola",'')
    l = l.replace("塑料",'')
    l = l.replace("玻璃",'')
    l = l.replace("纸",'')
    l = l.replace("GMX",'GM')
    l = l.replace("gmx",'gmx')
    l = strQ2B(l)
    l = l.replace(" ",'')
    return l

def brand_clean(l):
    l = l.replace("亚州", "亚洲")
    l = l.replace("【", "")
    l = l.replace("】", "")
    l = l.replace("@@",'')
    l = l.replace("@", "")
    l = l.replace("(", "")
    l = l.replace(")", "")
    l = l.replace("#", "")
    l = l.replace("{", "")
    l = l.replace("}", "")
    l = l.replace("$",'')
    l = l.replace("*",'')
    l = l.replace("/",';')
    l = strQ2B(l)
    return(l)


def debrand(l):
    l = str(l[0]).replace(str(l[1]), '')
    return l


import jieba
import jieba.analyse as analyse


import random
def extract_nr(text):
    allow_pos = ('nr','nz','nt','ns')
#allow_pos = ('nr','nz','nt', 'n')
    tags = analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=allow_pos)
    return tags

def extract_a(text):
    allow_pos = ('a','ad','an','d')
#allow_pos = ('nr','nz','nt', 'n')
    tags = analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=allow_pos)
    return tags

def extract_q(text):
    allow_pos = ('m','q')
#allow_pos = ('nr','nz','nt', 'n')
    tags = analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=allow_pos)
    return tags

def extract_n(text):
    allow_pos = ('n','vn')
#allow_pos = ('nr','nz','nt', 'n')
    tags = analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=allow_pos)
    return tags

def extract_v(text):
    allow_pos = ('v','vd', 'ad')
    tags = analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=allow_pos)
    return tags

import re
def brand_sp(text):
    UNESCAPED_SLASH_RE = re.compile(';|/')
    a = UNESCAPED_SLASH_RE.split(text)
    return a

def get_TF(words,topk = 5):
	
	tf_dic = {}
	for w in words:
		tf_dic[w] = tf_dic.get(w,0) + 1
	return sorted(tf_dic.items(),key = lambda x : x[1],reverse = True)[:topk]  #get top 10

def depart(s):
    eng = "".join(i for i in s if ord(i) < 256)
    chn = "".join(i for i in s if ord(i) >= 256)
    chn = chn.replace('包装','')
    chn = chn.replace('装','')
    return chn

def denumeric(s):
    result = ''.join([i for i in s if not i.isdigit()])
    return result


def text_prep(df, col='DESCRIPTION'):
    df[col]= df[col].apply(string_clean)
    df['textual'] = df[col].str.replace('\d+', '')
    df['textual'] = df['textual'].apply(depart)
    df['special'] =df['textual'].apply(extract_nr)
    df['nouns'] =df['textual'].apply(extract_n)
    df['ads'] =df['textual'].apply(extract_a)
    df['cut_DESCRIPTION']= eric_cat['textual'].apply(cut_word)
    
    return df
def numsplit(s):
    if 'mg' in s.lower():
        try:
            hd = re.findall(r"(\d+)mg", s.lower())[0]
            hd = int(hd)
        except IndexError:
            hd = 0
    if 'ml' in s.lower():
        try:
            hd = re.findall(r"(\d+)ml", s.lower())[0]
            hd = int(hd)
        except IndexError:
            hd = 0
    elif "g" in s.lower():
        try:
            hd = re.findall(r"(\d+)g", s.lower())[0]
            hd = int(hd)
        except IndexError:
            hd = 0
    elif "gm" in s.lower():
        try:
            hd = re.findall(r"(\d+)gm", s.lower())[0]
            hd = int(hd)
            try:
                hd2 = re.findall(r"gm(\d+)", s.lower())[0]
                hd = int(hd2)*hd
            except IndexError:
                pass
        except IndexError:
            hd = 0
    elif "克" in s:
        try:
            hd = re.findall(r"(\d+)克", s.lower())[0]
            hd = int(hd)
        except IndexError:
            hd = 0
    elif "条" in s:
        try:
            hd = re.findall(r"(\d+)条", s.lower())[0]
            hd = int(hd)
        except IndexError:
            hd = 0
    else:
        hd = 0
    return hd

import numpy as np

def levenshtein2(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in xrange(size_x):
        matrix [x, 0] = x
    for y in xrange(size_y):
        matrix [0, y] = y

    for x in xrange(1, size_x):
        for y in xrange(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    print (matrix)
    return (matrix[size_x - 1, size_y - 1])



from sentence_transformers import SentenceTransformer
#sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
#sentence_embeddings = model.encode(sentences)

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    

from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import operator

def tfidf_imdb(keyword, cc, col ='ITEM_DESCRIPTION'):
    texts =cc[col].tolist()
    texts_cut = [lcut(text) for text in texts]
    
    dictionary = Dictionary(texts_cut)
    num_features = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts_cut]
    kw_vector = dictionary.doc2bow(lcut(keyword))
    tfidf = TfidfModel(corpus)
    tf_texts = tfidf[corpus] 
    tf_kw = tfidf[kw_vector]
    sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
    similarities = sparse_matrix.get_similarities(tf_kw)
    
    sim_dict = dict(zip(texts, list(similarities)))
    txt = max(sim_dict.items(), key=operator.itemgetter(1))[0]
    
    barcode = cc[cc[col]== txt]['Barcode'].tolist()[0]
    return barcode
    



#print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
#print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])


#%%
writing_opt =0
CATEGORY = ['COFF','RTDC']
raw_path = r'C:\Users\base1001\Documents\O2O'

# imdb0604 = pd.read_csv(os.path.join(raw_path,'imdb.csv'), encoding='GB18030',
#                         usecols=['ITEMCODE','EAN', 'LONGDESC','MULTIPACK','PACKSIZE', 'CATCODE','CATEGORY', 'BRANDCODE','CBRAND'],
#                         dtype=str)

# source_ids = pd.read_excel(os.path.join(raw_path,'o2o_banner_list_update.xlsx'),sheet_name=0)
# src_ids =dict(zip(source_ids.SID, source_ids['Banner Name']))

# imdb_cat0 = imdb0604[imdb0604.CATCODE.isin(['COFF', 'RTDC'])]
# imdb_cat0 =imdb_cat0[~imdb_cat0.EAN.isin(['NO-EAN'])]


offline = pd.read_csv(os.path.join(raw_path,'o2o_bjsh_store_offline.csv'), encoding='utf-8', dtype={'STORECODE':str, 'ITEMCODE':str, 'PERIODCODE':str})
#                         usecols=['ITEMCODE','EAN', 'LONGDESC','MULTIPACK','PACKSIZE', 'CATCODE','CATEGORY', 'BRANDCODE','CBRAND'],
#                         dtype=str)
offline['unit_price'] = offline['RAW_SALESVALUE']/offline['RAW_SALESUNIT']
offline['unit_price'] = offline['unit_price'].apply(lambda x: round(x, 2))
itm_price = pd.pivot_table(offline, index = ["ITEMCODE"], values = "unit_price", aggfunc = 'mean').reset_index()

offline = offline[offline.CATEGORYCODE.isin(CATEGORY)]

periods = list(set(offline.PERIODCODE))

unit_avg = pd.pivot_table(offline, index = ["ITEMCODE", "STORECODE",'STORENAME'], values = "unit_price", aggfunc = 'mean').reset_index()
unit_avg['unit_price'] = unit_avg['unit_price'].apply(lambda x: round(x, 2))
unit_avg.sort_values('ITEMCODE', inplace=True, ascending=False)

unit_avg.to_csv(os.path.join(raw_path,'unit_price_coff.csv'),encoding='GB18030')
len(set(unit_avg.ITEMCODE))

itm_price = pd.pivot_table(offline, index = ["ITEMCODE"], values = "unit_price", aggfunc = 'mean').reset_index()


#%%
raw_path = r'C:\Users\base1001\Documents\O2O'
raw_path =r'/Users/irena/Documents/O2O'
#del imdb0604
imdb_kenny = pd.read_csv(os.path.join(raw_path,'o2o_imdb_19cat.csv'), dtype={'ITEMCODE':str})

#imdb = imdb_kenny.pivot(index=imdb_kenny.ITEMCODE, columns='ATTR')['ATTRVALUE', 'ATTRVALUE_CN']
imdb = imdb_kenny.groupby(['ITEMCODE', 'ATTR'])['ATTRVALUE'].aggregate('first').unstack()
imdb_cn = imdb_kenny.groupby(['ITEMCODE', 'ATTR'])['ATTRVALUE_CN'].aggregate('first').unstack()
imdb_cat= imdb[imdb['0001 PC'].isin(CATEGORY)]
imdb_cat_cn= imdb_cn[imdb_cn['0001 PC'].str.contains('咖啡')]



dup_imdb = imdb_cat[imdb_cat.duplicated(['0002_PC_ITEM'])]



cat_special_words = imdb_cat['SHORT_DESCRIPTION/BRAND'].tolist()
for i in cat_special_words:
    jieba.add_word(i,freq=10,tag='nz')


descol = [e for e in imdb_cat.columns if 'DESCR' in e]
descol.extend(['0001 PC', '0002_PC_ITEM','ACTUAL_MULTIPACK','ACTUAL_WEIGHT','BASE_MULTI','BASE_WEIGHT', 'BRAND_MKT','Barcode'])
imdb_cat = imdb_cat[descol]
imdb_cat = imdb_cat[~imdb_cat.Barcode.isin(['NO-EAN'])].reset_index()
imdb_cat['Barcode'] = imdb_cat['Barcode'].astype(str)

imdb_cat['ITEMCODE'] = imdb_cat['ITEMCODE'].astype(str)
imdb_cat = pd.merge(imdb_cat, itm_price, on='ITEMCODE', how='left')

#1- (imdb_w_price['unit_price'].isnull().values.sum()/len(imdb_w_price))

dup_st = imdb_cat[imdb_cat.duplicated(['Barcode'], keep=False)]
dup_st.sort_values('Barcode', inplace=True, ascending=False)


imdb_cat['ITEM_DESCRIPTION']= imdb_cat['ITEM_DESCRIPTION'].apply(string_clean)
imdb_cat['combined']= imdb_cat[['ITEM_DESCRIPTION','SHORT_DESCRIPTION/BRAND']].values.tolist()
imdb_cat['prod_debrand'] =imdb_cat['combined'].apply(debrand)
imdb_cat.drop("combined", axis=1, inplace=True)


imdb_cat['special'] =imdb_cat['prod_debrand'].apply(extract_nr)
imdb_cat['nouns'] =imdb_cat['prod_debrand'].apply(extract_n)
imdb_cat['ads'] =imdb_cat['prod_debrand'].apply(extract_a)



if writing_opt == 1:
    imdb_cat.to_csv(os.path.join(raw_path,'imdb_coff.csv'), index=False,encoding='GB18030')


imdb_cat['textual'] = imdb_cat['ITEM_DESCRIPTION'].str.replace('\d+', '')
imdb_cat['textual'] = imdb_cat['textual'].apply(depart)
imdb_cat['cut_textual'] = imdb_cat['textual'].apply(cut_word)

#key words detection
imdb_brands = set(imdb_cat['SHORT_DESCRIPTION/BRAND'].unique()) 


from jieba import analyse

tfidf = analyse.extract_tags
textrank = analyse.textrank

text = '|'.join(imdb_cat['textual'].tolist())
keywords1 = tfidf(text)

k_n = extract_n(text)
k_nr = extract_nr(text)
k_a = extract_a(text)
k_v = extract_v(text)
for i in keywords1:
    jieba.add_word(i,freq=5,tag='n')




gjz_count = CountVectorizer()
ls = cut_word(text)
textvect = gjz_count.fit_transform(ls)
textvect.todense()
g = gjz_count.vocabulary_

g = {k: v for k, v in sorted(g.items(), key=lambda item: item[1])}


# from itertools import islice

# def take(n, iterable):
#     return list(islice(iterable, n))

# top20 = take(70, g)

# gjz_count = CountVectorizer(
#         min_df=0, 
#         token_pattern=r'\b\w+\b') 


#%%
### loading item list from ERIC
eric = pd.read_csv(os.path.join(raw_path,'O2O_Retailer_Itemlist.csv'), dtype=str)
eric = eric.drop_duplicates()
eric = eric[~eric.DESCRIPTION.isnull()]

nobr_lst = ['5900649050440','4712277710150','8809257331774','8938515559084']

ee = eric[eric.RETAILERITEMCODE.isin(nobr_lst)]


ex_list = ['蛋','保温','趣多多','咖啡味','咖啡色','勺','咖啡杯','咖啡壶', '咖啡糖', '巾','曲奇','徐福记',
           '咖啡巧克力','饼','钻石杯','雪糕','冰激凌','冰淇淋','土司','玉米','口味棒','咖啡盖','糖宝',
           '充值卡','鸡','花生','咖啡机','脆卷','染发','家居','拖','套杯','漫画','汽水','咖啡因', '咖啡豆','壶','碟', '皂', '奶茶','内衣']
eric_cat = eric[eric.DESCRIPTION.str.contains('咖啡')]



eric_cat = eric_cat[~eric_cat.DESCRIPTION.str.contains('|'.join(ex_list))]
eric['NANKEY'] = eric['NANKEY'].astype(str)
eric= eric[eric.NANKEY !='-1']


eric_cat_key = list(set(eric_cat.NANKEY))


chk1 = imdb_cat[~imdb_cat.ITEMCODE.isin(eric_cat_key)]
chk2 = eric_cat[~eric_cat.NANKEY.isin(list(imdb_itcd))]

chk3 = chk2[chk2.DESCRIPTION.str.contains('|'.join(['雀巢','贝纳颂','麦斯威尔']))]

dup_eric = eric_cat_nankey[eric_cat_nankey.duplicated(['NANKEY'])]

s = eric_cat.DESCRIPTION.str.len().sort_values().index
eric_cat = eric_cat.reindex(s)

eric_cat.sort_values('RETAILERITEMCODE', inplace=True, ascending=False)

eric_cat.drop_duplicates(subset=['RETAILERITEMCODE'] ,keep='first', inplace=True)
eric_cat_prep = text_prep(eric_cat, col = 'DESCRIPTION')



# tt = pd.merge(imdb_cat, eric_cat, left_on='Barcode', right_on='RETAILERITEMCODE', suffixes = ('', '_eric'),how='right')
# nonmatch = tt[tt.SOURCEID.isnull()]  

if os.path.isfile(os.path.join(raw_path,'eric_imdbcat_coff.csv'))==False:

    eric_imdbcat = pd.merge(imdb_cat, eric_cat, left_on='Barcode', right_on='RETAILERITEMCODE', suffixes = ('', '_eric'),how='inner')
    
    eric_imdbcat['num'] = eric_imdbcat['DESCRIPTION'].str.lower().apply(numsplit)
    eric_imdbcat['num2'] = eric_imdbcat['ITEM_DESCRIPTION'].str.lower().apply(numsplit)
    eric_imdbcat.loc[eric_imdbcat.num2 ==0, 'num2'] = eric_imdbcat['num']
    text = '|'.join(eric_imdbcat['textual'].tolist())
    keywords2 = tfidf(text)
    
    import gensim
    desc_dict = dict(zip(eric_imdbcat.textual,eric_imdbcat.textual_eric))
    #for key, value in d.items():
    
    a = list(desc_dict.keys())[0]
    b = list(desc_dict.values())[0]
    
    string = [a,b]
    
    texts_list=[]
    for sentence in string:
        sentence_list=[ word for word in jieba.cut(sentence)]
        texts_list.append(sentence_list)
    
    dictionary=gensim.corpora.Dictionary(texts_list)
    print(dictionary)
    print(dictionary.token2id)
    
    eric_imdbcat['levenshtein_dist'] =0
    eric_imdbcat['levenshtein_ratio'] =0
    eric_imdbcat['levenshtein_jaro'] =0
    eric_imdbcat['levenshtein_jarowinkler'] =0
    import Levenshtein
    i=0
    for i in range(0, len(eric_imdbcat)-1):
        a =eric_imdbcat.ITEM_DESCRIPTION[i]
        b = eric_imdbcat.DESCRIPTION[i]
        eric_imdbcat.loc[i,'levenshtein_dist'] =Levenshtein.distance(a,b)
        eric_imdbcat.loc[i,'levenshtein_ratio'] =Levenshtein.ratio(a,b)
        eric_imdbcat.loc[i,'levenshtein_jaro'] =Levenshtein.jaro(a,b)
        eric_imdbcat.loc[i,'levenshtein_jarowinkler'] =Levenshtein.jaro_winkler(a,b)
    
    #eric_imdbcat.to_csv(os.path.join(raw_path,'eric_imdbcat_coff.csv'), index=False,encoding='GB18030')
else:
    eric_imdbcat = pd.read_csv(os.path.join(raw_path,'eric_imdbcat_coff.csv'),encoding='GB18030', dtype={'RETAILERITEMCODE':str, 'Barcode':str})

from difflib import SequenceMatcher
from Bio.Align import PairwiseAligner

aligner = PairwiseAligner()
eric_imdbcat['seq_match'] =0
for i in range(0, len(eric_imdbcat)-1):
    a =eric_imdbcat.ITEM_DESCRIPTION[i]
    b = eric_imdbcat.DESCRIPTION[i]
    eric_imdbcat.loc[i,'seq_match'] =aligner.score(a, b)


cor_df=eric_imdbcat[eric_imdbcat.columns[-5:]]
corr = cor_df.corr()
#%%
### load brand info

brands = pd.read_excel(os.path.join(raw_path,'eCom Brand Keywords.xlsx'),sheet_name = 0)
brands = brands[brands.CATCODE.isin(CATEGORY)]
brands['brand'] = brands['CSEGMENT']
brands.loc[brands.CSEGMENT.isin(['其他牌子','只为出DB使用']),'brand'] = brands['SHORTDESC']
brands['brand'] = brands['brand'].apply(brand_clean)

#brands.loc[brands.brand.str.endswith(';'), 'brands'] = brands['brand'][:-1]

brands['short_list'] = brands.SHORTDESC.str.lower().apply(brand_sp)

def f(lst):
    new_list=[x for x in lst if len(x)>0]
    return new_list
brands['short_list'] =brands['short_list'].apply(f)


brand_dic = dict(zip(brands.brand,brands.short_list))


##imbd brand from intern
xin = pd.read_excel(os.path.join(raw_path,'IMDB_COFF_Xin.xlsx'),sheet_name = 'all')



imbd_dic = dict((key,value) for key, value in brand_dic.items() if key in imdb_brands)

# test = pd.merge(imdb_cat, imdb_cat0, left_on='Barcode', right_on='EAN', how='outer')
# del imdb_kenny

#brands.to_csv(os.path.join(raw_path,'brands_coff.csv'), index=False,encoding='GB18030')
eric_list = list(brands.brand.unique())

#imdb_brands - set(brands.brand.unique())

#eric_imdb = pd.merge(imdb_cat, eric, left_on='Barcode', right_on='RETAILERITEMCODE', how='left')

#%%
#WEB DATA

web = pd.read_csv(os.path.join(raw_path,'o2o_online_coff.csv'))
web = web[web.CATCODE.isin(CATEGORY)]
#eric_imdb = pd.merge(eric, imdb0604, left_on='RETAILERITEMCODE', right_on='EAN', how='inner')
#eric_imdb.to_csv(os.path.join(raw_path,'eric_imdb.gz'),compression='gzip', index=False,na_rep='',encoding='GB18030')

web['PROD_DESC_RAW']=web['PROD_DESC_RAW'].apply(string_clean)

web['unit_price'] =web['SALES_VALUE']/web['SALES_UNIT']
web['unit_price'] = web['unit_price'].apply(lambda x: round(x, 2))
web['cut_desc'] = web['PROD_DESC_RAW'].apply(cut_word)
web['brand_pre']= web['cut_desc'].map(lambda x: x[0])



web['textual'] = web['PROD_DESC_RAW'].str.replace('\d+', '')
web['textual'] = web['textual'].apply(depart)

web['special'] =web['PROD_DESC_RAW'].apply(extract_nr)
web['nouns'] =web['PROD_DESC_RAW'].apply(extract_n)
web['ads'] =web['PROD_DESC_RAW'].apply(extract_a)

'''brista = 每日咖啡师'''

sp1 = ['每日咖啡师','火咖','法郎','乔雅','利趣','中原','雀巢','可比可','贝纳颂','飞可','炭仌', '统一','星巴克','Costa','三得利','可口可乐']
for element in sp1:
    web.loc[(web.PROD_DESC_RAW.str.contains(element))&(web.brand_pre != element), 'brand_pre']=element

web.loc[(web.PROD_DESC_RAW.str.contains(str("三德利"))),'brand_pre'] = '三得利'
web.loc[(web.PROD_DESC_RAW.str.contains(str("雅哈")))|(web.PROD_DESC_RAW.str.contains(str("统一"))), 'brand_pre']="统一"
web.loc[(web.PROD_DESC_RAW.str.contains(str("特浓即溶咖啡饮品91克")))&(web.brand_pre=='1'), 'brand_pre']='雀巢'
web.loc[(web.PROD_DESC_RAW.str.contains(str("奶香咖啡7条装")))&(web.brand_pre=='1'), 'brand_pre']='雀巢'


store_list = list(set(web.STORE_NAME))



#%%
from __future__ import print_function
import sys
import threading
from time import sleep
from itertools import islice

def take(n, iterable):
    return list(islice(iterable, n))




sample_dic = {'雅哈': ['a-ha', '雅哈', '统一', '雅哈']}
from textdistance import levenshtein


imbd_dic.pop('每日牌')
imbd_dic.pop('景蓝')
imbd_dic.pop('百杏林')
imbd_dic.pop('联达cobizco')
imbd_dic.pop('蓝山')

#%%
'''disance'''
coff_sum=pd.DataFrame()
n = 0
for key, value in imbd_dic.items():
    print(key)
    
    if '咖啡' in value:
        value.remove('咖啡')
    if '饮料' in value:
        value.remove('饮料')
    if '抖音' in value:
        value.remove('抖音')
    if '网红' in value:
        value.remove('网红')
    if '爆款' in value:
        value.remove('爆款')
    if '国进口' in value:
        value.remove('国进口')  
    if '越南' in value:
        value.remove('越南')  
    value = list(set(value))
    print(value)
    #eric_tmp = eric_cat_prep[eric_cat_prep.DESCRIPTION.str.contains('|'.join(value))]
    eric_tmp = eric_imdbcat[eric_imdbcat.DESCRIPTION.str.contains('|'.join(value))]
    web_tmp = web[web.PROD_DESC_RAW.str.contains('|'.join(value))].reset_index()
    if n >10:
        break
    else:
        if (len(web_tmp)>0) & (len(eric_tmp)>0):
            eric_tmp['DESCRIPTION'] = eric_tmp['DESCRIPTION'].astype(str)
            eric_tmp['DESCRIPTION'] = eric_tmp['DESCRIPTION'].str.lower()
            web_tmp['PROD_DESC_RAW'] = web_tmp['PROD_DESC_RAW'].apply(string_clean)
            web_tmp['eric_match'] = ''
            for i in range(0, len(web_tmp)):
                a =web_tmp.PROD_DESC_RAW[i]
                cat = web_tmp.CATCODE[i]
                eric_tmp['target'] = a.lower()
                eric_tmp['levenshtein_dist'] =eric_tmp.apply(lambda x: levenshtein.distance(x['target'],  x['DESCRIPTION']), axis=1)
                if len(eric_tmp[eric_tmp['0001 PC']==cat])>0:
                    
                    er = eric_tmp[eric_tmp['0001 PC']==cat].sort_values('levenshtein_dist',ascending=True)
                    if len(er)>10:
                        er = er[0:10]
                    
                    barcode = tfidf_imdb(a, er, col ='ITEM_DESCRIPTION')
                    
                    web_tmp.loc[i, 'barcode_levenshtein'] = er.iloc[0]['RETAILERITEMCODE']
                    web_tmp.loc[i, 'levenshtein_dist'] = er.iloc[0]['levenshtein_dist']
                    web_tmp.loc[i, 'eric_desc'] = er.iloc[0]['DESCRIPTION']
                    web_tmp.loc[i, 'barcode_tfidf'] = barcode
                    web_tmp.loc[i, 'eric_tfidf_desc'] = er[er['Barcode']== barcode]['ITEM_DESCRIPTION'].tolist()[0]
                else:
                    pass
            coff_sum=coff_sum.append(web_tmp[~web_tmp.eric_match.isnull()])
            n +=1
        else:
            print('Brand: ' + str(key) + ' has' + str(len(web_tmp)) +  ' web data')
            print('Brand: ' + str(key) + ' has' + str(len(eric_tmp)) +  ' eric data')
    


if writing_opt ==1:
    coff_sum.to_csv(os.path.join(raw_path,'output','0729_coff.csv'),encoding='GB18030')
else:
    pass


#tt = coff_sum.drop_duplicates(subset=['ITEMID','PROD_DESC_RAW'] ,keep='first', inplace=False)
#%%

'''ratio'''
for key, value in brand_dic.items():
    print(key)
    print(value)
    if '咖啡' in value:
        value.remove('咖啡')
    if '饮料' in value:
        value.remove('饮料')
    #eric_tmp = eric_cat_prep[eric_cat_prep.DESCRIPTION.str.contains('|'.join(value))]
    eric_tmp = eric_imdbcat[eric_imdbcat.DESCRIPTION.str.contains('|'.join(value))]
    web_tmp = web[web.PROD_DESC_RAW.str.contains('|'.join(value))].reset_index()
    if (len(web_tmp)>0) & (len(eric_tmp)>0):
        eric_tmp['DESCRIPTION'] = eric_tmp['DESCRIPTION'].astype(str)
        eric_tmp['DESCRIPTION'] = eric_tmp['DESCRIPTION'].str.lower()
        web_tmp['PROD_DESC_RAW'] = web_tmp['PROD_DESC_RAW'].apply(string_clean)
        web_tmp['eric_match'] = ''
        for i in range(0, len(web_tmp)):
            a =web_tmp.PROD_DESC_RAW[i]
            eric_tmp['target'] = a.lower()
            
            
            eric_tmp['levenshtein_dist'] =eric_tmp.apply(lambda x: levenshtein.distance(x['target'],  x['DESCRIPTION']), axis=1)
            eric_tmp.sort_values('levenshtein_dist', inplace=True, ascending=True)
            web_tmp.loc[i, 'eric_match'] = eric_tmp.iloc[0]['RETAILERITEMCODE']
            web_tmp.loc[i, 'levenshtein_dist'] = eric_tmp.iloc[0]['levenshtein_dist']
            web_tmp.loc[i, 'eric_desc'] = eric_tmp.iloc[0]['DESCRIPTION']
            coff_sum=pd.DataFrame()
            
    else:
        print('Brand: ' + str(key) + ' does not have web data')




i=0
a =web_tmp.PROD_DESC_RAW[i]
eric_tmp['target'] = a


eric_tmp['levenshtein_dist'] =eric_tmp.apply(lambda x: levenshtein.distance(x['target'],  x['DESCRIPTION']), axis=1)
eric_tmp[i,'levenshtein_ratio'] =eric_tmp.apply(lambda x: levenshtein.ratio(x['target'],  x['DESCRIPTION']), axis=1)

eric_tmp[i,'levenshtein_jaro'] =Levenshtein.jaro(a,b)
eric_tmp[i,'levenshtein_jarowinkler'] =Levenshtein.jaro_winkler(a,b)



for i in range(0, len(web_tmp)-1):
    a =web_tmp.PROD_DESC_RAW[i]
    eric_tmp['target'] = a.lower()
    eric_tmp['levenshtein_dist'] =eric_tmp.apply(lambda x: levenshtein.distance(x['target'],  x['DESCRIPTION']), axis=1)
    eric_tmp.sort_values('levenshtein_dist', inplace=True, ascending=True)
    web_tmp.loc[i, 'eric_match'] = eric_tmp.iloc[0]['RETAILERITEMCODE']
    web_tmp.loc[i, 'levenshtein_dist'] = eric_tmp.iloc[0]['levenshtein_dist']
    web_tmp.loc[i, 'eric_desc'] = eric_tmp.iloc[0]['DESCRIPTION']











#%%
#eric_imdb = pd.read_csv(os.path.join(raw_path,'eric_imdb.gz'), compression='gzip', encoding='GB18030', sep=',', quotechar='"', dtype={'RETAILERITEMCODE':str, 'EAN':str})
#eric_imdb = eric_imdb.iloc[:, : 16]

#ee ['combined']= ee[['LONGDESC','CBRAND']].values.tolist()
#ee['test'] =ee['combined'].apply(debrand)

#%%
#berry = eric_imdb[eric_imdb.RETAILERITEMCODE.str.startswith('697284473')]

noodle = imdb0604[imdb0604.CATCODE == 'COFF']


non_brand = noodle[noodle.CBRAND == '其他牌子']
non_brand['DESCRIPTION']=non_brand['DESCRIPTION'].apply(string_clean)
non_brand['cut_DESCRIPTION']=non_brand['DESCRIPTION'].apply(cut_word)



non_brand['LONGDESC']=non_brand['LONGDESC'].apply(string_clean)
non_brand['cut_LONGDESC']=non_brand['LONGDESC'].apply(cut_word)



noodle['LONGDESC']=noodle['LONGDESC'].apply(string_clean)

noodle['combined']= noodle[['LONGDESC','CBRAND']].values.tolist()
noodle['test'] =noodle['combined'].apply(debrand)

#noodle['cut_LONGDESC']=noodle['combined'].str.replace(' ':'').apply(cut_word)

from pandas import ExcelWriter
writer = ExcelWriter(os.path.join(raw_path, 'IMDB_NOOD.xlsx'))
noodle.to_excel(writer,'all', index = False)            
non_brand.to_excel(writer,'non_brand',index=False)
writer.save()

catsearch = ['方便面']
searchfor = ['方便面', '捞面','拌面']
#web_tmp = web[(web.CATEGORY_LEVEL_II.str.contains('方便面'))|(web['PROD_DESC_RAW'].str.contains('|'.join(searchfor)))|(web.CATEGORY_LEVEL_I.str.contains('|'.join(catsearch)))]
#web_tmp = web_tmp[~web_tmp['PROD_DESC_RAW'].str.contains('|'.join(['雪糕','冰激凌', '冰淇淋','冰棍','八喜','可口可乐','百事可乐','气泡水','啤酒','调料','调和油']))|(web_tmp.CATCODE =='NOOD')]
#web_tmp = web_tmp[web_tmp.CATCODE =='NOOD']

web_tmp = web.copy()





imdb_cat = imdb0604[(imdb0604.CATCODE=='NOOD')&(imdb0604.EAN !='NO-EAN')]
imdb_cat.loc[imdb_cat.LONGDESC.str.contains('康师傅'), 'CBRAND'] = '康师傅'
imdb_cat.loc[imdb_cat.LONGDESC.str.contains('康师傅'), 'BRANDCODE'] = 'KSFJD'
#imdb_cat[['brand']] = imdb_cat['CBRAND'].str.split(' ', 1)
imdb_cat.loc[:, 'brand_lv1'] = imdb_cat['CBRAND'].str.split(' ', 1).map(lambda x: x[0])

brand_list = list(imdb_cat.CBRAND.unique())
brand_list.remove('其他牌子')

def find_value_column(row):
    if isinstance(row['keywords'], list):
        for keyword in row['keywords']:
            return keyword in row.movie_title.lower()
    else:
        return False

imdb_cat[imdb_cat.apply(find_value_column, axis=1)][['LONGDESC', 'keywords']].head()

imdb_cat['LONGDESC'].str.contains('|'.join(brand_list))

imdb_cat['LONGDESC']= imdb_cat['LONGDESC'].apply(string_clean)
#imdb_cat[['A']] = imdb_cat['LONGDESC'].str.split(' ', 1)

imdb_cat['cut_desc'] = imdb_cat['LONGDESC'].apply(cut_word)
imdb_cat = pd.merge(eric, imdb_cat, left_on='RETAILERITEMCODE', right_on='EAN', how='right')

import barcode
barcode.PROVIDED_BARCODES
EAN = barcode.get_barcode_class('ean13')
ean = EAN('6972844730013')


