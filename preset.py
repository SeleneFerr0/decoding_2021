# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
vocab_dir = "/Users/seleneferro/Downloads/o2o/Torch-base/data/vocab.pkl"
emb_dim = 300
word_to_id = pkl.load(open(vocab_dir, 'rb'))
embeddings = np.random.rand(len(word_to_id), emb_dim)


#%%
import pandas as pd 
import jieba
import re

import numpy
from sklearn.preprocessing import StandardScaler

#from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
import csv
import sys

#from config_ch import *
import chardet
import numpy as np
import pandas as pd


import paddle
paddle.enable_static()


import jieba.posseg
import jieba.analyse
import jieba.posseg as pseg


import os
os.chdir('/Users/seleneferro/Downloads')



#%%

'''functions'''

def cut_word(word):
    #cw = jieba.cut_for_search(word)
    cw = jieba.cut(word)
    return list(cw)

def string_clean(l):
    l = str(l).replace('【领券满99减20】', "")
    l = l.replace('【领券满39减8】', "")
    l = l.replace('【领券满89减8】', "")
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
    l = l.replace('【网红】', "")
    l = l.replace("【", "")
    l = l.replace("】", "")
    l = l.replace("[EC]", "")
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
    l = l.replace("赠礼品袋",'')
    l = l.replace("3点1刻",'三点一刻')
    l = l.replace("店",'')
    l = l.replace("LUZHOULAOJIAO",'')
    l = l.replace("LUZHOU",'')
    l = l.replace("[", "")
    l = l.replace("]", "")
    l = l.replace("', '", "/")
    l = l.replace("@", "")
    l = l.replace("(", "")
    l = l.replace(")", "")
    l = l.replace("#", "")
    l = re.sub('\d+', " ", l).strip()   #words with numbers
    # l = re.sub('[^A-Za-z]+', ' ',l)        #special characters
    l = re.sub(r'[0-9]+', '', l) #numbers
    return l

def brand_clean(l):
    l = l.replace("亚州", "亚洲")
    l = l.replace("100年润发", "百年润发")
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
    l = l.replace("/",'')  
    l = l.replace("±",'')
    l = l.replace("-",'')
    l = l.replace("+",'')
    l = l.replace("g",'')
    l = l.replace("SH",'')
    l = l.replace("k",'')
    l = l.replace(" ",'')
    # l = l.replace("L",'')
    # l = l.replace("l",'')
    # l = l.replace("m",'')
    l = l.replace(",",'')
    l = l.replace(".",'')
    l = l.replace(">",'')
    l = l.replace("<",'')
    l = l.replace("=",'')
    l = l.replace("^",'')
    l = l.replace("。",'')
    l = l.replace("。",'')
    return(l)


def debrand(l):
    l = str(l[0]).replace(str(l[1]), '')
    return l


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

def cnt_var(df, var):
    var_lst = list(df[var].unique())
    print(var)
    for v in var_lst:
        print(str(v) + ' ' + str(len(df[df[var]== v])))


def cnt_per(df, var):
    var_lst = list(df[var].unique())
    print(var)
    for v in var_lst:
        print(str(v) + ' ' + "{:.0%}".format((len(df[df[var]== v])/len(df))))
        
      
def chain_identification(s, chain_list):
    print(s)
    ## remove space in store name
    s = re.sub(" ", "", str(s))
    ## segments the store name into separated words
    s_split = list(jieba.cut(s, cut_all = False))
    s_chain_name = [x for x in chain_list if str(x) in s_split]
    return s_chain_name



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

    


#%%
'''paddle test'''


from paddlenlp import Taskflow
import paddle
from paddlenlp.transformers import *

ernie = AutoModel.from_pretrained('ernie-3.0-medium-zh')
bert = AutoModel.from_pretrained('bert-wwm-chinese')
# albert = AutoModel.from_pretrained('albert-chinese-tiny')
# roberta = AutoModel.from_pretrained('roberta-wwm-ext')
# electra = AutoModel.from_pretrained('chinese-electra-small')
# gpt = AutoModelForPretraining.from_pretrained('gpt-cpm-large-cn')




tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
text = tokenizer('DOCTORLI李士祛痘修护精华液15毫升')



seg = Taskflow("word_segmentation")
a = seg('大自然泡椒莲藕^300g')





import os
wdd = r'/Users/seleneferro/Downloads/2022decode/model/o2o/data'
df = pd.read_csv(os.path.join(wdd,'trainSetFull0824.csv'),encoding='GB18030')


df['Clean_Text'] = df['TEXT'].map(string_clean)

df['Clean_Text'] = df['Clean_Text'].map(brand_clean)


df['label_0']=0

df.loc[df['LABEL'].str.contains("Non audit"), 'label_0'] = 1

df.loc[df.LABEL.str.contains(r'\\'), 'label_0'] = 2



#%%
'''data Aug'''

files = os.listdir('./0822')

files.remove('跨LPC bandpack_done.xlsx')

raw = pd.DataFrame()
for file in files:
    print(file)
    tmp = pd.read_excel('./0822/'+file)
    # print(tmp.columns)
    if len(raw)==0:
        raw = tmp
    else:
        raw= raw.append(tmp)



cat_list = list(raw.CATNAME.unique())


bl_chk1 = raw.groupby(['CATNAME', 'NANKEY']).size()

'''
RWC 61505473 872 散装糖果
RDT 78136304 360
LPChocolate 62426593 358


t = raw[raw.NANKEY == 62426593.0]
etc...
'''






#%%

# df = pd.read_excel('/Users/seleneferro/Downloads/o2o/data/Suguo Store Item Sample Data.xlsx',sheet_name ='Sheet1')
# itm_fresh = pd.read_excel('/Users/seleneferro/Downloads/o2o/data/item coding_fresh.xlsx',sheet_name ='Sheet1')
# itm_eric =  pd.read_excel('/Users/seleneferro/Downloads/o2o/data/item_coding.xlsx',sheet_name ='Sheet1')
import os
wdd = r'/Users/seleneferro/Downloads/2022decode/model/o2o/data'
df = pd.read_csv(os.path.join(wdd,'trainSetFull0824.csv'),encoding='GB18030')
# df =  pd.read_table(os.path.join(wdd, 'sample_0930.txt'),header= None, sep=',', encoding='utf-8', dtype={0: str, 1: str})
df.columns = ["PERIODCODE","PROD_DESC_RAW", "CATCODE"]

##
#O2O retailer_0720

df070 =  pd.read_excel(os.path.join(wdd, 'O2O retailer_0720.xlsx'))
democy =pd.read_excel(os.path.join(wdd, 'CN5_demo_to CuiYu.xlsx'))

rr25 = pd.read_excel(os.path.join(wdd, 'RR25 For All LPC.xlsx'))
t = rr25[rr25['Parent Nielsen Item Description'].isnull()]


rr25[['parent1','parent2','parent3']] = rr25['Parent Nielsen Item Description'].str.split(',',expand=True)

rr25['parent3'] = rr25['Parent Nielsen Item Description'].str.split().str[-1]

df2 = pd.read_table(os.path.join(wdd, 'sample_0429.txt'),header= None, sep=',', encoding='utf-8', dtype={0: str, 1: str})


df['product_desc'] = df['PROD_DESC_RAW'].map(string_clean)
df['product_desc'] = df['product_desc'].map(brand_clean)


df[['CAT1', 'CAT2', 'CAT3', 'CAT4', 'CAT5', 'CAT6']] = df['CATCODE'].str.split('/', expand=True)



cat = list(df.CATCODE.unique())

tst = df[df['CATCODE'].str.contains("CHISPIRITS")]



### test for 可口可乐
rr25 = rr25[~rr25['Parent Nielsen Item Description'].isnull()]
colaband= rr25[rr25['Parent Nielsen Item Description'].str.contains('可口可乐')]

df = df[~df['PROD_DESC_RAW'].isnull()]
o2ocola = df[df['PROD_DESC_RAW'].str.contains('可口可乐')]
o2ocola = o2ocola[~o2ocola.CAT2.isnull()]


writer = pd.ExcelWriter('colabanded.xlsx', engine='xlsxwriter')

o2ocola.to_excel(writer, sheet_name='sample0930')
colaband.to_excel(writer, sheet_name='rr25')
writer.save()


# tst = df[df['PROD_DESC_RAW'].str.contains("青菜")]
tst = df[~df.CAT3.isnull()]



keywrd = pd.read_excel(os.path.join(wdd,'Coding Key words_202206.xlsx'))
keywrd = keywrd[['CATCODE','CSEGMENT','SEGNAME','SHORTDESC']]
keywrd.head()
keywrd.loc[keywrd.SHORTDESC.isnull(), 'SHORTDESC'] = keywrd['CSEGMENT']

#%%
imdb = pd.read_csv('/Users/seleneferro/Downloads/IMDB0523.csv', encoding = 'GB18030')


imdb = imdb[~imdb.LONGDESC.isnull()]


'''	SUBBRAND_DESC_CN	ZZ2NA_其他品牌
	MANU_DESC_CN: 为了LHHT POSTING 创建
'''



imdb_cat = list(imdb.CATEGORYCODE.unique())


common_cat = list(set(cat) & set(imdb_cat))


single_itm = df[df.CAT2.isnull()]

single_itm = single_itm[['PROD_DESC_RAW', 'product_desc', 'CATCODE']]


imdb['product_desc'] = imdb['LONGDESC'].map(string_clean)
imdb['product_desc'] = imdb['product_desc'].map(brand_clean)
imdb.loc[imdb.BRAND_DESC_CN == 'NA_NA', 'BRAND_DESC_CN'] = imdb['SHORTDESC']



imdb_single = imdb[imdb.CATEGORYCODE != 'BANDEDPACK']
imdb_banded = imdb[imdb.CATEGORYCODE == 'BANDEDPACK']


imdb_single = imdb_single[~imdb_single['CATEGORYCODE'].str.contains("TSR")]

imdb_single = imdb_single[['LONGDESC', 'EAN', 'MULTIPACK', 'CATEGORYCODE', 'CATEGORY_DESC', 'CATEGORY_DESC_CN',
'BRANDNUM', 'BRAND_DESC', 'BRAND_DESC_CN','SHORTDESC', 'MANU_DESC',
'MANU_DESC_CN','product_desc']]

imdb_single.drop_duplicates(keep='first', inplace=True)

tsr = imdb[imdb['CATEGORYCODE'].str.contains("TSR")]




brand = imdb_single[['CATEGORYCODE','BRAND_DESC_CN','MANU_DESC_CN','SHORTDESC','CATEGORY_DESC_CN']]
brand = brand[brand.BRAND_DESC_CN != '为了LHHT POSTING 创建']
brand = brand[brand.MANU_DESC_CN != '为了LHHT POSTING 创建']
brand.loc[brand.BRAND_DESC_CN == '进口品牌', 'BRAND_DESC_CN'] = brand['MANU_DESC_CN']

brand.drop_duplicates(keep='first', inplace=True)

brand.sort_values(by = 'BRAND_DESC_CN', inplace=True)


brand_dict = brand[['CATEGORYCODE','BRAND_DESC_CN']].groupby('BRAND_DESC_CN')['CATEGORYCODE'].apply(list).to_dict()

add_word = list(imdb.SHORTDESC.unique())
add_brand = list(imdb_single.BRAND_DESC_CN.unique())
add_key = list(keywrd.SHORTDESC.unique())

add_word = [elm for elm in add_word if isinstance(elm, str)]
add_brand =  [elm for elm in add_brand if isinstance(elm, str)]
add_key = [elm for elm in add_key if isinstance(elm, str)]

for i in add_brand:
    jieba.add_word(i,freq=10,tag='nz')


for i in add_word:
    jieba.add_word(i,freq=10,tag='nz')
    
    
for i in add_key:
    jieba.add_word(i,freq=10,tag='nz') 



#%%



#cnt_per(df, 'cat2_name')


df['product_desc'] = df['product_name'].map(string_clean)
df['product_desc'] = df['product_desc'].map(brand_clean)
tt = df[0:300]



cat = pd.DataFrame(cat)


train = df[['product_desc','cat1_id']]
#train[0:120000].to_csv('/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/train.txt',index=False)

#with open('/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/train.txt', 'w') as f: train[0:120000].to_string(f, col_space=None,header=False, index=False)
#with open('/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/tev.txt', 'w') as f: train.to_string(f, col_space=None,header=False, index=False)

#%%
train[0:120000].to_csv('/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/train.txt',index=False, header=None)
train[120000:123801].to_csv('/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/test.txt',index=False,header=None)
train.to_csv('/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/dev.txt',index=False,header=None)
#cat.to_csv('/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/class.txt',index=False,header=None)








#%%
path = r'/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/train.txt'
contents = []
with open(path, 'r', encoding='UTF-8') as f:
    for line in f:
        lin = line.strip()
        print(lin)
        if not lin:
            continue
        content, label = lin.split(',')
        
        
        
        
        
        
        

#%%


def countDigitOne(n): # can not deal with n=-1 case!
    if n<=0: return 0
    
    iCount = 0
    iFactor = 1
    iLowerNum, iCurNum, iHigherNum = 0, 0, 0
    
    while n // iFactor != 0:
        iLowerNum = n - (n//iFactor) * iFactor
        iCurNum = (n//iFactor) % 10
        iHigherNum = n //(iFactor * 10)
        
        if iCurNum == 0:
            iCount += iHigherNum * iFactor
        elif iCurNum == 1:
            iCount += iHigherNum * iFactor + (iLowerNum + 1)
        else:
            iCount += (iHigherNum + 1) * iFactor
        iFactor *= 10
    return iCount



a = countDigitOne(n=13)

a
