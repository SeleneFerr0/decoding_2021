# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
# vocab_dir = "/Users/seleneferro/Downloads/o2o/Torch-base/data/vocab.pkl"
# emb_dim = 300
# word_to_id = pkl.load(open(vocab_dir, 'rb'))
# embeddings = np.random.rand(len(word_to_id), emb_dim)


#%%
import pandas as pd 
import jieba
import re
import extremeText
import numpy
from sklearn.preprocessing import StandardScaler

#from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
import csv
import sys

#from config_ch import *
# import chardet
import numpy as np


# import paddle
# paddle.enable_static()


import jieba.posseg
import jieba.analyse
import jieba.posseg as pseg
import extremeText

import os
os.chdir('/Users/seleneferro/Downloads')



#%%

'''functions'''

def cut_word(word):
    #cw = jieba.cut_for_search(word)
    cw = jieba.cut(word)
    return list(cw)

def splabel(l):
    l = str(l).replace(r'\\', '+')
    return(l)
    
    
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
    l = str(l).lower() 
    l.removesuffix('mlmlml')
    l.removesuffix('mlml')
    l.removesuffix('ml')
    l.removesuffix('kg')
    l.removesuffix('g')
    l.removesuffix('lll')
    l.removesuffix('ll')
    l.removesuffix('l')
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
    # l = l.replace("+",'')
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
    l = l.replace("^",' ')
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

    
def pad_seg(word):
    # tokenizer = AutoTokenizer.from_pretrained(method)
    seg = Taskflow("word_segmentation")
    a = seg(word)
    return a




#%%
'''paddle test'''


from paddlenlp import Taskflow
# import paddle
from paddlenlp.transformers import *

# ernie = AutoModel.from_pretrained('ernie-3.0-medium-zh')
# bert = AutoModel.from_pretrained('bert-wwm-chinese')
# albert = AutoModel.from_pretrained('albert-chinese-tiny')
# roberta = AutoModel.from_pretrained('roberta-wwm-ext')
# electra = AutoModel.from_pretrained('chinese-electra-small')
# gpt = AutoModelForPretraining.from_pretrained('gpt-cpm-large-cn')


t = 'DOCTORLI李士祛痘修护精华液15毫升'

tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
text = tokenizer('DOCTORLI李士祛痘修护精华液15毫升')



seg = Taskflow("word_segmentation")

a = seg(t)



tt['seg_Text'] = tt['Clean_Text'].map(pad_seg)
#%%


import os

wdd = r'/Users/seleneferro/Downloads/2022decode/model/o2o/data'
df = pd.read_csv(os.path.join(wdd,'trainSetFull1025.csv'),encoding='GB18030')
# add = pd.read_excel(os.path.join(wdd,'itemAdd0923.xlsx'))


df['Clean_Text'] = df['TEXT'].map(string_clean)
df['Clean_Text'] = df['Clean_Text'].map(brand_clean)



# add['Clean_Text'] = add['TEXT'].map(string_clean)
# add['Clean_Text'] = add['Clean_Text'].map(brand_clean)


df['label_0']=0

# for i in cat_code:
#     b[i] = b['LABEL'].apply(lambda x: 1 if i in x  else 0)
#     b.loc[b['FACIALMASK']==1, 'FACIA']



df.loc[df['LABEL'].str.contains("Non audit"), 'label_0'] = 1

df.loc[df.LABEL.str.contains(r'\\'), 'label_0'] = 2

df['seg_Text'] = df['Clean_Text'].map(cut_word)


single = df[~df.LABEL.str.contains(r'\\')]
bundle = df[df.LABEL.str.contains(r'\\')]

bundle['LABEL']=bundle['LABEL'].map(splabel)
# seg = bundle['LABEL'].str.split(r'\\', expand=True)
# seg.columns = ['CAT{}'.format(x+1) for x in seg.columns]

# bundle = pd.concat([bundle, seg], axis=1)
# bundle = bundle.reset_index()



cat_code = list(single.LABEL.unique())

for i in cat_code:
    bundle[i] = np.zeros((len(bundle),1))
    
    

for i in cat_code:
    bundle[i] = bundle['LABEL'].apply(lambda x: 1 if i in x  else 0)
    
    
df2 = df.copy()
for i in cat_code:
    df2[i] = np.zeros((len(df2),1))
    
    

for i in cat_code:
    df2[i] = df2['LABEL'].apply(lambda x: 1 if i in x  else 0)
    
    
df2.to_csv(os.path.join(wdd, 'expand1025.csv'))




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
# import os
# wdd = r'/Users/seleneferro/Downloads/2022decode/model/o2o/data'
# df = pd.read_csv(os.path.join(wdd,'trainSetFull0902.csv'),encoding='GB18030')
# # df =  pd.read_table(os.path.join(wdd, 'sample_0930.txt'),header= None, sep=',', encoding='utf-8', dtype={0: str, 1: str})
# df.columns = ["PERIODCODE","PROD_DESC_RAW", "CATCODE"]

##
#O2O retailer_0720

df070 =  pd.read_excel(os.path.join(wdd, 'O2O retailer_0720.xlsx'))
democy =pd.read_excel(os.path.join(wdd, 'CN5_demo_to CuiYu.xlsx'))

rr25 = pd.read_excel(os.path.join(wdd, 'RR25 For All LPC.xlsx'))
t = rr25[rr25['Parent Nielsen Item Description'].isnull()]


rr25[['parent1','parent2','parent3']] = rr25['Parent Nielsen Item Description'].str.split(',',expand=True)

rr25['parent3'] = rr25['Parent Nielsen Item Description'].str.split().str[-1]

df2 = pd.read_table(os.path.join(wdd, 'sample_0429.txt'),header= None, sep=',', encoding='utf-8', dtype={0: str, 1: str})


# df['product_desc'] = df['PROD_DESC_RAW'].map(string_clean)
# df['product_desc'] = df['product_desc'].map(brand_clean)


df[['CAT1', 'CAT2', 'CAT3', 'CAT4', 'CAT5', 'CAT6']] = df['LABEL'].str.split('/', expand=True)




# cat = list(df.CATCODE.unique())

# tst = df[df['CATCODE'].str.contains("CHISPIRITS")]



### test for 可口可乐
# rr25 = rr25[~rr25['Parent Nielsen Item Description'].isnull()]
# colaband= rr25[rr25['Parent Nielsen Item Description'].str.contains('可口可乐')]

# df = df[~df['PROD_DESC_RAW'].isnull()]
# o2ocola = df[df['PROD_DESC_RAW'].str.contains('可口可乐')]
# o2ocola = o2ocola[~o2ocola.CAT2.isnull()]


# writer = pd.ExcelWriter('colabanded.xlsx', engine='xlsxwriter')

# o2ocola.to_excel(writer, sheet_name='sample0930')
# colaband.to_excel(writer, sheet_name='rr25')
# writer.save()


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



# single_itm = single_itm[['PROD_DESC_RAW', 'product_desc', 'CATCODE']]


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
    jieba.add_word(i,freq=30,tag='nz')


for i in add_word:
    jieba.add_word(i,freq=30,tag='nz')
    
    
for i in add_key:
    jieba.add_word(i,freq=30,tag='nz') 



keywrd = pd.read_excel(os.path.join(wdd,'Coding Key words_202206.xlsx'))
keywrd = keywrd[['CATCODE','CSEGMENT','SEGNAME','SHORTDESC']]
keywrd.head()
keywrd.loc[keywrd.SHORTDESC.isnull(), 'SHORTDESC'] = keywrd['CSEGMENT']


del imdb, imdb_single



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
# path = r'/Users/seleneferro/Downloads/o2o/Torch-base/vegetables/data/train.txt'
# contents = []
# with open(path, 'r', encoding='UTF-8') as f:
#     for line in f:
#         lin = line.strip()
#         print(lin)
#         if not lin:
#             continue
#         content, label = lin.split(',')
        




import argparse
import os

import numpy as np
import scipy.sparse as smat
from pecos.pecos.utils import smat_util
from pecos.pecos.utils.cli import SubCommand, str2bool

vectorizer_dict = {}

LOGGER = logging.getLogger(__name__)


class VectorizerMeta(ABCMeta):
    """Metaclass for keeping track of all `Vectorizer` subclasses."""

    def __new__(cls, name, bases, attr):
        cls = super().__new__(cls, name, bases, attr)
        if cls.__name__ != "Vectorizer":
            vectorizer_dict[cls.__name__.lower()] = cls
        return cls


class Vectorizer(metaclass=VectorizerMeta):
    """Wrapper class for all vectorizers."""

    def __init__(self, config, model):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer.
            model (Vectorizer): Trained vectorizer.
        """

        self.config = config
        self.model = model

    def save(self, vectorizer_folder):
        """Save trained vectorizer to disk.

        Args:
            vectorizer_folder (str): Folder to save to.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.config))
        self.model.save(vectorizer_folder)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load a saved vectorizer from disk.

        Args:
            vectorizer_folder (str): Folder where `Vectorizer` was saved to using `Vectorizer.save`.

        Returns:
            Vectorizer: The loaded object.
        """

        config_path = os.path.join(vectorizer_folder, "config.json")
        if not os.path.exists(config_path):
            # to maintain compatibility with previous versions of pecos models
            config = {"type": "tfidf", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())
        vectorizer_type = config.get("type", None)
        assert vectorizer_type is not None, f"{vectorizer_folder} is not a valid vectorizer folder"
        assert vectorizer_type in vectorizer_dict, f"invalid vectorizer type {config['type']}"
        model = vectorizer_dict[vectorizer_type].load(vectorizer_folder)
        return cls(config, model)

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer. Default behavior is to use tfidf vectorizer with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Vectorizer: Trained vectorizer.
        """

        config = config if config is not None else {"type": "tfidf", "kwargs": {}}
        LOGGER.debug(f"Train Vectorizer with config: {json.dumps(config, indent=True)}")
        vectorizer_type = config.get("type", None)
        assert (
            vectorizer_type is not None
        ), f"config {config} should contain a key 'type' for the vectorizer type"
        assert vectorizer_type in vectorizer_dict, f"invalid vectorizer type {config['type']}"
        assert (
            isinstance(trn_corpus, list) or vectorizer_type == "tfidf"
        ), "only tfidf support from file training"
        model = vectorizer_dict[vectorizer_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        return cls(config, model)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list or str): List of strings to vectorize or path to text file.
            **kwargs: Keyword arguments to pass to the trained vectorizer.

        Returns:
            numpy.ndarray or scipy.sparse.csr.csr_matrix: Matrix of features.
        """

        if isinstance(corpus, str) and self.config["type"] != "tfidf":
            raise ValueError("Iterable over raw text expected for vectorizer other than tfidf.")
        return self.model.predict(corpus, **kwargs)

    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `vectorizer_config_path` (path to a json file) or `vectorizer_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """

        if args.vectorizer_config_path is not None:
            with open(args.vectorizer_config_path, "r", encoding="utf-8") as fin:
                vectorizer_config_json = fin.read()
        else:
            vectorizer_config_json = args.vectorizer_config_json

        try:
            vectorizer_config = json.loads(vectorizer_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                "Failed to load vectorizer config json from {} ({})".format(
                    vectorizer_config_json, jex
                )
            )
        return vectorizer_config


class Tfidf(Vectorizer):
    """Multithreaded tfidf vectorizer with C++ backend.

    Supports 'word', 'char' and 'char_wb' tokenization.
    """

    def __init__(self, model=None):
        """Initialization

        Args:
            model (ctypes.c_void_p): pointer to C instance tfidf::Vectorizer
        """
        self.model = model

    def __del__(self):
        """Destruct self model instance"""
        clib.tfidf_destruct(self.model)

    def save(self, save_dir):
        """Save trained tfidf vectorizer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """
        os.makedirs(save_dir, exist_ok=True)
        clib.tfidf_save(self.model, save_dir)

    @classmethod
    def load(cls, load_dir):
        """Load a Tfidf vectorizer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            Tfidf: The loaded object.
        """
        if not os.path.exists(load_dir):
            raise ValueError(f"tfidf model not exist at {load_dir}")
        return cls(clib.tfidf_load(load_dir))

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list of str or str): Training corpus in the form of a list of strings or path to corpus file/folder.
            config (dict): Dict with keyword arguments to pass to C++ class tfidf::Vectorizer.
                The keywords are:
                    ngram_range (tuple of int): (min_ngram, max_ngram)
                    truncate_length (int): sequence truncation length, set to negative to disable
                    max_feature (int): maximum number of features allowed, set to 0 to disable
                    min_df_ratio (float, [0, max_df_ratio)): min ratio for document frequency truncation
                    max_df_ratio (float, (min_df_ratio, 1]): max ratio for document frequency truncation
                    min_df_cnt (int, [0, max_df_cnt)): min count for document frequency truncation
                    max_df_cnt (int, (min_df_cnt, Inf)): max count for document frequency truncation. Default -1 to disable.
                    binary (bool): whether to binarize term frequency, default False
                    use_idf (bool): whether to use inverse document frequency, default True
                    smooth_idf (bool): whether to smooth IDF by adding 1 to all DF counts, default True
                    add_one_idf (bool): whether to smooth IDF by adding 1 to all IDF scores, default False
                    sublinear_tf (bool): whether to use sublinear mapping (log) on term frequency, default False
                    keep_frequent_feature (bool): if max_feature > 0, will only keep max_feature features by
                                    ignoring features with low document frequency (if True, default),
                                    ignoring features with high document frequency (if False)
                    norm (str, 'l1' or 'l2'): feature vector will have unit l1 or l2 norm
                    analyzer (str, 'word', 'char' or 'char_wb'): Whether to use word or character n-grams.
                                    Option ‘char_wb’ creates character n-grams only from text inside word boundaries,
                                    n-grams at the edges of words are padded with single space.
                    buffer_size (int): if train from file, number of bytes allocated for file I/O. Set to 0 to use default value.
                    threads (int): number of threads to use, set to negative to use all
            dtype (np.dtype): The data type to use. Default to `np.float32`.

        Note:
            stop word removal: simultaneously satisfy count and ratio constraint.
                i.e. will use max(min_df_cnt, min_df_ratio * nr_doc) as final min_df_cnt
                and min(max_df_cnt, max_df_ratio * nr_doc) as final max_df_cnt

        Returns:
            Tfidf: Trained vectorizer.
        """
        DEFAULTS = {
            "ngram_range": (1, 1),
            "truncate_length": -1,
            "max_feature": 0,
            "min_df_ratio": 0.0,
            "max_df_ratio": 1.0,
            "min_df_cnt": 0,
            "max_df_cnt": -1,
            "binary": False,
            "use_idf": True,
            "smooth_idf": True,
            "add_one_idf": False,
            "sublinear_tf": False,
            "keep_frequent_feature": True,
            "norm": "l2",
            "analyzer": "word",
            "buffer_size": 0,
            "threads": -1,
        }

        DEFAULTS_META = {
            "norm_p": 2,
            "buffer_size": 0,
            "threads": -1,
            "base_vect_configs": [DEFAULTS],
        }

        def check_base_config_key(base_config):
            unexpected_keys = []
            for key in base_config:
                if key not in DEFAULTS:
                    unexpected_keys.append(key)
            if len(unexpected_keys) > 0:
                raise ValueError(f"Unknown argument: {unexpected_keys}")
            return {**DEFAULTS, **base_config}

        if "base_vect_configs" not in config:
            config = check_base_config_key(config)
        else:
            for idx, base_config in enumerate(config["base_vect_configs"]):
                base_config = check_base_config_key(base_config)
                config["base_vect_configs"][idx] = base_config
            config = {**DEFAULTS_META, **config}

        cmodel = clib.tfidf_train(trn_corpus, config)

        return cls(cmodel)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.
            **kwargs:
                threads (int, default -1): number of threads to use for predict, set to negative to use all

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        return clib.tfidf_predict(
            self.model,
            corpus,
            buffer_size=kwargs.get("buffer_size", 0),
            threads=kwargs.get("threads", -1),
        )


class SklearnTfidf(Vectorizer):
    """Sklearn tfidf vectorizer"""

    def __init__(self, model=None):
        """Initialization

        Args:
            model (sklearn.feature_extraction.text.TfidfVectorizer, optional): The trained tfidf vectorizer. Default is `None`.
        """

        self.model = model

    def save(self, vectorizer_folder):
        """Save trained sklearn Tfidf vectorizer to disk.

        Args:
            vectorizer_folder (str): Folder to store serialized object in.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load a saved sklearn Tfidf vectorizer from disk.

        Args:
            vectorizer_folder (str): Folder where `SklearnTfidf` object was saved to using `SklearnTfidf.save`.

        Returns:
            SklearnTfidf: The loaded object.
        """

        vectorizer_path = os.path.join(vectorizer_folder, "vectorizer.pkl")
        assert os.path.exists(vectorizer_path), "vectorizer path {} does not exist".format(
            vectorizer_path
        )
        with open(vectorizer_path, "rb") as fvec:
            return cls(pickle.load(fvec))

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's TfidfVectorizer.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Tfidf: Trained vectorizer.

        Raises:
            Exception: If `config` contains keyword arguments that the tfidf vectorizer does not accept.
        """
        defaults = {
            "encoding": "utf-8",
            "strip_accents": "unicode",
            "stop_words": None,
            "ngram_range": (1, 1),
            "min_df": 1,
            "lowercase": True,
            "norm": "l2",
            "dtype": dtype,
        }
        try:
            model = TfidfVectorizer(**{**defaults, **config})
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(model)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.
            **kwargs: Ignored.

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        result = self.model.transform(corpus)
        # Indices must be sorted for C++ batch code to work
        result.sort_indices()
        return result


class SklearnHashing(Vectorizer):
    """Sklearn hashing vectorizer"""

    def __init__(self, model=None):
        """Initialization

        Args:
            model (sklearn.feature_extraction.text.HashingVectorizer, optional): The trained hashing vectorizer. Default is `None`.
        """
        self.model = model

    def save(self, vectorizer_folder):
        """Save trained sklearn hashing vectorizer to disk.

        Args:
            vectorizer_folder (str): Folder to store serialized object in.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load a saved sklearn hashing vectorizer from disk.

        Args:
            vectorizer_folder (str): Folder where `SklearnHashing` object was saved to using `SklearnHashing.save`.

        Returns:
            SklearnHashing: The loaded object.
        """

        vectorizer_path = os.path.join(vectorizer_folder, "vectorizer.pkl")
        assert os.path.exists(vectorizer_path), "vectorizer path {} does not exist".format(
            vectorizer_path
        )
        with open(vectorizer_path, "rb") as fvec:
            return cls(pickle.load(fvec))

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's HashingVectorizer.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Hashing: Trained vectorizer.

        Raises:
            Exception: If `config` contains keyword arguments that the hashing vectorizer does not accept.
        """

        defaults = {
            "encoding": "utf-8",
            "strip_accents": "unicode",
            "stop_words": None,
            "ngram_range": (1, 2),
            "lowercase": True,
            "norm": "l2",
            "dtype": dtype,
            "n_features": 1048576,  # default number in HashingVectorizer
        }
        try:
            model = HashingVectorizer(**{**defaults, **config})
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for HashingVectorizer"
            )
        model.fit(trn_corpus)
        return cls(model)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.
            **kwargs: Ignored.

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        result = self.model.transform(corpus)
        # Indices must be sorted for C++ batch code to work
        result.sort_indices()
        return result


class Preprocessor(object):
    """Preprocess text to numerical values"""

    def __init__(self, vectorizer=None):
        """Initialization

        Args:
            vectorizer (Vectorizer): Text vectorizer class instance.
        """
        self.vectorizer = vectorizer

    def save(self, preprocessor_folder):
        """Save the preprocess object to a folder

        Args:
            preprocessor_folder (str): The saving folder name
        """
        self.vectorizer.save(preprocessor_folder)

    @classmethod
    def load(cls, preprocessor_folder):
        """Load preprocessor

        Args:
            preprocess_folder (str): The folder to load

        Returns:
            cls: An instance of Preprocessor
        """

        vectorizer = Vectorizer.load(preprocessor_folder)
        return cls(vectorizer)

    @classmethod
    def train(cls, corpus, vectorizer_config, dtype=np.float32):
        """Train a preprocessor

        Args:
            corpus (list of strings or a string): Training text input.
                If given a list of strings, it's the list of training inputs.
                If given a string, it's the path to a file with lines of text inputs to be trained.
            vectorizer_config (dict): Config file for the vectorizer
            dtype (scipy.dtype): Data type for the vectorized output

        Returns:
            A Preprocessor
        """

        vectorizer = Vectorizer.train(corpus, vectorizer_config, dtype=dtype)
        return cls(vectorizer)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus

        Args:
            corpus (list of strings or a string): Predicting text input.
                If given a list of strings, it's the list of text input to be vectorized.
                If given a string, it's the path to a file with lines of text inputs to be vectorized.
            kwargs (optional): Args to be passed to Vectorizer

        Returns:
            csr_matrix: Vectorized output
        """

        return self.vectorizer.predict(corpus, **kwargs)

    @staticmethod
    def load_data_from_file(
        data_path,
        label_text_path=None,
        split_sep="\t",
        maxsplit=-1,
        text_pos=1,
        label_pos=0,
        return_dict=True,
    ):
        """Parse a tab-separated text file to a CSR label matrix and a list of text strings.

        Text format for each line:
        <comma-separated label indices><TAB><space-separated text string>
        Example: l_1,..,l_k<TAB>w_1 w_2 ... w_t
            l_k can be one of two format:
                (1) the zero-based index for the t-th relevant label
                (2) double colon separated label index and label relevance
            w_t is the t-th token in the string

        Args:
            data_path (str): Path to the text file
            label_text_path (str, optional): Path to the label text file.
                The main purpose is to obtain the number of labels. Default: None
            split_sep (str, optional): The separator. Default: "\t".
            maxsplit (int, optional): The max number of splits for each line. Default: -1 to denote full split
            text_pos (int, optional): The position of the text part in each line. Default: 1.
            label_pos (int, optional): The position of the text part in each line. Default: 0.
            return_dict (bool, optional): if True, return the parsed results in a dictionary. Default True

        Returns:
            if return_dict:
                {
                    "label_matrix": (csr_matrix) label matrix with shape (N, L),
                    "label_relevance": (csr_matrix) label relevance matrix with shape (N, L)
                                        have same sparsity pattern as label_matrix.
                    "corpus": (list of str) the parsed instance text with length N.
                }
            else:
                (label_matrix, label_relevance, corpus)
        """
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"cannot find input text file at {data_path}")
        with open(data_path, "r", encoding="utf-8") as fin:
            label_strings, corpus = [], []
            for line in fin:
                parts = line.strip("\n")
                parts = parts.split(split_sep, maxsplit)
                if len(parts) < max(label_pos, text_pos) + 1:
                    raise ValueError(f"corrupted line from input text file:\n{line}")
                label_strings.append(parts[label_pos])
                text_string = parts[text_pos]
                corpus.append(text_string)

        def parse_label_strings(label_strings, L):
            rows, cols, vals, rels = [], [], [], []

            # determine if relevance is provided
            has_rel = ":" in label_strings[0]

            for i, label in enumerate(label_strings):
                if has_rel:
                    label_tuples = [tp.split(":") for tp in label.split(",")]
                    label_list = list(map(int, [tp[0] for tp in label_tuples]))
                    # label values are currently not being used.
                    val_list = list(map(float, [tp[1] if tp[1] else 1.0 for tp in label_tuples]))
                    rel_list = list(map(float, [tp[2] for tp in label_tuples]))
                else:
                    label_list = list(map(int, label.split(",")))
                    val_list = [1.0] * len(label_list)
                    rel_list = []

                rows += [i] * len(label_list)
                cols += label_list
                vals += val_list
                rels += rel_list

            Y = smat.csr_matrix(
                (vals, (rows, cols)), shape=(len(label_strings), L), dtype=np.float32
            )
            if has_rel:
                R = smat.csr_matrix(
                    (rels, (rows, cols)), shape=(len(label_strings), L), dtype=np.float32
                )
            else:
                R = None

            return Y, R

        if label_text_path is not None:
            if not os.path.isfile(label_text_path):
                raise FileNotFoundError(f"cannot find label text file at: {label_text_path}")
            # this is used to obtain the total number of labels L to construct Y with a correct shape
            L = sum(1 for line in open(label_text_path, "r", encoding="utf-8") if line)
            label_matrix, label_relevance = parse_label_strings(label_strings, L)
        else:
            label_matrix = None
            label_relevance = None

        if return_dict:
            return {
                "label_matrix": label_matrix,
                "label_relevance": label_relevance,
                "corpus": corpus,
            }
        else:
            return label_matrix, label_relevance, corpus


class BuildPreprocessorCommand(SubCommand):
    """Command to train a preprocessor"""

    @staticmethod
    def run(args):
        """Train a preprocessor.

        Args:
            args (argparse.Namespace): Command line argument parsed by `parser.parse_args()`
        """
        if not args.from_file:
            corpus = Preprocessor.load_data_from_file(
                args.input_text_path,
                maxsplit=args.maxsplit,
                text_pos=args.text_pos,
            )["corpus"]
        else:
            corpus = args.input_text_path
        vectorizer_config = Vectorizer.load_config_from_args(args)
        preprocessor = Preprocessor.train(corpus, vectorizer_config, dtype=args.dtype)
        preprocessor.save(args.output_model_folder)

    @classmethod
    def add_parser(cls, super_parser):
        """Add parser to the run.

        Args:
            super_parser (argparse.ArgumentParser): Argument parser.
        """
        parser = super_parser.add_parser("build", aliases=[], help="Build a preprocessor")
        cls.add_arguments(parser)
        parser.set_defaults(run=cls.run)

    @staticmethod
    def add_arguments(parser):
        """Add arguments for the build.

        Args:
            parser (argparse.ArgumentParser): Argument parser.
        """
        parser.add_argument(
            "-i", "--input-text-path", type=str, required=True, help="text input file name"
        )

        vectorizer_config_group_parser = parser.add_mutually_exclusive_group()
        vectorizer_config_group_parser.add_argument(
            "--vectorizer-config-path",
            type=str,
            default=None,
            metavar="VECTORIZER_CONFIG_PATH",
            help="Json file for vectorizer config (default tfidf vectorizer)",
        )

        vectorizer_config_group_parser.add_argument(
            "--vectorizer-config-json",
            type=str,
            default='{"type":"tfidf", "kwargs":{}}',
            metavar="VECTORIZER_CONFIG_JSON",
            help=f'Json-format string for vectorizer config (default {{"type":"tfidf", "kwargs":{{}}}}). Other type option: {list(vectorizer_dict.keys())}',
        )

        parser.add_argument(
            "-m", "--output-model-folder", type=str, required=True, help="model folder name"
        )

        parser.add_argument(
            "--maxsplit",
            type=int,
            default=-1,
            help="the max number of splits used to partition each line. (default -1 to denote full split)",
        )

        parser.add_argument(
            "--text-pos",
            type=int,
            default=1,
            help="the position of the text part in each line. (default 1)",
        )

        parser.add_argument(
            "-t",
            "--dtype",
            type=lambda x: np.float32 if "32" in x else np.float64,
            default=np.float32,
            help="data type for the output csr matrix. float32 | float64. (default float32)",
        )

        parser.add_argument(
            "--from-file",
            type=str2bool,
            metavar="[true/false]",
            default=False,
            help="[Only support tfidf vectorizer] training without preloading corpus to memory. If true, --input-text-path is expected to be a file or a folder containing files that each line contains only input text. Default false",
        )


class RunPreprocessorCommand(SubCommand):
    """Command to preprocess text using an existing preprocessor"""

    @staticmethod
    def run(args):
        """Preprocess text using an existing preprocessor.

        Args:
            args (argparse.Namespace): Command line argument parsed by `parser.parse_args()`
        """
        preprocessor = Preprocessor.load(args.input_preprocessor_folder)
        if args.from_file and not args.output_label_path and not args.output_rel_path:
            Y, R = None, None
            corpus = args.input_text_path
        else:
            result = Preprocessor.load_data_from_file(
                args.input_text_path,
                label_text_path=args.label_text_path,
                maxsplit=args.maxsplit,
                text_pos=args.text_pos,
                label_pos=args.label_pos,
            )
            Y = result["label_matrix"]
            R = result["label_relevance"]
            corpus = result["corpus"]

        X = preprocessor.predict(
            corpus,
            batch_size=args.batch_size,
            use_gpu_if_available=args.use_gpu,
            buffer_size=args.buffer_size,
            threads=args.threads,
        )

        smat_util.save_matrix(args.output_inst_path, X)

        if args.output_label_path and Y is not None:
            smat_util.save_matrix(args.output_label_path, Y)
        if args.output_rel_path and R is not None:
            smat_util.save_matrix(args.output_rel_path, R)

    @classmethod
    def add_parser(cls, super_parser):
        """Add parser to the run.

        Args:
            super_parser (argparse.ArgumentParser): Argument parser.
        """
        parser = super_parser.add_parser("run", aliases=[], help="Run a pre-built preprocessor")
        cls.add_arguments(parser)
        parser.set_defaults(run=cls.run)

    @staticmethod
    def add_arguments(parser):
        """Add arguments for the run.

        Args:
            parser (argparse.ArgumentParser): Argument parser.
        """
        parser.add_argument(
            "-p",
            "--input-preprocessor-folder",
            type=str,
            required=True,
            help="preprocessor folder name",
        )
        parser.add_argument(
            "-i", "--input-text-path", type=str, required=True, help="text input file name"
        )
        parser.add_argument(
            "-x",
            "--output-inst-path",
            type=str,
            required=True,
            help="output inst file name",
        )
        parser.add_argument(
            "--maxsplit",
            type=int,
            default=-1,
            help="the number of splits used to partition each line. (default -1 to denote full split))",
        )
        parser.add_argument(
            "--text-pos",
            type=int,
            default=1,
            help="the position of the text part in each line. (default 1)",
        )
        parser.add_argument(
            "-l", "--label-text-path", type=str, default=None, help="label text file name"
        )
        parser.add_argument(
            "-y", "--output-label-path", type=str, default=None, help="output label file name"
        )
        parser.add_argument(
            "-r", "--output-rel-path", type=str, default=None, help="output relevance file name"
        )
        parser.add_argument(
            "--label-pos",
            type=int,
            default=0,
            help="the position of the text part in each line. (default 0)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="batch size for Transformer vectorizer embedding evaluation (default 8)",
        )
        parser.add_argument(
            "--use-gpu",
            type=str2bool,
            metavar="[true/false]",
            default=True,
            help="if true, use CUDA training if available. Default true",
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=-1,
            help="number of threads to use for predict (default -1 to use all)",
        )
        parser.add_argument(
            "--from-file",
            type=str2bool,
            metavar="[true/false]",
            default=False,
            help="[Only support tfidf vectorizer] predict without preloading corpus to memory. If true, --input-text-path is expected to be a file that each line contains only input text. Default false",
        )
        parser.add_argument(
            "--buffer-size",
            type=int,
            default=0,
            help="number of bytes to use as file I/O buffer if --from-file (set to 0 to use default value)",
        )


def get_parser():
    """Get a parser for training preprocessor and preprocessing text"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subcommands", metavar="SUBCOMMAND")
    subparsers.required = True
    BuildPreprocessorCommand.add_parser(subparsers)
    RunPreprocessorCommand.add_parser(subparsers)
    return parser


#%%
from pecos.pecos.utils import smat_util

wdd = '/Users/seleneferro/pecos/'


X = smat_util.load_matrix(wdd + "test/tst-data/xmc/xtransformer/train_feat.npz")
Y = smat_util.load_matrix(wdd + "test/tst-data/xmc/xtransformer/train_label.npz")
# load training text features
# from pecos.pecos.utils.featurization.text.preprocess import Preprocessor
text = Preprocessor.load_data_from_file(wdd + "test/tst-data/xmc/xtransformer/train.txt", text_pos=0)["corpus"]
        
from pecos.pecos.xmc.xtransformer.model import XTransformer
from pecos.pecos.xmc.xtransformer.module import MLProblemWithText
prob = MLProblemWithText(text, Y, X_feat=X)
xtf = XTransformer.train(prob)

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
