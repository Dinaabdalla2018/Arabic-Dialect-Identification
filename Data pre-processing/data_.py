import nltk
nltk.download('stopwords')
nltk.download('punkt')
import requests
import pandas as pd
import re
import pickle
import numpy as np
import string
from random import shuffle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class data_:
    def __init__(self,Path_,number):
        self.data = pd.read_csv(Path_)
        indcies   = [i for i in range(len(self.data))]
        shuffle(indcies)
        self.data = self.data.iloc[indcies[:number]]
        self.data['id']  = self.data['id'].apply(str)
        self.data_       = self.data.id.tolist()
    
    # To get Text (represent X)
    def Call_API(self,URL_Link,number_thos,number_m):
        self.X     = [] 
        start = 0
        for end in range(1,number_thos+1):
            data0_   = self.data_[start:end*1000]
            start    = end * 1000
            request_ = requests.post(URL_Link,json=data0_)
            request_ = request_.json()
            for value in request_.values():
                self.X.append(value)

        data0_    = self.data_[start:start+number_m]       
        request_  = requests.post(URL_Link,json=data0_)
        request_  = request_.json()
        for value in request_.values():
            self.X.append(value)
    
    def clean_text (self,text):
        emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002500-\U00002BEF"  # chinese char
         u"\U00002702-\U000027B0"
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         u"\U0001f926-\U0001f937"
         u"\U00010000-\U0010ffff"
         u"\u2640-\u2642" 
         u"\u2600-\u2B55"
         u"\u200d"
         u"\u23cf"
         u"\u23e9"
         u"\u231a"
         u"\ufe0f"  # dingbats
         u"\u3030"
         "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
    
        arabic_punctuations  = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        english_punctuations = string.punctuation
        punctuations_list    = arabic_punctuations + english_punctuations

        #Replace @username with empty string
        text = re.sub('@[^\s]+', ' ', text)
        #Convert www.* or https?://* to " "
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
        #Replace #word with word
        text = re.sub(r'#([^\s]+)', r'\1', text)
        # to remove numeric digits from string
        text = ''.join([i for i in text if not i.isdigit()])
        # Repalce Some Word in Arabic
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        re.sub(r'(.)\1+', r'\1', text)
    
        text = "".join([i for i in text if i not in punctuations_list])
    
        stopwords_list = stopwords.words('arabic')
        text = [word for word in text.split() if word not in stopwords_list]
        text = " ".join(text)
        return text


    def Transform_data(self,method='No'):
        self.vectorizer        = CountVectorizer(analyzer = self.clean_text)
        X_train_counts         = self.vectorizer.fit_transform(self.X)
        self.tfidf_transformer = TfidfTransformer()
        self.X                 = self.tfidf_transformer.fit_transform(X_train_counts).toarray()
        self.Y                 = self.data['dialect']
        self.num_classes       = int((len(set(self.Y ))))
        if method == 'DL':
            encoder = LabelEncoder()
            self.Y  = encoder.fit_transform(self.Y)
            self.Y  = tensorflow.keras.utils.to_categorical(self.Y,self.num_classes)


    def save_trans_models(self,countvector_file, tf_idf_file):
        self.save_model(countvector_file,self.vectorizer.vocabulary_ )
        self.save_model(tf_idf_file,self.tfidf_transformer)

    
    def Split_Data(self):
        #split the data into 80% training and 20% testing
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X,self.Y,test_size=0.20,random_state=0)

    def save_model(self,name_file,name_model):
        # Save Model to a file
        pickle.dump(name_model, open(name_file, 'wb'))
    
    def load_model(self,name_file):
        # Load Model from a file
        model = pickle.load(open(name_file, 'rb'))
        return model   

