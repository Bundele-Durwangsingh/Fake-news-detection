import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# procesing the data
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')
data_fake['class'] = 0
data_true['class'] = 1
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop(([i]), axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop(([i]), axis=0, inplace=True)

data_fake_manual_testing['class']=0
data_true_manual_testing ['class']=1
data_merge =pd.concat([data_fake,data_true], axis =0)
data =data_merge.drop(['title','subject','date'],axis=1)
data=data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True)
print("clear 1")

#processing the content
def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+','',text)
    text=re.sub('<,*?>+','',text)
    text=re.sub('[%s]' % re.escape (string.punctuation),'' , text)
    text=re.sub('\n','', text)
    text=re.sub('\w*\d\w*', '', text)
    return text
data['text']=data['text'].apply(wordopt)
x=data['text']
y=data['class']
print("clear 2")


#traning the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)
print("clear 3")

LR=LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)



def output_label(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not a fake news"

def manual_testing(news):
    testing_news={'text':[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test['text']=new_def_test['text'].apply(wordopt)
    new_x_test=new_def_test['text']
    new_xv_test=vectorization.transform(new_x_test)
    pred_lr=LR.predict(new_xv_test)
    return print("\n\nLR Prediction: {} ".format(output_label(pred_lr[0])))

news=str(input("Enter your News"))
manual_testing(news)
