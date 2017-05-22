
# coding: utf-8

# In[19]:

import pandas as pd
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows',None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth',0)
import unicodedata

from pandas import DataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
np.set_printoptions(threshold='nan')

import math
from __future__ import division


# In[2]:

df = pd.read_csv('train.tsv', sep='\t', index_col =None)
df


# In[3]:

good = df[df["Label"]==1]    #good
bad = df[df["Label"]==2]    #bad


# In[4]:

#Numericals
Attributes = ["Attribute2","Attribute5","Attribute8","Attribute11","Attribute13","Attribute16","Attribute18"]

for attribute in Attributes: 
    Attribute_good = good[[attribute]]
    Attribute_bad = bad[[attribute]]
    
    #convert data into list
    Attribute2_good = list(Attribute_good.values.flatten())
    Attribute2_bad = list(Attribute_bad.values.flatten())
    
    
    data_to_plot = [Attribute2_good, Attribute2_bad]

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)


    # Create the boxplot
    ## change the style of fliers and their fill
    flierprops = dict(marker='o', markerfacecolor='#e7298a', markersize=12,
                  linestyle='none', markeredgecolor='g') 
    
    #Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True, flierprops= flierprops)
    
    #Change color

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#9e1b90' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#20ef04', linewidth=2)

    ## change the style of fliers and their fill
    #for flier in bp['fliers']:
     #   flier.set(marker='o', color='#e7298a', alpha=0.5)
        
      

    #Set grid
    ax.grid()

    #Set labels
    ax.set_xticklabels(['Good', 'Bad'])    

    #Set title
    plt.title(attribute)
    plt.show()

    # Save the figure
    #fig.savefig('fig1.png', bbox_inches='tight')


# In[5]:

#for categorical data


attributesCategorical = ['Attribute1','Attribute3','Attribute4','Attribute6','Attribute7','Attribute9','Attribute10'
              ,'Attribute12','Attribute14','Attribute15','Attribute17','Attribute19','Attribute20']

for item in attributesCategorical:

    attributeGood = pd.Series(good[item])
    attributeBad = pd.Series(bad[item])
    stringGood = item + " Good"
    stringBad = item + " Bad"

    plt.title(item)
    attributeGood.value_counts().plot(kind='bar',label= stringGood, color = '#9129e7' )
    attributeBad.value_counts().plot(kind='bar',label= stringBad, color = '#eae727')
    plt.legend()
    plt.show()


# In[6]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

withoutLabel = df.ix[:, df.columns != 'Label' ]
withoutLabelAndId = withoutLabel.ix[:, withoutLabel.columns != 'Id' ]

Count_Row = withoutLabelAndId.shape[0]

listOfDocuments= []
for i in range(Count_Row):
    documents = withoutLabelAndId.iloc[i]
    documents = documents.to_string(index=False , header = False).encode('utf8')
    listOfDocuments.append(documents)
    
    
tupleOfDocuments = tuple(listOfDocuments)


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tupleOfDocuments)

#SVD
tfidf_matrix = np.array(TruncatedSVD(n_components=10).fit_transform(tfidf_matrix))
print tfidf_matrix


# In[7]:

X = np.array(tfidf_matrix)
y = []
for i in range(Count_Row):
    y.append(df["Label"].iloc[i]) 
    

y = np.array(y)

print y


# In[8]:

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import *


random_state = np.random.RandomState(0)
k_fold = KFold(n_splits=10,shuffle = True)

clf = SVC(kernel = 'linear' , C = 1.0, probability=True, random_state=random_state)
clf.fit(X, y)
accuracySVM = cross_val_score(clf, X, y, cv=k_fold, n_jobs=-1 , scoring = 'accuracy')
accuracyMeanSVM =np.mean(accuracySVM)
print "SVM"
print accuracySVM
print accuracyMeanSVM

clf = RandomForestClassifier(n_estimators=80)
clf.fit(X, y)
accuracyRF = cross_val_score(clf, X, y, cv=k_fold, n_jobs=-1 , scoring = 'accuracy')
accuracyMeanRF =np.mean(accuracyRF)
print "RF"
print accuracyRF
print accuracyMeanRF

clf = GaussianNB()
clf.fit(X, y)
accuracyNB = cross_val_score(clf, X, y, cv=k_fold, n_jobs=-1 , scoring = 'accuracy')
accuracyMeanNB =np.mean(accuracyNB)
print "NB"
print accuracyNB
print accuracyMeanNB


# In[9]:

import csv

with open('EvaluationMetric_10fold.csv', 'wb') as fp:
    writer = csv.writer(fp, delimiter='\t') 
    l = [(' ','Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM'),
         (' ','Accuracy', accuracyMeanNB, accuracyMeanRF, accuracyMeanSVM)]
    writer.writerows(l)
    
evalDf = pd.read_csv('EvaluationMetric_10fold.csv',sep='\t',encoding="utf-8",index_col=0)
evalDf


# In[10]:

dfTest = pd.read_csv('test.tsv',sep='\t',encoding="utf-8")

withoutId = dfTest.ix[:, dfTest.columns != 'Id' ]

withoutId


# In[11]:

#figuring out categories for testSet

Count_RowTest = dfTest.shape[0]

listOfDocumentsTest= []
IdList = []
for i in range(Count_RowTest):
    documents = withoutId.iloc[i]
    IdList.append(dfTest['Id'].iloc[i])
    documents = documents.to_string(index=False , header = False).encode('utf8')
    listOfDocumentsTest.append(documents)
    
tupleOfDocumentsTest = tuple(listOfDocumentsTest)    

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrixTest = tfidf_vectorizer.fit_transform(tupleOfDocumentsTest)

#SVD
tfidf_matrixTest = np.array(TruncatedSVD(n_components = 10).fit_transform(tfidf_matrixTest))


clf = GaussianNB()
clf.fit(X, y)

categories = clf.predict(tfidf_matrixTest)

TestList = []

for i in range(len(categories)):
    if (categories[i]==1):
         TestList.append("Good")
    elif (categories[i]==2):
         TestList.append("Bad")

FinalTestList = zip(IdList, TestList)

with open('testSet_Predictions.csv', 'wb') as fp:
    writer = csv.writer(fp, delimiter='\t') 
    l = [(' ','Client_ID', 'Predicted_Label')]
    
    for item in FinalTestList:
        l.append((' ',item[0],item[1]))
         
    writer.writerows(l)
    
dfPredict = pd.read_csv('testSet_Predictions.csv',sep='\t',encoding="utf-8",index_col=0)
dfPredict    


# In[13]:

dfAttr = pd.read_csv('train.tsv',sep='\t',encoding="utf-8")

dfAttr = dfAttr.ix[:, dfAttr.columns != 'Id' ]
dfAttr = dfAttr.ix[:, dfAttr.columns != 'Label' ]
#dfAttrInitial = dfAttr

accuracyMeanList = []

Attributes = ["Attribute1","Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16","Attribute17","Attribute18","Attribute19"]
for attribute in Attributes:

    #dfAttr = dfAttrInitial.ix[:, dfAttrInitial.columns != attribute ]
    dfAttr = dfAttr.ix[:, dfAttr.columns != attribute ]
    print attribute
    print dfAttr

    Count_RowAttr = dfAttr.shape[0]

    listOfDocumentsAttr= []
    for i in range(Count_RowAttr):
        documents = dfAttr.iloc[i]
        documents = documents.to_string(index=False , header = False).encode('utf8')
        listOfDocumentsAttr.append(documents)


    tupleOfDocumentsAttr = tuple(listOfDocumentsAttr)


    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrixAttr = tfidf_vectorizer.fit_transform(tupleOfDocuments)

    #SVD
    tfidf_matrixAttr = np.array(TruncatedSVD(n_components=10).fit_transform(tfidf_matrixAttr))

    X = np.array(tfidf_matrixAttr)
    y = []
    for i in range(Count_RowAttr):
        y.append(df["Label"].iloc[i]) 


    y = np.array(y)

    #print y


    k_fold = KFold(n_splits=10,shuffle = True)

    clf = GaussianNB()
    clf.fit(X, y)
    accuracyNB = cross_val_score(clf, X, y, cv=k_fold, n_jobs=-1 , scoring = 'accuracy')
    accuracyMeanNB =np.mean(accuracyNB)
    #print accuracyNB
    #print accuracyMeanNB
    accuracyMeanList.append(accuracyMeanNB)
    
print "accuracyMeanList" 
print accuracyMeanList


# In[20]:

def entropy(num1,num2):
    a = num1/(num1+num2)
    b = num2/(num1+num2)
    mathem = -a*math.log(a,2) -b*math.log(b,2)
    return mathem


# In[21]:

Count_Row = df.shape[0]
print Count_Row

labelFeature = df['Label']
labelFeature = pd.Series(labelFeature)
#feature

counts = np.array(labelFeature.value_counts());

print counts

entropiesLabel = []

entropiesLabel.append(entropy(counts[0],counts[1]))
       

parentEntropy = sum(entropiesLabel)
print parentEntropy


# In[27]:

attributesCategorical = ['Attribute1','Attribute3','Attribute4','Attribute6','Attribute7','Attribute9','Attribute10'
              ,'Attribute12','Attribute14','Attribute15','Attribute17','Attribute19','Attribute20']
infoGain = []
for attribute in attributesCategorical:   
    
    rating_probs = df.groupby(attribute).size().div(len(df)) 
   
    a = df.groupby([attribute,'Label']).size()

    keys = list([a.keys()[x][0] for x in range(len(a.keys()))])

    values = list([i for i in a])

    allVal = zip(keys,values)

    terms =[]

    r = zip(allVal,allVal[1:])[::2]
   
    for i in r:
        entr1 = entropy(i[0][1],i[1][1])
        result = rating_probs[i[0][0]]*entr1
        terms.append(result)


    entropyAttr = sum(terms)
    numAttribute = attribute.replace('Attribute', '')
    numAttribute = int(numAttribute)

    infoGain.append(((parentEntropy - entropyAttr),numAttribute))
print infoGain


# In[35]:

AttributesNumerical = ["Attribute2","Attribute5","Attribute8","Attribute11","Attribute13","Attribute16","Attribute18"]
group_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

for attribute in AttributesNumerical:  
    
   
    feature = df[attribute]
    attributeCut = pd.cut(feature, 5, labels = group_names)

    d = {attribute : pd.Series(attributeCut),
            'Label' : pd.Series(df['Label'])}

    d = pd.DataFrame(d)

    a = d.groupby([attribute,'Label']).size()   
    rating_probs = d.groupby(attribute).size().div(len(d))
 

    keys = list([a.keys()[x][0] for x in range(len(a.keys()))])

    values = list([i for i in a])

    allVal = zip(keys,values)

    terms =[]

    r = zip(allVal,allVal[1:])[::2]
    print r

    for i in range(len(r)):

        entr2 = entropy(r[i][0][1],r[i][1][1])
        result = rating_probs[i]*entr2
        terms.append(result)


    entropyAttr = sum(terms)
    
    numAttribute = attribute.replace('Attribute', '')
    numAttribute = int(numAttribute)
    
    print "AAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    print parentEntropy
    print entropyAttr

    infoGain.append(((parentEntropy - entropyAttr),numAttribute))
#print infoGain


# In[33]:

sortInfoGain = sorted(infoGain, key=lambda x: x[1])
print sortInfoGain


# In[ ]:



