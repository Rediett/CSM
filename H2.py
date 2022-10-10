from nltk.stem import *
import numpy as np
import sklearn as sk
import pandas as pd
from scipy import spatial
import csv
from sklearn.feature_extraction.text import CountVectorizer

# implment countVectorization
def NPcountVector(Docs, ans=None):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(Docs)
    ra = X.toarray()
    nra = []
    # print(ra,nra)
    c = 0
    for i in ra:
        l = list(i)
        if ans != None:
            l.append(ans[c])
        nra.append(l)
        c += 1
    npCV = np.array(nra)
    npcvT = npCV.T
    sumc = np.sum(npcvT,axis=1)
    mx = max(sumc)
    nwl = []
    for s in range(len(sumc)-1):
        if sumc[s] <= (0.5*mx):
            nwl.append(npcvT[s])
    nwl=np.array(nwl)
    nwlt = nwl.T

    return (npCV,sumc)
def stemData(path,max=None):
    file = open(path, newline='')
    Data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    DataStem = []
    p = 0
    ansKey = []
    pars = []
    stemmer = PorterStemmer()
    for l in Data:
        if(max != None and p == max):
            break
        par = ""
        for s in l[-1].split(" "):
            
            if(par != ""):
                par += " "
            par += stemmer.stem(s.split("\\n")[-1])
        DataStem.append([l[0],par])
        pars.append(par)
        ansKey.append(l[0])
        p += 1
    return (DataStem,pars,ansKey)

def uploadData(val):
    with open('ans.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(val)
    
def dist(to,fromm):
    d = np.subtract(to,fromm)
    ans = np.dot(d.T,d)
    ans = np.sqrt([ans])
    return ans[0]
    
def Knn(k, train, test, ans):
    correct = 0
    wrong = 0
    for i in test:
        diff = train - i
        dist = np.sqrt(np.sum(diff**2,axis=-1))
        indx = np.argmin(dist)
        l = {}
        maxsize = 0
        pred = train[indx][-1]
        if pred == i[-1]:
            correct += 1
        else:
            wrong += 1
    ans = correct/(correct + wrong)
    return ans

def Knn2(k, train, test):
    correct = 0
    wrong = 0
    c =0
    with open('ans.csv', 'a', encoding='UTF8') as f:
        for i in test:
            diff = train - i
            dist = np.sqrt(np.sum(diff**2,axis=-1))
            indx = np.argmin(dist)
            maxsize = 0
            pred = train[indx][-1]
            writer = csv.writer(f)
            writer.writerow([int(pred)])

def cleanup(ra,row):
    ind = []
    for i in range(len(row)-1):
        if(row[i] >= (0.5*max(row)) or row[i] <= 4.0):
            ind.append(i)
    op = np.delete(ra,ind,1)
    return op

# word processing
maxL = 1000
train = stemData("/Users/rediettadesse/Desktop/CS/CS484/H2/1663973187_4812264_new_train.csv",maxL)
trainingData = train[0]
trainingDocs = train[1]
trainAnskey = train[2]
tp = NPcountVector(trainingDocs,trainAnskey)
print(tp[0].shape,tp[0][0]) 
traingVectorVals = tp[0]
traingVectorVals = cleanup(traingVectorVals,tp[1])

print(traingVectorVals.shape,traingVectorVals[0])
test = stemData("/Users/rediettadesse/Desktop/CS/CS484/H2/1664308636_4631202_new_test.csv",maxL)
testingData = train[0]
testingDocs = train[1]
testp = NPcountVector(trainingDocs)
print(testp[0].shape,testp[0][0]) 
testingVectorVals = testp[0]
testingVectorVals = cleanup(testingVectorVals,testp[1])
print("processing")

# Cross validation
partitionSize = round(len(traingVectorVals)*0.1)
optimalK = {}

for k in range(1,5):
    print(k)
    for l in range(0,len(traingVectorVals),partitionSize):
        testPartition = traingVectorVals[l:l+partitionSize]
        trainPartition = np.concatenate((traingVectorVals[:l],traingVectorVals[l+partitionSize:]),axis=0)
        a = Knn(k,trainPartition,testPartition,trainAnskey)
        if k in optimalK:
            optimalK[k].append(a)
        else:
            optimalK[k] = [a]
bestK = 0
bestavg = 0
for m in optimalK.keys():
    avg = sum(optimalK[m])/(len(optimalK[m]))
    if(bestavg < avg):
        bestK = m
        bestavg = avg
print("BestK:",bestK,"avg:",(bestavg*100))

fl = open('ans.csv', 'w')
fl.close() 
ini = 0
perc = 0.10
for l in range(0,(testingVectorVals.shape[0]),round(testingVectorVals.shape[0]*perc)):
    print("From",l,"To",l+round(testingVectorVals.shape[0]*perc))
    Knn2(bestK,traingVectorVals,testingVectorVals[l:l+round(testingVectorVals.shape[0]*perc)])
