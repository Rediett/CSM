
from nltk.stem import *
import numpy as np
import sklearn as sk
import pandas as pd
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
    # print(npCV)
    npcvT = npCV.T
    # print(npCV)
    sumc = np.sum(npcvT,axis=1)
    # print(sumc)
    mx = max(sumc)
    nwl = []
    for s in range(len(sumc)-1):
        if sumc[s] <= (0.5*mx):
            nwl.append(npcvT[s])
    nwl=np.array(nwl)
    nwlt = nwl.T

    # # print(nwlt)

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
#         print("I",i)
        nn = {}
        for j in train:
#             print("J",j)
            dis = dist(j,i)
            if len(nn.keys()) < k and dis not in nn:
                nn[dis] = j
            elif len(nn.keys()) == k and dis not in nn and min(nn.keys()) > dis:
                del nn[min(nn.keys())]
                nn[dis] = j
        l ={}
        maxsize = 0
        pred = 0
        for m in nn.keys():
            if(nn[m][-1] in l):
                l[nn[m][-1]].append(m)
            else:
                l[nn[m][-1]] = [m]
            if len(l[nn[m][-1]]) > maxsize:
                maxsize = len(l[nn[m][-1]])
                pred = nn[m][-1]
            # print(pred)
        if pred == i[-1]:
            correct += 1
        else:
            wrong += 1
    ans = correct/(correct + wrong)
    return ans

def Knn2(k, train, test):
    correct = 0
    wrong = 0
    open('ans.csv', 'w')
    
    with open('ans.csv', 'a', encoding='UTF8') as f:
        for i in test:
            nn = {}
            for j in train:
                dis = dist(j[:(len(j))],i)
                if len(nn.keys()) < k and dis not in nn:
                    nn[dis] = j
                elif len(nn.keys()) == k and dis not in nn and min(nn.keys()) > dis:
                    del nn[min(nn.keys())]
                    nn[dis] = j
            l = {}
            maxsize = 0
            pred = 0
            for m in nn.keys():
                if(nn[m][-1] in l):
                    l[nn[m][-1]].append(m)
                else:
                    l[nn[m][-1]] = [m]
                if len(l[nn[m][-1]]) > maxsize:
                    maxsize = len(l[nn[m][-1]])
                    pred = nn[m][-1]
            writer = csv.writer(f)
            writer.writerow([int(pred)])
            
def cleanup(ra,row):
    ind = []
    for i in range(len(row)-1):
        if(row[i] >= (0.5*max(row)) or row[i] <= 2.0):
            ind.append(i)
    op = np.delete(ra,ind,1)
    return op



# word processing
maxL = 6000
train = stemData("1663973187_4812264_new_train.csv",maxL)
trainingData = train[0]
trainingDocs = train[1]
trainAnskey = train[2]
tp = NPcountVector(trainingDocs,trainAnskey)
print(tp[0].shape,tp[0][0]) 
traingVectorVals = tp[0]
traingVectorVals = cleanup(traingVectorVals,tp[1])

print(traingVectorVals.shape,traingVectorVals[0])
test = stemData("1664308636_4631202_new_test.csv",maxL)
testingData = train[0]
testingDocs = train[1]
testp = NPcountVector(trainingDocs)
testingVectorVals = testp[0]
testingVectorVals = cleanup(testingVectorVals,testp[1])


Knn2(1,traingVectorVals,testingVectorVals)


# # Cross validation
# partitionSize = round(len(traingVectorVals)*0.1)
# optimalK = {}

# for k in range(1,5):
#     print(k)
#     for l in range(0,len(traingVectorVals),partitionSize):
#         testPartition = traingVectorVals[l:l+partitionSize]
#         trainPartition = np.concatenate((traingVectorVals[:l],traingVectorVals[l+partitionSize:]),axis=0)
#         a = Knn(k,trainPartition,testPartition,trainAnskey)
#         if k in optimalK:
#             optimalK[k].append(a)
#         else:
#             optimalK[k] = [a]
# bestK = 0
# bestavg = 0
# for m in optimalK.keys():
#     avg = sum(optimalK[m])/(len(optimalK[m]))
#     if(bestavg < avg):
#         bestK = m
#         bestavg = avg
# print("BestK:",bestK,"avg:",(bestavg*100))



