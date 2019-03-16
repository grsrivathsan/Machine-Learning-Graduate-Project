# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:54:39 2018

@author: SRIVATHSAN
"""
import numpy as np;
import math;

def getNumFeaturesNRows(inputFile):
 rows = 0;maxval = 0;
 for line in inputFile:
    split = line.split();
    #print(split);print(len(split));
    for i in range(1,len(split)):
        pair = split[i].split(':');
        fval = int(pair[0]);
        #print(fval);
        if fval > maxval:
            maxval = fval;
    rows = rows + 1;        
    #maxval = 74481;
 return maxval,rows;

def loadData(trainingFile,features,rows):
    #print("inside loadDAta");    
    #Initialiazing XArray(features) and YArray(labels)
    xdata = np.zeros((rows,features),"double"); #rows X features
    #print(x)
    ydata = np.zeros((rows,1),int);#rows X 1.. Since the default type is float in np.zeros
    #print(trainingFile.tell()); #Current Position is in End of file.. Initialize it
    #Initialize the pointer position in File
    trainingFile.seek(0);
    pos = 0;
   # print(trainingFile.tell())
    for line in trainingFile:
        val = line.split();
        #print(val[0]);
        ydata[pos] = val[0];
        xrow = np.zeros((1,features),"float");
        #print("xrowShape:",xrow.shape);
        #xrow = np.array((features), 'float');
        for entry in val[1:]:
            fValue = entry.split(":");#print(fValue[1]);
            xrow[0][int(fValue[0]) - 1] = float(fValue[1]);
        #print("xrow:",xrow);print("xsize:",len(xrow));
        xdata[pos] = xrow[0];
        pos = pos + 1;
    
    return xdata,ydata;

def calcDistance(xtrain,ytrain,f,x,k):
    dist = 0;rms = 0;kdict = [];
    for i in range(len(ytrain)):
        dist = 0;
        if(i % 5000) == 0:
            print("Trained:",i," examples");
        temp = [];
        for j in range(0,f-1):
#            if(j % 50000) == 0:
#                print("Computed Dist for ",j, "columns");
            dist +=  pow((x[j] - xtrain[i][j]),2);
        rms = math.sqrt(dist);
        temp.append(rms);
        temp.append(ytrain[i]);
        kdict.append(temp);
    
    kdict.sort();
    print("Dictionary Sorted");
    positive = negative = 0;
    print("K value:",k);
    for i in range(0,k):
        if kdict[i][1] == 1:
            positive += 1;
        elif kdict[i][1] == -1:
            negative += 1;
    
    if positive > negative:
        return 1;
    else:
        return 0;

def calcHammingDistance(xtrain,ytrain,f,x,k):
    kdict = [];
    for i in range(len(ytrain)):
        
        featureSimilarities = 0;
        if(i % 5000) == 0:
            print("Trained:",i," examples");
        temp = [];
        for j in range(0,f-1):
#            if(j % 50000) == 0:
#                print("Computed Dist for ",j, "columns");
            if (x[j] == xtrain[i][j]):
                featureSimilarities += 1;
                    
        temp.append(featureSimilarities);
        temp.append(ytrain[i]);
        kdict.append(temp);
    
    kdict.sort();
    print("Dictionary Sorted");
    positive = negative = 0;
    print("K value:",k);
    for i in range(0,k):
        if kdict[i][1] == 1:
            positive += 1;
        elif kdict[i][1] == -1:
            negative += 1;
    
    if positive > negative:
        return 1;
    else:
        return 0;
    
def calcAccuracy(ytest,prediction):
    count = 0;
    for i in range(len(ytest)):
        if (ytest[i] == prediction):
            count = count + 1;
    accuracy = (count / len(ytest)) * 100;
    return accuracy;
        

def main():
    print("K Nearest Neighbours");
    print("====================");
    #Choose k as an odd number to avoid ties
    k = 17;
    #Load Training Set:
    #trainingFile = open('../movie-ratings/datasetPrepared/data.train',"r",encoding="utf8");
    trainingFile = open('../movie-ratings/datasetPrepared/data.train',"r",encoding="utf8"); 
    features,rows = getNumFeaturesNRows(trainingFile);
    print(features,rows);    
    #Form an Array of Feature vector and labels
    x,y = loadData(trainingFile,features,rows);
    print("Loaded training Set");
#    #Load Test Set:
#    #testFile = open('../data/train.liblinear',"r",encoding="utf8"); 
#    testFile = open('../movie-ratings/data-splits/data.test',"r",encoding="utf8");
#    featuresTest,rowsTest = getNumFeaturesNRows(testFile);
#    xtest,ytest = loadData(testFile,featuresTest,rowsTest);
#    
#    prediction = [];
#    for i in range(len(ytest)):
#        examplePredict = calcDistance(x,y,features,xtest[i],k);
#        print("examplePredict:",examplePredict);
#        prediction.append(examplePredict);
#        
#    del x,y,xtest;
#    #Predict Accuracy on test set/eval set
#    accuracy = calcAccuracy(ytest,prediction);
#    
#    print("Accuracy on Test Set for KNN:",accuracy);
#    
    #Compute Accuracy on Eval Set:
    print("******************************************");
    print("Predicting accuracy of Evaluation set");
    evalFile = open('../movie-ratings/data-splits/data.eval.anon',"r",encoding="utf8");
    featuresEval,rowsEval = getNumFeaturesNRows(evalFile);
    xeval,yeval = loadData(evalFile,featuresEval,rowsEval);
    
    prediction = [];
    
    fopen = open('../movie-ratings/results/KNN-hamming-results.txt', 'w');
    
    for i in range(len(yeval)):
        examplePredict = calcHammingDistance(x,y,features,xeval[i],k);
        print("label of line ",i,":",examplePredict);       
        prediction.append(examplePredict);
        fopen.write(str(prediction[i]));
        fopen.write("\n");
    
    #write the predicted label to file
#    fopen = open('../movie-ratings/results/KNN-results.txt', 'w');
#    for i in len(prediction):
#        
#        fopen.write(prediction[i]);
#        fopen.write("\n");
    print("Eval file successfully written");

if __name__ == '__main__':
    main();
