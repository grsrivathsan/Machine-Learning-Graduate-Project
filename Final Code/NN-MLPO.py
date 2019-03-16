# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:26:02 2018

@author: SRIVATHSAN
"""

from sklearn.preprocessing import StandardScaler
import numpy as np;
from sklearn.neural_network import MLPClassifier;
from sklearn.metrics import classification_report,confusion_matrix;
from scipy import sparse;

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
    maxval = 74481;
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


print("Neural Network - MultiLayer Perceptron");
print("======================================");
scaler = StandardScaler()
# Fit only to the training data
trainingFile = open('../movie-ratings/datasetPrepared/data.train',"r",encoding="utf8"); 
features,rows = getNumFeaturesNRows(trainingFile);
print(features,rows);    
    #Form an Array of Feature vector and labels
xtrain,ytrain = loadData(trainingFile,features,rows);
xtrain = sparse.csr_matrix(xtrain);
ytrain = ytrain.ravel();
print("Loaded Training Set");
#scaler.fit(xtrain); print("Normalization done on training");
#xtrain = scaler.transform(xtrain); print("Scalar Transformation on Training Done");
mlp = MLPClassifier(hidden_layer_sizes=(40,30,20));print("MLP declared");
mlp.fit(xtrain,ytrain); print("MLP Fit with training done");

#Load Test Set
testFile = open('../movie-ratings/data-splits/data.test',"r",encoding="utf8");
featuresTest,rowsTest = getNumFeaturesNRows(testFile);
xtest,ytest = loadData(testFile,featuresTest,rowsTest);
print(featuresTest,rowsTest);
#xtest = scaler.transform(xtest); print("Scalar Transformation on Test done");
xtest = sparse.csr_matrix(xtest);
ytest = ytest.ravel();
print("Loaded Test Set");
predictions = mlp.predict(xtest); print("Prediction done on xtest set");
print("type:",type(predictions));print("len:",len(predictions));
print("=======");
print(predictions);

print(confusion_matrix(ytest,predictions)); print("Confustion Matrix");

print("Classification Report")
print(classification_report(ytest,predictions));

#Prediction in eval set
evalFile = open('../movie-ratings/data-splits/data.eval.anon',"r",encoding="utf8");     
feval,reval = getNumFeaturesNRows(evalFile);
print(feval,reval);    
    
xeval,yeval = loadData(evalFile,feval,reval);
xeval = sparse.csr_matrix(xeval);
del evalFile;
print("****************************************");
print("Predicting accuracy of Evaluation set");    
predictionEval = mlp.predict(xeval); print("Prediction done on xeval set");

fopen = open('../movie-ratings/results/NN-MLP3.txt', 'w');
for i in np.nditer(predictionEval):
    fopen.write(str(i));
    fopen.write("\n");
    
print("Successfully Written to file");



