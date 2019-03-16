# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:42:33 2018

@author: SRIVATHSAN
"""


import numpy as np
import random;
def getNumFeaturesNRows(inputFile):
    features = 0;rows = 0;firstline='';
    for line in inputFile:
        firstline = line;
        val = line.split();
        if(len(val) > features):
            features = len(val);
        rows = rows + 1;
    #Excluding the label
    features = features - 1;
    features = 74481;
    #print(firstline);
    return features,rows;

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

def randomNum():
    randomNumber = random.uniform(-0.01,0.01);
    return randomNumber;

def trainModel(xval,yval,b,wval,eta,rows,avgW,avgB):
    #print("wvaltransposeshape:",wval.transpose().shape);
    #print("xval.shape:",xval.shape);print("xval[0].shape:",np.asarray(xval[0]).shape);
    #print("rows:",rows);
    updates = 0;
    for i in range(0,rows): #zip to make values iterable
       #print(yval[i]*(np.dot(xval[i],wval.transpose())+b));
       if yval[i] == 0:
           yvalue = -1;
       else:
           yvalue = yval[i];
       if (yvalue*(np.dot(xval[i],wval.transpose())+b)) < 0:
            #update wval and b
            wval= wval+(eta*yvalue*xval[i]);#print("wval:",wval);
            b = b+(eta*yvalue);#print("b:",b);
            updates = updates + 1;
       #For each example update avgW and avgB    
       avgW = avgW + wval;
       avgB = avgB + b;
       #updates = updates + 1;
    #print("wval.shapeH:",wval.shape);
    #print("b:",b);
    #print("updates:",updates);
    return wval,b,updates,avgW,avgB;

def accuracyCalc(xdev,ydev,wNew,bias,rowsDev):
    count = 0;
    for i in range(0,rowsDev):
        if ydev[i] == 0:
            yval = -1;
        else:
            yval = ydev[i];
        if (yval*(np.dot(xdev[i],wNew.transpose())+bias)) < 0:
            #count the number of misprediction
            count = count + 1;
     
    #correct predictions = totalrows - noOfFalsePredictions
    correct_pred = rowsDev - count;
    accuracy = (correct_pred/rowsDev)*100;
    #print("Accuracy:",accuracy);
    return accuracy;
    
def labelPredict(xeval,yeval,bestWeight,bestBias,rowsEval):
    
    print("Predicting eval set label");
    fopen = open('../movie-ratings/results/averagedPerceptron-results.txt', 'w');
    for i in range(0,rowsEval):
        if (np.dot(xeval[i],bestWeight.transpose())+bestBias) < 0:
            label = "0";
        else:
            label = "1";
        fopen.write(label);
        fopen.write("\n");
    print("Eval file successfully written");    
    
def getParamsTrain(cvCount):
    if cvCount == 0:
        split1 = "../movie-ratings/datasetPrepared/CVSplits/training0123.data";
        fileopen = open(split1,"r",encoding="utf8");
        f1,row1 = getNumFeaturesNRows(fileopen);
        xval1234,yval1234 = loadData(fileopen,f1,row1);
        print("xval1234.shape:",xval1234.shape,"yval1234.shape:",yval1234.shape);
        del split1,fileopen;
        return xval1234,yval1234,row1,f1;
        del xval1234,yval1234,f1;
    elif cvCount == 1:
        split2 = "../movie-ratings/datasetPrepared/CVSplits/training0124.data";
        fileopen = open(split2,"r",encoding="utf8");
        f2,row2 = getNumFeaturesNRows(fileopen);
        xval1235,yval1235 = loadData(fileopen,f2,row2);
        print("xval1235.shape:",xval1235.shape,"yval1235.shape:",yval1235.shape);
        del split2,fileopen;
        return xval1235,yval1235,row2,f2;
        del xval1235,yval1235;
    elif cvCount == 2:
        split3 = "../movie-ratings/datasetPrepared/CVSplits/training0134.data";
        fileopen = open(split3,"r",encoding="utf8");
        f3,row3 = getNumFeaturesNRows(fileopen);
        xval1245,yval1245 = loadData(fileopen,f3,row3);
        print("xval1245.shape:",xval1245.shape,"yval1245.shape:",yval1245.shape);
        del split3,fileopen;
        return xval1245,yval1245,row3,f3;
        del xval1245,yval1245;
    elif cvCount == 3:
        split4 = "../movie-ratings/datasetPrepared/CVSplits/training0234.data";
        fileopen = open(split4,"r",encoding="utf8");
        f4,row4 = getNumFeaturesNRows(fileopen);
        xval1345,yval1345 = loadData(fileopen,f4,row4);
        print("xval1345.shape:",xval1345.shape,"yval1345.shape:",yval1345.shape);
        del split4,fileopen;
        return xval1345,yval1345,row4,f4;
        del xval1345,yval1345;
    elif cvCount == 4:
        split5 = "../movie-ratings/datasetPrepared/CVSplits/training1234.data";
        fileopen = open(split5,"r",encoding="utf8");
        f5,row5 = getNumFeaturesNRows(fileopen);
        xval2345,yval2345 = loadData(fileopen,f5,row5);
        print("xval2345.shape:",xval2345.shape,"yval2345.shape:",yval2345.shape);
        del split5,fileopen;
        return xval2345,yval2345,row5,f5;
        del xval2345,yval2345;
    else:
        print("Count Value:",cvCount," not recognized");

def getParamsTest(cvCount):
    if cvCount == 0:
        split5 = "../movie-ratings/datasetPrepared/CVSplits/training04.data";
        fileopen = open(split5,"r",encoding="utf8");
        f5,row5 = getNumFeaturesNRows(fileopen);
        xval5,yval5 = loadData(fileopen,f5,row5);
        print("xval5.shape:",xval5.shape,"yval5.shape:",yval5.shape);
        del split5,fileopen,f5;
        return xval5,yval5,row5;
        del xval5,yval5,row5;
    elif cvCount == 1:
        split4 = "../movie-ratings/datasetPrepared/CVSplits/training03.data";
        fileopen = open(split4,"r",encoding="utf8");
        f4,row4 = getNumFeaturesNRows(fileopen);
        xval4,yval4 = loadData(fileopen,f4,row4);
        print("xval4.shape:",xval4.shape,"yval4.shape:",yval4.shape);
        del split4,fileopen,f4;
        return xval4,yval4,row4;
        del xval4,yval4,row4;
    elif cvCount == 2:
        split3 = "../movie-ratings/datasetPrepared/CVSplits/training02.data";
        fileopen = open(split3,"r",encoding="utf8");
        f3,row3 = getNumFeaturesNRows(fileopen);
        xval3,yval3 = loadData(fileopen,f3,row3);
        print("xval3.shape:",xval3.shape,"yval3.shape:",yval3.shape);
        del split3,fileopen,f3;
        return xval3,yval3,row3;
        del xval3,yval3,row3;
    elif cvCount == 3:
        split2 = "../movie-ratings/datasetPrepared/CVSplits/training01.data";
        fileopen = open(split2,"r",encoding="utf8");
        f2,row2 = getNumFeaturesNRows(fileopen);
        xval2,yval2 = loadData(fileopen,f2,row2);
        print("xval2.shape:",xval2.shape,"yval2.shape:",yval2.shape);
        del split2,fileopen,f2;
        return xval2,yval2,row2;
        del xval2,yval2,row2;
    elif cvCount == 4:
        split1 = "../movie-ratings/datasetPrepared/CVSplits/training00.data";
        fileopen = open(split1,"r",encoding="utf8");
        f1,row1 = getNumFeaturesNRows(fileopen);
        xval1,yval1 = loadData(fileopen,f1,row1);
        print("xval1.shape:",xval1.shape,"yval1.shape:",yval1.shape);
        del split1,fileopen,f1;
        return xval1,yval1,row1;
        del xval1,yval1,row1;
    else:
        print("Count Value:",cvCount," not recognized");
    
    
def getBestHyperparameter(w,b):
    
    wNew = w;biasNew = b;
    
        
    xval1234 = xval1235 = xval1245 = xval1345 = xval2345 = 0;
    
    cvTrainx = np.array([xval1234,xval1235,xval1245,xval1345,xval2345]);
    #cvTestx = np.array
    #print("shape of y:",cvTest[0].shape)
    #for every eta value.. compute the average accuracy
    #etas = np.array([1, 0.1, 0.01], 'float');
    etas = np.array([0.01, 0.1, 1], 'float');
    #print(rows[0]);
    avgAcc = [];etaTrack = [];
   
    for eta in etas:
        cvCount = 0;accuracy = 0;totAcc=0;etaCount = 0;
        #wNew = w;biasNew = b;
        for trainSet in cvTrainx:
            wNew = w;biasNew = b;
            cvtrx,cvtry,rowtr,f = getParamsTrain(cvCount);
            avgW = np.zeros((1, f));
            avgB = 0;
            #print("cvCount",cvCount);
            for epoch in range(0,10):
                wNew,biasNew,updates,avgW,avgB = trainModel(cvtrx,cvtry,biasNew,wNew,eta,rowtr,avgW,avgB);
                #wNew,biasNew,updates,avgW,avgB = trainModel(trainSet,cvTrainy[cvCount],biasNew,wNew,eta,rows[cvCount],avgW,avgB);
            #After ten epoch calculate the accuracy
            del cvtrx,cvtry,rowtr;
            cvttx,cvtty,rowte = getParamsTest(cvCount);
            accuracy = accuracyCalc(cvttx,cvtty,avgW,avgB,rowte); print("accuracy:",accuracy);
            del cvttx,cvtty,rowte,avgW,avgB;
            #accuracy = accuracyCalc(cvTestx[cvCount],cvTesty[cvCount],avgW,avgB,rows[cvCount]);
            totAcc = totAcc + accuracy;
            cvCount = cvCount + 1;
        #Compute Average Accuracy of 5Fold CV for each eta value
        #print("totAcc:",totAcc);
        avgAcc.append(totAcc/(5));
        etaTrack.append(etaCount);
        etaCount = etaCount + 1;
    #choose the eta with highest avgAcc    
    #print(avgAcc);
    maxAccIndex = np.argmax(np.array(avgAcc));#print(maxAccIndex)
    maxAcc = avgAcc[maxAccIndex];#print(maxAcc);
    maxEta = etas[maxAccIndex];#print(maxEta);
    
    return maxEta,maxAcc;

    
        
    
    
    
        
def main():
    print("Perceptron Model - Averaged");
    print("===========================");
    #Load the training set
    #trainingFile = open('../dataset/diabetes.train',"r",encoding="utf8");
    trainingFile = open('../movie-ratings/datasetPrepared/data.train',"r",encoding="utf8");
    features,rows = getNumFeaturesNRows(trainingFile);
    #print(features,rows);    
    #Form an Array of Feature vector and labels
    x,y = loadData(trainingFile,features,rows);
    #print(type(x));
    print("here");
    print("xshape:",x.shape);
    print("yshape:",y.shape);
    
    #Loading the dev set:
    devFile = open('../movie-ratings/datasetPrepared/data.dev',"r",encoding="utf8");
    featuresDev,rowsDev = getNumFeaturesNRows(devFile);
    xdev,ydev = loadData(devFile,featuresDev,rowsDev);
    print("xdev:",xdev.shape);print("ydev:",ydev.shape);
   
    #eta = 0.01;
    #Initialize weight vector w and bias b to a random number between -0.01 and 0.01
    bias = random.uniform(-0.01,0.01);
    #variables to store the parameters for each epoh to choose the best epoh
    biasArray = [];
    wArray = [];
    updateArray = [];epochArray = [];
    accuracyArray = [];
    w = np.zeros((1,features),"float");
    #Load the weight vector with Random Value between -0.01 and 0.01
    for i in range(0,features):
        w[0][i] = randomNum();
    updates = 0;biasNew = bias;wNew = w;    
    print("w before update:",w);
    #print(w.shape);
    
    avgWeight = np.zeros((1, features));
    avgBias = 0;
    
     #Choose eta within {1,0.1,0.01} giving highest accuracy using Cross - Validation
    eta,etaAcc = getBestHyperparameter(wNew,biasNew);
    print("BestHyperParameter Chosen:")
    print("==========================");
    print("eta:",eta);
    print("Hyperparameter Accuracy:",etaAcc);
    print("==========================");
    #Train the classifier for 20 epochs as given in the question
    
    for epoch in range(0,20):
        wNew,biasNew,updates,avgWeight,avgBias = trainModel(x,y,biasNew,wNew,eta,rows,avgWeight,avgBias);
        #Appending the avg bias and avgWeight
        biasArray.append(avgBias);wArray.append(avgWeight);updateArray.append(updates);
        epochArray.append(epoch);
        #print("wval in main:",type(wNew));
        #print("wval in main ac",wNew);
        #predict the model for accuracy using the parameters of current epoch
        accuracy = accuracyCalc(xdev,ydev,avgWeight,avgBias,rowsDev);
        accuracyArray.append(accuracy);
    
    #Select the parameters with highest accuracy
    #print(epochArray);
    print("accuracyArray:",accuracyArray);
    index = 0;highestAcc = 0;
    for i in epochArray:
        if accuracyArray[i] > highestAcc:
            highestAcc = accuracyArray[i];
            index = i;
            
    #print(index,highestAcc);
    beshEpoch = index;
    bestWeight = wArray[index];
    bestBias = biasArray[index];
    bestAccuracy = highestAcc;
    bestUpdates = updateArray[index];
    
    print("Final Perceptron Parameters after Training");
    print("==========================================");
    
    print("beshEpoch:",beshEpoch);
    print("bestWeight:",bestWeight);
    print("bestBias:",bestBias);
    print("bestAccuracy:",bestAccuracy);
    print("bestUpdates:",bestUpdates);

    
    #Calculate Test Set Accuracy:
    #testFile = open('../dataset/diabetes.test',"r",encoding="utf8");
    testFile = open('../movie-ratings/data-splits/data.test',"r",encoding="utf8");
    featuresTest,rowsTest = getNumFeaturesNRows(testFile);
    xtest,ytest = loadData(testFile,featuresTest,rowsTest);
    accuracyTest = accuracyCalc(xtest,ytest,bestWeight,bestBias,rowsDev);
    del xtest,ytest,featuresTest,rowsTest,testFile; 
    print("Accuracy on Test Set:",accuracyTest);

 # =============================================================================
    #Predicting the eval set
    print("******************************************");
    print("Predicting accuracy of Evaluation set");
    evalFile = open('../movie-ratings/data-splits/data.eval.anon',"r",encoding="utf8");
    featuresEval,rowsEval = getNumFeaturesNRows(evalFile);
    xeval,yeval = loadData(evalFile,featuresEval,rowsEval);
    labelPredict(xeval,yeval,bestWeight,bestBias,rowsEval);
    
if __name__ == '__main__':
    main();