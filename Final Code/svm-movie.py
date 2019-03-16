# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 20:15:16 2018

@author: SRIVATHSAN
"""

import numpy as np
import random;
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
     
     
#    features = 0;rows = 0;firstline='';
#    for line in inputFile:
#        firstline = line;
#        val = line.split();
#        if(len(val) > features):
#            features = len(val);
#        rows = rows + 1;
#    #Excluding the label
#    features = features - 1;
#    #print(firstline);
    maxval = 74481;
 return maxval,rows;

#Split training and dev set


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

def trainModel(xval,yval,b,wval,eta,rows,mu):
    #print("wvaltransposeshape:",wval.transpose().shape);
    #print("xval.shape:",xval.shape);print("xval[0].shape:",np.asarray(xval[0]).shape);
    #print("rows:",rows);
    #updates = 0;
    #eta = gamma and mu = C
    #print("Inside Traing:");
    for i in range(0,rows):
        if (i % 4000) == 0:
            print("Processed:",i," rows");
        #print("yval[i].shape:",yval[i].shape);
        #print("xval[i].shape:",xval[i].shape);
        #print("wval.transpose().shape",wval.shape);
        #zip to make values iterable
        #print("Here",yval[i]*(np.dot(xval[i],wval.transpose())+b));
        if yval[i] == 0:
           yvalue = -1;
        else:
           yvalue = yval[i];
        #if yvalue*(np.dot(xval[i],wval.transpose())+b) <= 1:
        
        #print(yval[i]*(np.dot(xval[i],wval.transpose())+b));
        #print("yval[i].shape:",yval[i].shape);
        #print("yval[i]:",yval[i]);
        #print("xval[i].shape:",xval[i].shape)
        #print("wval[i].shape:",wval.shape);
        #print("wval[i].transpose.shape:",wval.transpose().shape);
        
        #print("v:",np.dot(xval[i],wval.transpose())+b);
        term1 = np.dot(xval[i],wval.transpose());
        #term2 = np.dot(term1);
        term2 = term1 + b;
        term3 = yvalue * term2;
        #if (yval[i]*(np.dot(xval[i],wval.transpose())+b)) <= 1:  
        if term3 <= 1:  
        
            temp1 = np.array([(1-eta) * x for x in wval]);
            temp2 = np.array([(eta*mu*yvalue) * t for t in xval[i]]);
            #print("temp1.shape:",temp1.shape);
            #print("temp2.shape:",temp2.shape);
            #wval = np.array([(1-eta) * x for x in wval]) + np.array([(eta*mu*yval[i]) * t for t in xval[i]]);
            wval = temp1 + temp2.transpose();
            #print("Sa1:",wval.shape);
            wval = np.array(wval);
            #b = b + (eta * yval[i]);
            b = ((1 - eta) * b) + (eta*mu*yvalue);
           
        else:
          
            wval = [(1-eta) * x for x in wval];
            wval = np.array(wval);
            b = (1 - eta) * b;
            
    return wval,b;

def accuracyCalc(xdev,ydev,wNew,bias,rowsDev,mu):
    count = 0;#tp = 0;fp = 0;fn = 0;
    #prediction
    for i in range(0,rowsDev):
        yval = ydev[i];
        if ydev[i] == 0:
            yval = -1;
        else:
            yval = ydev[i];
        #prediction = np.dot(xdev[i],wNew.transpose())+bias;
    
        if (yval*(np.dot(xdev[i],wNew.transpose())+bias)) < 0:
        #if yval*(np.dot(xdev[i],wNew.transpose())+bias)) <= 1:
            #count the number of misprediction
            count = count + 1;
 
    correct_pred = rowsDev - count;
    accuracy = (correct_pred/rowsDev)*100;
    #print("Accuracy:",accuracy);
    return accuracy;

def accuracyCalcTest(xdev,ydev,wNew,bias,rowsDev,mu):
    count = 0;#tp = 0;fp = 0;fn = 0;
    for i in range(0,rowsDev):
        yval = ydev[i];
        #print("yval:",yval);
        if ydev[i] == 0:
            yval = -1;
        else:
            yval = ydev[i];
        prediction = np.dot(xdev[i],wNew.transpose())+bias;
#        #print("prediction:",prediction);
#        if yval > 0 and prediction > 0:
#            tp = tp + 1;
#        if yval < 0 and prediction > 0:
#            fp = fp + 1;
#        if yval > 0 and prediction < 0:
#            fn = fn + 1;
            
        if (yval*(prediction)) < 0:
        #if yval*(np.dot(xdev[i],wNew.transpose())+bias)) <= 1:
            #count the number of misprediction
            count = count + 1;
     
    #correct predictions = totalrows - noOfFalsePredictions
   
    correct_pred = rowsDev - count;
    accuracy = (correct_pred/rowsDev)*100;
    #print("Accuracy:",accuracy);
    return accuracy;

def labelPredict(xeval,yeval,bestWeight,bestBias,rowsEval,mu):
    
    print("Predicting eval set label");
    fopen = open('../movie-ratings/results/svm-results2.txt', 'w');
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
        #print("xval1234.shape:",xval1234.shape,"yval1234.shape:",yval1234.shape);
        del split1,fileopen,f1;
        return xval1234,yval1234,row1;
        del xval1234,yval1234;
    elif cvCount == 1:
        split2 = "../movie-ratings/datasetPrepared/CVSplits/training0124.data";
        fileopen = open(split2,"r",encoding="utf8");
        f2,row2 = getNumFeaturesNRows(fileopen);
        xval1235,yval1235 = loadData(fileopen,f2,row2);
        #print("xval1235.shape:",xval1235.shape,"yval1235.shape:",yval1235.shape);
        del split2,fileopen,f2;
        return xval1235,yval1235,row2;
        del xval1235,yval1235;
    elif cvCount == 2:
        split3 = "../movie-ratings/datasetPrepared/CVSplits/training0134.data";
        fileopen = open(split3,"r",encoding="utf8");
        f3,row3 = getNumFeaturesNRows(fileopen);
        xval1245,yval1245 = loadData(fileopen,f3,row3);
        #print("xval1245.shape:",xval1245.shape,"yval1245.shape:",yval1245.shape);
        del split3,fileopen,f3;
        return xval1245,yval1245,row3;
        del xval1245,yval1245;
    elif cvCount == 3:
        split4 = "../movie-ratings/datasetPrepared/CVSplits/training0234.data";
        fileopen = open(split4,"r",encoding="utf8");
        f4,row4 = getNumFeaturesNRows(fileopen);
        xval1345,yval1345 = loadData(fileopen,f4,row4);
        #print("xval1345.shape:",xval1345.shape,"yval1345.shape:",yval1345.shape);
        del split4,fileopen,f4;
        return xval1345,yval1345,row4;
        del xval1345,yval1345;
    elif cvCount == 4:
        split5 = "../movie-ratings/datasetPrepared/CVSplits/training1234.data";
        fileopen = open(split5,"r",encoding="utf8");
        f5,row5 = getNumFeaturesNRows(fileopen);
        xval2345,yval2345 = loadData(fileopen,f5,row5);
        #print("xval2345.shape:",xval2345.shape,"yval2345.shape:",yval2345.shape);
        del split5,fileopen,f5;
        return xval2345,yval2345,row5;
        del xval2345,yval2345;
    else:
        print("Count Value:",cvCount," not recognized");
        
def getParamsTest(cvCount):
    if cvCount == 0:
        split5 = "../movie-ratings/datasetPrepared/CVSplits/training04.data";
        fileopen = open(split5,"r",encoding="utf8");
        f5,row5 = getNumFeaturesNRows(fileopen);
        xval5,yval5 = loadData(fileopen,f5,row5);
        #print("xval5.shape:",xval5.shape,"yval5.shape:",yval5.shape);
        del split5,fileopen,f5;
        return xval5,yval5,row5;
        del xval5,yval5,row5;
    elif cvCount == 1:
        split4 = "../movie-ratings/datasetPrepared/CVSplits/training03.data";
        fileopen = open(split4,"r",encoding="utf8");
        f4,row4 = getNumFeaturesNRows(fileopen);
        xval4,yval4 = loadData(fileopen,f4,row4);
        #print("xval4.shape:",xval4.shape,"yval4.shape:",yval4.shape);
        del split4,fileopen,f4;
        return xval4,yval4,row4;
        del xval4,yval4,row4;
    elif cvCount == 2:
        split3 = "../movie-ratings/datasetPrepared/CVSplits/training02.data";
        fileopen = open(split3,"r",encoding="utf8");
        f3,row3 = getNumFeaturesNRows(fileopen);
        xval3,yval3 = loadData(fileopen,f3,row3);
        #print("xval3.shape:",xval3.shape,"yval3.shape:",yval3.shape);
        del split3,fileopen,f3;
        return xval3,yval3,row3;
        del xval3,yval3,row3;
    elif cvCount == 3:
        split2 = "../movie-ratings/datasetPrepared/CVSplits/training01.data";
        fileopen = open(split2,"r",encoding="utf8");
        f2,row2 = getNumFeaturesNRows(fileopen);
        xval2,yval2 = loadData(fileopen,f2,row2);
        #print("xval2.shape:",xval2.shape,"yval2.shape:",yval2.shape);
        del split2,fileopen,f2;
        return xval2,yval2,row2;
        del xval2,yval2,row2;
    elif cvCount == 4:
        split1 = "../movie-ratings/datasetPrepared/CVSplits/training00.data";
        fileopen = open(split1,"r",encoding="utf8");
        f1,row1 = getNumFeaturesNRows(fileopen);
        xval1,yval1 = loadData(fileopen,f1,row1);
        #print("xval1.shape:",xval1.shape,"yval1.shape:",yval1.shape);
        del split1,fileopen,f1;
        return xval1,yval1,row1;
        del xval1,yval1,row1;
    else:
        print("Count Value:",cvCount," not recognized");
    
def getBestHyperparameter(w,b):
    
    wNew = w;biasNew = b;
    
    xval1234 = xval1235 = xval1245 = xval1345 = xval2345 = 0;
    #Build the train and test splits
    #xval1234 = np.array([xval1,xval2,xval3,xval4]);
    #print("type:",type(xval1234));
    #print("xval1234.shape:",xval1234.shape);
  
    cvTrainx = np.array([xval1234,xval1235,xval1245,xval1345,xval2345]);

    #cvTestx = np.array
    #print("shape of y:",cvTest[0].shape)
    #for every eta value.. compute the average accuracy
    #etas = np.array([1, 0.1, 0.01], 'float');
    #Uncomment below lines
    gamma = np.array([0.1,0.01,0.001,0.0001], 'float');
    #regC = np.array([10,1,0.1,0.01,0.001,0.0001], 'float');
    
    #gamma = np.array([0.001,0.0001], 'float');
    regC = np.array([10,100,10000,100000], 'float');
    #rows = np.array([row1,row2,row3,row4,row5]);
    #print(rows[0]);
    etaTrack = [];muTrack = [];
    #accF = wF = bF = 0;
    muAccuracy = np.empty((0, 3), 'float');print(muAccuracy.shape)
    for mu in regC:
        print("C:",mu);
        muCount = 0;avgAcc = [];maxEta = 0;maxAcc = 0.00;maxAccIndex=0;
        for eta in gamma:
            print("gamma:",eta);
            cvCount = 0;accuracy = 0;totAcc=0;etaCount = 0;F1=0.0000;totF1 = 0.0000;
            #wNew = w;biasNew = b;
            for trainSet in cvTrainx:
                wNew = w;biasNew = b;
                #print("cvCount",cvCount);
                cvtrx,cvtry,rowtr = getParamsTrain(cvCount);
                #print("w.shape:",wNew.shape);
                for epoch in range(0,1):
                    
                    #wNew,biasNew,updates = trainModel(trainSet,cvTrainy[cvCount],biasNew,wNew,eta,rows[cvCount],mu);
                    wNew,biasNew = trainModel(cvtrx,cvtry,biasNew,wNew,eta,rowtr,mu);
                del cvtrx,cvtry,rowtr;
                cvttx,cvtty,rowte = getParamsTest(cvCount);
                    #After ten epoch calculate the accuracy
                #accuracy = accuracyCalc(cvTestx[cvCount],cvTesty[cvCount],wNew,biasNew,rows[cvCount],mu);
                accuracy = accuracyCalc(cvttx,cvtty,wNew,biasNew,rowte,mu);
                #print("Accuracy:",accuracy);
#                if accuracy > accF :
#                    accF = accuracy;
#                    wF = wNew;
#                    bF= biasNew;
                del cvttx,cvtty,rowte;
                totF1 = totF1 + F1;
                totAcc = totAcc + accuracy;
                cvCount = cvCount + 1;
                #Compute Average Accuracy of 5Fold CV for each eta value
            #print("totAcc:",totAcc);
            avgAcc.append(totAcc/(5));
            #avgF1.append(totF1/(5));
            etaTrack.append(etaCount);
            etaCount = etaCount + 1;
                #choose the eta with highest avgAcc    
        print("avgAcc:",avgAcc);
        maxAccIndex = np.argmax(np.array(avgAcc));print(maxAccIndex);
        #maxF1Index = np.argmax(np.array(avgF1));#print(maxAccIndex);
        maxAcc = avgAcc[maxAccIndex];#print(maxAcc);
        ##maxF1 = avgF1[maxF1Index]; print("maxF1 Chosen:",maxF1);
        maxEta = gamma[maxAccIndex];#print(maxEta);
        muTrack.append(muCount);
        muCount = muCount + 1;
        #print(np.array([mu,maxEta,maxAcc]));
        #muBestEta.append(mu,maxEta,maxAcc);
        #print("muAcc:",muAccuracy);
        #Store the combination having mn,bestEta,BestAcc;
        #print("mu:",mu," mu.shape:",mu.shape);
        #print("maxEta:",maxEta," maxEta.shape:",maxEta.shape);
        #print("maxAcc:",maxAcc);
        muAccuracy = np.vstack((muAccuracy, np.array([mu,maxEta,maxAcc])));
        #muAccuracy = np.vstack((muAccuracy, np.array([mu,maxEta,maxF1])));
        
    
    #print(muAccuracy);print(muAccuracy.shape);
    accIndex = np.argmax(muAccuracy[:, 2]);print(accIndex);
    #F1Index = np.argmax(muAccuracy[:, 2]);print(F1Index);
    #Choose the best Eta and mu combination;
    bestMu = muAccuracy[accIndex][0];#print("bestMu:",bestMu);
    bestEta = muAccuracy[accIndex][1];#print("bestEta:",bestEta);
    bestAcc = muAccuracy[accIndex][2];#print("bestAcc:",bestAcc);
    #bestF1 = muAccuracy[accIndex][2];
    return bestMu,bestEta,bestAcc;

    
        
    
    
    
        
def main():
    print("SVM Model Training");
    print("==================");
    #Load the training set
    trainingFile = open('../movie-ratings/datasetPrepared/data.train',"r",encoding="utf8");
    #trainingFile = open('../data/train.liblinear',"r",encoding="utf8");
    
    features,rows = getNumFeaturesNRows(trainingFile);
    print(features,rows);    
    #Form an Array of Feature vector and labels
    x,y = loadData(trainingFile,features,rows);
    #print(type(x));
    print("here");
    print("xshape:",x.shape);
    print("yshape:",y.shape);
    print("Training Set Loaded");
    #Loading the dev set:
    devFile = open('../movie-ratings/datasetPrepared/data.dev',"r",encoding="utf8");
    featuresDev,rowsDev = getNumFeaturesNRows(devFile);
    xdev,ydev = loadData(devFile,featuresDev,rowsDev);
    print("xdev:",xdev.shape);print("ydev:",ydev.shape);
    print("Dev Set Loaded");
    #eta = 0.01;
    #Initialize weight vector w and bias b 
    bias = 0;
    #print(bias);
    #variables to store the parameters for each epoh to choose the best epoh
    biasArray = [];
    wArray = [];
    updateArray = [];epochArray = [];
    accuracyArray = [];#tpArray = [];fpArray = [];fnArray = [];
    w = np.zeros((1,features),"float");
    #Load the weight vector with Random Value between -0.01 and 0.01
    for i in range(0,features):
        #w[0][i] = randomNum();
        w[0][i] = 0;
    updates = 0;biasNew = bias;wNew = w;    
    #print("w before update:",w);
    #print(w.shape);
    
    #Choose eta within giving highest accuracy using Cross - Validation
    print("Hyper Parameter - Cross Validation");
    print("==================================");
    mu,eta,etaAcc = getBestHyperparameter(wNew,biasNew);
    #eta = gamma = the learning rate; C = mu = regularization parameter
    #mu = 50000.0;
    #eta = 0.001;
    #etaAcc = 70.53;
    print("BestHyperParameter Chosen:")
    print("==========================");
    print("C (Trade off):",mu);    
    print("gamma (LR):",eta);
    print("HyperParameter Accuracy:",etaAcc);
    print("===============================");
    
#    #Load the test set to find which epoch performs best on test set
#    testFile = open('../data/test.liblinear',"r",encoding="utf8");
#    featuresTest,rowsTest = getNumFeaturesNRows(testFile);
#    xtest,ytest = loadData(testFile,featuresTest,rowsTest);
    
    #Train the classifier for 20 epochs as given in the question
   
    for epoch in range(0,15):
        
        #Shuffle x and y for each epoch
        c = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)];
        np.random.shuffle(c);
        x = c[:, :x.size//len(x)].reshape(x.shape);
        y = c[:, x.size//len(x):].reshape(y.shape);
        print("Data Shuffle Over");
        #Shuffle over
        
        #Decaying the learning rate
        eta = eta/(1+epoch);
        
        wNew,biasNew = trainModel(x,y,biasNew,wNew,eta,rows,mu);
        print("Completed Training for epoch:",epoch);
        biasArray.append(biasNew);wArray.append(wNew);updateArray.append(updates);
        epochArray.append(epoch);
        #tp,fp,fn,accuracyTest = accuracyCalcTest(xtest,ytest,wNew,biasNew,rowsTest,mu);
        #accuracyTest = accuracyCalc(xtest,ytest,wNew,biasNew,rowsTest,mu);
        #accuracyArray.append(accuracyTest);
        #tpArray.append(tp);fpArray.append(fp);fnArray.append(fn);
        #print("wval in main:",type(wNew));
        #print("wval in main ac",wNew);
        #predict the model for accuracy using the parameters of current epoch
        print("Computing accuracy on dev set for epoch:",epoch);
        accuracy = accuracyCalc(xdev,ydev,wNew,bias,rowsDev,mu);
        accuracyArray.append(accuracy);
    
    #Select the parameters with highest accuracy
    #print(epochArray);
    print("accuracyArray:",accuracyArray);
    index = 0;highestAcc = 0;
    for i in epochArray:
        if accuracyArray[i] > highestAcc:
           highestAcc = accuracyArray[i];
           index = i;
            
   
    #Display p,r and F1 for all the epochs
#    for i in epochArray:
#        if fpArray[i] == 0 and tpArray[i] == 0 :
#            p = 0;
#        else:
#            p = (tpArray[i] / (tpArray[i] + fpArray[i]));
#    #p = (tpFinal / (tpFinal + fpFinal));
#        if tpArray[i] == 0 and fnArray[i] == 0:
#            r = 0;
#        else:
#            r = (tpArray[i] / (tpArray[i] + fnArray[i]));
#        F1 = 2 * ((p * r) / (p + r));
#        print("Epoch: %s; p: %s; r: %s; F1: %s; Accuracy: %s;" %(i,p,r,F1,accuracyArray[i]));
    print(index,highestAcc);
    beshEpoch = index;
    bestWeight = wArray[index];
    bestBias = biasArray[index];
    bestAccuracy = highestAcc;
#    tpFinal = tpArray[index];
#    fpFinal = fpArray[index];
#    fnFinal = fnArray[index];
    bestUpdates = updateArray[index];
    #Find the highest bias and weighht
    #bestWeight = wNew;
    #bestBias = biasNew;
#    if fpFinal == 0 and tpFinal == 0 :
#        p = 0;
#    else:
#        p = (tpFinal / (tpFinal + fpFinal));
#    #p = (tpFinal / (tpFinal + fpFinal));
#    if tpFinal == 0 and fnFinal == 0:
#        r = 0;
#    else:
#        r = (tpFinal / (tpFinal + fnFinal));
#    F1 = 2 * ((p * r) / (p + r));
#    
    print("Final SVM Model Parameters after Training");
    print("=========================================");
    
    print("beshEpoch:",beshEpoch);
    print("bestWeight:",bestWeight);
    print("bestBias:",bestBias);
    print("bestAccuracy:",bestAccuracy);
    print("bestUpdates:",bestUpdates);
    
    del xdev,ydev,devFile,featuresDev,rowsDev;
    del x,y,trainingFile,features,rows;
    
    #Calculate Test Set Accuracy:
    testFile = open('../movie-ratings/data-splits/data.test',"r",encoding="utf8");
    featuresTest,rowsTest = getNumFeaturesNRows(testFile);
    xtest,ytest = loadData(testFile,featuresTest,rowsTest);
    print("Loaded Test Set. Predicting Accuracy on Test Set");
    accuracyTest = accuracyCalc(xtest,ytest,bestWeight,bestBias,rowsTest,mu);
    del xtest,ytest,featuresTest,rowsTest,testFile; 
    print("Accuracy on Test Set:",accuracyTest);
    
    # =============================================================================
    #Predicting the eval set
    print("******************************************");
    print("Predicting accuracy of Evaluation set");
    evalFile = open('../movie-ratings/data-splits/data.eval.anon',"r",encoding="utf8");
    featuresEval,rowsEval = getNumFeaturesNRows(evalFile);
    xeval,yeval = loadData(evalFile,featuresEval,rowsEval);
    labelPredict(xeval,yeval,bestWeight,bestBias,rowsEval,mu);

if __name__ == '__main__':
    main();
