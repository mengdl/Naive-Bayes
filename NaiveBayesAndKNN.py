import numpy
import math
from operator import itemgetter
 
# load data files
trainData = numpy.loadtxt("train.txt",delimiter=",",dtype=float)
trainData = numpy.array(trainData)
testData = numpy.loadtxt("test.txt",delimiter=",",dtype=float)
testData = numpy.array(testData)

# get the features from data set
def preData(data):
    data = numpy.array(data)
    data = data[:,0:-1]
    return data


# separate the features to each class,  
# so that it is convinent to calculate the standerd variance, mean,
# and posterior probability
def separateClass(data):
    separated = {}
    for i in range(len(data)):
        if data[i][-1] not in separated:
            separated[data[i][-1]] = []
        separated[data[i][-1]].append(data[i])
    return separated


# calculate the mean by class
def meanM(sep):
    mean = {}
    for classV, attribute in sep.iteritems():
        data = numpy.vstack((sep[classV]))
        n = len(data)
        sumData = data.sum(axis = 0)
        mean[classV] = sumData/float(n)
    return mean

# calculate the standerd variance by class
def standVarM(sep):
    standV = {}
    for classV, attribute in sep.iteritems():
        x = meanM(sep)[classV]
        data = numpy.vstack((sep[classV]))
        x.shape = (1,len(data[0,:]))
        avg = numpy.transpose(x)
        n = len(data[:,0])
        data.shape = (len(data[:,0]),len(data[0,:]))
        y = numpy.transpose(data)
        var = ((y - avg)*(y - avg)).sum(axis = 1)/float(n)
        standV[classV] = numpy.sqrt(var)
    return standV

# calculate the prior probability p(y)
def priorProb(data,sep):
    prioriProb = {}
    for classV,attribute in sep.iteritems():
        prioriProb[classV] = float(len(attribute))/float(len(data))
    return prioriProb

# Assume these features is normal distribution
def gaussian(x,mean,standv):
    if mean == 0 and standv == 0:
        return 0
    e = math.exp(-(math.pow(x-mean,2))/(2*math.pow(standv,2)))
    return (1/(math.sqrt(2*math.pi)*standv))*e


# calculate the posterior probability p(y|x)
def postProb(x,mean,standv,sep,prior):
    prob = {}
    for classV in sep.keys():
        prob[classV] = 1
        for i in range(len(x)):
            prob[classV] = gaussian(x[i],mean[classV][i],standv[classV][i])*prob[classV]
        prob[classV] = prob[classV]*prior[classV]
    return prob

# for naive bayes, find the most likely class for 
# each test data by compare the probability
def predict(prob):
    maxProb = -1
    classPre = 0
    for classV in prob.keys():
        if prob[classV]> maxProb:
            classPre = classV
            maxProb = prob[classV]
    return classPre

# for evaluation
def getAccuracy(data,predictions):
    correct = 0
    n = len(data)
    for i in range(n):
        if data[i][-1] == predictions[i]:
            correct = correct+1
    return (correct/float(n))*100.0


def bayes(trainData,testData):
    ndata = trainData[:,1:]
    sep = separateClass(ndata)
    data = preData(ndata)
    test = preData(testData[:,1:])
    mean = meanM(sep)
    standv = standVarM(sep)
    prior = priorProb(data,sep)
    predictions=[]
    for i in range(len(test)):
        prob = postProb(test[i],mean,standv,sep,prior)
        result = predict(prob)
        predictions.append(result)
        pred = predictions
    TestAccuracy = getAccuracy(testData,pred)
    predictions =[]
    for i in range(len(data)):
        prob = postProb(data[i],mean,standv,sep,prior)
        result = predict(prob)
        predictions.append(result)
        pred = predictions
    TrainAccuracy = getAccuracy(trainData,pred)
    print "Naive Bayes: "
    print "the result of testAccuracy is", TestAccuracy
    print "the result of trainAccuracy is", TrainAccuracy


#.........................................................
# the KNN part


# for l1_distance
def L1_distance(data1,x):
    sum = 0
    for i in range(1,len(data1)-1):
        if data1[0] != x[0]:    #leave one out
            sum = abs(data1[i] - x[i])+sum
    return sum

# for l2_distance
def L2_distance(data1,x):
    sum = 0
    for i in range(1,len(data1)-1):
        if data1[0] != x[0]:
            sum = math.pow(data1[i] - x[i],2)+sum
    return math.sqrt(sum)

# sort each comparison within a data record,and find the min distance
def findMax(data):
    separated = {}
    preR =[]
    for i in range(len(data)):
        if data[i][-1] not in separated:
            separated[data[i][-1]] = []
        separated[data[i][-1]].append(data[i])
    for classV, attrbute in separated.iteritems():
        attrbute = numpy.c_[attrbute]
        for i in range(len(attrbute)):
            preR.append([classV,-attrbute[i,0],len(attrbute)])   #(number,distance)
    preR = sorted(preR, key = itemgetter(2,1),reverse=True)
    maxR = preR[0][0]
    return maxR

# make prediction
def predictL1(K,data,x):
    result = []
    for i in range(len(data)):
        result.append([L1_distance(data[i],x),data[i][-1]])
    result = sorted(result,key = lambda result:result[0])
    return numpy.vstack((result[:K]))  #result([distance, class])

# make prediction
def predictL2(K,data,x):
    result = []
    for i in range(len(data)):
        result.append([L2_distance(data[i],x),data[i][-1]])
    result = sorted(result,key = lambda result:result[0])
    return numpy.vstack((result[:K]))

# for normalization
def meanMKNN(data):
    n = len(data)
    sumData = data.sum(axis = 0)
    return sumData/float(n)

# for normalization 
def standVarMKNN(data):
    x = meanMKNN(data)
    x.shape = (1,len(data[0,:]))
    avg = numpy.transpose(x)
    n = len(data[:,0])
    data.shape = (len(data[:,0]),len(data[0,:]))
    y = numpy.transpose(data)
    var = ((y - avg)*(y - avg)).sum(axis = 1)/float(n-1)
    return numpy.sqrt(var)

def normalization(data,x,y):
    x.shape = (1,len(data[0,:]))
    mean = numpy.transpose(x)
    y.shape = (1,len(data[0,:]))
    st = numpy.transpose(y)
    cdata = numpy.transpose(data)
    return numpy.transpose((cdata - mean)/st)


def knn(trainData,testData,K,testD):
    if testD == True:
        ntraindata = normalization(trainData[:,1:-1],meanMKNN(trainData[:,1:-1]),standVarMKNN(trainData[:,1:-1]))
        ntcdata = numpy.insert(ntraindata, 0 ,values =trainData[:,0],axis =1)
        data = numpy.c_[ntcdata,trainData[:,-1]]
        ntestdata = normalization(testData[:,1:-1],meanMKNN(trainData[:,1:-1]),standVarMKNN(trainData[:,1:-1]))
        nctestdata = numpy.insert(ntestdata, 0 ,values =testData[:,0],axis =1)
        test = numpy.c_[nctestdata, testData[:,-1]]
        predictions1=[]
        predictions2=[]
        for i in range(len(test)):   #predict test data
            pre1 = predictL1(K,data,test[i])
            pre1 = findMax(pre1)
            predictions1.append(pre1)
            pre2 = predictL2(K,data,test[i])
            pre2 = findMax(pre2)
            predictions2.append(pre2)
        TestAccuracyL1 = getAccuracy(testData,predictions1)
        TestAccuracyL2 = getAccuracy(testData,predictions2)
        print 'K = %d '%(K)
        print 'Test Accuracy Using L1 Distance: %f' %(TestAccuracyL1)
        print 'Test Accuracy Using L2 Distance: %f' %(TestAccuracyL2)
    if testD == False:
        predictions1=[]
        predictions2=[]
        for i in range(len(trainData)):
            if trainData[:,0][i] == testData[:,0][i]:
                trainDataUp = trainData[0:i,:]
                trainDataDown = trainData[i+1:,:]
                train = numpy.vstack((trainDataUp,trainDataDown))
                ntraindata = normalization(train[:,1:-1],meanMKNN(trainData[:,1:-1]),standVarMKNN(train[:,1:-1]))
                ntcdata = numpy.insert(ntraindata, 0 ,values =train[:,0],axis =1)
                data = numpy.c_[ntcdata,train[:,-1]]
                ntestdata = normalization(testData[:,1:-1],meanMKNN(train[:,1:-1]),standVarMKNN(train[:,1:-1]))
                nctestdata = numpy.insert(ntestdata, 0 ,values =testData[:,0],axis =1)
                test = numpy.c_[nctestdata, testData[:,-1]]
            pre1 = predictL1(K,data,test[i])
            pre1 = findMax(pre1)
            predictions1.append(pre1)
            pre2 = predictL2(K,data,test[i])
            pre2 = findMax(pre2)
            predictions2.append(pre2)
        TrainAccuracyL1 = getAccuracy(trainData,predictions1)
        TrainAccuracyL2 = getAccuracy(trainData,predictions2)
        print 'Training Accuracy Using L1 Distance: %f' %(TrainAccuracyL1)
        print 'Training Accuracy Using L2 Distance: %f' %(TrainAccuracyL2)

def main():
    bayes(trainData,testData)
    for K in range(1,8,2):
        knn(trainData,testData,K,True)
        knn(trainData,trainData,K,False)

main()
