clear

% load data
% load('.\data\ALLAML.mat','classes012','D');
% classes=classes012;
% clear('classes012');

load('C:\YifengLi\Reseach Program\dataset\Colon\ColonCancer.mat');
dataStr='Colon';

% load('C:\YifengLi\Reseach Program\dataset\Leukemia\Leukemia.mat','trainSet','testSet','trainSampleClasses','testSampleClasses');
% dataStr='Leukemia2';
% D=[trainSet,testSet];
% classes=[trainSampleClasses;testSampleClasses];
% clear('trainSet','testSet','trainSampleClasses','testSampleClasses');


% load('C:\YifengLi\Reseach Program\lisa\Hu2006UniqueUnigene.mat','DOrdered','shortenClassesOrdered','classesOrdered');
% dataStr='Hu2006';
% % remove Claudin
% rmInd=strcmp(shortenClassesOrdered,'Claudin');
% shortenClassesOrdered=shortenClassesOrdered(~rmInd);
% classesOrdered=classesOrdered(~rmInd);
% DOrdered=DOrdered(:,~rmInd);
% D=DOrdered;
% % D(isnan(D))=0;
% D=knnimpute(D);
% classes=changeClassLabels01(classesOrdered);

% load fisheriris
% dataStr='fisheriris';
% D=meas';
% classes=changeClassLabels01(species);
% clear('meas','species');


% load(':\YifengLi\Reseach Program\Classification\kfda\data\ionosphere.mat','ionosphere','classes');
% dataStr='ionosphere';
% D=ionosphere;
% classes=changeClassLabels01(classes);

% normalize D
D=normc(D);
% CV
kfold=4;
ind=crossvalind('Kfold',classes,kfold);
indTest=(ind==1);
trainSet=D(:,~indTest);
testSet=D(:,indTest);
trainClass=classes(~indTest);
testClass=classes(indTest);

% option
option.lambda=0.2;
option.SRMethod='l1nnls';
option.kernel='rbf';
option.param=2^0;
k=10;

% learn dictionary
trainTrain=computeKernelMatrix(trainSet,trainSet,option);
 [AtA,Y,numIter,tElapsed,finalResidual]=KSRDL(trainTrain,k,option);
 
% sparse coding
trainTest=computeKernelMatrix(trainSet,testSet,option);
testTest=computeKernelMatrix(testSet,testSet,option);
testTest=diag(testTest);
AtTest=Y'\trainTest;
[X,tElapsed]=KSRSC(AtA,AtTest,testTest,option);
method='knn';
[testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,method,option);
classPerformKNN=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])))

method='ksrcl1ls';
[testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,method,option);
classPerformKl1LS=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])))

method='ksrcnnls';
[testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,method,option);
classPerformKNNLS=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])))

method='ksrcl1nnls';
[testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,method,option);
classPerformKl1NNLS=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])))

method='hdlm';
option.kernel='rbf';
option.param=2^0;
[testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,method,option);
classPerformKSVM=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])))

