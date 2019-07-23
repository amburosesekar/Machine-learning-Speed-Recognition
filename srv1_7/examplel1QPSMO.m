% example of testing l1QPSMO

clear

%         load('C:\YifengLi\Reseach Program\lisa\Hu2006UniqueUnigene.mat','DOrdered','shortenClassesOrdered','classesOrdered');
%         dataStr='Hu2006';
%         % remove Claudin
%         rmInd=strcmp(shortenClassesOrdered,'Claudin');
%         shortenClassesOrdered=shortenClassesOrdered(~rmInd);
%         classesOrdered=classesOrdered(~rmInd);
%         DOrdered=DOrdered(:,~rmInd);
%         D=DOrdered;
%         classes=changeClassLabels01(classesOrdered);
%         D=knnimpute(D);

% load('C:\YifengLi\Reseach Program\dataset\SRBCT\smallRoundBlueCellTumorOFChildhood.mat');
% dataStr='SRBCT';
% D=Data;
% clear('Data');

% load('C:\YifengLi\Reseach Program\dataset\Leukemia\Leukemia.mat','trainSet','testSet','trainSampleClasses','testSampleClasses');
% dataStr='Leukemia2';
% D=[trainSet,testSet];
% classes=[trainSampleClasses;testSampleClasses];
% clear('trainSet','testSet','trainSampleClasses','testSampleClasses');

% load('C:\YifengLi\Reseach Program\dataset\meta-analysis microarray data\Dawany2011\Dawany2011.mat','D','classes');

% load(':\YifengLi\Reseach Program\Classification\kfda\data\ionosphere.mat','ionosphere','classes');
% dataStr='ionosphere';
% D=ionosphere;
% classes=changeClassLabels01(classes);

% load(':\YifengLi\Reseach Program\Classification\kfda\data\pima.mat','pima','classes');
%         dataStr='pima';
%         D=pima;
%         classes=changeClassLabels01(classes);
        
% load('C:\YifengLi\Reseach Program\dataset\UC Irvine Machine Learning Repository\wave2\wave2.mat','D','classes');

% load('C:\YifengLi\Reseach Program\dataset\USPS\USPS.mat');
% dataStr='USPS';
% D=[trainSet,testSet];
% classes=[trainClass;testClass];
% classes=changeClassLabels01(classes);



% load('C:\YifengLi\Reseach Program\dataset\MSData\Prostate_Cancer_Y.mat','Y','grp');
%         dataStr='Prostate_Cancer_Y';
%         D=Y;
%         classes=changeClassLabels01(grp);
%         clear('Y','grp');

load('C:\YifengLi\Reseach Program\dataset\meta-analysis microarray data\Dawany2011\Dawany2011.mat','D','classes');
classes=changeClassLabels01(classes');

% normalize D
D=normc(D);
% CV
kfold=10;
ind=crossvalind('Kfold',classes,kfold);
indTest=(ind==1);
trainSet=D(:,~indTest);
testSet=D(:,indTest);
trainClass=classes(~indTest);
testClass=classes(indTest);

% trainSet=[trainSet,-trainSet];
testSet=testSet(:,end);

% compute kernel
option.kernel='linear';
option.param=2^0;
H=computeKernelMatrix(trainSet,trainSet,option);
G=computeKernelMatrix(trainSet,testSet,option);
B=computeKernelMatrix(testSet,testSet,option);
lambda=0;
ts1=tic;
%[X1, Pset] = l1NNQPActiveSet(H, -G,lambda);
t1=toc(ts1)

ts2=tic;
[X2,S2] = NNQPSMOMulti(H, lambda-G);
t2=toc(ts2)

ts3=tic;
%[X3, U] = l1QP(H, -G,lambda);
t3=toc(ts3)

ts4=tic;
X4=l1QPSMOMulti(H,-G,lambda);
t4=toc(ts4)

ts5=tic;
lambda=0.8;
option.lambda=lambda;
option.L=size(trainSet,2);
option.residual=1e-7;
option.tof=1e-7;
% X5=l1LSProximal(H,G,diag(B),option);
t5=toc(ts5)







