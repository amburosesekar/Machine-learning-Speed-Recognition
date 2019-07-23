clc
clear

% load data
% suppose the current folder is the one containing the NMF toolbox
% load('C:\YifengLi\Reseach Program\dataset\Leukemia\ALLAML.mat');
% classes=classes012;
% clear('classes012');

% load('C:\YifengLi\Reseach Program\dataset\SRBCT\smallRoundBlueCellTumorOFChildhood.mat');
%         dataStr='SRBCT';
%         D=Data;
%         clear('Data');
        
% load('C:\YifengLi\Reseach Program\dataset\MLL\MLL.mat','data','classes');
% dataStr='MLL';
% D=data;
        
% load('C:\YifengLi\Reseach Program\dataset\Colon\ColonCancer.mat');
% dataStr='Colon';

% load('C:\YifengLi\Reseach Program\dataset\MSData\OvarianCancerLow4_3_02.mat','Y','grp');
% dataStr='OvarianCancerLow4_3_02';
% D=Y;
% classes=changeClassLabels01(grp);
% clear('Y','grp');

% load fisheriris
% dataStr='fisheriris';
% D=meas';
% classes=changeClassLabels01(species);
% clear('meas','species');

% 
% load('C:\YifengLi\Reseach Program\dataset\UC Irvine Machine Learning Repository\wave2\wave2.mat','D','classes');
% dataStr='wave2';
   
% load(':\YifengLi\Reseach Program\Classification\kfda\data\pima.mat','pima','classes');
% dataStr='pima';
% D=pima;
% classes=changeClassLabels01(classes);

load(':\YifengLi\Reseach Program\Classification\kfda\data\ionosphere.mat','ionosphere','classes');
dataStr='ionosphere';
D=ionosphere;
classes=changeClassLabels01(classes);

D=normc(D);
numCl=numel(unique(classes));
kfold=3;
classPerformkfold=zeros(kfold,numCl+2);
classPerformSMOkfold=zeros(kfold,numCl+2);
classPerformQPkfold=zeros(kfold,numCl+2);
for i=1:kfold
ind=crossvalind('Kfold',classes,kfold);
indTest=(ind==i);
trainSet=D(:,~indTest);
testSet=D(:,indTest);
trainClass=classes(~indTest);
testClass=classes(indTest);

% classification
disp('LRC classifier...');
numFeat=size(trainSet,1);
paramDefault=log2(sqrt(numFeat))-0;
option.kernel='rbf';
option.param=2^paramDefault;
[testClassPredicted,otherOutput]=lrc(trainSet,trainClass,testSet,testClass,option);
classPerform=perform(testClassPredicted,testClass,numCl);
classPerformkfold(i,:)=classPerform;

end
mean(classPerformkfold,1)



