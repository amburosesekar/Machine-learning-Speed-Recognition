% examples of using universe sparse representation (universe sparse matrix factorization)
clear

% load data
% suppose the current folder is the one containing the NMF toolbox
load('.\data\ALLAML.mat','classes012','D');
classes=classes012;
clear('classes012');
k=3;

% Directly call a NMF algorithm to obtain the factor matrices.
[A,Y]=nmfnnls(D,k);
% Then clustering. 
indCluster10=NMFCluster(Y);

% both basis vectors and coefficient vectors are sparse and non-negative 
Dn=normc(D);
optionsnmf2.alpha2=1;
optionsnmf2.alpha1=0.01;
optionsnmf2.lambda2=0;
optionsnmf2.lambda1=0.01;
% [A,Y]=sparsenmf2rule(Dn,k,optionsnmf2);
[A,Y]=sparsenmf2nnqp(Dn,k,optionsnmf2);
% Then clustering. 
indCluster11=NMFCluster(Y);

%%
clear
% USR/USMF
load('C:\YifengLi\Reseach Program\lisa\Hu2006UniqueUnigene.mat','DOrdered','shortenClassesOrdered','classesOrdered');
dataStr='Hu2006';
D=DOrdered;
classes=changeClassLabels01(classesOrdered);
Dn=knnimpute(D);
% Dn=Dn(1:2000,:);
Dn=normc(Dn);
optionusr.alpha2=0;
optionusr.alpha1=0.01;
optionusr.lambda2=0;
optionusr.lambda1=0.01;
optionusr.t1=false;
optionusr.t2=true;
k=6;
% [A,Y]=sparsenmf2rule(Dn,k,optionsnmf2);
[A,Y]=usr(Dn,k,optionusr);
