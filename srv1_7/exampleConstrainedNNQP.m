clear
load('C:\YifengLi\Reseach Program\dataset\Colon\ColonCancer.mat');
dataStr='Colon';
classes(classes==0)=-1;
X=D;
L=classes;
k=8;
[AtA,Y,numIter,tElapsed,finalResidual]=superkernelseminmfnnls(X,L,k);

