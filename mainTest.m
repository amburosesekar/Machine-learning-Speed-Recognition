%% Developed By Amburose Sekar.S
%% Works in WinterGreen Technologies,
%% Marthandam,KanyaKumari Dist,India. 
%% Thanks For Third Party Codes.

clc;
clear all;
close all;

restoredefaultpath;
addpath(genpath(pwd));


[File,Path] = uigetfile('*','Select the Image file');
y=imread(strcat(Path,File));
y=imresize(y,[250 300]);
figure,
imshow(y)
title('Input image')
y1=y;


s1=size(y);
src_img=y;
if(numel(s1) > 2)
    y=rgb2gray(y);
end    
figure,
imshow(y)
title('Grayscale image')




%% find color MSER
R=double(src_img(:,:,1));
G=double(src_img(:,:,2));
B=double(src_img(:,:,3));

ohmRB=max(R./(R+G+B),B./(R+G+B));

figure,
imshow(uint8(ohmRB),[])
title('Normalized RB')

I = uint8(mat2gray(ohmRB)) ;  

%% Connected Component Analysis
f=im2bw(ohmRB);
f=bwareaopen(f,50);

figure,
imshow(f)
title('Morphology Filter')   
connComp = bwconncomp(f); % Find connected components
stats = regionprops(connComp,'Area','Eccentricity','Solidity');
disp(stats)
%% High Area Detected
clear s
s=regionprops(f,{'Area';'EquivDiameter';'BoundingBox';'Eccentricity'})
[v ind]=max([s.Area]);
D=s(ind).EquivDiameter;
% 
A=pi.*D.^2.0/4.0;
% 
Diff=abs(A-s(ind).Area)
zk=imcrop(y1,s(ind).BoundingBox);
figure,
imshow(zk)

s(ind).Eccentricity

zk1=imcrop(f,s(ind).BoundingBox);
yk=imfill((zk1),'holes');
figure,imshow(yk)
title('Filling Holes')


%% Verify Circle
clear s
Ibw1=yk;
s1  = regionprops(Ibw1,'MajorAxisLength','MinorAxisLength','Area','centroid');

ind=find([s1.Area]==max([s1.Area]));
centroids = cat(1, s1.Centroid);
Router=s1.MajorAxisLength./2.0;
Rinner=s1.MinorAxisLength./2.0;

[B,L] = bwboundaries(Ibw1,'noholes');

imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
end



%% MIC Model
R=Router;

Xc=centroids(ind,1);
Yc=centroids(ind,2);
[m1,n1]=size(Ibw1);


Xi=boundary(:,2);%-round(m1/2);%Xl+(Xu-Xl).*rand(1,10);
Yi=boundary(:,1);%-round(n1/2);%Yl+(Yu-Yl).*rand(1,10);

eMIC= max(sqrt((Xi-Xc).^2 + (Yi-Yc).^2)-R);

fprintf('eMIC Value is --> %3.2f\n',eMIC);

if(eMIC<6.0)
   
    fprintf('Circle Detected\n');
    
    
    yk1=(zk(:,:,1)+zk(:,:,2)+zk(:,:,3))./3;
    
    yk1(yk1<=20)=0;
    yk1(yk1>20)=255;
    
    
    yk1=im2bw(rgb2gray(zk));
   
    
    yk1=imresize(yk1,[128 128]);
    
    figure,
    imshow(yk1)
    fprintf('Circle Detected\n');
    title('Circle Sign Detected')
    idk=1;
    
else
    
    idk=0;
    
    figure,
    imshow(zk)
    fprintf('Triangle Sign Detected\n');
    title('Triangle Sign Detected')
      
    
end


ik1=0;
if(idk==1)
    
%% Extract the Features

cellSize = 8 ;
%hog = vl_hog(im2single(yk1), cellSize, 'verbose') ;
hog=HOG(im2single(yk1));
load Train2.mat

for ik=1:numel(H)
P(:,ik)=double(H{ik}(:));
end

PP=double(hog(:));
T1=cell2mat(T);

trainSet=double(P).*0.5;

%%%%%%imp

trainClass=T1;
 testSet=double(PP).*0.5;
 testClass=1;
%%%%


[model,OtherOutput1]=classificationTrain(trainSet,trainClass,'lsvm');    
% end
[result21,OtherOutput1]=classificationPredict(model,trainSet,trainClass);
[result2,OtherOutput]=classificationPredict(model,testSet,1);


if(result2==1)
    msgbox('Vechile Speed Set----> 20 Km/h');
% elseif(result2==2)
%     msgbox('Vechile Speed Set----> 30 Km/h');
elseif(result2==2)
    msgbox('Vechile Speed Set----> 40 Km/h');
elseif(result2==3)
    msgbox('Vechile Speed Set----> 60 Km/h');
else
    msgbox('Vechile Speed Normal');
    
end


    
    
    
    
    
    
    
    
    
    
    
    ik1=1;
    
end


