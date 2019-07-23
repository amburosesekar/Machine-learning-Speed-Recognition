



clc;
clear all;
close all;




ij1=1;
  ac=1;
  ab=1;
  akk=1;
abc=1;
  
  
folder_name = uigetdir;
dk=[];
dk = dir(strcat(folder_name,'/*.jpg'));
numfiles = length(dk);

figure,
for ika=1:numfiles 
 
y=imread(strcat(folder_name,'/',dk(ika,1).name));
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


I = uint8(y) ;  
[r,f] = vl_mser(I,'MinDiversity',0.2,...
                'MaxVariation',0.1,...
                'Delta',10) ;

f = vl_ertr(f) ;
vl_plotframe(f) ;
hold on
title('White MSER')




%% find color MSER
R=double(src_img(:,:,1));
G=double(src_img(:,:,2));
B=double(src_img(:,:,3));

ohmRB=max(R./(R+G+B),B./(R+G+B));

figure,
imshow(uint8(ohmRB),[])
title('Normalized RB')


I = uint8(mat2gray(ohmRB)) ;  
[r,f] = vl_mser(I,'MinDiversity',0.2,...
                'MaxVariation',0.1,...
                'Delta',10) ;
            
         
% f = vl_ertr(f) ;
vl_plotframe(f) ;
hold on
title('RB MSER')

%  end

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


%% Verify Cicrcle


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
% Rl=8;
% Ru=10;
% R=Rl+(Ru-Rl).*rand(1);  % This is inner R

R=Router;

Xc=centroids(ind,1);
%Xu=centroids(ind,1)+6;
%Xc=Xl+(Xu-Xl).*rand(1);

Yc=centroids(ind,2);
%Yu=centroids(ind,2)-6;

%Yc=Yl+(Yu-Yl).*rand(1);

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
    
    
    
    figure,
    imshow(zk)
    fprintf('Triangle Sign Detected\n');
    title('Triangle Sign Detected')
    
    
    
    
    
end



if(idk==1)
    
%% Extract the Features

cellSize = 8 ;
%hog = vl_hog(im2single(yk1), cellSize, 'verbose') ;
hog=HOG(im2single(yk1));


H{abc}=hog;

T1=input('Enter Train ID-->')

T{abc}=T1;


abc=abc+1;


    
    
    
    
    
    
    
    
    
    
    
    
    
end


end



save('Train2.mat','H','T');




