clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Medical Image Segmentation and Tumor Extraction
%  Algorithm       : Fuzzy C-Means (FCM) Clustering – from scratch
%  
%  Description     : Application of Fuzzy C-Means clustering on grayscale 
%                    medical images for intensity-based soft segmentation.
%                    After clustering, morphological operations are used 
%                    to extract the tumor region and highlight its border.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/pattern-season08-clustering/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background and video explanations, 
%  please visit OnlineBME.
%% ========================================================================
% I= imread('benign.jpg');
I= imread('malignant.jpg');
I= im2double(I);
if size(I,3)==3
    I= rgb2gray(I);
end
imshow(I)
%% clusstering
%% step 1: extract feature from each pixcel, convert image to vector
vectI= I(:)';
[m,n]= size(I);
k=4;
Nepoch=35;
[centers,U] = myfcm(vectI,k,Nepoch);
%%
[mx,labels]= max(U);
J= zeros(size(labels));
for i=1:k
    indx= find(labels==i);
    J(indx)= centers(i);
end
%%
J= reshape(J,m,n);
figure
imshow(J)

%% tumor extraction
mx= max(J(:));
BW= (J==mx);
figure
imshow(BW)
strl= strel('sphere',2);
BW= imerode(BW,strl);
BW= imdilate(BW,strl);
BW= bwareaopen(BW,100);
figure
imshowpair(I,BW,'montage')
%% boreder ploting
BW2= imerode(BW,strl);
border= BW-BW2;
[x,y]= find(border==1);
figure
imshow(I)
hold on
plot(y,x,'.r')



