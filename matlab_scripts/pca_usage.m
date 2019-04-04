clc;clear;

feature = rand(30,100);

[coeff,score,latent,tsquare] = pca(feature,'Economy',false,'Centered',false);
% feature = bsxfun(@minus,feature,mean(feature,1));
feature_after_PCA = feature*coeff;
