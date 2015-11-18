function [zPowMat] = zScorePow(PowMat,BasePowMean,BasePowSTD)
% function [zPowMat] = zScorePow(pow,BasePowMean,BasePowSTD)
%
% function to zscore a 3D power matrix with the given baseline mean and
% std
%
% last updated:
%       10/03/14    YE - created function


zPowMat = (PowMat - repmat(BasePowMean,[1 size(PowMat,2), size(PowMat,3)]))...
     ./repmat(BasePowSTD,[1 size(PowMat,2), size(PowMat,3)]);