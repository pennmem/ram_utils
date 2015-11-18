function [binPowMat] = BinPow(powMat,timeWin,timeStep,samplingFreq,freQ,freqBins)
% function [binPowMat] = BinPow(powMat,timeWin,timeStep,samplingFreq,freQ,freqBins)
%
% This function calculates average power within frequency and time bins.
%
% Inputs
% powMat                matrix of power values
% timeWin               length of sliding window, in milliseconds
% timeStep              ms by which to advance the sliding window
% samplingFreq          samplingFreq of the powMat (in Hz)
% freQ                  frequencies to use for wavelets
% freqBands             vector of frequencies at which to bin, or a N x 2
%                       matrix specifying the min and max of N frequency
%                       bands
%                       
% Outputs
%   binPowMat            power binned in frequency and time
%

% Last modified:
%   10/23/14             YE created function (from AGR pow2timefreqbin)


% create time bins
NanTimeBins = false;
if ~(isnan(timeStep) || isnan(timeWin))
    timeWin = timeWin*samplingFreq/1000;
    timeStep = timeStep*samplingFreq/1000;
    timeBins = (timeWin:timeStep:size(powMat,2));
else
    timeBins = [];
    NanTimeBins = true;
end

%Initialize power mat 
binPowMat = nan([length(freqBins) length(timeBins) size(powMat,3)]);

if isempty(timeBins)
    if ~NanTimeBins
        warning('BinPow.m: unable to bin powMat at given timebin params. returning unbinned powMat.');
    end
    binPowMat = powMat;
else
    % loop through freqBins
    for f = 1:length(freqBins)
        
        if isvector(freqBins)
            fInd = find((freQ >= min(freqBins(f))) & (freQ <= max(freqBins(f))));
        else
            fInd = find((freQ >= min(freqBins(f,:))) & (freQ <= max(freqBins(f,:))));
        end
        
        %loop through timeBins
        for t = 1:length(timeBins)
            
            %identify time windows
            tInd = (timeBins(t)-timeWin+1):timeBins(t);
            
            %calculate mean power
            binPowMat(f,t,:) = nanmean(nanmean(powMat(fInd,tInd,:),1),2);
        end
        
    end
end
