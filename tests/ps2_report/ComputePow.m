function [PowMat,PhaseMat] = ComputePow(EEG,params)
% function [PowMat,PhaseMat] = ComputePow(EEG,params)
%
% function to extract raw eeg data and compute power. this is the standard
% function that uses the multitaper method (multiphasevec)
%
% inputs:
%       EEG         - a nEvents x nSamples matrix of voltages (e.g. as is
%                   output by ComputeEEG.m)
%
%       params      - struct with several fields determining how to extract and
%                   compute power on the data
%                       params.eeg.sampFreq     sampling frequency of the input EEG signal
%                       params.eeg.bufferMS     length of buffer (in milliseconds) surrounding the EEG signal input (only necessary for 'wavelet' method)
%
%                       params.pow.type         string indicating method for computing power
%                                               'wavelet'   = Morlet wavelet
%                                               'fft_slep'  = FFT, using Slepian windows
%
%                       params.pow.freqs        frequencies at which to compute power (e.g. [2 4 8...] or logspace(log10(2), log10(200),50)
%                       params.pow.logTrans     1/0 flag indicating whether to log transform power vals
%
%                       params.pow.wavenum      number of wavelets (typical = 7)
%
%                       params.pow.bandwidth    bandwidth of slepian windows, if computing power using 'fft_slep' (typical=2)
%                       params.pow.winSize      for 'multitaper' and
%                       'FFT_slep', this sets the width of the windows (in seconds) that are used for computing the FFT
%
%                       params.pow.doPhase      1/0 whether to get phase
%                       info
%
%
%
% outputs:
%       PowMat      - if 'wavelet', a freq X time X event matrix of power
%                   - if 'fft_slep', a 1 x nfreqs vector for the chosen 
% 
% last updated:
%       07/23/15    added params.pow.doPhase support to 'wavelet' option
%
%       01/09/15    YE major update: removed EEG-extraction part of the
%                   code (wrapper for gete_ms) and put that in ComputeEEG.m
%                   this function now only computes power, but switches the
%                   way power is computed on the basis of the input
%
%       changes to previous iterations of ComputePow.m:
%       
%       01/09/15    YE updated to incorporate the artifact removal
%                   process from ComputePow_ArtifactRem.m as an
%                   option--this is flagged in params.ArtifactCorrect.
%                   ComputePow_ArtifactRem.m has been renamed to
%                   ComputePow_ArtifactRem_old.m and is now obsolete.
%       10/23/14    YE added phase output, removed binning
%       10/02/14    YE created function

%% initialize variables
PowMat = [];
PhaseMat = [];
%% switch over power calculation method
switch params.pow.type
    
    % power using wavelets
    case 'wavelet'
        % compute power and phase
        %Initialize variables
        PowMat = single(nan(length(params.pow.freqs),size(EEG,2),size(EEG,1)));
        doPhase = 0;PhaseMat = [];
        if isfield(params.pow,'doPhase')
            if params.pow.doPhase
                doPhase = 1;
                PhaseMat = PowMat;
            end
        end
        
        %Loop through  trials
        if doPhase
            for iEvent = 1:size(EEG,1)
                [PhaseMat(:,:,iEvent), PowMat(:,:,iEvent)]=multiphasevec3(params.pow.freqs,EEG(iEvent,:),params.eeg.sampFreq,params.pow.wavenum);
            end
        else
            for iEvent = 1:size(EEG,1)
                [~, PowMat(:,:,iEvent)]=multiphasevec3(params.pow.freqs,EEG(iEvent,:),params.eeg.sampFreq,params.pow.wavenum);
            end
        end
        
        %Remove buffer at beginning and at the end
        buffToRem = params.eeg.bufferMS*params.eeg.sampFreq/1000;
        buffInd = [1:buffToRem, (size(EEG,2)-buffToRem+1):size(EEG,2)];
        PowMat(:,buffInd,:) = [];
        if doPhase; PhaseMat(:,buffInd,:) = []; end
        
        % Log if logBefMean Flag is 1
        if params.pow.logTrans==1
            PowMat = log10(PowMat);
        end

    % power using slepian multitapers and fft
    case 'multitaper'
        
        % do slepian multitaper using eeg_toolbox fn mtenergyvec
        PowMat = single(nan(length(params.pow.freqs),size(EEG,2),size(EEG,1)));
        PhaseMat = PowMat;
        
        work = [];
        for iEvent = 1:size(EEG,1)
            [PowMat(:,:,iEvent),PhaseMat(:,:,iEvent),work] = mtenergyvec(EEG(iEvent,:),params.pow.freqs,params.eeg.sampFreq,params.pow.bandwidth,params.pow.winSize,work);
        end
        
        buffToRem = params.eeg.bufferMS*params.eeg.sampFreq/1000;
        buffInd = [1:buffToRem, (size(EEG,2)-buffToRem+1):size(EEG,2)];
        PowMat(:,buffInd,:) = [];
        PhaseMat(:,buffInd,:) = [];
        
        if params.pow.logTrans==1
            PowMat = log10(PowMat);
        end
        
        
    case 'pmtm_slep'
        % do slepian multitaper by FFT (i.e. don't use the convolution
        % method used in 'multitaper' mtenergyvec.m function)

        % first, determine winSize in samples and determine how many fit
        % into SignalLen (excluding the buffer)
        winSize = params.pow.winSize*params.eeg.sampFreq; % winSize in samples
        
        buffToRem = params.eeg.bufferMS*params.eeg.sampFreq/1000;
        buffInd = [1:buffToRem, (size(EEG,2)-buffToRem+1):size(EEG,2)];
        EEG(:,buffInd) = [];
        
        
        
        % loop through the signal to compute FFT in windows of size params.pow.winSize
        SignalLen = size(EEG,2);
        nWins = SignalLen/winSize;
        nEvents = size(EEG,1);
        if mod(nWins,1)~=0
            error('signal length is not an integer multiple of params.pow.winSize');
        else
            EEGrs = reshape(EEG,nEvents,winSize,[]);
        end
        PowMat = nan(length(params.pow.freqs),nWins,nEvents);
        PhaseMat = PowMat;
        
        for iWin = 1:nWins
            % use pmtm to compute multitaper estimate
            PowMat(:,iWin,:) = pmtm(EEGrs(:,:,iWin)',params.pow.bandwidth,params.pow.freqs,params.eeg.sampFreq);
        end
        
        % Log if logBefMean Flag is 1
        if params.pow.logTrans==1
            PowMat = log10(PowMat);
        end
        
    %%%%%%%%%%%%%
    case 'fft_slep'

        winSize = params.pow.winSize*params.eeg.sampFreq; % winSize in samples
        
        % figure out the length to make the fft, to get the freq resolution
        % you need to return the freqs specified in params.pow.freqs
        nfft = max(2^nextpow2(params.eeg.sampFreq/min(diff(params.pow.freqs))),winSize);
        %nfft = max(2^nextpow2(winSize),winSize);
        
        
        % don't include any buffer in the fft data
        buffToRem = params.eeg.bufferMS*params.eeg.sampFreq/1000;
        buffInd = [1:buffToRem, (size(EEG,2)-buffToRem+1):size(EEG,2)];
        EEG(:,buffInd) = [];
        
        
        % loop through the signal to compute FFT in windows of size params.pow.winSize
        SignalLen = size(EEG,2);
        
        % Compute the positions of the windows
        if isnan(params.pow.winStep) % do every sample
            winStarts = 1:((SignalLen-winSize)+1);
            winEnds = winSize:SignalLen;
        else % use a timestep
            winStep = params.pow.winStep*params.eeg.sampFreq;
            winStarts = 1:winStep:((SignalLen-winSize)+1);
            winEnds = winSize:winStep:SignalLen;
        end
        
        nEvents = size(EEG,1);
        
        % define the sleppian windows
        sWins = dpss(winSize, params.pow.bandwidth, (2*params.pow.bandwidth)-1); %sWins has dimensions (N x numSWins)
        numSWins = size(sWins, 2);
        sWins = sWins'; %now sWins has dimensions (numSWins x numSamples)
        
        sWins    = sWins(:,:,ones(1,nEvents));
        sWins    = permute(sWins,[3 2 1]);
        
        % allocate space for fft, define freqs & nyquist
        %nFreqs = length(params.pow.freqs);
        Fs = 0:params.eeg.sampFreq/nfft:params.eeg.sampFreq;
        Fs = Fs(1:nfft);
        NyquistInds = Fs<=floor(params.eeg.sampFreq/2);
        % get the fft freq index that is closest to our desired freq index,
        % and also the one prior and after. take the average across all 3
        % to get estimate of desired frequency
        freqInds = nan(length(params.pow.freqs),1);
        for iFreq = 1:length(params.pow.freqs)
            [v,c] = min(abs(Fs-params.pow.freqs(iFreq)));
            freqInds(iFreq) = c;
        end
        
        
        % check to remove any above Nyquist rate)
        if any(freqInds>find(NyquistInds,1,'last'))
            error('requested a frequency above the Nyquist limit');
            %freqInds(freqInds>max(find(NyquistInds))) = [];
        end
        
        
        % initialize PowMat
        %PowMat = nan(sum(NyquistInds), nWins, nEvents);
        PowMat = nan(length(params.pow.freqs), length(winStarts), nEvents);
        %PhaseMat = PowMat;
        
        % expand EEG to 3D for ease of multiplication w/slep wins
        EEG = EEG(:,:,ones(1,numSWins));

        for iInd = 1:length(winStarts)
            StartInd = winStarts(iInd);
            EndInd = winEnds(iInd);
        %for iInd = 1:(SignalLen-winSize)
        %for iWin = 1:nWins
            %StartInd = 1+((iWin-1)*(winSize));
            %EndInd = iWin*winSize;
            %StartInd = iInd;
            %EndInd = StartInd+winSize-1;
            
            % multiply signal by windows
            tmpEEG = EEG(:,StartInd:EndInd,:);
            tmpEEGProj = tmpEEG.*sWins;
            
            % compute the fft
            %FT = fft(tmpEEGProj,nfft,2)/params.eeg.sampFreq;
            FT = fft(tmpEEGProj,nfft,2)/nfft;
            FT = abs(FT).^2;
            FT = mean(FT,3);
            
            %FT = FT(:,freqInds)'; % select out freqs closest to those requested. also get prior and next freq, and avg the 3 together
            FT = cat(3,FT(:,freqInds),FT(:,freqInds-1),FT(:,freqInds+1));
            FT = mean(FT,3)';
            
            PowMat(:,iInd,:) = FT;
        end

        if params.pow.logTrans==1
            PowMat = log10(PowMat);
        end
        
        
end % switch params.pow.type



