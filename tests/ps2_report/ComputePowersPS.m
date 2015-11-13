function  [a]=ComputePowersPS(Subject, output_dir)

%%% [a] is needed as a dummy var to avois too many output arguments error when calling from python


a=10;

params = createParams(output_dir);

computePowers(params,Subject); % this has artifact removal

save(fullfile(output_dir,'paramsPS.mat'),'params')



end




function computePowers(params,subj)

% set the baseline period from params struct
BaseBins = params.pow.baseBins;

% Part 1: view individual EEG traces of all stimulation events on stimulated channel and neighboring channel(s)
events = get_sub_events('RAM_PS',subj);
evInd = params.events(events); % function handle to filter events
events = events(evInd);
SessList = unique([events.session]);
events = events(ismember([events.session],SessList));

bp = getBipolarSubjElecs(subj,1,1,1);

% Part 2: compute power 
parfor iElec = 1:length(bp)
    RAM_PS_ComputePower_PAR(subj,bp(iElec).channel,params,events,BaseBins,SessList);
end

end

function [params] = createParams(saveDir)
% Function to generate a struct of parameters for use in PS2 analysis
% (analysis of RAM Parameter Search experiments).
%
% Inputs:
%               saveDir                 directory to save parameters. this
%                                       is also the location where the
%                                       results of PS2 analysis will be
%                                       stored.
%
% Outputs:
%               params.eeg.durationMS   length of the window within which to compute power
%               params.eeg.offsetMS     length of time by which durationMS precedes/follows event onset
%               params.eeg.bufferMS     size of buffer placed around epoch of interest
%               params.eeg.filtfreq     specific frequency to filter (e.g. [58 62] for 60Hz noise)
%               params.eeg.filttype     filter type (default in gete_ms = 'stop')
%               params.eeg.filtorder    filter order (default in gete_ms = 1)
%               params.eeg.sampFreq     frequency at which to resample (e.g. 1000 or 500)
%               params.eeg.kurtThr      kurtosis threshold to use for
%                                       discarding bad events
%               params.pow.type         string indicating method for computing power
%                                           'wavelet'   = Morlet wavelet
%                                           'fft_slep'  = FFT, using Slepian windows
%               params.pow.freqs        frequencies at which to compute power (e.g. [2 4 8...] or logspace(log10(2), log10(200),50)
%               params.pow.logTrans     1/0 flag indicating whether to log transform power vals
%               params.pow.wavenum      number of wavelets (typical = 7)
%               params.pow.bandwidth    bandwidth of slepian windows, if computing power using 'fft_slep' (typical=2)
%               params.pow.winSize      for 'multitaper' and 'FFT_slep'
%                                           this sets the width of the windows (in seconds) that are used for computing the FFT
%               params.pow.timeWin      binning: window size
%               params.pow.timeStep     binning: window step
%               params.pow.zscore       1/0 flag whether to zscore power
%
%               params.ArtifactCorrect  flag indicating whether to run artifact removal (e.g. 0 or 1)
%
%
% Last updated:
%   06/05/15    YE created function
% as of 06/05/15
%   min stim pulse duration = 250ms
%   min inter-pulse gap     = 2750ms
%   pre stim-pulse period   = 1000 (for up to 800ms window, 200ms gap)
params.eeg.durationMS   = 4000; 
params.eeg.offsetMS     = -1000;
params.eeg.bufferMS     = 1000;
params.eeg.filtfreq     = [58 62];
params.eeg.filttype     = 'stop';
params.eeg.filtorder    = 1;
params.eeg.sampFreq     = 500;
params.eeg.kurtThr      = 4;

params.pow.type         = 'fft_slep';
params.pow.freqs        = logspace(log10(1),log10(200),50);
params.pow.logTrans     = 1;
params.pow.wavenum      = nan;
params.pow.bandwidth    = 2;

% the following two parameters are specific fft_slep and are different from
% the parameters below that specifiy post-power binning.
params.pow.winSize      = .4;  % units in seconds
params.pow.winStep      = .05; % units in seconds. use nan if computing power for every sample.

params.pow.timeWin      = nan;
params.pow.timeStep     = nan;
params.pow.freqBins     = params.pow.freqs;
params.pow.zscore       = 1;


% Set up indices for getting stim onset and the pre-stim baseline bins. 
SignalLen = params.eeg.durationMS;
winSize = 1000*params.pow.winSize; % winSize in samples
if isnan(params.pow.winStep) % do every sample
    winStarts = 1:((SignalLen-winSize)+1);
    winEnds = winSize:SignalLen;
else % use a timestep
    winStep = 1000*params.pow.winStep;
    winStarts = 1:winStep:((SignalLen-winSize)+1);
    winEnds = winSize:winStep:SignalLen;
end
winStarts = winStarts+params.eeg.offsetMS-1;
winEnds = winEnds+params.eeg.offsetMS-1;

params.pow.timeBins = [winStarts' winEnds'];

% for the onset bin, take the bin that is closest to centered around 0
params.pow.onsetInd     = find(winEnds<(winSize/2),1,'last'); 

% for the pre-stimulation baseline bins, take the interval from baseStart
% to baseEnd
baseStart = -650; baseEnd = -50;
params.pow.baseBins     = intersect(find(winStarts>=baseStart),find(winEnds<=baseEnd));

params.events           = @(events)strcmp({events.type},'STIMULATING') & strcmp({events.experiment},'PS2');

params.savedir = saveDir;

end

