function [params] = CreateParamsDemo(params_path)

load(params_path)
whos
params=10

save('params_serialized_from_matlab.mat')

% params.eeg.durationMS   = 4000; 
% params.eeg.offsetMS     = -1000;

% params.eeg.bufferMS     = 1000;
% params.eeg.filtfreq     = [58 62];
% params.eeg.filttype     = 'stop';
% params.eeg.filtorder    = 1;
% params.eeg.sampFreq     = 500;
% params.eeg.kurtThr      = 4;

% params.eeg.HilbertBands = [4 8;...
%                             8 12;...
%                             30 50;...
%                             70 100];
                        


% params.eeg.HilbertNames = {'Theta (4-8 Hz)','Alpha (8-12 Hz)','Low Gamma (30-50 Hz)','High Gamma (70-100 Hz)'};

% params.pow.type         = 'fft_slep';
% params.pow.freqs        = logspace(log10(1),log10(200),50);
% params.pow.logTrans     = 1;
% params.pow.wavenum      = nan;
% params.pow.bandwidth    = 2;


% params.pow.wavenum


% % the following two parameters are specific fft_slep and are different from
% % the parameters below that specifiy post-power binning.
% params.pow.winSize      = .4;  % units in seconds
% params.pow.winStep      = .05; % units in seconds. use nan if computing power for every sample.

% params.pow.timeWin      = 100;
% params.pow.timeStep     = 50;
% params.pow.freqBins     = params.pow.freqs;
% params.pow.zscore       = 1;


% params.events           = @(events)ismember({events.type},{'STIMULATING','BEGIN_BURST','STIM_SINGLE_PULSE'});
% params.savedir          = SaveDir;

% cd_mkdir(SaveDir); 

% save('params','params');