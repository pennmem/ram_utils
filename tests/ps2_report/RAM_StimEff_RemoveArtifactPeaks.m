function [FiltSig,StorePeaks] = RAM_StimEff_RemoveArtifactPeaks(RawSig,SampleRate,StimFreq,ArtifactWin,NPeaksAvg,AvgKernel,InputPeaks)
% function [FiltSig,StorePeaks] = RAM_StimEff_RemoveArtifactPeaks(RawSig,SampleRate,StimFreq,PeakWin,NPeaksAvg,AvgKernel,InputPeaks)
%
% Function that reads data from RAM stimulation sessions and removes
% artifacts. Using the stimulation frequency, the code detects peaks in the
% raw EEG that are driven by artifacts and steps through the data,
% averaging within small temporal windows to create an artifact 'template'
% that is subtracted from the midpoint of the window to correct the signal.
%
%
% Inputs:
%   RawSig      - nEvents x nSamples raw EEG data (as from gete_ms)
%   SampleRate  - sampling rate of the RawSig input (e.g. 500 Hz)
%   StimFreq    - frequency (in Hz) at which stimulation was applied. This
%                 is IMPORTANT b/c the code uses the stim frequency as a
%                 prior for the inter-artifact gap (e.g. 50Hz stim = 20ms
%                 gap).
%   ArtifactWin - # of samples to grab before and after each peak (i.e. how
%                 long does the artifact last?)
%   NPeaksAvg   - the number of instances of the artifact to include in the average.
%   AvgKernel   - weighting function for averaging (if empty = uniform).
%                 length of the kernel must be equal to NPeaksAvg
%   InputPeaks  - nEvent x 2 (peaks & locs) cell array of peak locations, identified
%                 from a previous call to RAM_StimEff_RemoveArtifactPeaks
%   
%   
% Last updated:
%   01/08/14    YE updated to include second-phase median filtering of the
%               peaks that remain after template subtraction
%
%   12/23/14    YE created function


%% Set some parameters for first-phase template subtraction
basedir = GetBaseDir;
MinPeakDist = ((1/StimFreq)*SampleRate)-1;  % minimum peak distance = (ideal distance - 1 sample)
nEvents = size(RawSig,1);
nSamples = size(RawSig,2);
FiltSig = RawSig;

if ~exist('AvgKernel','var')
    AvgKernel = ones(NPeaksAvg,1);
end
if length(AvgKernel) ~= NPeaksAvg
    error('*** length of averaging kernel must equal the # of peaks to use in averaging ***');
end

% set the # of peaks before and after the center peak to use in averaging.
% this will differ depending on whether NPeaksAvg is odd/even.
if rem(NPeaksAvg,2)==1 % odd
    NPrePeaks = (NPeaksAvg-1)/2;
    NPostPeaks = NPrePeaks;
else  % even
    NPrePeaks = NPeaksAvg/2;
    NPostPeaks = NPrePeaks-1;
end

PeaksVect = nan(nEvents,1);

%% set some parameters for second-phase zscore-based median filtering.
% this second phase was added b/c the basic moving-window average
% subtraction was leaving behind pretty big spikes at the points where the
% artifact transitions from one shape to another
%buffsamp = buffMS*SampleRate/1000;

WinSize = 50; % size of the window (in samples) 
WinStep = 5;
zThresh = 4; MedFiltWin = 3;


%% Find peaks in the signal
for iEvent = 1:nEvents
    % NOTE: there is a function in the eeg_toolbox called findpeaks.m that does
    % not have the same functionality as the Matlab version. The eeg_toolbox
    % version is used in alignment (location: eeg_toolbox/align/), so
    % temporarily remove this folder from your path in order to run the Matlab
    % version.
%     S = which('findpeaks');
%     if strfind(S,fullfile('eeg_toolbox','align','findpeaks.m'))
%         error('*** remove the eeg_toolbox version of findpeaks.m from your path ***');
%     end
%     
    % peaks not previously defined/provided as input--identify now
    if ~exist('InputPeaks','var')
        [peaks,locs] = findpeaks(RawSig(iEvent,:),'MinPeakDistance',MinPeakDist,'MinPeakProminence',1000);
        % any peaks that are missed (i.e. where the inter-peak distance is too
        % large), find and insert the missing peak
        InterPeak = diff(locs);
        if any(InterPeak>MinPeakDist+3) % use +2 to ignore the ideal inter-peak dist (+1) and one more than that (+2).
            Gaps = find(InterPeak>MinPeakDist+3);
            for iGap = 1:length(Gaps)
                NMissing = floor(InterPeak(Gaps(iGap))/MinPeakDist);
                NewLocs = round(linspace(locs(Gaps(iGap)),locs(Gaps(iGap)+1),NMissing+1));
                NewLocs(1) = []; NewLocs(end) = [];
                %NewLoc = locs(Gaps(iGap))+round(InterPeak(Gaps(iGap))/2);
                NewPeaks = RawSig(iEvent,NewLocs);
                locs = [locs,NewLocs]; peaks = [peaks,NewPeaks];
            end
            [locs,i] = sort(locs);
            peaks = peaks(i);
        end
        StorePeaks{iEvent,1} = peaks; StorePeaks{iEvent,2} = locs;
        
    % peaks already identified and provided as input--load for current event
    else
        peaks   = InputPeaks{iEvent,1};
        locs    = InputPeaks{iEvent,2};
        StorePeaks = InputPeaks;
    end

    PeaksVect(iEvent) = length(peaks);
    
    %% Loop through the locations of the peaks and take an average of the artifact
    for iPeak = 1:length(peaks)
        
        % condition for start of the list
        if iPeak <= NPrePeaks
            
            % find index of start peak to include
            StartPeakAdj = iPeak-1;
            PeaksToAvg = locs(iPeak-StartPeakAdj:iPeak+NPostPeaks);
            MidInd = round(length(AvgKernel)/2); %MidInd = find(AvgKernel==1);
            TmpAvgKernel = AvgKernel(MidInd-StartPeakAdj:MidInd+NPostPeaks);
            AvgMat = repmat(TmpAvgKernel,1,(2*ArtifactWin)+1);
            
            TmpSig = [];
            for iData = 1:length(PeaksToAvg)
                % put in a check to make sure that
                % PeaksToAvg(iData)-ArtifactWin isn't going out of the
                % bounds of RawSig. if it is, just skip this peak
                if (PeaksToAvg(iData)-ArtifactWin > 0) && (PeaksToAvg(iData)+ArtifactWin <= size(RawSig,2))
                    TmpSig(iData,:) = RawSig(iEvent,PeaksToAvg(iData)-ArtifactWin:PeaksToAvg(iData)+ArtifactWin)-RawSig(iEvent,PeaksToAvg(iData)-ArtifactWin);
                end
            end
            TmpSigAvg = sum(TmpSig.*AvgMat)./sum(TmpAvgKernel);
            if (locs(iPeak)-ArtifactWin > 0) && (locs(iPeak)+ArtifactWin <= size(RawSig,2))
                FiltSig(iEvent,locs(iPeak)-ArtifactWin:locs(iPeak)+ArtifactWin) = FiltSig(iEvent,locs(iPeak)-ArtifactWin:locs(iPeak)+ArtifactWin)-TmpSigAvg;
            end
            
        % condition for end of the list
        elseif iPeak > length(peaks)-NPostPeaks
            
            % find index of end peak to include
            EndPeakAdj = length(peaks)-iPeak;
            PeaksToAvg = locs(iPeak-NPrePeaks:iPeak+EndPeakAdj);
            MidInd = round(length(AvgKernel)/2); %MidInd = find(AvgKernel==1);
            TmpAvgKernel = AvgKernel(MidInd-NPrePeaks:MidInd+EndPeakAdj);
            AvgMat = repmat(TmpAvgKernel,1,(2*ArtifactWin)+1);
            
            TmpSig = [];
            for iData = 1:length(PeaksToAvg)
                if (PeaksToAvg(iData)-ArtifactWin > 0) && (PeaksToAvg(iData)+ArtifactWin <= size(RawSig,2))
                    TmpSig(iData,:) = RawSig(iEvent,PeaksToAvg(iData)-ArtifactWin:PeaksToAvg(iData)+ArtifactWin)-RawSig(iEvent,PeaksToAvg(iData)-ArtifactWin);
                end
            end
            TmpSigAvg = sum(TmpSig.*AvgMat)./sum(TmpAvgKernel);
            if (locs(iPeak)-ArtifactWin > 0) && (locs(iPeak)+ArtifactWin <= size(RawSig,2))
                FiltSig(iEvent,locs(iPeak)-ArtifactWin:locs(iPeak)+ArtifactWin) = FiltSig(iEvent,locs(iPeak)-ArtifactWin:locs(iPeak)+ArtifactWin)-TmpSigAvg;
            end
            
            
        % middle of the list
        else
            PeaksToAvg = locs(iPeak-NPrePeaks:iPeak+NPostPeaks);
            AvgMat = repmat(AvgKernel,1,(2*ArtifactWin)+1);
            TmpSig = [];
            for iData = 1:length(PeaksToAvg)
                if (PeaksToAvg(iData)-ArtifactWin > 0) && (PeaksToAvg(iData)+ArtifactWin <= size(RawSig,2))
                    TmpSig(iData,:) = RawSig(iEvent,PeaksToAvg(iData)-ArtifactWin:PeaksToAvg(iData)+ArtifactWin)-RawSig(iEvent,PeaksToAvg(iData)-ArtifactWin);
                end
            end
            TmpSigAvg = sum(TmpSig.*AvgMat)./sum(AvgKernel);
            if (locs(iPeak)-ArtifactWin > 0) && (locs(iPeak)+ArtifactWin <= size(RawSig,2))
                FiltSig(iEvent,locs(iPeak)-ArtifactWin:locs(iPeak)+ArtifactWin) = FiltSig(iEvent,locs(iPeak)-ArtifactWin:locs(iPeak)+ArtifactWin)-TmpSigAvg;
            end
        end
    end
    
end % iEvent


%% do second-phase median filtering based on zscored signal during stim interval
% Take the stimulation epoch of the filtered signal and check
% for remaining peaks--e.g. the ones that seemed to still be
% present after peak detection. these seemed to be at
% timepoints where the stimulation artifact was changing shape.

% use zscored signal to identify peaks
FiltSigNew = FiltSig;
for iEvent = 1:size(FiltSig,1);
    locs = StorePeaks{iEvent,2};
    StartLoc = min(locs)-10; EndLoc = max(locs)+10;
    WinMids = [StartLoc:WinStep:EndLoc];
    for iWin = 1:length(WinMids)
        if WinMids(iWin)-(WinSize/2) > 1
            zPeakLocs = find(abs(zscore(FiltSigNew(iEvent,WinMids(iWin)-(WinSize/2):WinMids(iWin)+(WinSize/2))))>zThresh);
            if ~isempty(zPeakLocs)
                zPeakLocs = zPeakLocs+WinMids(iWin)-(WinSize/2)-1;
                for iPeak = 1:length(zPeakLocs)
                    %FiltSigNew(iEvent,zPeakLocs(iPeak)-MedFiltWin:zPeakLocs(iPeak)+MedFiltWin) = medfilt1(FiltSig(iEvent,zPeakLocs(iPeak)-MedFiltWin:zPeakLocs(iPeak)+MedFiltWin),);
                    if zPeakLocs(iPeak)-MedFiltWin > 1
                        FiltSigNew(iEvent,zPeakLocs(iPeak)) = median(FiltSigNew(iEvent,zPeakLocs(iPeak)-MedFiltWin:zPeakLocs(iPeak)+MedFiltWin));
                    end
                end
            end
        end
    end
end
FiltSig = FiltSigNew;

