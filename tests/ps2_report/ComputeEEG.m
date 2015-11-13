function [EEG] = ComputeEEG(Elec,events,params,InputPeaks)
% function [EEG] = ComputeEEG(Elec,events,params,InputPeaks)
%
% wrapper function to for gete_ms (and artifact correction) to return eeg
% voltage signal (which can then be subsequently passed to ComputPow.m,
% multiphasevec3, etc...
%
% inputs:
%       Elec        - electrode number from which to extract
%                   [3 4]   bipolar
%                   [1]     monopolar
%
%       events      - events (struct) for which to extract power
%
%       params      - struct with several fields determining how to extract and compute power on the data
%                       params.eeg.durationMS   length of the window within which to compute power
%                       params.eeg.offsetMS     length of time by which durationMS precedes/follows event onset
%                       params.eeg.bufferMS     size of buffer placed around epoch of interest
%                       params.eeg.filtfreq     specific frequency to filter (e.g. [58 62] for 60Hz noise)
%                       params.eeg.filttype     filter type (default in gete_ms = 'stop')
%                       params.eeg.filtorder    filter order (default in gete_ms = 1)
%                       params.eeg.sampFreq     frequency at which to resample (e.g. 1000 or 500)
%                       params.ArtifactCorrect  flag indicating whether to run artifact removal (e.g. 0 or 1)
%
%       InputPeaks  - cell array with peak values and locations for a set of
%                       events for a particular electrode. these are the
%                       locations of (stimulation) artifacts and are
%                       generated by RAM_StimEFF_RemoveArtifactPeaks.m
%
%
% outputs:
%       EEG      - nEvents x samples matrix of voltages
% 
% last updated:
%       01/09/15    created function from ComputePow (i.e. separated out
%                   the EEG and power components of ComputePow)
%
%       changes to previous iteration of the ComputePow function:
%
%       01/09/15    YE updated to incorporate the artifact removal
%                   process from ComputePow_ArtifactRem.m as an
%                   option--this is flagged in params.ArtifactCorrect.
%                   ComputePow_ArtifactRem.m has been renamed to
%                   ComputePow_ArtifactRem_old.m and is now obsolete.
%       10/23/14    YE added phase output, removed binning
%       10/02/14    YE created function


%% set the correct location of the raw data if running locally
basedir = GetBaseDir;

for k=1:length(events)
    if ~isempty(events(k).eegfile)
        if length(Elec)==2
            events(k).eegfile=[regexprep(events(k).eegfile,'eeg.reref','eeg.noreref')];
        end
        if strcmp(events(k).subject,'UT009a')
            events(k).eegfile=[basedir strrep(events(k).eegfile,'/data10/RAM/subjects/','/data/eeg/')];
        else
            events(k).eegfile=[basedir regexprep(events(k).eegfile,['/data.*/' events(1).subject '/eeg'],['/data/eeg/' events(1).subject '/eeg'])];
        end
    end
end

%% check for artifact removal inputs
if ~exist('InputPeaks','var')
    InptPksVar = 0;
elseif isempty(InputPeaks)
    InputPksVar = 0;
else
    InptPksVar = 1;
end

if ~isfield(params,'ArtifactCorrect')
    if InptPksVar==1
        error('*** InputPeaks passed for artifact removal, but params.ArtifactCorrect is not set ***');
    end
    DoArtRem = 0;
elseif params.ArtifactCorrect==1
    if InptPksVar==1
        DoArtRem = 1;
    else
        error('*** artifact removal indicated but no InputPeaks passed as input ***');
    end
else
    if InptPksVar==1
        error('*** params.ArtifactCorrect = 0, but InputPeaks passed as input ***');
    else
        DoArtRem = 0;
    end
end


%% get EEG
% adjust duration and offset to include buffer, for use in gete_ms.
% this will ensure the buffer interval is returned in the raw EEG, so that
% it can then be passed to multiphasevec3
durMS = params.eeg.durationMS+2*params.eeg.bufferMS;
offMS = params.eeg.offsetMS-params.eeg.bufferMS;

% monopolar referencing
if length(Elec)==1
    [EEG,resampleFreq] = gete_ms(Elec,events,durMS,offMS,params.eeg.bufferMS,...
        params.eeg.filtfreq,params.eeg.filttype,params.eeg.filtorder,...
        params.eeg.sampFreq);
    
    if DoArtRem
        EEGToFilt = EEG([events.isStim]==1,:);
        [EEGFilt,AllPeaks] = RAM_StimEff_RemoveArtifactPeaks(EEGToFilt,params.eeg.sampFreq,params.StimFreq,params.ArtifactWin,params.NPeaksAvg,params.AvgKernel,InputPeaks);
        
        EEG([events.isStim]==1,:) = EEGFilt;
    end 
    
% bipolar referencing
elseif length(Elec)==2
    [EEG1,resampleFreq] = gete_ms(Elec(1),events,durMS,offMS,params.eeg.bufferMS,...
        params.eeg.filtfreq,params.eeg.filttype,params.eeg.filtorder,...
        params.eeg.sampFreq);

    [EEG2,resampleFreq] = gete_ms(Elec(2),events,durMS,offMS,params.eeg.bufferMS,...
        params.eeg.filtfreq,params.eeg.filttype,params.eeg.filtorder,...
        params.eeg.sampFreq);

    if DoArtRem
        % pass in the events with stimulation for artifact removal
        EEG1ToFilt = EEG1([events.isStim]==1,:); EEG2ToFilt = EEG2([events.isStim]==1,:);
        [EEG1Filt,AllPeaks] = RAM_StimEff_RemoveArtifactPeaks(EEG1ToFilt,params.eeg.sampFreq,params.StimFreq,params.ArtifactWin,params.NPeaksAvg,params.AvgKernel,InputPeaks);
        [EEG2Filt,AllPeaks] = RAM_StimEff_RemoveArtifactPeaks(EEG2ToFilt,params.eeg.sampFreq,params.StimFreq,params.ArtifactWin,params.NPeaksAvg,params.AvgKernel,InputPeaks);
        
        EEG1([events.isStim]==1,:) = EEG1Filt;
        EEG2([events.isStim]==1,:) = EEG2Filt;
    end
    
    EEG = EEG1-EEG2;
end


%% Artifact removal Step 2: filter signal at stimulation freq and harmonics
if DoArtRem
    Multiples = floor(max(params.pow.freqs)/params.StimFreq);
    Harmonics = params.StimFreq.*[1:1:Multiples];
    freqrange = [Harmonics-2;Harmonics+2]';
    EEGorig = EEG;
    [EEG,filters] = buttfilt(EEG,freqrange,params.eeg.sampFreq,'stop',1);
end

