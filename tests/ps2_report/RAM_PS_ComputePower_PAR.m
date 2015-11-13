function [] = RAM_PS_ComputePower_PAR(Subject,channels,params,events,BaseBins,SessList)


for iSess = 1:length(SessList)
    
    SessNum = SessList(iSess);
    if SessNum < 10
        SessName = sprintf('Sess0%d',SessNum);
    else
        SessName = sprintf('Sess%d',SessNum);
    end
    
    % check that power hasn't already been computed
    doPow = false;
    
    % check top-level subject power folder
    if ~isdir(fullfile(params.savedir,'power',Subject,SessName));
        doPow = true;
    else
        cd(fullfile(params.savedir,'power',Subject,SessName));
        
        % check individual power files
        %if ~exist(sprintf('%d-%d_Pow_bin_BaseSubtr.mat',channels(1),channels(2)),'file') ||...
        if        ~exist(sprintf('%d-%d_Pow_bin_zs.mat',channels(1),channels(2)),'file')
            doPow = true;
            
            % power exists but not all events have been computed
            %         elseif exist(sprintf('%d-%d_Pow_bin_BaseSubtr.mat',channels(1),channels(2)),'file')
            %             load(sprintf('%d-%d_Pow_bin_BaseSubtr.mat',channels(1),channels(2)));
            %             if size(PowMat,3)~=length(events)
            %                 events = events(size(PowMat,3)+1:end);
            %                 doPow = 2;
            %             end
            %         elseif exist(sprintf('%d-%d_Pow_bin_zs.mat',channels(1),channels(2)),'file')
            %             load(sprintf('%d-%d_Pow_bin_zs.mat',channels(1),channels(2)));
            %             if size(PowMat,3)~=length(events)
            %                 events = events(size(PowMat,3)+1:end);
            %                 doPow = 2;
            %             end
        end
    end
    
    
    if ~doPow
        display(sprintf('Power Found: Sess %d, %s Elec: %d-%d...Skipping',SessNum,Subject,channels(1),channels(2)));
        
    else
        SessInds = [events.session]==SessNum;
        % Get EEG traces, w/buffer appended
        EEG = ComputeEEG(channels,events(SessInds),params);
        
        % Get power, buffer removed
        [PowMat,~] = ComputePow(EEG,params);
        
        % Downsample into averaged bins
        [PowMat] = BinPow(PowMat,params.pow.timeWin,params.pow.timeStep,params.eeg.sampFreq,params.pow.freqs,params.pow.freqBins);
        
        % Save un-baseline corrected power data
        %cd_mkdir(fullfile(params.savedir,'power',Subject,SessName));
        %fname = sprintf('%d-%d_Pow_bin.mat',channels(1),channels(2));
        %save(fname,'PowMat');
        
        % try two types of baseline correction: (1) use trial-specific pre-stimulation
        % onset bins and (2) use the mean of the pre-stimulation period
        % across all events. use -500-0ms pre-stimulation onset.
        BaseVec = nanmean(PowMat(:,BaseBins,:),2);
        BaseMat = repmat(BaseVec,1,size(PowMat,2),1);
        
        PowMat = PowMat-BaseMat;
        %cd_mkdir(fullfile(params.savedir,'power',Subject,SessName));
        %fname = sprintf('%d-%d_Pow_bin_BaseSubtr.mat',channels(1),channels(2));
        %if doPow     % computing power from scratch, so save whole matrix
        %    save(fname,'PowMat');
        %end
        clear CurrPow;
        
        % z-score method (2):
        % zscore within session. do z-score across all stimulation events
        % in the session for the interval -500-0 ms pre-stimulation onset
        PowMat = PowMat+BaseMat;
        
        %for iSess = 1:length(SessList)
        %SessInds = [events.session]==SessList(iSess);
        BasePowMean = nanmean(nanmean(PowMat(:,BaseBins,:),3),2);
        BasePowSTD = nanstd(nanmean(PowMat(:,BaseBins,:),2),[],3);
        [PowMat] = zScorePow(PowMat,BasePowMean,BasePowSTD);
        %end
        cd_mkdir(fullfile(params.savedir,'power',Subject,SessName));
        fname = sprintf('%d-%d_Pow_bin_zs.mat',channels(1),channels(2));
        if doPow     % computing power from scratch, so save whole matrix
            save(fname,'PowMat');
        end
        
        display(sprintf('completed %s: %d-%d',Subject,channels(1),channels(2)));
    end
    
    
end