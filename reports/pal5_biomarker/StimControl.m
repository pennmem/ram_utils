% Subject: FILL_IN
% Version: FILL_IN
%
% Stimulation Parameters:
%   elec1: FILL_IN
%   elec2: FILL_IN
%   duration: FILL_IN ms
%   trainFrequency: FILL_IN Hz
%   trainCount: FILL_IN
%   pulseFrequency: FILL_IN Hz
%   pulseCount: FILL_IN
%   state: FILL_IN
%
% ClosedLoop Parameters:
%   thresh: FILL_IN
%   fs: FILL_IN
%   winSize: FILL_IN
%   windowFunc: FILL_IN
%   probLims: FILL_IN
%
% Classifier Parameters:
%   Task: FILL_IN
%   Sessions: FILL_IN
%   timebinsLabel: FILL_IN
%   FeatType: FILL_IN
%   Classifier: FILL_IN
%   CV: FILL_IN
%   Penalty: FILL_IN
%
% Copyright: University of Pennsylvania
% Workfile: StimControl.m
% Purpose: Control stimulation decision for RAM system.

%{
The class and methods will be called as follows:

    control = StimControl.getInstance();        % Gets a handle to the single instance
    control.initialize();                       % Call the initialize method.  Do not pass the 'this' parameter

    ...                                         % Initialize is called only once per experiment

    experimentState = struct;                   % A struct is used to allow for future additions
    experimentState.phase = StimControl.PHASE_ENCODING; % It's better to use the pre-defined constants
    experimentState.sample = 2001;              % This would be true for the second call
    sampleSize = 2000;                          % Two seconds of data at 1K sample per second
    dataByChannel = zeros(sampleSize, StimControl.NSP_CHANNELS);     % Would normally be filled by the Neuroport
    control = StimControl.getInstance();        % Gets a handle to the single instance
    stimDecision = control.stimChoice(experimentState, dataByChannel);  % Make a stim choice.  Do not pass 'this'.

    ...                                         % Cleanup is called only once per experiment

    control = StimControl.getInstance();        % Gets a handle to the single instance
    control.cleanup();                          % Call the cleanup method.  Do not pass the 'this' parameter

%}
classdef StimControl < handle

    % ---- Add any constants here rather than embedding them inside code below
    properties (Access = public, Constant = true)
        % Number of channels recorded
        % NSP_CHANNELS        = 144; % 128 channels recorded from the patient + 16 analog channels

        STIMCONTROL_VERSION = 2.00;

        % Tested at the start of a session, to ensure that states can be retrieved
        POSSIBLE_STATES = {'PRACTICE',...
                           'STIM ENCODING', ...
                           'NON-STIM ENCODING', ...
                           'RETRIEVAL', ...
                           'DISTRACT', ...
                           'INSTRUCT', ...
                           'COUNTDOWN', ...
                           'WAITING', ...
                           'WORD', ...
                           'ORIENT', ...
                           'MIC TEST'};

    end

    % ---- Add or read in any persistent variables here
    properties(Access = public)
        % Initialize any variables here
        Subject;
        StimParams;
        zscorer; % StatAccum to accumulate data for zscoring
        n_freqs; % Number of frequencies, 8 for PAL3 design
        bpmat;   % matrix to convert monopolar to bipolar eeg
        freqs;   % frequencies used in wavelet transform
        fs;      % sampling rate = 1000 Hz for Sys 2.x
        winsize; % 1700 for PAL3
        total_winsize; % 3700 for PAL3
        wait_after_word_on; % 2000 ms for PAL3
        bufsize; % 1000 for PAL3
        phase;   % experiment phase
        W_in;    % classifier weights
        thresh;  % prob threshold
        savedFileName; % where zscorer and ks_done flag are saved
        session_eeg;   % raw monopolar eeg stored for postmortem
        session_pows;  % collected for KS test and postmortem
        powFileName;   % file with saved powers
        Filter;        % Butterworth filter [58 62] Hz
        trainingProb;  % collection of classifier probabilities from PAL1 sessions
        ks_test_done;  % flag that becomes true when KS test is performed and passed
        current_word_analyzed;  % flag that says not to compute on the present word again if true
        wavelet_transformer;
    end

    % ---- Public instance methods
    methods (Access = public)

        % ---- Initialize
        function initialize(this)
           % This method will be called wherever an experiment starts.  Set all persistent variables here.
           % Remember to include 'this.' in front of all variables that you want to be persistent.
           load FILL_IN  % loads Bio struct of biomarker information

           this.Subject = Bio.Subject;
           this.StimParams = Bio.StimParams;

           this.zscorer = StatAccum();
           
           % Real-time analysis parameters
           this.freqs = logspace(log10(3), log10(180), 8);
           this.n_freqs = 8;
           this.bpmat = Bio.bpmat;
           this.W_in = [Bio.W Bio.W0];      % weights, w/constant term to the end of the weight vector
           this.trainingProb = Bio.trainingProb;
           this.thresh = Bio.thresh;
           this.fs = Bio.fs;                  % sampling freq.
           this.winsize = 1700;
           this.total_winsize = 3700;
           this.bufsize = 1000;
           this.wait_after_word_on = 2000;
           [B,A] = butter(4, [58.0 62.0]/(this.fs/2.0), 'stop');
           this.Filter.coeffs = [B;A];

           % check for existing zscore data (which happens if the session is
           % stopped and re-started) and load it. If not, initialize zscore
           % variables.
           control = RAMControl.getInstance();
           this.savedFileName = fullfile(control.getDataFolder,[Bio.Subject,'StateSave.mat']);
           if exist(this.savedFileName,'file')
               savedData = load(this.savedFileName);
               this.zscorer = savedData.zscorer;
               this.ks_test_done = savedData.ks_test_done;
           else
               this.zscorer.initialize();
               this.ks_test_done = false;
           end
           this.powFileName = fullfile(control.getDataFolder,[Bio.Subject,'PowSave.mat']);
           if exist(this.powFileName,'file')
               powData = load(this.powFileName);
               this.session_eeg = powData.session_eeg;
               this.session_pows = powData.session_pows;
           else
               this.session_eeg = [];
               this.session_pows = [];
           end
           this.current_word_analyzed = false;
           this.wavelet_transformer = morlet_interface();
           this.wavelet_transformer.init();
        end

        function setPhase(this, newPhase)
            % Set the phase
            this.phase = newPhase;
        end

        function value = getPhase(this)
            % Get the phase
            value = this.phase;
        end

        function clear(this)
            clearvars this;
        end

        % ---- Make a decision whether or not to stim based on the data provided
        function [decision, stopSession, errMsg] = stimChoice(this, experimentState, dataByChannel)
            % This method makes a stim decision based on the experiment state and data provided.
            %
            % This method will be called every 50 ms to determine whether to stimulate or not.
            % The computations in this methods must 'keep up', so the budget for computations
            % is 30 ms.  This is an average, so it is possible for the method to take longer than
            % 30 ms, at which time more data accumlates in the buffer.
            %
            % Inputs:
            %   this.Bio            structure of info loaded from patient's
            %                       biomarker file
            %   experimentState:    a structure that contains information about the experiment state.
            %                       elements include:
            %       .phase          integer with one of the following values (ENCODING(1), DISTRACTOR(2),
            %                       RETRIEVAL(3), NONE(4) indicating the experimental phase
            %       .sample         number indicating the number of samples collected from the start
            %                       of the experiment
            %
            %   dataByChannel:      is number-of-samples x #channels matrix.  Each column contains a single channel.
            %                       The first 128 channels come from the patient.  The next 16 channels
            %                       are analog input channels, which could include sync pulses or
            %                       stim pulses (TBD). Repeated #neuroports times.
            % Outputs:
            %   decision (no-stim=0 or stim=1)
            decision = 0;
            stopSession = false;
            errMsg = [];

            control = RAMControl.getInstance();

            if ~this.ks_test_done && control.isStateActive('WAITING') && (experimentState.list==4)
                n_samples = size(this.session_pows,2);
                probs = zeros(1,n_samples);
                for i=1:n_samples
                    normalized_pow = (this.session_pows(:,i) - this.zscorer.mean) ./ this.zscorer.stdev;
                    probs(i) = 1.0 / (1.0+exp(-(this.W_in*[normalized_pow;1])));
                    fprintf('prob=%f, threshold=%f\n', probs(i), this.thresh);
                end
                [~,p] = kstest2(this.trainingProb,probs);
                if p>=0.05
                    fprintf('KS test does not reject the null hypothesis, p = %f\n', p);
                else
                    stopSession = true;
                    errMsg = sprintf('KS test rejects the null hypothesis, p = %f\n', p);
                    fprintf(errMsg);
                    return;
                end
                this.ks_test_done = true;
            end

            [state_is_word, time_since_change] = control.isStateActive('WORD');
            state_is_retrieval = control.isStateActive('RETRIEVAL');

            if ~this.current_word_analyzed && state_is_word && ~state_is_retrieval && time_since_change>=this.wait_after_word_on
                this.current_word_analyzed = true;
                is_stim_encoding = control.isStateActive('STIM ENCODING');

                % decoding procedure

                n_channels = size(dataByChannel,2);
                if n_channels==144
                    dataByChannel = dataByChannel(end-this.winsize+1:end,1:128);
                elseif n_channels==288
                    dataByChannel = dataByChannel(end-this.winsize+1:end,[1:128,145:272]);
                else
                    stopSession = true;
                    errMsg = sprintf('ERROR: %d channels detected, unknown number of neuroports\n', n_channels);
                    fprintf(errMsg);
                    return;
                end

                this.session_eeg = cat(3, this.session_eeg, dataByChannel);

                dataByChannel = dataByChannel*this.bpmat;

                % mirroring happens here
                flipdata = flipud(dataByChannel);
                dataByChannel = [flipdata(end-this.bufsize:end-1,:); dataByChannel; flipdata(2:this.bufsize+1,:)];

                n_bps = size(dataByChannel,2);

                b = this.Filter.coeffs(1,:);
                a = this.Filter.coeffs(2,:);

                % apply Butterworth filter
                dataByChannel = filtfilt2(b,a,dataByChannel);

                % compute powers
                pow = zeros(this.n_freqs, n_bps);
                pow_j = zeros(this.total_winsize, this.n_freqs);
                for j=1:n_bps
                    signal = dataByChannel(:,j)';
                    this.wavelet_transformer.multiphasevec(signal, pow_j);
                    pow_j_stripped = pow_j(this.bufsize+1:end-this.bufsize,:);
                    pow_j_stripped = log10(pow_j_stripped);
                    pow_j_stripped = nanmean(pow_j_stripped, 1);
                    pow(:,j) = pow_j_stripped';
                end
                pow = pow(:);

                % saving for KS test or post-mortem
                this.session_pows = [this.session_pows pow];

                if is_stim_encoding
                    % zscore powers
                    pow = (pow - this.zscorer.mean) ./ this.zscorer.stdev;

                    % apply classifier here
                    prob = 1.0 / (1.0+exp(-(this.W_in*[pow;1])));
                    fprintf('prob=%f, threshold=%f\n', prob, this.thresh);
                    if prob<this.thresh
                        decision = 1;
                    end
                else
                    % update zscorer
                    this.zscorer.receive(pow);
                    fprintf('#samples for zscoring = %d\n', this.zscorer.n);

                    if this.ks_test_done
                        % enough samples for zscoring; output probability in the regular way
                        pow = (pow - this.zscorer.mean) ./ this.zscorer.stdev;
                        prob = 1.0 / (1.0+exp(-(this.W_in*[pow;1])));
                        fprintf('prob=%f, threshold=%f\n', prob, this.thresh);
                    end
                end
            end

            if ~state_is_word
                this.current_word_analyzed = false;
            end;
        end

        % ---- Cleanup
        function cleanup(this)
            % Will be called to cleanup anything
            zscorer = this.zscorer;
            ks_test_done = this.ks_test_done;
            save(this.savedFileName, 'zscorer', 'ks_test_done');

            session_eeg = this.session_eeg;
            session_pows = this.session_pows;
            save(this.powFileName, 'session_eeg', 'session_pows');
        end

    end % end public methods

    % ---- Private methods.
    methods (Access = private)
        % Create private constructor to force use of StimControl.getInstance() to access methods
        % of this class.
        function this = StimControl
            % Do any one-time per program execution operations here, for example opening a log file.
        end
    end

    % ---- Static methods.  Use StimControl.getInstance() to create the singleton instance of this class.
    methods (Access = public, Static)

        % Create or get the singleton instance
        function instance = getInstance
            persistent localObj
            if isempty(localObj) || ~isvalid(localObj)
                localObj = StimControl;
            end
            instance = localObj;
        end
    end
end
