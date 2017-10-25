%Subject: $subject
%Version: $version
%
%Stimulation Parameters:
%  elec1: $anode, $anode_num
%  elec2: $cathode, $cathode_num
%  target amplitude: $target_amplitude uA
%  duration: $pulse_width ms
%  train frequency: $burst_frequency Hz
%  trainCount: $burst_count
%  pulse frequency: $pulse_frequency Hz
%  pulseCount: $pulse_count
%  wait after word on: $wait_after_word_on ms
%
%Copyright: University of Pennsylvania
%Workfile: StimControl.m
%Purpose: Control stimulation decision for RAM system.

%
%The class and methods will be called as follows:
%
%    control = StimControl.getInstance();        %Gets a handle to the single instance
%    control.initialize();                       %Call the initialize method.  Do not pass the 'this' parameter
%
%    ...                                         %Initialize is called only once per experiment
%
%    experimentState = struct;                   %A struct is used to allow for future additions
%    experimentState.phase = StimControl.PHASE_ENCODING; %It's better to use the pre-defined constants
%    experimentState.sample = 2001;              %This would be true for the second call
%    sampleSize = 2000;                          %Two seconds of data at 1K sample per second
%    dataByChannel = zeros(sampleSize, StimControl.NSP_CHANNELS);     %Would normally be filled by the Neuroport
%    control = StimControl.getInstance();        %Gets a handle to the single instance
%    stimDecision = control.stimChoice(experimentState, dataByChannel);  %Make a stim choice.  Do not pass 'this'.
%
%    ...                                         %Cleanup is called only once per experiment
%
%    control = StimControl.getInstance();        %Gets a handle to the single instance
%    control.cleanup();                          %Call the cleanup method.  Do not pass the 'this' parameter
%
%
classdef StimControl < handle

    %---- Add any constants here rather than embedding them inside code below
    properties (Access = public, Constant = true)
        %Number of channels recorded
        NSP_CHANNELS        = 144; %128 channels recorded from the patient + 16 analog channels

        STIMCONTROL_VERSION = 2.00;

       %Tested at the start of a session, to ensure that states can be retrieved
        POSSIBLE_STATES = {'STIM ENCODING', ...
                           'WORD'};

    end

    %---- Add or read in any persistent variables here
    properties(Access = public)
        %Initialize any variables here
        Subject;
        StimParams;
        wait_after_word_on; %1366 ms for FR3 1.0.2
        current_word_analyzed;  %flag that says not to compute on the present word again if true
        current_list_analyzed; 
        stimCategories; %1: stim odd, 2: stim even
        stimList_i; %Number of stim list that we are on
        word_i; %Number of word that we are on
        savedFileName; 
    end

    methods (Access = public, Static)
        function stimCategories = createStimCategories(~)
            nStimEvenFirstHalf = randi(2)+2; %either 3 or 4 even stims in first half
            nStimOddFirstHalf = 7-nStimEvenFirstHalf;
            firstHalf = [ones(1, nStimOddFirstHalf) ...
                         ones(1, nStimEvenFirstHalf)*2];
            firstHalf = firstHalf(randperm(length(firstHalf)));

            secondHalf = [ones(1, 7 - nStimOddFirstHalf) ...
                          ones(1, 7 - nStimEvenFirstHalf)*2];
            secondHalf = secondHalf(randperm(length(secondHalf)));

            stimCategories = [firstHalf secondHalf];
        end
    end
    
    %---- Public instance methods
    methods (Access = public)


                     
        
        %---- Initialize
        function initialize(this)
           %This method will be called wherever an experiment starts.  Set all persistent variables here.
           %Remember to include 'this.' in front of all variables that you want to be persistent.

           %oad R1124J_1_2016-01-22_RAM_FR1.biomarker.mat  %loads Bio struct of biomarker information

           this.Subject = '$subject';
           this.StimParams = struct('amplitude', $target_amplitude,...
                                    'duration', $pulse_width,...
                                    'trainFrequency', $burst_frequency,...
                                    'trainCount', $burst_count,...
                                    'pulseFrequency', $pulse_frequency,...
                                    'pulseCount', $pulse_count,...
                                    'elec1', $anode_num,...
                                    'elec2', $cathode_num);

           this.wait_after_word_on = $wait_after_word_on;
           
           this.stimCategories = this.createStimCategories();
           this.stimList_i = 0;
           this.word_i = 0;
           this.current_word_analyzed = false;
           this.current_list_analyzed = false;

           %check for existing zscore data (which happens if the session is
           %stopped and re-started) and load it. If not, initialize zscore
           %variables.
           control = RAMControl.getInstance();
           this.savedFileName = fullfile(control.getDataFolder,['FR4','StateSave.mat']);
           if exist(this.savedFileName,'file')
               savedData = load(this.savedFileName);
               this.stimList_i = savedData.stimList_i;
               this.stimCategories = savedData.stimCategories;
           end
        end

        function clear(this)
            clearvars this;
        end

        %---- Make a decision whether or not to stim based on the data provided
        function [decision, stopSession, errMsg] = stimChoice(this, experimentState, dataByChannel)
            %This method makes a stim decision based on the experiment state and data provided.
            %
            %Inputs:
            %  this.Bio            structure of info loaded from patient's
            %                      biomarker file
            %  experimentState:    a structure that contains information about the experiment state.
            %                      elements include:
            %      .phase          integer with one of the following values (ENCODING(1), DISTRACTOR(2),
            %                      RETRIEVAL(3), NONE(4) indicating the experimental phase
            %      .sample         number indicating the number of samples collected from the start
            %                      of the experiment
            %
            %  dataByChannel:      is number-of-samples x 144 matrix.  Each column contains a single channel.
            %                      The first 128 channels come from the patient.  The remaining 16 channels
            %                      are analog input channels, which could include sync pulses or
            %                      stim pulses (TBD)
            %Outputs:
            %  decision (no-stim=0 or stim=1)

            decision = 0;
            stopSession = false;
            errMsg = [];

            control = RAMControl.getInstance();
            
            [is_stim_list, ~] = control.isStateActive('STIM ENCODING');
            if ~is_stim_list
                this.current_list_analyzed = false;
                return
            end
            
            if ~this.current_list_analyzed
                this.stimList_i = this.stimList_i + 1;
                this.word_i = 0;
                this.current_list_analyzed = true;
            end
            
            [state_is_word, time_since_change] = control.isStateActive('WORD');

            if ~this.current_word_analyzed && ...
                    state_is_word && ...
                    time_since_change>=this.wait_after_word_on && ...
                    this.word_i < 10
            
                this.current_word_analyzed = true;
                this.word_i = this.word_i + 1;
                
                if xor(this.stimCategories(this.stimList_i)==1, mod(this.word_i, 2) == 0)
                    decision = 1;
                end
                
                return
                
            end

            if ~state_is_word
                this.current_word_analyzed = false;
            end
        end

        %---- Cleanup
        function cleanup(this)
            %Will be called to cleanup anything
            stimList_i = this.stimList_i;   %ok<NASGU,PROP>
            stimCategories = this.stimCategories; %ok<PROP,NASGU>
            save(this.savedFileName, 'stimList_i', 'stimCategories');
        end

    end %end public methods

    %---- Private methods.
    methods (Access = private)
        %Create private constructor to force use of StimControl.getInstance() to access methods
        %of this class.
        function this = StimControl
            %Do any one-time per program execution operations here, for example opening a log file.
        end
    end

    %---- Static methods.  Use StimControl.getInstance() to create the singleton instance of this class.
    methods (Access = public, Static)

        %Create or get the singleton instance
        function instance = getInstance
            persistent localObj
            if isempty(localObj) || ~isvalid(localObj)
                localObj = StimControl;
            end
            instance = localObj;
        end
    end
end
