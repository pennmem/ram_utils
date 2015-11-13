function [GroupStruct,varargout] = RAM_GetPSL(Task, SubjList)
% function [GroupStruct,varargout] = RAM_GetPSL(Task)
%
% Function takes a task name as input and returns a GroupStruct, a struct
% with one element per unique [Patient--Stim Location] (PSL) value. Each
% element of the struct has fields with various pieces of info relating to
% that particular unique [Patient--Stim Location].
%
% Inputs:
%       Task            string with the task name (e.g. 'RAM_FR2')
%
% Outputs:
%       GroupStruct     structure with one element per unique Pt-Stimloc
%                       datapoint for the experiment given as input Task.
%       varargout       if requested, this outputs err_struct, which
%                       captures any error information
%
% last updated:
%       03/13/15    YE added varargout to output error struct
%       02/19/15    YE created function


% get patients, filter to only TJ and R ones
%SubjList = get_subs(Task);
KeepSubjs = strncmp(SubjList,'TJ',2) | strncmp(SubjList,'R1',2) | strncmp(SubjList,'UT',2);
SubjList = SubjList(KeepSubjs);clear KeepSubjs;

% loop through patients. load events to determine unique set of stimulated
% locations
c = 1; ec = 1;
GroupStruct = []; err_struct = [];
for iSubj = 1:length(SubjList)
    
    try
        % initialize vars
        tagNames = {};
        Anodes = tagNames; Cathodes = tagNames;
        
        % load subject, events, bipolar struct (for RAM_GetMTLROIs) and
        % monopolar struct (for matching channels to tagNames)
        Subject = SubjList{iSubj};%if strcmp(Subject,'R1052E_1');keyboard;end;
        events = get_sub_events(Task,Subject);
        bpStruct = getBipolarSubjElecs(Subject);
        [ROIList,bpLabels] = RAM_GetMTLROIs(Subject,bpStruct);
        monoStruct = getBipolarSubjElecs(Subject,0);
        chans = [monoStruct.channel];
        
        % loop through sessions to find the anode-cathode pairs for each
        % session
        Anodes = []; Cathodes = [];
        SessList = unique([events.session]);
        for iSess = 1:length(SessList)
            % find # of different stim locations for this patient & task
            if isfield(events,'stimAnodeTag')
                Anodes = [Anodes unique({events([events.session]==SessList(iSess)).stimAnodeTag})];
                Cathodes = [Cathodes unique({events([events.session]==SessList(iSess)).stimCathodeTag})];
            elseif isfield(events,'stimAnode') % FR2, CatFR2, YC2
                Anodes = [Anodes; nanunique([events([events.session]==SessList(iSess)).stimAnode])];
                Cathodes = [Cathodes; nanunique([events([events.session]==SessList(iSess)).stimCathode])];
                %Anodes = nanunique([events.stimAnode]);
                %Cathodes = nanunique([events.stimCathode]);
            else % PAL2
                stimLocs = [events.stimLoc];
                Anodes = [Anodes; nanunique(stimLocs(1,[events.session]==SessList(iSess)))];
                Cathodes = [Cathodes; nanunique(stimLocs(2,[events.session]==SessList(iSess)))];
                %Anodes = nanunique(stimLocs(1,:));
                %Cathodes = nanunique(stimLocs(2,:));
            end
        end % iSess
        
        if iscell(Anodes)
            Anodes(cellfun('isempty',Anodes)) = [];
        end
        if iscell(Cathodes)
            Cathodes(cellfun('isempty',Cathodes)) = [];
        end
        
        if ~iscolumn(Anodes)
            Anodes = Anodes';
        end
        if ~iscolumn(Cathodes)
            Cathodes = Cathodes';
        end
        
        if ~isempty(Anodes)
            if iscell(Anodes)
                [~,ia,~] = uniqueRowsCA([Anodes Cathodes]);
                Anodes = Anodes(ia); Cathodes = Cathodes(ia);
                tmpAnodes = Anodes; tmpCathodes = Cathodes;
                Anodes = nan(length(tmpAnodes),1); Cathodes = nan(length(tmpCathodes),1);
                for i=1:length(tmpAnodes);
                    Anodes(i) = monoStruct(strcmpi({monoStruct.tagName},tmpAnodes{i})).channel;
                    Cathodes(i) = monoStruct(strcmpi({monoStruct.tagName},tmpCathodes{i})).channel;
                end
            else
                [~,ia,~] = unique([Anodes Cathodes],'rows');
                Anodes = Anodes(ia); Cathodes = Cathodes(ia);
            end
        end
        clear tmpAnodes tmpCathodes;
        
        % deal with two exception conditions
        if strcmp(Subject,'R1030J') && strcmp(Task,'RAM_PS')
            Anodes = 50;
            Cathodes = 51;
            %StimTag = 'LQ2-LQ3';
        elseif strcmp(Subject,'R1044J') && strcmp(Task,'RAM_PS')
            Anodes = 164;
            Cathodes = 165;
            %StimTag = 'LA2-LA3';
        end
        nLocs = length(Anodes);
        

        % loop over the unique locations of stimulation (tags)
        for iLoc = 1:nLocs
            tagNames = {monoStruct.tagName};
            AnodeTag = tagNames{chans==Anodes(iLoc)};
            CathodeTag = tagNames{chans==Cathodes(iLoc)};
            
            GroupStruct(c).Subject = Subject;
            GroupStruct(c).Task = Task;
            if isfield(events,'stimAnodeTag')
                GroupStruct(c).Sessions = unique([events(strcmpi({events.stimAnodeTag},AnodeTag)).session]);
            elseif isfield(events,'stimAnode') % FR2, CatFR2, YC2
                GroupStruct(c).Sessions = unique([events([events.stimAnode]==Anodes(iLoc)).session]);
            else % PAL2
                GroupStruct(c).Sessions = unique([events(stimLocs(1,:)==Anodes(iLoc)).session]);
            end
            
            % exception condition
            if (strcmp(Subject,'R1030J') || strcmp(Subject,'R1044J')) && strcmp(Task,'RAM_PS')
               GroupStruct(c).Sessions = unique([events.session]); 
            end
            
            GroupStruct(c).StimElecTag = [AnodeTag '-' CathodeTag];
            GroupStruct(c).StimElecChans = [Anodes(iLoc) Cathodes(iLoc)];
            
            % anode ROI label
            if sum(strcmp(ROIList(:,1),AnodeTag))==0 ||...
                    isempty(ROIList{strcmp(ROIList(:,1),AnodeTag),2})
                GroupStruct(c).Tag1MTL = 'undef';
            else 
                GroupStruct(c).Tag1MTL = ROIList{strcmp(ROIList(:,1),AnodeTag),2};
            end
            
            % cathode ROI label
            if sum(strcmp(ROIList(:,1),CathodeTag))==0 ||...
                    isempty(ROIList{strcmp(ROIList(:,1),CathodeTag),2})
                GroupStruct(c).Tag2MTL = 'undef';
            else
                GroupStruct(c).Tag2MTL = ROIList{strcmp(ROIList(:,1),CathodeTag),2};
            end
            
            % midpoint ROI label. first check the .locTag field of the bpStruct
            if ismember([AnodeTag '-' CathodeTag],{bpStruct.tagName}) && isfield(bpStruct,'locTag')
                if isempty(bpStruct(strcmp([AnodeTag '-' CathodeTag],{bpStruct.tagName})).locTag)
                    GroupStruct(c).TagMid = 'undef';
                else
                    GroupStruct(c).TagMid = bpStruct(strcmp([AnodeTag '-' CathodeTag],{bpStruct.tagName})).locTag;
                end
                
            elseif sum(strcmp(ROIList(:,1),[AnodeTag '-' CathodeTag]))==0  ||...
                    isempty(ROIList{strcmp(ROIList(:,1),[AnodeTag '-' CathodeTag]),2})
                GroupStruct(c).TagMid = 'undef';
            else
                GroupStruct(c).TagMid = ROIList{strcmp(ROIList(:,1),[AnodeTag '-' CathodeTag]),2};
            end
            
            % x,y,z coordinates
            GroupStruct(c).Tag1_TalXYZ = get_xyz(monoStruct(strcmp({monoStruct.tagName},AnodeTag)));
            GroupStruct(c).Tag2_TalXYZ = get_xyz(monoStruct(strcmp({monoStruct.tagName},CathodeTag)));
            
            % hemisphere (assumes anode/cathode are in same hemisphere)
            Loc1Names = {monoStruct.Loc1};
            if strncmp('Left',Loc1Names{chans==Anodes(iLoc)},4)
                GroupStruct(c).StimHemisphere = 'Left';
            elseif strncmp('Right',Loc1Names{chans==Anodes(iLoc)},5)
                GroupStruct(c).StimHemisphere = 'Right';
            else
                GroupStruct(c).StimHemisphere = 'undef';
            end
            
            % electrode spacing
            if ismember([AnodeTag '-' CathodeTag],{bpStruct.tagName})
                % check if indivSurf localization is available
                if ~any(isnan(get_xyz(monoStruct(strcmp({monoStruct.tagName},AnodeTag)).indivSurf)))
                    GroupStruct(c).Spacing = sqrt(sum((get_xyz(monoStruct(strcmp({monoStruct.tagName},AnodeTag)).indivSurf)...
                        -get_xyz(monoStruct(strcmp({monoStruct.tagName},CathodeTag)).indivSurf)).^2));
                else % no indivSurf, use tal
                    GroupStruct(c).Spacing = sqrt(sum((get_xyz(monoStruct(strcmp({monoStruct.tagName},AnodeTag)))...
                        -get_xyz(monoStruct(strcmp({monoStruct.tagName},CathodeTag)))).^2));
                end
            else
                GroupStruct(c).Spacing = nan;
            end

            % amplitude
            if strcmp(Subject,'R1006P') && strcmp(Task,'RAM_YC2')
                GroupStruct(c).StimAmp = [1 .5];
            elseif strcmp(Task,'RAM_PS')
                GroupStruct(c).StimAmp = nan;
            else
                GroupStruct(c).StimAmp = unique([events([events.stimAnode]==Anodes(iLoc)).stimAmp],'stable');
            end
            
            c = c+1;
        end
        
        % hard-code TJ083 FR2 condition
        if strcmp(Subject,'TJ083') && strcmp(Task,'RAM_FR2')
            GroupStruct(c).Subject = Subject;
            GroupStruct(c).Task = Task;
            GroupStruct(c).Sessions = 0;
            GroupStruct(c).StimElecTag = 'LPH1-LPH2';
            GroupStruct(c).StimElecChans = [1 2];
            GroupStruct(c).Tag1MTL = 'EC';
            GroupStruct(c).Tag2MTL = 'EC';
            GroupStruct(c).TagMid = 'Left EC';
            GroupStruct(c).Tag1_TalXYZ = [-20 -12 -23];
            GroupStruct(c).Tag2_TalXYZ = [-26 -12 -22];
            GroupStruct(c).StimHemisphere = 'Left';
            GroupStruct(c).Spacing = 7.1414;
            GroupStruct(c).StimAmp = 1.5;
            %GroupStruct(c).MTLROI = {'EC' 'EC'};
            c = c+1;
        end
        
    catch e
        err_struct(ec).e = e;
        err_struct(ec).Subject = Subject;
        err_struct(ec).Task = Task;
        ec = ec+1;
    end

end

% set, if error output was requested
if nargout == 2
    varargout{1} = err_struct;
end
 
        
    
