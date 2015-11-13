function [ROIList,bpLabels] = RAM_GetMTLROIs(Subject,bpStruct)
% function [ROIList,bpLabels] = RAM_GetMTLROIs(Subject,bpStruct)
%
% fFnction to get MTL ROI label for a given patient's channels. makes use
% of the RAM_Patient_MTLROIs.xlsx spreadsheet that is a collection of the
% info from the neurorad localization spreadsheets, along additional
% columns indicating the ROI 'label' for each contact. each contact is
% assigned a label from a set of MTL regions, or left blank if it is
% unclear or outside the core MTL regions.
%
% Inputs:
%   Subject     subject code (e.g. 'R1001P')
%   bpStruct    the given subject's bipolar talairach structure, obtained
%               from bpStruct = getBipolarSubjElecs(Subject)
%
% Outputs:
%   ROIList     a cell array in which the first column is the electrode tag
%               name, and the second column is the 'strict' ROI label (see
%               below for info on strict vs liberal).
%   bpLabels    a cell array the same length as bpStruct with the ROI label
%               for each channel of the bipolar pair
%
% List of regions:
%   CA1
%   CA2
%   CA3
%   DG      dentate gyrus
%   SUB     subiculum
%   EC      entorhinal cortex
%   AMY     amygdala
%   PRC     perirhinal cortex
%
% As of 02/09/15, labels were included for an 'ROI:strict' column, which
% tried to use a conservative threshold for including a contact in a ROI
% bin. Specifically, the Das Volumetric Atlas Location and the Comments
% should have been unambiguous that the contact was in a particular region.
%
% Last updated;
%   02/09/15    YE created function

basedir = GetBaseDir;

%cd([basedir '/RAM/']);
if isdir('/home2/yezzyat/RAM/')
    [n,s] = xlsread('/home2/yezzyat//RAM/RAM_Patient_MTLROIs.xlsx'); s(1,:) = []; % header row
else
    [n,s] = xlsread([basedir '/RAM/RAM_Patient_MTLROIs.xlsx']); s(1,:) = []; % header row
end

SubjRows = strcmp(s(:,1),Subject);

ROIList = s(SubjRows,2:3);

bpLabels = cell(length(bpStruct),2); bpLabels(:,:) = {''};
for iElec = 1:length(bpStruct)
    
    DashPos = strfind(bpStruct(iElec).tagName,'-');
    tag1 = bpStruct(iElec).tagName(1:DashPos-1);
    tag2 = bpStruct(iElec).tagName(DashPos+1:end);

    ind1 = strcmp(tag1,ROIList(:,1));
    if sum(ind1)==1
        bpLabels(iElec,1) = ROIList(ind1,2);
    end
    
    ind2 = strcmp(tag2,ROIList(:,1));
    if sum(ind2)==1
        bpLabels(iElec,2) = ROIList(ind2,2);
    end
    
end


