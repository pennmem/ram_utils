function  [a]=ComputePowersAndClassifier(subject, workspace_dir ,param_file_path)

%%% [a] is needed as a dummy var to avois too many output arguments error when calling from python

% if ~exist('scratchDir','var') || isempty(scratchDir)
%     scratchDir = '/scratch/reports/ecog/PS2';
% end

% if ~exist('reportDir','var') || isempty(reportDir)
%     reportDir = '/data10/eeg/ecogReports/PS2';
% end

% scratchDir = fullfile(scratchDir, Subject);
% cd_mkdir(scratchDir);

if isempty(gcp('nocreate'))
    num_nodes = 25;mem = '10G';
    open_rhino2_pool(num_nodes,mem);
end


a=10
load (param_file_path) % loads BM

% fprintf bm_params

% return
% bm_params = params

% fprintf 
% bm_params = RAM_FR3_CreateParams(scratchDir);
[errors.pow] = RAM_FR3_ComputePowersPy(bm_params,subject);
[errors.classifier] = RAM_FR3_ComputeClassifer(bm_params.classifier,subject);


% classifierDir = fullfile(workspace_dir, 'biomarker/L2LR/Feat_Freq');

% cd(classifierDir);


% d = dir([Subject '_RAM_FR1_*']);
% if isempty(d.name)
%     error('no classifier weights found in %s',classifierDir);
% else
%     if size(d,1)>1
%         d = d(end);
%     end
%     load(d.name);
% end

% if exist('res','var')
%     Weights = res.Weights;
% end

% save(fullfile(workspace_dir,'Weights.mat'),'Weights')



% if ~exist('classifierDir','var') || isempty(classifierDir)
%     errors = [];
%     bm_params = RAM_FR3_CreateParams(scratchDir);
%     [errors.pow] = RAM_FR3_ComputePowers(bm_params,Subject);
%     if ~isempty(errors.pow)
%         error('Unable to compute FR1 powers for %s', Subject);
%     end
%     [errors.classifier] = RAM_FR3_ComputeClassifer(bm_params.classifier,Subject);
%     if ~isempty(errors.classifier)
%         error('Unable to compute FR1 classifier for %s', Subject);
%     end
%     classifierDir = fullfile(scratchDir, 'biomarker/L2LR/Feat_Freq');
end


% % % function  [a]=py_report_wrapper_1(Subject, scratchDir, reportDir, classifierDir)

% % % %%% [a] is needed as a dummy var to avois too many output arguments error when calling from python

% % % % if ~exist('scratchDir','var') || isempty(scratchDir)
% % % %     scratchDir = '/scratch/reports/ecog/PS2';
% % % % end

% % % % if ~exist('reportDir','var') || isempty(reportDir)
% % % %     reportDir = '/data10/eeg/ecogReports/PS2';
% % % % end

% % % % scratchDir = fullfile(scratchDir, Subject);
% % % % cd_mkdir(scratchDir);

% % % if isempty(gcp('nocreate'))
% % %     num_nodes = 25;mem = '10G';
% % %     open_rhino2_pool(num_nodes,mem);
% % % end




% % % a=10
% % % bm_params = RAM_FR3_CreateParams(scratchDir);
% % % [errors.pow] = RAM_FR3_ComputePowers(bm_params,Subject);
% % % [errors.classifier] = RAM_FR3_ComputeClassifer(bm_params.classifier,Subject);



% % % % if ~exist('classifierDir','var') || isempty(classifierDir)
% % % %     errors = [];
% % % %     bm_params = RAM_FR3_CreateParams(scratchDir);
% % % %     [errors.pow] = RAM_FR3_ComputePowers(bm_params,Subject);
% % % %     if ~isempty(errors.pow)
% % % %         error('Unable to compute FR1 powers for %s', Subject);
% % % %     end
% % % %     [errors.classifier] = RAM_FR3_ComputeClassifer(bm_params.classifier,Subject);
% % % %     if ~isempty(errors.classifier)
% % % %         error('Unable to compute FR1 classifier for %s', Subject);
% % % %     end
% % % %     classifierDir = fullfile(scratchDir, 'biomarker/L2LR/Feat_Freq');
% % % end

function pool = open_rhino2_pool(num_nodes,mem)
% function pool = open_rhino2_pool(num_nodes,mem)
%
% Opens an interactive parallel pool on rhino2 (i.e. instead of using
% parmgo)
%
% Inputs:
%       num_nodes       # of processors (e.g. 10)
%       mem             memory per processor (e.g. '4G')
%
% Usage:
%       pool = open_rhino2_pool(10,'4G');
%
% Last updated
%       07/02/2015      YE

OpenPool = 0;
pool = gcp('nocreate');
TimeOut = 30; % length of IdleTimeout in minutes

if isempty(pool)
    OpenPool = 1;
elseif pool.NumWorkers ~= num_nodes
    delete(gcp('nocreate'));
    OpenPool = 1;
end

if OpenPool
   LoadRhinoProfile;
   cluster = parcluster();
   homedir = getenv('HOME');
   cluster.JobStorageLocation = fullfile(homedir, 'parruns');
   cd_mkdir(cluster.JobStorageLocation);
   cluster.CommunicatingSubmitFcn = {@communicatingSubmitFcn,mem};
   pool = parpool(cluster,num_nodes,'IdleTimeout',TimeOut);
    
end

end
