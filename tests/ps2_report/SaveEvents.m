function  [a]=SaveEvents(Subject, output_dir)

%%% [a] is needed as a dummy var to avois too many output arguments error when calling from python


% cd_mkdir(scratchDir);

% if isempty(gcp('nocreate'))
%     num_nodes = 25;mem = '10G';
%     open_rhino2_pool(num_nodes,mem);
% end


a=10
% load (param_file_path)
% bm_params = params


[GroupPSL,~] = RAM_GetPSL('RAM_PS', {Subject});
save(fullfile(output_dir,'GroupPSL.mat'),'GroupPSL')

PS2Events = get_sub_events('RAM_PS',Subject);
PS2Events = PS2Events(strcmp({PS2Events.experiment},'PS2'));
save(fullfile(output_dir,'PS2Events.mat'),'PS2Events')



% % bpFull = getBipolarSubjElecs(Subject);
% % bp = getBipolarSubjElecs(Subject,1,1,1);
% % nElecs = length(bp);

% EvAmps = []; EvFreqs = [];
% PostProb.Post = []; PostProb.Pre = []; PostProb.Diff = [];

% PS2Events = get_sub_events('RAM_PS',Subject);
% PS2Events = PS2Events(strcmp({PS2Events.experiment},'PS2'));

% save('PS2Events.mat','PS2Events')
% save(GroupPSL)



end



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
