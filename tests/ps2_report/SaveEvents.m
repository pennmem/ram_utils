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


% [GroupPSL,errors] = RAM_GetPSL('RAM_PS', {Subject});

% b = output_dir

% c= fullfile(output_dir,'GroupPSL.mat')

% if ~isempty(errors)
%     for i=1:length(errors)
%         fprintf(errors)
%     end
% end

% save(fullfile(output_dir,'GroupPSL.mat'),'GroupPSL')

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



