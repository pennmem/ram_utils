function  [a]=CreateParams(subject, workspace_dir)

%%% [a] is needed as a dummy var to avois too many output arguments error when calling from python

a=10;
bm_params = RAM_FR3_CreateParams(workspace_dir);
save(fullfile(workspace_dir,'bm_params.mat'),'bm_params')


end

