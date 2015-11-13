function  [a]=PrepareBPS(Subject, output_dir)

%%% [a] is needed as a dummy var to avois too many output arguments error when calling from python


a=10;


bpFull = getBipolarSubjElecs(Subject);
bp = getBipolarSubjElecs(Subject,1,1,1);

save(fullfile(output_dir,'bpFull.mat'),'bpFull')
save(fullfile(output_dir,'bp.mat'),'bp')



end




