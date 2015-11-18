function  [a]=SaveEventsPy(Subject, Experiment, output_dir)

%%% [a] is needed as a dummy var to avoid too many output arguments error when calling from python

a=10;

PSEvents = get_sub_events('RAM_PS',Subject);
PSEvents = PSEvents(strcmp({PSEvents.experiment},Experiment));
save(fullfile(output_dir,'PSEvents.mat'),'PSEvents')

end
