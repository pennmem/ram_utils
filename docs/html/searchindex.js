Search.setIndex({docnames:["classifier","cli","data","events","index","misc","pipeline"],envversion:53,filenames:["classifier.rst","cli.rst","data.rst","events.rst","index.rst","misc.rst","pipeline.rst"],objects:{"ramutils.classifier":{cross_validation:[0,0,0,"-"],utils:[0,0,0,"-"]},"ramutils.classifier.cross_validation":{perform_lolo_cross_validation:[0,1,1,""],perform_loso_cross_validation:[0,1,1,""],permuted_lolo_cross_validation:[0,1,1,""],permuted_loso_cross_validation:[0,1,1,""]},"ramutils.classifier.utils":{reload_classifier:[0,1,1,""],train_classifier:[0,1,1,""]},"ramutils.events":{add_field:[3,1,1,""],calculate_repetition_ratio:[3,1,1,""],clean_events:[3,1,1,""],coerce_study_pair_to_word_event:[3,1,1,""],combine_retrieval_events:[3,1,1,""],concatenate_events_across_experiments:[3,1,1,""],concatenate_events_for_single_experiment:[3,1,1,""],extract_event_metadata:[3,1,1,""],extract_experiment_from_events:[3,1,1,""],extract_lists:[3,1,1,""],extract_sample_rate:[3,1,1,""],extract_sessions:[3,1,1,""],extract_stim_information:[3,1,1,""],extract_subject:[3,1,1,""],find_free_time_periods:[3,1,1,""],find_subjects:[3,1,1,""],get_all_retrieval_events_mask:[3,1,1,""],get_baseline_retrieval_mask:[3,1,1,""],get_encoding_mask:[3,1,1,""],get_fr_retrieval_events_mask:[3,1,1,""],get_math_events_mask:[3,1,1,""],get_nonstim_events_mask:[3,1,1,""],get_pal_retrieval_events_mask:[3,1,1,""],get_partition_masks:[3,1,1,""],get_recall_events_mask:[3,1,1,""],get_required_columns:[3,1,1,""],get_session_mask:[3,1,1,""],get_stim_list_mask:[3,1,1,""],get_stim_table_event_mask:[3,1,1,""],get_time_between_events:[3,1,1,""],get_vocalization_mask:[3,1,1,""],get_word_event_mask:[3,1,1,""],initialize_empty_event_reccarray:[3,1,1,""],initialize_empty_stim_reccarray:[3,1,1,""],insert_baseline_retrieval_events:[3,1,1,""],load_events:[3,1,1,""],normalize_pal_events:[3,1,1,""],partition_events:[3,1,1,""],remove_bad_events:[3,1,1,""],remove_incomplete_lists:[3,1,1,""],remove_intrusions:[3,1,1,""],remove_negative_offsets:[3,1,1,""],remove_nonresponses:[3,1,1,""],remove_practice_lists:[3,1,1,""],rename_correct_to_recalled:[3,1,1,""],select_all_retrieval_events:[3,1,1,""],select_baseline_retrieval_events:[3,1,1,""],select_column_subset:[3,1,1,""],select_encoding_events:[3,1,1,""],select_math_events:[3,1,1,""],select_retrieval_events:[3,1,1,""],select_session_events:[3,1,1,""],select_stim_table_events:[3,1,1,""],select_vocalization_events:[3,1,1,""],select_word_events:[3,1,1,""],separate_stim_events:[3,1,1,""],subset_pal_events:[3,1,1,""],update_pal_retrieval_events:[3,1,1,""],update_recall_outcome_for_retrieval_events:[3,1,1,""],update_subject:[3,1,1,""],validate_single_experiment:[3,1,1,""],validate_single_session:[3,1,1,""]},"ramutils.exc":{CommandLineError:[5,2,1,""],MissingFileError:[5,2,1,""],MultistimNotAllowedException:[5,2,1,""],TooManyExperimentsError:[5,2,1,""],TooManySessionsError:[5,2,1,""],UnableToReloadClassifierException:[5,2,1,""],UnsupportedExperimentError:[5,2,1,""]},"ramutils.log":{WarningAccumulator:[5,3,1,""],get_logger:[5,1,1,""],get_warning_accumulator:[5,1,1,""]},"ramutils.log.WarningAccumulator":{format_all:[5,4,1,""]},"ramutils.parameters":{ExperimentParameters:[2,3,1,""],FRParameters:[2,3,1,""],FilePaths:[2,3,1,""],PALParameters:[2,3,1,""],StimParameters:[2,3,1,""]},"ramutils.tasks":{classifier:[6,0,0,"-"],events:[6,0,0,"-"],make_task:[6,1,1,""],misc:[6,0,0,"-"],montage:[6,0,0,"-"],odin:[6,0,0,"-"],powers:[6,0,0,"-"],summary:[6,0,0,"-"],task:[6,1,1,""]},"ramutils.tasks.classifier":{perform_cross_validation:[6,1,1,""],post_hoc_classifier_evaluation:[6,1,1,""],reload_used_classifiers:[6,1,1,""],serialize_classifier:[6,1,1,""]},"ramutils.tasks.events":{build_test_data:[6,1,1,""],build_training_data:[6,1,1,""]},"ramutils.tasks.misc":{read_index:[6,1,1,""],store_results:[6,1,1,""]},"ramutils.tasks.odin":{generate_electrode_config:[6,1,1,""],generate_ramulator_config:[6,1,1,""]},"ramutils.tasks.powers":{compute_normalized_powers:[6,1,1,""]},"ramutils.tasks.summary":{summarize_math:[6,1,1,""],summarize_nonstim_sessions:[6,1,1,""],summarize_ps_sessions:[6,1,1,""],summarize_stim_sessions:[6,1,1,""]},"ramutils.utils":{bytes_to_str:[5,1,1,""],combine_tag_names:[5,1,1,""],extract_experiment_series:[5,1,1,""],extract_subject_montage:[5,1,1,""],is_stim_experiment:[5,1,1,""],mkdir_p:[5,1,1,""],reindent_json:[5,1,1,""],safe_divide:[5,1,1,""],sanitize_comma_sep_list:[5,1,1,""],save_array_to_hdf5:[5,1,1,""],timer:[5,1,1,""],touch:[5,1,1,""]},ramutils:{constants:[5,0,0,"-"],events:[3,0,0,"-"],exc:[5,0,0,"-"],log:[5,0,0,"-"],parameters:[2,0,0,"-"],utils:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","exception","Python exception"],"3":["py","class","Python class"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:exception","3":"py:class","4":"py:method"},terms:{"boolean":[0,1,3],"byte":5,"case":[1,6],"catch":[3,5],"class":[3,4,5],"default":[0,1,2,3,5,6],"float":[0,5,6],"function":[0,3,6],"import":6,"int":[0,3,5,6],"new":[3,5],"return":[0,3,5,6],"true":[3,5,6],"try":[1,3,5,6],"while":6,ENS:6,For:[1,3],Not:3,The:[1,3,6],There:[3,6],These:[0,3,6],Use:1,Used:6,Useful:5,Uses:3,abil:6,abov:1,absolut:[1,2],accept:6,accommod:5,account:1,accumul:5,across:[0,3],activ:4,actual:[0,1,3,5,6],add:[2,3,5],add_field:3,added:3,adding:[3,6],addit:3,affect:6,after:[0,3,5],ahead:3,all:[0,1,2,3,4,5,6],all_ev:[3,6],all_pair:6,allow:6,along:1,alreadi:3,also:3,alter:3,altern:1,although:6,alwai:1,amount:3,amplitud:1,amplitudedetermin:1,analyz:3,ani:[3,6],anod:[1,5,6],appear:5,appli:[2,6],area:[2,6],area_fil:[1,2],arg:6,argument:[0,1,2,6],around:2,arrai:[0,1,2,3,5,6],array_lik:[0,6],artifact:6,arug:6,assess:6,assign:6,associ:[2,3,6],assum:[1,3,6],attempt:5,auc:0,autogener:1,automat:3,avoid:[0,3],backend:6,base:[0,2,3,6],base_path:0,baselin:3,basi:3,basic:3,bask:3,becaus:3,been:3,befor:[3,6],begin:3,behavior:[2,6],being:6,belong:3,below:6,better:3,between:[3,6],binari:6,biomark:3,bipolar:6,blank:[1,6],block:5,bool:[3,5,6],both:3,bound:3,buffer:6,build:3,build_test_data:6,build_training_data:6,built:6,bytes_to_str:5,cach:[1,6],cachedir:1,calcul:[3,6],calculate_repetition_ratio:3,call:[0,3,6],callabl:[5,6],can:[0,2,3,5,6],cannot:5,care:0,cat:3,catfr1:1,catfr3:1,catfr5:1,catfr6:1,catfr:3,cathod:[1,5,6],caus:1,chang:[0,6],channel:[1,2,6],check:[3,6],classif:2,classifi:[2,3,4,5],classifier_contain:0,classifier_summari:6,classifiercontain:[0,6],classifiersummari:6,classififercontain:6,classiflib:[0,6],clean:[3,5],clean_ev:3,clear:1,cli:5,clone:4,code:[3,5],coerc:3,coerce_study_pair_to_word_ev:3,collect:[0,3],column:3,com:4,combin:[1,3,6],combine_retrieval_ev:3,combine_tag_nam:5,come:6,comma:5,command:[4,5],commandlineerror:5,common:[2,4,6],comparison:3,complet:3,compon:6,compute_normalized_pow:6,compute_pow:6,concaten:3,concatenate_events_across_experi:3,concatenate_events_for_single_experi:3,conda:4,conf:1,config:[1,2,6],configur:[2,4,5,6],consid:3,construct:6,contact:[1,5,6],contain:[0,3,5,6],context:5,convert:[2,3,5],convolut:3,copi:0,correct:[3,6],correctli:6,correspond:[1,3,6],could:[1,5,6],count:3,crash:5,creat:[3,4,5,6],creation:[2,3],cross:[3,4,6],cross_session_summari:6,cross_valid:0,csv:[1,6],current:3,dask:6,data:[1,3,4,5,6],data_nam:5,datafram:3,dataset:[3,5],datatyp:3,debug:5,decod:5,decor:[5,6],default_area:1,default_surface_area:6,default_v:3,defin:[1,4,5],delai:6,denomin:5,depend:[3,4],describ:1,design:3,desir:3,dest:[1,2],detail:[0,6],detect:6,determin:[1,3],dict:[0,6],dictionari:[5,6],did:1,differ:[1,3,6],dimens:1,dimension:3,dir:[1,5],directori:[1,2,6],disabl:[1,6],disk:[3,6],displai:5,divis:5,doc:[1,3],document:6,doe:6,doesn:[1,3,5],don:1,done:[3,6],draw:3,due:1,durat:3,dure:[0,6],each:[0,3,6],easili:[2,3],eeg:[1,3],eegoffset:3,elaps:5,electrod:[1,2,6],electrode_config_fil:[1,2],element:5,els:4,empti:3,enabl:6,encod:[0,3,5,6],encoding_classifier_summari:6,encoding_onli:3,end:[3,5,6],end_tim:3,enough:3,ensur:[2,3,6],entri:[3,6],environ:4,epoch:3,epoch_arrai:3,equival:6,error:[1,3,5],estim:6,etc:0,evalu:[3,6],event:[0,4,5],event_list:3,everi:3,everyth:4,exampl:1,exc:5,except:[1,4],exclud:[3,6],excluded_pair:[2,6],exist:[1,3,5],exit:1,exp_param:6,expect:0,experi:[1,2,3,5,6],experiment:4,experimentparamet:[2,6],explicitli:1,extend:[1,6],extended_blank:6,extra:[3,6],extract:[3,5,6],extract_event_metadata:3,extract_experiment_from_ev:3,extract_experiment_seri:5,extract_list:3,extract_sample_r:3,extract_sess:3,extract_stim_inform:3,extract_subject:3,extract_subject_montag:5,extran:5,failur:3,fals:[3,5,6],featur:[0,6],few:2,field:3,field_nam:3,file:[0,1,2,3,4,5,6],file_typ:3,filepath:[2,6],filesystem:1,filter:3,find:[3,6],find_free_time_period:3,find_subject:3,fit:0,flag:3,flatten:3,flush:5,follow:[1,3,6],forc:1,format:[1,5,6],format_al:5,former:6,found:[0,1,3,6],fr1:1,fr3:1,fr5:[0,1],fr6:[0,1],free:[2,3],frequent:2,from:[0,1,3,5,6],frparamet:2,full:6,func:6,futur:[3,4,6],gener:[2,3,4,5,6],generate_electrode_config:6,generate_ramulator_config:6,get:[2,3],get_all_retrieval_events_mask:3,get_baseline_retrieval_mask:3,get_encoding_mask:3,get_fr_retrieval_events_mask:3,get_logg:5,get_math_events_mask:3,get_nonstim_events_mask:3,get_pal_retrieval_events_mask:3,get_partition_mask:3,get_recall_events_mask:3,get_required_column:3,get_sample_weight:[0,6],get_session_mask:3,get_stim_list_mask:3,get_stim_table_event_mask:3,get_time_between_ev:3,get_vocalization_mask:3,get_warning_accumul:5,get_word_event_mask:3,git:4,github:4,given:[1,2,3,5,6],goal:3,goe:6,graph:1,group:3,guess:6,handler:5,happen:3,hardwar:6,have:[0,3,6],hdf5:5,help:1,helper:3,here:[3,5],hoc:[3,6],homogen:3,how:[3,6],http:4,identifi:[3,5,6],includ:[3,5,6],incomplet:3,incorpor:3,indent:5,independ:6,index:[1,6],indexerror:1,indic:[3,5,6],infer:6,inform:5,inherti:2,initialize_empty_event_reccarrai:3,initialize_empty_stim_reccarrai:3,input:5,input_list:5,insert:3,insert_baseline_retrieval_ev:3,instanc:6,instead:3,intend:6,intern:0,intrus:3,is_post_stim_item:3,is_stim_experi:5,is_stim_item:3,issu:[1,6],istr:5,item:3,iter:[0,3],jacksheet:[1,2],joblib:6,joint:6,joint_report:6,json:[2,3,5,6],json_fil:5,jsonindexread:6,just:6,kept:3,keyword:[0,2,6],kwarg:[0,2,6],label:[1,6],lacd:1,lad8:1,lad8_lad9:1,lad9:1,lad:1,lah1:1,lah1_lah2:1,lah2:1,lahcd:1,latter:6,lead:1,least:3,leav:0,leftov:5,legaci:6,length:[3,6],liblinear:0,lid:1,like:[1,5,6],line:4,list:[0,3,5,6],live:1,lmcd:1,load:[0,1,3,5,6],load_ev:3,local:[1,6],locat:0,lofd:1,log:[1,4,6],log_arg:6,logger:5,logic:[1,3],logist:0,logisticregress:0,lolo:6,look:[0,1,2,3,6],loso:6,lphcd:1,lptd:1,maco:1,macro:1,mai:[0,3,5],make:[0,3],make_task:6,manag:5,mandatori:3,mani:[2,5],manner:2,manual:[1,3],map:6,mark:3,mask:3,match:[1,3],math:[3,6],math_ev:3,mathsummari:6,matrix:[0,6],max:1,max_amplitud:1,maximum:1,mean:[0,6],medtron:6,memori:3,mere:3,merg:3,messag:[1,5],meta:6,method:[0,2,3],metric:6,micro:1,might:1,mimic:5,min:1,min_amplitud:1,minimum:1,mirror:3,misc:6,miscellan:4,missingfileerror:5,mkdir:5,mkdir_p:5,mode:[5,6],model:[0,2,6],modifi:3,montag:[1,5],more:[0,2,3,6],mount:[0,1,2],mount_point:[0,6],multipl:[5,6],multisessionclassifiersummari:6,multistimnotallowedexcept:5,must:[1,3,6],n_permut:[0,6],n_session:6,name:[2,3,5,6],natur:3,ndarrai:[0,3,5,6],necessari:[3,5],need:[3,6],neg:3,neurorad:6,neutral:5,next:3,non:3,none:[0,3,5,6],normal:[3,6],normalize_pal_ev:3,note:[0,3,5,6],noth:3,nout:6,number:[0,1,3,5,6],numer:5,numpi:[3,6],object:[0,5,6],occur:3,odin:2,offset:3,one:[0,3,6],onli:[0,3,6],onset:3,oper:3,option:[0,1,6],order:[1,5,6],ordereddict:6,origin:6,other:[1,3,6],otherwis:1,out:[0,3],outcom:[0,3],output:[1,5,6],outstand:3,overrid:3,overwrit:5,packag:[2,4,5,6],pair:[2,6],pairs_data:6,pal1:[0,1],pal3:1,pal5:[1,3],pal:[3,6],palparamet:2,parallel:6,param:3,paramet:[0,3,4,5,6],part:[3,6],partial:6,particular:[0,3],partit:3,partition_ev:3,pass:[0,1,2,6],path:[1,2,5,6],peform:3,penalti:0,penalty_param:0,penalty_typ:0,pennmem:4,percent:5,perform:[0,3,5,6],perform_cross_valid:6,perform_lolo_cross_valid:0,perform_loso_cross_valid:0,period:[3,6],permut:[0,6],permuted_lolo_cross_valid:0,permuted_loso_cross_valid:0,piec:6,pipelin:4,plot:5,plu:6,point:[0,2],popul:3,possibl:[2,3,6],post:[3,6],post_hoc_classifier_evalu:6,post_stim:6,post_stim_ev:6,post_stim_pow:6,post_stim_predicted_prob:6,pow_mat:[0,6],power:[0,1],practic:3,pre:3,preced:1,predefin:6,predict:0,prerequisit:4,present:[3,5,6],preserv:5,prevent:5,primarili:[5,6],prior:6,prob:0,probabl:0,problem:0,process:[4,5,6],product:1,promin:5,proper:6,ps2:5,ps4:3,ps4_catfr5:1,ps4_event:3,ps4_fr5:1,ps_event:6,ptsa:3,purpos:5,put:6,python:4,r1347d:1,r1374t:1,racd:1,rad:1,rahcd:1,rais:[3,5,6],ram_clin:1,ram_util:4,ramul:[0,4],ramutil:[0,1,2,3,5,6],random:3,rather:[3,6],ratio:[3,6],raw:3,read:[3,6],read_index:6,reader:6,rec:3,rec_append_field:3,rec_bas:3,recal:[0,2,3],recall_ev:3,recarrai:[0,3,6],reccari:3,reccarrai:3,reconvert:3,record:6,redund:6,refer:4,regress:0,regular:0,reindent:5,reindent_json:5,rel:[1,2,3],relat:[5,6],releas:4,relev:[1,2],reload:6,reload_classifi:0,reload_used_classifi:6,remov:[3,5],remove_bad_ev:3,remove_incomplete_list:3,remove_intrus:3,remove_negative_offset:3,remove_nonrespons:3,remove_practice_list:3,rename_correct_to_recal:3,repetit:[3,6],repetition_ratio_dict:6,report:[3,5],repositori:4,repres:6,request:[3,5],requir:[3,4,5],rerun:1,respons:3,restart:6,result:[3,5,6],resum:6,retrain:6,retrained_classifi:6,retriev:[3,6],return_stim_ev:3,rhino:[0,1,2,6],rid:1,rmcd:1,rofd:1,root:[1,2,5,6],rootdir:3,rphcd:1,rptd:1,run:[1,6],safe_divid:5,same:[1,2,3],sample_weight:[0,6],sampler:3,sanitize_comma_sep_list:5,save:[1,2,5],save_array_to_hdf5:5,schema:[2,6],scheme:6,scratch:1,script:1,search:6,second:3,see:[0,6],seem:[1,3],select:3,select_all_retrieval_ev:3,select_baseline_retrieval_ev:3,select_column_subset:3,select_encoding_ev:3,select_math_ev:3,select_retrieval_ev:3,select_session_ev:3,select_stim_table_ev:3,select_vocalization_ev:3,select_word_ev:3,sens:3,sensibl:5,separ:[3,5],separate_stim_ev:3,seri:[5,6],serial:[2,6],serializ:4,serialize_classifi:6,serv:3,sesses:6,session:[0,3,5,6],session_summari:6,session_summaries_stim_t:6,sessionsummari:6,set:[0,3,6],setup:4,share:1,shell:5,should:[1,3,4,6],show:1,simpli:6,sinc:[0,1,3],singl:[0,2,3,6],site:5,size:1,skip:3,sklearn:[0,6],smatter:3,solver:0,some:6,someth:[5,6],sometim:1,sourc:[0,2,3,4,5,6],span:6,spec:6,special:5,specif:[3,6],specifi:[4,6],split:3,sshf:1,standard:3,start:[3,6],start_tim:3,state:0,step:3,stim:[1,3,5,6],stim_df:3,stim_on:3,stim_param:[3,6],stimparamet:[2,6],stimul:[1,2,3,5,6],storag:6,store:6,store_result:6,str:[0,2,3,5,6],string:5,strsuctur:3,structur:[3,4],study_pair:3,subclass:2,subjebct:6,subject:[0,1,3,5,6],subject_id:5,subsequ:6,subset:[3,6],subset_pal_ev:3,success:3,summar:6,summari:3,summarize_math:6,summarize_nonstim_sess:6,summarize_ps_sess:6,summarize_stim_sess:6,suppli:6,support:[5,6],surfac:[2,6],surrog:3,tag:6,tag_name_list:5,taken:3,target:1,target_amplitud:1,task:[0,1,2,3,4],task_ev:[3,6],templat:5,test:6,than:[1,3,6],them:[3,5],therefor:6,thi:[0,1,3,6],those:[3,6],throughout:5,time:[0,2,3,5,6],timer:5,tmp:6,too:5,toomanyexperimentserror:[5,6],toomanysessionserror:5,touch:5,train:[2,3,4,6],train_classifi:0,trait:2,traitschema:[2,6],treat:1,trial:[0,3],trivial:3,tupl:5,two:3,txt:[1,2,4],type:[0,2,3,4,6],unabl:6,unabletoreloadclassifierexcept:5,underli:5,uniqu:[1,3,6],unix:5,unsupportedexperimenterror:5,updat:[3,6],update_pal_retrieval_ev:3,update_recall_outcome_for_retrieval_ev:3,update_subject:3,uri:6,url:6,usag:4,use:[0,1,2,5,6],used:[0,1,2,3,5,6],useful:5,user:6,using:[1,2,6],usual:0,utf:5,util:[3,4],valid:[3,4,6],validate_single_experi:3,validate_single_sess:3,valu:[0,1,2,3,5,6],vector:0,version:5,via:[1,4,6],vispath:1,visual:1,vocal:3,wai:[1,3,5],warn:5,warningaccumul:5,weight:6,were:[3,5],what:[1,3],when:[0,2,3,5,6],whenev:2,where:[0,1,6],wherev:3,whether:[3,5,6],which:[1,2,3,5,6],whitespac:5,who:3,whose:3,width:5,window:2,within:3,without:3,word:[1,3,6],work:3,workaround:1,would:[1,3],wrap:[5,6],write:[1,2,6],wrong:6,yet:5,you:[0,3],zero:5,zip:6,zip_path:6},titles:["Classifier training, cross validation, and utilities","Command-line usage","Serializable data structures","Event processing","ramutils","Miscellaneous utilities","Pipelines"],titleterms:{"class":2,"function":5,area:1,classifi:[0,6],command:1,common:5,comput:6,configur:1,constant:5,content:4,cross:0,data:2,defin:[2,6],els:5,event:[3,6],everyth:5,except:5,experiment:2,gener:1,instal:4,line:1,log:5,miscellan:[5,6],montag:6,odin:6,paramet:2,pipelin:6,power:6,process:3,ramul:[1,6],ramutil:4,refer:[0,6],report:6,serializ:2,specifi:1,structur:2,summari:6,surfac:1,task:6,train:0,troubleshoot:1,type:5,usag:1,util:[0,5],valid:0}})