Search.setIndex({docnames:["classifier","cli","data","events","getting_started_report_data","index","misc","pipeline"],envversion:53,filenames:["classifier.rst","cli.rst","data.rst","events.rst","getting_started_report_data.ipynb","index.rst","misc.rst","pipeline.rst"],objects:{"ramutils.classifier":{cross_validation:[0,0,0,"-"],utils:[0,0,0,"-"]},"ramutils.classifier.cross_validation":{perform_lolo_cross_validation:[0,1,1,""],perform_loso_cross_validation:[0,1,1,""],permuted_lolo_cross_validation:[0,1,1,""],permuted_loso_cross_validation:[0,1,1,""]},"ramutils.classifier.utils":{reload_classifier:[0,1,1,""],save_classifier_weights_plot:[0,1,1,""],train_classifier:[0,1,1,""]},"ramutils.events":{add_field:[3,1,1,""],build_dtype_list:[3,1,1,""],calculate_repetition_ratio:[3,1,1,""],clean_events:[3,1,1,""],coerce_study_pair_to_word_event:[3,1,1,""],combine_retrieval_events:[3,1,1,""],concatenate_events_across_experiments:[3,1,1,""],concatenate_events_for_single_experiment:[3,1,1,""],correct_fr2_stim_item_identification:[3,1,1,""],dataframe_to_recarray:[3,1,1,""],extract_event_metadata:[3,1,1,""],extract_experiment_from_events:[3,1,1,""],extract_lists:[3,1,1,""],extract_sample_rate_from_eeg:[3,1,1,""],extract_sessions:[3,1,1,""],extract_stim_information:[3,1,1,""],extract_subject:[3,1,1,""],find_free_time_periods:[3,1,1,""],find_subjects:[3,1,1,""],get_all_retrieval_events_mask:[3,1,1,""],get_baseline_retrieval_mask:[3,1,1,""],get_encoding_mask:[3,1,1,""],get_fr_retrieval_events_mask:[3,1,1,""],get_math_events_mask:[3,1,1,""],get_nonstim_events_mask:[3,1,1,""],get_pal_retrieval_events_mask:[3,1,1,""],get_partition_masks:[3,1,1,""],get_recall_events_mask:[3,1,1,""],get_required_columns:[3,1,1,""],get_session_mask:[3,1,1,""],get_stim_list_mask:[3,1,1,""],get_stim_table_event_mask:[3,1,1,""],get_time_between_events:[3,1,1,""],get_vocalization_mask:[3,1,1,""],get_word_event_mask:[3,1,1,""],initialize_empty_event_reccarray:[3,1,1,""],initialize_empty_stim_reccarray:[3,1,1,""],insert_baseline_retrieval_events:[3,1,1,""],insert_baseline_retrieval_events_deprecated:[3,1,1,""],insert_baseline_retrieval_events_logan:[3,1,1,""],load_events:[3,1,1,""],lookup_sample_rate:[3,1,1,""],normalize_pal_events:[3,1,1,""],partition_events:[3,1,1,""],remove_bad_events:[3,1,1,""],remove_incomplete_lists:[3,1,1,""],remove_intrusions:[3,1,1,""],remove_negative_offsets:[3,1,1,""],remove_nonresponses:[3,1,1,""],remove_practice_lists:[3,1,1,""],remove_session_number_offsets:[3,1,1,""],remove_voice_detection:[3,1,1,""],rename_correct_to_recalled:[3,1,1,""],select_all_retrieval_events:[3,1,1,""],select_baseline_retrieval_events:[3,1,1,""],select_column_subset:[3,1,1,""],select_encoding_events:[3,1,1,""],select_math_events:[3,1,1,""],select_retrieval_events:[3,1,1,""],select_session_events:[3,1,1,""],select_stim_table_events:[3,1,1,""],select_vocalization_events:[3,1,1,""],select_word_events:[3,1,1,""],separate_stim_events:[3,1,1,""],subset_pal_events:[3,1,1,""],update_pal_retrieval_events:[3,1,1,""],update_recall_outcome_for_retrieval_events:[3,1,1,""],update_subject:[3,1,1,""],validate_single_experiment:[3,1,1,""],validate_single_session:[3,1,1,""]},"ramutils.exc":{CommandLineError:[6,2,1,""],DataLoadingError:[6,2,1,""],MissingArgumentsError:[6,2,1,""],MissingFileError:[6,2,1,""],MultistimNotAllowedException:[6,2,1,""],RamException:[6,2,1,""],RetrievalBaselineError:[6,2,1,""],TooManyExperimentsError:[6,2,1,""],TooManySessionsError:[6,2,1,""],UnableToReloadClassifierException:[6,2,1,""],UnsupportedExperimentError:[6,2,1,""],ValidationError:[6,2,1,""]},"ramutils.log":{RamutilsFileHandler:[6,3,1,""],RamutilsStreamHandler:[6,3,1,""],get_logger:[6,1,1,""]},"ramutils.parameters":{ExperimentParameters:[2,3,1,""],FRParameters:[2,3,1,""],FilePaths:[2,3,1,""],PALParameters:[2,3,1,""],PS5Parameters:[2,3,1,""],StimParameters:[2,3,1,""]},"ramutils.pipelines.aggregated_report":{make_aggregated_report:[7,1,1,""]},"ramutils.pipelines.ramulator_config":{make_ramulator_config:[7,1,1,""]},"ramutils.pipelines.report":{make_report:[7,1,1,""]},"ramutils.reports":{summary:[2,0,0,"-"]},"ramutils.reports.summary":{CatFRSessionSummary:[2,3,1,""],ClassifierSummary:[2,3,1,""],FRSessionSummary:[2,3,1,""],FRStimSessionSummary:[2,3,1,""],MathSummary:[2,3,1,""],PSSessionSummary:[2,3,1,""],SessionSummary:[2,3,1,""],StimSessionSummary:[2,3,1,""],Summary:[2,3,1,""]},"ramutils.reports.summary.CatFRSessionSummary":{irt_between_category:[2,4,1,""],irt_within_category:[2,4,1,""],populate:[2,5,1,""],raw_repetition_ratios:[2,4,1,""],repetition_ratios:[2,4,1,""],subject_ratio:[2,4,1,""]},"ramutils.reports.summary.ClassifierSummary":{auc:[2,4,1,""],classifier_activation:[2,4,1,""],confidence_interval_median_classifier_output:[2,4,1,""],false_positive_rate:[2,4,1,""],high_tercile_diff_from_mean:[2,4,1,""],low_tercile_diff_from_mean:[2,4,1,""],median_classifier_output:[2,4,1,""],mid_tercile_diff_from_mean:[2,4,1,""],permuted_auc_values:[2,4,1,""],populate:[2,5,1,""],predicted_probabilities:[2,4,1,""],pvalue:[2,4,1,""],regions:[2,4,1,""],thresholds:[2,4,1,""],true_outcomes:[2,4,1,""],true_positive_rate:[2,4,1,""]},"ramutils.reports.summary.FRSessionSummary":{intrusion_events:[2,4,1,""],num_correct:[2,4,1,""],num_extra_list_intrusions:[2,4,1,""],num_lists:[2,4,1,""],num_prior_list_intrusions:[2,4,1,""],num_words:[2,4,1,""],percent_recalled:[2,4,1,""],populate:[2,5,1,""],serialpos_probabilities:[2,6,1,""]},"ramutils.reports.summary.FRStimSessionSummary":{combine_sessions:[2,6,1,""],delta_recall:[2,6,1,""],lists:[2,6,1,""],num_nonstim_lists:[2,6,1,""],num_stim_lists:[2,6,1,""],pre_stim_prob_recall:[2,6,1,""],prob_first_recall_by_serialpos:[2,6,1,""],prob_recall_by_serialpos:[2,6,1,""],prob_stim_by_serialpos:[2,6,1,""],recall_test_results:[2,6,1,""],recalls_by_list:[2,6,1,""],stim_columns:[2,4,1,""],stim_events_by_list:[2,6,1,""],stim_parameters:[2,6,1,""],stim_params_by_list:[2,6,1,""]},"ramutils.reports.summary.MathSummary":{events:[2,4,1,""],num_correct:[2,4,1,""],num_lists:[2,4,1,""],num_problems:[2,4,1,""],percent_correct:[2,4,1,""],populate:[2,5,1,""],problems_per_list:[2,4,1,""],session_number:[2,4,1,""],to_dataframe:[2,5,1,""],total_num_correct:[2,6,1,""],total_num_problems:[2,6,1,""],total_percent_correct:[2,6,1,""],total_problems_per_list:[2,6,1,""]},"ramutils.reports.summary.PSSessionSummary":{decision:[2,4,1,""],location_summary:[2,4,1,""]},"ramutils.reports.summary.SessionSummary":{bipolar_pairs:[2,4,1,""],events:[2,4,1,""],excluded_pairs:[2,4,1,""],experiment:[2,4,1,""],normalized_powers:[2,4,1,""],normalized_powers_plot:[2,4,1,""],num_lists:[2,4,1,""],populate:[2,5,1,""],session_datetime:[2,4,1,""],session_length:[2,4,1,""],session_number:[2,4,1,""],subject:[2,4,1,""],to_dataframe:[2,5,1,""]},"ramutils.reports.summary.StimSessionSummary":{populate:[2,5,1,""],post_stim_prob_recall:[2,4,1,""],subject:[2,4,1,""]},"ramutils.reports.summary.Summary":{create:[2,7,1,""],events:[2,4,1,""],populate:[2,5,1,""],raw_events:[2,4,1,""]},"ramutils.tasks":{classifier:[7,0,0,"-"],events:[7,0,0,"-"],make_task:[7,1,1,""],misc:[7,0,0,"-"],montage:[7,0,0,"-"],odin:[7,0,0,"-"],powers:[7,0,0,"-"],summary:[7,0,0,"-"],task:[7,1,1,""]},"ramutils.tasks.classifier":{post_hoc_classifier_evaluation:[7,1,1,""],reload_used_classifiers:[7,1,1,""],serialize_classifier:[7,1,1,""],summarize_classifier:[7,1,1,""]},"ramutils.tasks.events":{build_test_data:[7,1,1,""],build_training_data:[7,1,1,""]},"ramutils.tasks.misc":{load_existing_results:[7,1,1,""],read_index:[7,1,1,""],save_all_output:[7,1,1,""]},"ramutils.tasks.odin":{generate_electrode_config:[7,1,1,""],generate_ramulator_config:[7,1,1,""]},"ramutils.tasks.summary":{summarize_math:[7,1,1,""],summarize_nonstim_sessions:[7,1,1,""],summarize_ps_sessions:[7,1,1,""],summarize_stim_sessions:[7,1,1,""]},"ramutils.utils":{bytes_to_str:[6,1,1,""],combine_tag_names:[6,1,1,""],encode_file:[6,1,1,""],extract_experiment_series:[6,1,1,""],extract_report_info_from_path:[6,1,1,""],extract_subject_montage:[6,1,1,""],get_completed_sessions:[6,1,1,""],get_session_str:[6,1,1,""],is_stim_experiment:[6,1,1,""],load_event_test_data:[6,1,1,""],mkdir_p:[6,1,1,""],reindent_json:[6,1,1,""],safe_divide:[6,1,1,""],sanitize_comma_sep_list:[6,1,1,""],save_array_to_hdf5:[6,1,1,""],show_log_handlers:[6,1,1,""],tempdir:[6,1,1,""],timer:[6,1,1,""],touch:[6,1,1,""]},ramutils:{constants:[6,0,0,"-"],events:[3,0,0,"-"],exc:[6,0,0,"-"],log:[6,0,0,"-"],parameters:[2,0,0,"-"],utils:[6,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","exception","Python exception"],"3":["py","class","Python class"],"4":["py","attribute","Python attribute"],"5":["py","method","Python method"],"6":["py","staticmethod","Python static method"],"7":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:exception","3":"py:class","4":"py:attribute","5":"py:method","6":"py:staticmethod","7":"py:classmethod"},terms:{"0x1c76ae7400":4,"0x1cb803f7b8":4,"abstract":2,"boolean":[0,1,2,3],"byte":6,"case":[1,2,4,6,7],"catch":[3,6],"class":[3,4,5,6,7],"default":[0,1,2,3,4,6,7],"float":[0,2,4,6,7],"function":[0,3,4,7],"import":[4,7],"int":[0,2,3,6,7],"long":7,"new":[2,3,4],"return":[0,2,3,4,6,7],"static":[2,7],"true":[2,3,4,7],"try":[1,3,4,6,7],"while":7,ENS:7,For:[1,2,3,4,7],Not:3,One:4,The:[0,1,2,3,4,7],Their:2,There:[3,4],These:[0,3,4,7],Use:[1,7],Used:[2,6,7],Useful:6,Uses:3,With:4,_ax:4,_event:4,_predicted_prob:4,_subplot:4,_wrapper:4,abil:[4,7],abl:7,about:[2,4],abov:1,absolut:[1,2,7],accept:7,access:6,accommod:6,account:[1,4],across:[0,3,5,7],activ:2,actual:[0,1,2,3,4,6,7],add:[2,3,4],add_field:3,add_loc:3,added:[3,4],adding:[3,6,7],addit:[2,3],advers:4,affect:7,after:[0,2,3],again:4,agg:4,agg_report:7,aggreg:[5,7],aggregated_report:7,ahead:3,algorithm:[2,3],all:[0,1,2,3,4,6,7],all_ev:[3,7],all_pair:7,all_relev:3,allow:[1,4,6,7],almost:4,along:1,alreadi:[3,4,7],also:[2,3,7],alter:3,altern:[1,4],although:7,alwai:1,amiss:2,amount:[2,3],amplitud:[1,4],amplitudedetermin:1,analys:[2,4],analysi:[2,4],analyz:[3,4],ani:[3,4,6,7],anlays:4,anod:[1,4,6,7],anoth:4,answer:[2,4],api:4,appear:6,append:[4,7],appli:[2,4,7],approach:[4,7],arbitrari:6,area:[2,7],area_fil:[1,2],arg:[6,7],argument:[0,1,2,3,6,7],around:[2,4],arrai:[0,1,2,3,4,6],array_lik:[0,2,3,7],artifact:[4,7],artifactu:4,arug:7,ask:4,assert:4,assess:[2,4,7],assign:7,associ:[2,3,4,7],assum:[1,2,3,4,7],assumpt:4,attempt:[6,7],attribut:2,auc:[0,2,4],autogener:1,automat:[3,7],avail:[4,7],averag:[2,4],avoid:[0,3],awai:4,awar:2,axes:4,axessubplot:4,axisgrid:4,base64:[2,6],base:[0,2,3,4,6,7],base_path:0,baselin:[3,4,6],basi:3,basic:[3,4],bask:3,bayesian:[2,7],bear:4,becaus:[3,4,7],becom:4,been:[3,4,7],befor:[3,4,7],begin:3,behavior:[2,4,7],behavioral_result:7,being:7,belong:3,below:[2,7],benefit:4,best:4,better:3,between:[2,3,7],biggest:4,binari:[2,7],biomark:[2,3],bipolar:[1,2,7],bipolar_pair:[2,7],bird:3,blank:[1,7],block:6,bool:[2,3,6,7],both:[3,4,7],bound:3,broad:4,build:[3,5,7],build_dtype_list:3,build_test_data:7,build_training_data:7,built:[4,7],bytes_to_str:6,cach:[1,2,7],cachedir:1,calcul:[2,3,4,7],calculate_repetition_ratio:3,call:[0,2,3,4,7],callabl:[6,7],can:[0,2,3,4,6,7],candid:3,cannot:[4,6],capabl:7,care:0,cat:3,categor:[2,3],categori:2,catfr1:1,catfr3:[1,7],catfr5:[1,4,7],catfr6:1,catfr:[1,2,3,7],catfrsessionsummari:2,cathod:[1,4,6,7],caudalmiddlefront:4,caus:[1,3],center:2,cerebr:4,certain:6,chang:[0,2,4,7],channel:[1,2,4,7],check:[3,4,6,7],chi:2,child:2,classif:2,classifi:[1,2,3,5,6],classifier_:7,classifier_activ:2,classifier_contain:[0,4],classifier_data:4,classifier_df:4,classifier_evaluation_result:[4,7],classifier_excluded_lead:7,classifier_info:4,classifier_loc:4,classifier_output:4,classifier_sess:4,classifier_summari:[2,4,7],classifier_summary_data:4,classifiercontain:[0,4,7],classifiersummari:[2,4,7],classififercontain:7,classiflib:[0,4,7],classmethod:2,clean:[3,6],clean_ev:3,clear:1,cli:6,code:[3,4,6],coeffici:2,coerc:3,coerce_study_pair_to_word_ev:3,cohort:4,collect:[0,3],column:[3,4],combin:[1,2,3,5,6,7],combine_retrieval_ev:3,combine_sess:[2,4],combine_tag_nam:6,combined_df:4,combined_item_df:4,come:[4,7],comma:6,command:[5,6],commandlineerror:6,common:[1,2,4,5,7],compar:4,comparison:[2,3],complet:[2,3,4,6,7],complic:4,compon:6,comprehens:4,comput:[2,4,5],concaten:3,concatenate_events_across_experi:3,concatenate_events_for_single_experi:3,concept:7,concern:4,conda:5,condit:4,conduct:[2,4],conf:[1,7],confid:2,confidence_interval_median_classifier_output:2,config:[1,2,7],configur:[2,5,6,7],consid:3,consist:4,construct:7,contact:[1,6,7],contain:[0,2,3,4,6,7],content:[5,6,7],context:6,continu:4,convent:[2,4,6],convert:[2,3,6],convolut:3,copi:0,core:4,correct:[2,3,7],correct_fr2_stim_item_identif:3,correctli:[2,3,4,7],correl:4,correspond:[1,2,3,4,7],could:[1,4,6,7],count:[3,4],cover:4,crash:6,creat:[2,3,4,6,7],creation:[2,3],cross:[3,4,5,7],cross_session_summari:7,cross_valid:0,csv:[1,2,7],current:[2,3,7],curv:2,custom:[4,6],dask:7,data:[1,3,5,6,7],data_db:2,data_nam:6,data_typ:[2,7],databas:[4,7],datafram:[2,3,4,7],dataframe_to_recarrai:3,dataloadingerror:6,datapath:6,dataset:[3,5,6],datatyp:3,date:4,datetim:2,db_loc:[4,7],dboy1:1,deal:4,debug:6,decis:2,decod:6,decor:[6,7],deduc:4,def:4,default_area:1,default_surface_area:7,default_v:3,defin:[1,4,5,6],delai:7,delta:4,delta_recal:[2,4],denomin:6,densiti:4,depend:3,deprec:4,deriv:7,describ:[0,1],design:3,desir:3,despit:4,dest:[1,2],destin:7,detail:[0,7],detect:[4,7],determin:[1,3],develop:4,dict:[0,2,7],dictionari:[2,4,6,7],did:[1,4],differ:[1,3,5,7],differenti:[2,4],dimens:1,dimension:3,dir:[1,6],directori:[1,2,4,6,7],disabl:[1,7],disk:[1,2,3,7],displai:6,dissimilar:7,distplot:4,distractor:2,divis:6,doc:[1,3],document:[2,4,7],doe:[4,7],doesn:[1,3,4,6],doing:4,don:[1,4],done:[2,3,4,7],door:4,doubter:4,draw:3,dtype:3,due:[1,4],dump:2,duplic:4,durat:3,dure:[0,2,4,6,7],each:[0,2,3,4,7],easi:4,easier:2,easili:[2,3,4],eeg:[1,3],eegfil:6,eegoffset:3,effect:[2,7],either:[0,2,4],elaps:6,electrod:[1,2,7],electrode_config_fil:[1,2],electrode_typ:4,electrophysiolog:4,element:[3,6],els:5,emb:[4,7],embed:4,empti:[3,7],enabl:7,encod:[0,2,3,4,6,7],encode_fil:6,encoding_classifier_summari:7,encoding_onli:3,end:[2,3,4],end_tim:3,enhanc:7,enough:3,enrich:4,ensur:[2,3,7],entri:[3,7],env:4,epoch:3,epoch_arrai:3,error:[1,3,6],estim:7,etc:[0,2,4,7],evalu:[3,4,7],even:4,event:[0,2,4,5,6],event_list:3,eventu:[2,7],everi:[3,4],everyth:[5,7],exactli:4,exampl:[1,2,4,6,7],exc:6,except:[1,5],excit:4,exclud:[1,2,3,4,7],excluded_contact:1,excluded_pair:[2,7],exist:[1,3,5,6],exit:1,exp_param:7,expect:[0,4,6],experi:[1,3,4,5,6,7],experiment:[2,4],experimentparamet:[2,7],explicitli:1,exploit:6,extend:[1,2,4,7],extended_blank:7,extra:[2,3,7],extract:[3,5,6,7],extract_event_metadata:3,extract_experiment_from_ev:3,extract_experiment_seri:6,extract_list:3,extract_report_info_from_path:[4,6],extract_sample_rate_from_eeg:3,extract_sess:3,extract_stim_inform:3,extract_subject:3,extract_subject_montag:6,extran:6,fact:4,fail:6,failur:3,fals:[2,3,4,6,7],false_positive_r:2,far:4,featur:[0,2,4,7],few:2,field:[2,3,4,6],field_nam:3,field_type_str:3,figur:6,file:[0,1,2,3,5,6,7],file_:0,file_nam:4,file_path:6,file_typ:[2,3,4,7],filenam:[2,6,7],filepath:[2,7],filesystem:1,filter:3,find:[3,5,7],find_free_time_period:3,find_subject:3,first:[2,4,7],fit:[0,7],fit_model:7,flag:3,flatten:3,follow:[1,2,3,7],forc:[1,2],forg:5,form:3,format:[1,2,4,6,7],former:7,formerli:7,forward:2,found:[0,1,2,3,4,7],fr1:1,fr2:[3,7],fr3:[1,7],fr5:[0,1,4,7],fr5_session_summary_loc:4,fr6:[0,1],free:[2,3,4],frequenc:[0,2,4],frequent:2,friend:4,from:[0,1,2,3,5,6,7],from_dict:4,from_hdf:4,from_record:4,frparamet:2,frsessionsummari:2,frstimsessionsummari:[2,4],full:7,fulli:3,func:7,futur:[3,7],gener:[2,3,4,5,6,7],generate_electrode_config:7,generate_ramulator_config:7,get:[2,3,5,6],get_all_retrieval_events_mask:3,get_baseline_retrieval_mask:3,get_completed_sess:6,get_encoding_mask:3,get_fr_retrieval_events_mask:3,get_logg:6,get_math_events_mask:3,get_nonstim_events_mask:3,get_pal_retrieval_events_mask:3,get_partition_mask:3,get_recall_events_mask:3,get_required_column:3,get_sample_weight:[0,7],get_session_mask:3,get_session_str:6,get_stim_list_mask:3,get_stim_table_event_mask:3,get_time_between_ev:3,get_vocalization_mask:3,get_word_event_mask:3,give:6,given:[1,2,3,4,6,7],glob:4,goal:3,goe:[6,7],going:4,gone:4,good:4,graph:[1,7],great:4,group:[3,4],groupbi:4,guess:7,guid:4,gyru:4,handi:4,handler:6,happen:3,hard:2,hardwar:7,has:[4,6],hauf:2,have:[0,3,4,7],hdf5:6,head:4,help:1,helper:[2,3,4,7],helpfer:4,here:[3,6],hierach:7,high:4,high_tercile_diff_from_mean:2,highest:2,hmm_result:4,hoc:[3,4,7],homogen:3,how:[2,3,4],howev:[4,7],hstack:4,html:7,idea:4,ident:7,identif:3,identifi:[2,3,6,7],imag:7,implicitli:4,includ:[1,2,3,4,6,7],incomplet:[3,4],incorpor:3,increas:7,indent:6,independ:7,index:[1,7],indexerror:1,indic:[2,3,4,6,7],infer:7,inferiortempor:4,inform:[2,4,6],inherti:2,initialize_empty_event_reccarrai:3,initialize_empty_stim_reccarrai:3,inlin:4,input:[2,4,6,7],input_list:6,insert:3,insert_baseline_retrieval_ev:3,insert_baseline_retrieval_events_deprec:3,insert_baseline_retrieval_events_logan:3,insid:6,instanc:2,instead:[1,3,4,7],intent:4,interact:4,interest:4,intern:[0,4],interv:2,intrus:[2,3],intrusion_ev:2,invalid:6,irt_between_categori:2,irt_within_categori:2,is_post_stim_item:[3,4],is_stim_experi:6,is_stim_item:[3,4],is_stim_list:4,isnul:4,isol:4,issu:[1,7],istr:6,item:[2,3,5,7],iter:[0,3],its:[4,6],itself:[4,7],jacksheet:[1,2],joblib:7,join:4,joint:[1,7],joint_report:7,jointgrid:4,jointplot:4,json:[2,3,6,7],json_fil:6,jsonindexread:7,just:[2,4],kei:[2,4,7],kept:3,keyword:[0,2,3,7],kind:4,know:4,known:7,kwarg:[0,2,4,6,7],la1_la2:1,label:[1,7],lacd:1,lad8:1,lad8_lad9:1,lad9:1,lad:1,lah1:1,lah1_lah2:1,lah2:1,lahcd:1,lambda:4,last:2,latter:7,lazili:4,lead:1,least:[2,3],leav:[0,4],left:4,leftov:6,legaci:7,len:[0,2,4],length:[3,7],less:4,let:4,level:5,lib:4,liblinear:0,lid:1,like:[0,1,2,6],line:[4,5],linger:4,list:[0,2,3,4,6,7],live:1,lmcd:1,load:[0,1,3,5,6,7],load_ev:3,load_event_test_data:6,load_existing_result:[2,4,7],local:[1,7],locat:[0,2,4,6,7],location_summari:2,lofd:1,log:[1,5,7],log_arg:7,logger:6,logic:[1,3],logist:0,logisticregress:0,lolo:7,longer:4,look:[0,1,2,3,4,7],lookup_sample_r:3,loso:7,lot:4,low:2,low_tercile_diff_from_mean:2,lowest:2,lphcd:1,lptd:1,luckili:4,maco:1,macro:1,magic:5,mai:[0,3,4,6,7],main:[4,7],maintain:3,make:[0,3,4,7],make_aggregated_report:7,make_ramulator_config:7,make_report:7,make_task:7,manag:6,mandatori:3,mani:[2,4,6],manipul:[2,4,6],manner:2,manual:[1,3],map:7,mark:3,mask:3,match:[1,3,4],math:[2,3,4,7],math_data:4,math_ev:3,math_summari:[2,4,7],mathsummari:[2,4,7],matplotlib:4,matrix:[0,2,4,7],matter:4,max:[1,4],max_amplitud:1,maximum:1,mcg:4,mean:[0,2,4,7],measur:4,mechan:4,median:2,median_classifier_output:2,medtron:7,memori:[3,7],mere:3,merg:[3,4],messag:[1,6],meta:7,metadata:[2,5],method:[0,2,3,4],metric:[2,4,7],micro:1,mid_tercile_diff_from_mean:2,middl:[2,4],middletempor:4,might:1,mimic:6,min:[1,4],min_amplitud:1,miniconda3:4,minimum:1,mirror:[3,7],misc:[2,4,7],miscellan:5,miss:[3,4],missingargumentserror:6,missingfileerror:6,mkdir:6,mkdir_p:6,mode:[6,7],model:[0,2,7],model_metadata:2,modifi:[3,6],montag:[1,2,4,5,6],more:[0,2,3,4,7],morletwaveletfilt:4,morletwaveletfiltercpplegaci:4,morletwavelettransform:4,most:[2,4],mount:[0,1,2,7],mount_point:[0,4,7],move:7,mstime:4,much:4,multilvel:7,multipl:[2,5,6,7],multisessionclassifiersummari:7,multistimnotallowedexcept:6,must:[1,2,3,7],n_bipolar_pair:2,n_event:[2,4],n_featur:4,n_frequenc:2,n_permut:[0,7],n_session:7,name:[2,3,4,6,7],nan:4,nanmean:4,natur:[3,7],nauc:4,ndarrai:[0,2,3,6,7],necessari:[2,3,7],need:[3,4,7],neg:3,neurorad:7,neutral:6,next:[3,4],nice:4,non:[2,3,4,7],none:[0,2,3,6,7],nonstim:2,nonstim_low_bio_recal:4,norm:4,normal:[2,3,4,7],normalize_pal_ev:3,normalized_delta_recal:4,normalized_item_delta_recal:4,normalized_pow:[2,7],normalized_powers_plot:2,normalzi:2,note:[0,2,3,4,6,7],noth:3,notic:4,nout:7,now:4,nsession:4,nsubject:4,num_correct:2,num_extra_list_intrus:2,num_list:2,num_nonstim_list:2,num_prior_list_intrus:2,num_problem:2,num_stim_list:2,num_word:2,number:[0,1,2,3,4,6,7],numer:6,numpi:[2,3,4],obj:4,object:[0,2,3,4,6,7],occur:[3,4],odin:[2,5],off:1,offset:3,oftentim:4,older:4,onc:4,one:[0,2,3,4,7],onli:[0,2,3,4,7],onset:3,open:4,oper:[3,4],optim:2,option:[0,1,2,6,7],order:[1,4,6,7],ordereddict:7,origin:[2,3,7],other:[1,3,4,7],otherwis:[1,2],our:4,out:[0,3,4,6,7],out_of_sample_pow:4,outcom:[0,3],output:[1,2,6,7],outstand:3,over:[2,4,7],overal:2,overall_recal:4,overrid:[3,7],overriden:2,overwrit:6,own:4,packag:[2,4,6,7],pair:[0,1,2,6,7],pairs_data:7,pal1:[0,1],pal3:1,pal5:[1,3],pal:3,palparamet:2,panda:5,parallel:7,param:[3,6],paramet:[0,3,4,5,6,7],paramt:7,pars:6,parstriangulari:4,part:[2,3,4,7],partial:[2,7],particular:[0,2,3,4,7],partit:3,partition_ev:3,pass:[0,1,2,6,7],path:[0,1,2,5,6,7],pattern:4,peform:3,penalti:0,penalty_param:0,penalty_typ:0,pennmem:5,per:[2,4],percent:[2,6],percent_correct:[2,4],percent_recal:2,percentag:2,perform:[0,2,3,4,6,7],perform_lolo_cross_valid:0,perform_loso_cross_valid:0,period:[2,3,6,7],perman:2,permut:[0,2,7],permuted_auc_valu:2,permuted_lolo_cross_valid:0,permuted_loso_cross_valid:0,phase:4,piec:4,pipelin:[4,5],pipeline_nam:7,plot:[2,4,6,7],plu:7,point:[0,2,4,7],popul:[2,3],pose:4,posit:2,possibl:[2,3,4,7],post:[2,3,4,7],post_hoc_classifier_evalu:7,post_stim:7,post_stim_eeg:[2,7],post_stim_ev:7,post_stim_item:[2,3],post_stim_pow:7,post_stim_predicted_prob:7,post_stim_prob_recal:2,post_stim_trigger_output:7,potenti:4,pow_mat:[0,7],power:[0,1,2,4,5],practic:[2,3],pre:[2,3],pre_stim_prob_recal:2,preced:1,precentr:4,predefin:7,predict:[0,2,4],predict_proba:4,predicted_prob:[2,4],present:[3,7],preserv:6,pretti:4,prevent:6,previou:1,previous:[4,7],primarili:[6,7],print:[4,6],prior:2,prob:0,prob_first_recall_by_serialpo:2,prob_recall_by_serialpo:2,prob_stim_by_serialpo:2,probabl:[0,2,4],problem:[0,2,4],problems_per_list:2,procedur:3,proceed:4,process:[4,5,6,7],produc:[4,6,7],product:1,program:1,proper:7,properti:[2,4],provid:7,ps2:6,ps4:3,ps4_catfr5:1,ps4_event:3,ps4_fr5:1,ps5:[2,6,7],ps5_catfr:1,ps5_fr:1,ps5paramet:2,ps_event:7,pssessionsummari:2,ptsa:3,pulse_freq:4,purpos:[6,7],pvalu:[2,4],python3:4,python:4,question:4,quick:4,quickli:4,r1275:4,r1275d:4,r127:4,r1292:4,r1292e:4,r129:4,r1304:4,r1304n:4,r1308:4,r1308t:4,r130:4,r1315t:4,r131:4,r1347d:1,r1374t:1,r1387e:4,r1387e_catfr5_all_retrained_classifi:4,r1389j:4,r1xxx:2,r1xxx_xyz_1_2_3_target_selection_t:2,racd:1,rad:1,rahcd:1,rais:[3,6,7],ram:[1,5],ram_clin:1,ramexcept:6,ramul:[0,4,5],ramulator_config:7,ramutil:[0,1,2,3,4,6,7],ramutils_dev:4,ramutils_test:4,ramutilsfilehandl:6,ramutilsstreamhandl:6,random:[3,4],rate:[2,3,4],rather:[1,3,4,7],ratio:[2,3,7],raw:3,raw_ev:2,raw_repetition_ratio:2,read:[1,3,4,7],read_index:7,reader:7,realli:4,rebuild:2,rec:[2,3],rec_append_field:3,rec_bas:3,recaarrai:3,recal:[0,2,3,4],recall_ev:3,recall_prob:2,recall_test_result:2,recalls_by_list:2,recarrai:[0,2,3,4,7],reccari:3,reccarrai:3,recent:4,reconvert:3,record:[2,3,4,7],recreat:2,redund:7,refer:[1,5],referenc:7,reg:4,region:[2,4],regress:0,regular:0,reindent:6,reindent_json:6,rel:[1,2,3,4,7],relat:[2,6,7],relev:[1,2],reli:4,reload:[2,5,7],reload_classifi:[0,4],reload_used_classifi:7,remaind:4,remov:[3,6],remove_bad_ev:3,remove_incomplete_list:3,remove_intrus:3,remove_negative_offset:3,remove_nonrespons:3,remove_practice_list:3,remove_session_number_offset:3,remove_voice_detect:3,rename_correct_to_recal:3,repeat:4,repetit:[2,3,7],repetition_ratio:2,repetition_ratio_dict:[2,7],replac:4,replic:4,report:[2,3,5,6],report_databas:[4,7],report_db_loc:4,report_path:7,repres:[4,7],represent:7,reproduc:4,request:[3,7],requir:[3,4,6,7],requri:4,rerun:[1,7],reset_index:4,resid:4,respons:[2,3],restart:7,result:[1,2,3,4,6,7],results_dict:4,results_fil:7,resum:7,retrain:[1,4,7],retrained_classifi:[4,7],retrained_classifier_loc:4,retriev:[3,6],retrievalbaselineerror:6,return_stim_ev:3,reus:1,rhino:[0,1,2,4,6,7],rhino_root:4,rid:1,right:4,rmcd:1,rofd:1,rom:6,root:[1,2,6,7],rootdir:[3,4,6,7],rostralmiddlefront:4,rotat:6,routin:4,row:4,rphcd:1,rptd:1,run:[1,4,6,7],runtim:6,safe_divid:6,sake:4,same:[1,2,3,4,7],sampl:[3,4],sample_weight:[0,7],sampler:3,sanitize_comma_sep_list:6,save:[1,2,4,6,7],save_all_output:[2,7],save_array_to_hdf5:6,save_classifier_weights_plot:0,save_loc:7,saved_result:7,schema:2,scheme:7,scratch:[1,2,4,7],script:1,seaborn:4,search:[2,7],second:[2,3,4],section:4,see:[0,2,4,7],seem:[1,3],select:[2,3,4,7],select_all_retrieval_ev:3,select_baseline_retrieval_ev:3,select_column_subset:3,select_encoding_ev:3,select_math_ev:3,select_retrieval_ev:3,select_session_ev:3,select_stim_table_ev:3,select_vocalization_ev:3,select_word_ev:3,sens:[3,7],sensibl:[6,7],separ:[1,2,3,6],separate_stim_ev:3,seraializ:4,seri:[4,6,7],serial:[2,7],serializ:5,serialize_classifi:7,serialpo:4,serialpos_prob:2,serv:3,sesses:7,session:[0,1,2,3,4,6,7],session_data:4,session_datetim:[2,4],session_df:4,session_length:2,session_list:6,session_numb:2,session_summari:[2,4,7],session_summary_loc:4,sessions_summari:7,sessionsummari:[2,7],set:[0,2,3,4,7],shape:[2,4],share:1,shell:6,should:[0,1,2,3,4,7],shouldn:6,show:1,show_log_handl:6,simpl:4,simpli:7,sinc:[0,1,3,4,7],singl:[0,2,3,5,7],site:[4,6],size:[1,4],skip:3,sklearn:[0,7],slightli:4,slow:7,smatter:3,sniff:2,sns:4,solv:2,solver:0,some:[0,4,7],someth:[2,6,7],sometim:1,sort:4,sound:4,sourc:[0,2,3,6,7],span:7,special:3,specif:[2,3,4,7],specifi:[2,6,7],split:[3,4],squar:2,sshf:1,standard:[2,3,4],start:[3,4],start_tim:3,state:0,statist:4,statrt:5,statu:7,step:3,still:4,stim:[1,2,3,4,6,7],stim_column:2,stim_df:3,stim_dur:4,stim_events_by_list:2,stim_item:3,stim_items_onli:2,stim_list_onli:2,stim_on:3,stim_param:[3,4,7],stim_param_df:3,stim_paramet:[2,4],stim_params_by_list:2,stim_recal:4,stim_report:[4,7],stimanodetag:4,stimcathodetag:4,stimparamet:[2,7],stimsessionsummari:2,stimul:[1,2,3,4,6,7],stop:4,storag:[4,7],store:[2,4,6,7],str:[0,2,3,6,7],straightforward:4,strang:4,stream:6,strict:2,string:[0,2,6],strsuctur:3,structur:[3,4,5],study_pair:3,subclass:2,subjebct:7,subject:[0,1,2,3,4,6,7],subject_df:4,subject_id:6,subject_ratio:2,subsequ:[3,7],subset:[3,7],subset_pal_ev:3,success:3,suit:6,summar:[2,7],summari:[2,3,5],summarize_classifi:7,summarize_math:7,summarize_nonstim_sess:7,summarize_ps_sess:7,summarize_stim_sess:7,summary_dir:4,summary_loc:4,superiorfront:4,superiortempor:4,suppli:7,support:[6,7],surfac:[2,7],surrog:3,swath:4,tabl:[2,3,4,7],tag:[2,7],tag_name_list:6,take:4,taken:3,target:[1,2,4,7],target_amplitud:1,target_selection_t:[2,4,7],task:[0,1,2,3,4,5],task_ev:[3,7],tempdir:6,templat:[6,7],tempor:4,temporari:6,tercil:2,term:7,test:[2,6,7],than:[1,3,4,7],thei:[3,4],them:3,therefor:[3,7],thi:[0,1,2,3,4,7],thing:4,those:[3,4,7],three:4,thresh:4,threshold:2,through:4,throughout:6,time:[0,2,3,4,6],timer:6,timezon:2,to_datafram:2,togeth:4,too:6,toomanyexperimentserror:[6,7],toomanysessionserror:6,total:2,total_num_correct:2,total_num_problem:2,total_percent_correct:2,total_problems_per_list:2,touch:6,trace:7,train:[2,3,5,7],train_classifi:0,trained_classifier_df:4,training_auc:4,training_classifier_summary_data:4,trait:2,traitschema:2,transform:4,treat:1,trial:[0,3],trigger:[1,6,7],trigger_electrod:7,trigger_output:7,trigger_pair:[1,7],trivial:[3,4],true_outcom:2,true_positive_r:2,tupl:[3,6],tutori:4,two:[3,4],txt:[1,2,7],type:[0,2,3,5,7],typic:7,unabl:[6,7],unabletoreloadclassifierexcept:6,underli:[4,5,6],underscor:[1,2],undo:3,unfortun:4,uniqu:[1,2,3,7],unit:2,unix:6,unless:4,unsupportedexperimenterror:6,updat:[3,7],update_pal_retrieval_ev:3,update_recall_outcome_for_retrieval_ev:3,update_subject:3,upon:6,usag:[4,5],use:[0,1,2,4,6,7],use_classifier_excluded_lead:7,use_common_refer:7,use_deprec:3,use_retrain:7,used:[0,1,2,3,4,6,7],useful:[2,4,6,7],user:[4,7],userwarn:4,uses:4,using:[1,2,4,7],usual:0,utc:2,utf:6,util:[3,4,5],valid:[3,5,6,7],validate_single_experi:3,validate_single_sess:3,validationerror:6,valu:[0,1,2,3,4,6,7],variabl:4,varianc:2,variou:4,vector:0,ver:2,veri:7,version:[1,4,6,7],versu:2,via:[1,7],vispath:[1,7],visual:[0,1,7],vocal:3,volum:4,wai:[0,1,3,4,6],want:[3,4],warn:4,weight:[0,2,7],well:4,were:[2,3,4],what:[1,3,4,6],when:[0,2,3,4,6,7],whenev:2,where:[0,1,2,3,4,7],wherev:3,whether:[2,3,6,7],which:[0,1,2,3,4,6,7],white:4,whitespac:6,who:[3,7],whose:[2,3,4,7],wide:4,wider:4,width:6,wil:2,wild:4,window:2,within:[2,3],without:3,word:[1,2,3,4],work:[3,4,6],workaround:1,world:4,worth:[4,6],would:[1,2,3,4,7],wrap:[6,7],write:[1,2,4],wrong:[6,7],xyz:2,yet:6,you:[0,3,4],your:4,zduei:4,zero:6,zip:7,zip_path:7},titles:["Classifier training, cross validation, and utilities","Command-line usage","Serializable data structures","Event processing","Getting Statrted: RAM Report Data","ramutils","Miscellaneous utilities","Pipelines"],titleterms:{"class":2,"function":6,across:4,aggreg:4,area:1,build:4,classifi:[0,4,7],combin:4,command:1,common:6,comput:7,configur:1,constant:6,cross:0,data:[2,4],dataset:4,defin:[2,7],differ:4,els:6,event:[3,7],everyth:6,except:6,exist:7,experi:2,extract:4,file:4,find:4,from:4,gener:1,get:4,instal:5,item:4,level:4,line:1,load:4,log:6,magic:4,metadata:4,miscellan:[6,7],montag:7,multipl:4,odin:7,panda:4,paramet:2,path:4,pipelin:7,power:7,process:3,ram:4,ramul:[1,7],ramutil:5,refer:[0,7],reload:4,report:[1,4,7],serializ:2,singl:4,specifi:1,statrt:4,structur:2,summari:[4,7],surfac:1,task:7,train:[0,4],troubleshoot:1,type:[4,6],underli:2,usag:1,util:[0,6],valid:0}})