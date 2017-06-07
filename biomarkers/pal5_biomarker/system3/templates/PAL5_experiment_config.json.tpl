{
    "subject": "{{subject}}",
    "experiment": {
        "type": "{{experiment}}",
        "experiment_specific_data": {
            "classifier_file": "{{classifier_file}}",
            "stim_electrode_pair":"{{fr5_stim_channel}}",
            "stim_amplitude" : {{stim_params_dict[fr5_stim_channel]['stim_amplitude']}},
            "stim_frequency" : 200,
            "stim_duration"  : 500,
            "random_stim_prob": false,
            "save_debug_output": true
        },
        "experiment_specs": {
            "version": "3.0.0",
            "experiment_type": "{{experiment}}",
            "biomarker_sample_start_time_offset": "300",
            "biomarker_sample_time_length": "1700",
            "buffer_time": "1198",
            "stim_duration": "500",
            "freq_min": "6",
            "freq_max": "180",
            "num_freqs": "8",
            "num_items": "300"
        }
    },
    "biomarker_threshold":{{biomarker_threshold}},
    "retrieval_biomarker_threshold":{{retrieval_biomarker_threshold}},
    "electrode_config_file": "{{electrode_config_file}}",
    "montage_file": "{{montage_file}}",
    "excluded_montage_file": "{{excluded_montage_file}}",
    "global_settings": {
        "data_dir": "SET_AUTOMATICALLY_AT_A_RUNTIME",
        "experiment_config_filename": "SET_AUTOMATICALLY_AT_A_RUNTIME",
        "plot_fps": 5,
        "plot_window_length": 20000,
        "plot_update_style": "Sweeping",
        "max_session_length": 120,
        "sampling_rate": 1000,
        "odin_lib_debug_level": 0,
        "connect_to_task_laptop": true
    },
    "AUC_all_electrodes": {{auc_all_electrodes}},
    "AUC_no_stim_adjacent_electrodes": {{auc_no_stim_adjacent_electrodes}}
}
