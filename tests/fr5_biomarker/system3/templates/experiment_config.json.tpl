{
    "subject": "{{subject}}",
    "experiment": {
        "type": "{{experiment}}",
        "experiment_specific_data": {
            "classifier_file": "{{classifier_file}}",
            "stim_channels": {
                {% for stim_channel,stim_params in stim_params_dict.iteritems() %}"{{stim_channel}}":
                {
                    "min_stim_amplitude": {{ stim_params['min_stim_amplitude'] }},
                    "max_stim_amplitude": {{ stim_params['max_stim_amplitude'] }},
                    "stim_frequency": {{ stim_params['stim_frequency'] }},
                    "stim_duration": {{ stim_params['stim_duration'] }}
                },
                {% end %}
            "random_stim_prob": false,
            "save_debug_output": true
        },
        "experiment_specs": {
            "version": "3.0.0",
            "experiment_type": "FR5",
            "biomarker_sample_start_time_offset": "0",
            "biomarker_sample_time_length": "1366",
            "buffer_time": "1000",
            "stim_duration": "500",
            "retrieval_biomarker_sample_start_time_offset": "0",
            "retrieval_biomarker_sample_time_length": "525",
            "retrieval_buffer_time": "524",
            "post_stim_biomarker_sample_time_length": "500",
            "post_stim_buffer_time": "499",
            "post_stim_wait_time":"100",
            "freq_min": "6",
            "freq_max": "180",
            "num_freqs": "8",
            "num_items": "300"
        }
    },
    "electrode_config_file": "{{electrode_config_file}}",
    "montage_file": "{{montage_file}}",
    "excluded_montage_file": "{{excluded_montage_file}}"
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
        }
    }
}