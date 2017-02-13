{
    "subject": "{{subject}}",
    "experiment": {
        "type": "{{experiment}}",
        "experiment_specific_data": {
            "classifier_file": "{{classifier_file}}",
            "stim_electrode_pair": "{{stim_electrode_pair}}",
            "stim_amplitude": {{stim_amplitude}},
            "stim_frequency": {{stim_frequency}},
            "stim_duration": 500,
            "random_stim_prob": false,
            "save_debug_output": true
        },
        "experiment_specs": {
            "version": "3.0.0",
            "experiment_type": "FR3",
            "biomarker_sample_start_time_offset": "0",
            "biomarker_sample_time_length": "1366",
            "buffer_time": "1000",
            "stim_duration": "500",
            "freq_min": "3",
            "freq_max": "180",
            "num_freqs": "8",
            "num_items": "300"
        }
    },
    "electrode_config_file": "{{electrode_config_file}}",
    "montage_file": "{{montage_file}}",
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