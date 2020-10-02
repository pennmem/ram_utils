'''
In order to make event normalization extendable, we use a set of classes that are responsible
for individual experiments, as well as a class that handles the logic for creating
super sets of events for multiple experiments. Experiments with stimulation use the
StimNormalization in addition to the experiment normalization to retain both sets of fields.

Experiment classes may split their behaviour based on the experiment series, in which case
this should be transparent to the MultipleNormalization class, that handles the logic for 
creating super sets of events.
'''

# TODO: remap misnamed columns
name_remap = {}

class AlignEvents(EventNormalization):
    def __init__(self, normalizers):
        pass


# TODO: should this return or modify in place?
class EventNormalization(object):
    def __init__(self, events):
        self.events = events

        # tuples of dtype, default value
        self.required_columns = {}

        self.experiment = self.extract_experiment() # TODO
        self.series_num = self.extract_series(self.experiment) # TODO


    def update_subject(self):
        """ Ensure subject field is populated for all events """
        subject = extract_subject(self.events)
        self.events.subject = subject


    def normalize_columns(self):
        pass

    
    def select_column_subset(self):
        final_columns = []
        for col in self.required_columns:
            if col in events.dtype.names:
                final_columns.append(col)

        # Explicitly ask for a copy since a view is returned in numpy 1.13 and later
        self.events = events[final_columns].copy()


    def _add_field(self, field_name, default_val, dtype):
    """ Add field to the recarray

    Notes
    -----
    Converting to a dataframe, adding the field, and reconverting to a
    recarray because the rec_append_fields function in numpy doesn't seem to
    work

    """
    events_df = pd.DataFrame(self.events)
    events_df[field_name] = default_val
    orig_dtypes = build_dtype_list(events.dtype)

    # Add the given field and type to dtype list
    orig_dtypes.append((field_name, dtype))
    self.events = dataframe_to_recarray(events_df, orig_dtypes)


    def _dataframe_to_recarray(dataframe, dtypes):
        """
            Convert from dataframe to recarray maintaining the original datatypes
        """
        names = [dt[0] for dt in dtypes]
        events = dataframe.to_records(index=False)
        # Make sure that all the columns are in the correct order
        events = events[names].astype(dtypes)
        events.dtype.names = [str(name) for name in events.dtype.names]
        return events

    def _remove_incomplete_lists(self, events):
        pass

    def _remove_practice_list(self, events):
        pass

# TODO: is this a reasonable pattern?
class MultipleNormalization(EventNormalization):
    def __init__(self, normalizers):
        self.required_columns = {*cols for cols in normalizers.required_columns}

        # TODO: raise error if dictionaries of default values don't align
        # and can't easily be coerced (eg int16 vs string)

    def combine_columns(self):
        pass

class PSNormalization(EventNormalization):
    pass

class PALNormalization(EventNormalization):
    pass

class FRNormalization(EventNormalization):
    pass

class StimNormalization(EventNormalization):
    pass

class CatFRNormalization(EventNormalization):
    pass 

class RepFRNormalization(EventNormalization):
    def __init__(self):

    pass
