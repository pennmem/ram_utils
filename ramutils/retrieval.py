"""RetrievalCreationHelper.py, author = LoganF, updated May 16, 2018

A script used to create retrieval events and their corresponding "deliberation" matches. A deliberation match being
a point in time in a different list where a subject is actively trying, but failing to recall a word.

Works using data formatted for the Kahana Computational Memory Lab.

The code has been built for functionality with all RAM data, scalp ltpFR2, and pyFR. However, the author
only tested out the functionality on the FR1, catFR1, ltpFR2, and pyFR data.

For help please see docstrings (function_name? or help(function_name) or contact LoganF
For errors please contact LoganF

Example usage and import to use:

from RetrievalCreationHelper import create_matched_events

#------> Example FR1 usage
events = create_matched_events(subject='R1111M', experiment='FR1', session=0,
                               rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                               remove_before_recall = 2000, remove_after_recall = 2000,
                               recall_eeg_start = -1250, recall_eeg_end = 250,
                               match_tolerance = 2000, verbose=True)

#------> Example ltpFR2
events = create_matched_events(subject='LTP093', experiment='ltpFR2', session=1,
                               rec_inclusion_before = 3000, rec_inclusion_after = 1000,
                               remove_before_recall = 3000, remove_after_recall = 3000,
                               recall_eeg_start = -1500, recall_eeg_end = 500,
                               match_tolerance = 3000)
"""
# General imports
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd

from ptsa.data.readers import BaseEventReader

# ----------> Main function to import
def create_matched_events(events,
                          rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                          recall_eeg_start = -1250, recall_eeg_end = 250,
                          match_tolerance = 3000,
                          remove_before_recall = 2000, remove_after_recall = 2000,
                          verbose=False):
    """Creates behavioral events for recall and matched-in-time deliberation points

    Parameters
    ----------
    :events:
        set of events to build matched deliberation periods for
    :rec_inclusion_before:
        int,  time in ms before each recall that must be free
        from other events (vocalizations, recalls, stimuli, etc.) to count
        as a valid recall
    :rec_inclusion_after:
        int, time in ms after each recall that must be free
        from other events (vocalizations, recalls, stimuli, etc.) to count
        as a valid recall
    :remove_before_recall:
        int, time in ms to remove before a recall/vocalization
        used to know when a point is "valid" for a baseline
    :remove_after_recall:
        int, time in ms to remove after a recall/vocalization
        used to know when a point is "valid" for a baseline
    :recall_eeg_start:
        int, time in ms of eeg that we would start at
        relative to recall onset, note the sign (e.g. -1500 vs 1500) does not matter
    :recall_eeg_end:
        int, time in ms of eeg that we would stop at relative to recall onset
    :match_tolerance:
        int, time in ms that a deliberation may deviate from the
        retrieval of an item and still count as a match.
    :desired_duration:
        ###FOR NOW DO NOT USE THIS PARAMETER!!!! TODO: ADD IN FULL FUNCTIONALITY###
        int, default = None, time in ms of eeg that we would like the deliberation
        period to last for. If None, the code assumes the desired duration is calculated
        using recall_eeg_start and recall_eeg_end
    :verbose:
        bool, by default False, whether or not to print out the steps of the code along the way, additionally if set to
        true then the code will output a report on the goodness of fit for the created events
    :goodness_fit_check:
        bool, by default False, whether or not to display information regarding the goodness of
        fit of the code

    Returns
    ----------
    :matched_events:
        np.recarray, array of behavioral events corresponding to included recalls with matches
        and their corresponding matches

    Example Usage
    ---------

    #------> Example FR1 usage
    events = create_matched_events(subject='R1111M', experiment='FR1', session=0,
                                   rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                                   remove_before_recall = 2000, remove_after_recall = 2000,
                                   recall_eeg_start = -1250, recall_eeg_end = 250,
                                   match_tolerance = 2000, verbose=True)

    #------> Example ltpFR2
    events = create_matched_events(subject='LTP093', experiment='ltpFR2', session=1,
                                   rec_inclusion_before = 3000, rec_inclusion_after = 1000,
                                   remove_before_recall = 3000, remove_after_recall = 3000,
                                   recall_eeg_start = -1500, recall_eeg_end = 500,
                                   match_tolerance = 3000, goodness_fit_check=True)

    Notes
    -------
    Code will do a exact match in time first, afterwards will do a "tolerated" match. Any recalls that
    are not matched are dropped. If there are multiple possibles matches (either exact or tolerated) for a
    recall then the code will select the match that is closest in trial number to the recall.
    """
    subject_instance = DeliberationEventCreator(events=events,
                                                rec_inclusion_before=rec_inclusion_before,
                                                rec_inclusion_after=rec_inclusion_after,
                                                recall_eeg_start=recall_eeg_start,
                                                recall_eeg_end=recall_eeg_end,
                                                match_tolerance=match_tolerance,
                                                remove_before_recall=remove_before_recall,
                                                remove_after_recall=remove_after_recall,
                                                desired_duration=None,  # UNTIL IMPLEMENTED SET AS NONE!
                                                verbose=verbose)

    return subject_instance.create_matched_recarray()


class RetrievalEventCreator(object):
    """An object used to create recall behavioral events that are formatted in a consistent way regardless of the CML
    experiment.

    PARAMETERS
    -------
    INPUTS:
        :subject:
            str; subject id, e.g. 'R1111M'
        :experiment:
            str, experiment id, e.g. 'FR1'
            valid intracranial experiments:
                ['FR1', 'FR2', 'FR3', 'FR5', 'FR6', 'PAL1', 'PAL2', 'PAL3', 'PAL5',
                 'PS1', 'PS2', 'PS2.1', 'PS3', 'PS4_FR', 'PS4_catFR', 'PS5_catFR',
                 'TH1', 'TH3', 'THR', 'THR1', 'YC1', 'YC2', 'catFR1', 'catFR2',
                 'catFR3', 'catFR5', 'catFR6', 'pyFR']
            valid scalp experiments:
                ['ltpFR2']
        :session:
            int, session to analyze
        :inclusion_time_before:
            int, by default 1500, time in ms before each recall that must be free
            from other events (vocalizations, recalls, stimuli, etc.) to count
            as a valid recall
        :inclusion_time_after:
            int, by default 250, time in ms after each recall that must be free
            from other events (vocalizations, recalls, stimuli, etc.) to count
            as a valid recall
        :verbose:
            bool, by default False, whether or not to print out steps along the way


    EXAMPLE USAGE
    --------------
    old_ieeg_data = RetrievalEventCreator(subject='BW022', experiment='pyFR',
                                 session=2, inclusion_time_before=1500,
                                inclusion_time_after=500, verbose=True)
    # Set all the attributes of the object, which can then be used.
    old_ieeg_data.initialize_recall_events()


    self = RetrievalEventCreator(subject='R1111M', experiment='FR1', session=2,
                                 inclusion_time_before=1500, inclusion_time_after=500,
                                 verbose=True)
    """
    # Shared by the class
    jr = None # JsonIndexReader
    jr_scalp = None # JsonIndexReader


    # -----> Initialize the instance
    def __init__(self, events,
                 inclusion_time_before, inclusion_time_after,
                 verbose=False):

        # Initialize passed arguments
        experiments = np.unique(events['experiment'])
        subjects = np.unique(events['subject'])
        if len(experiments) != 1 or len(subjects) != 1:
            raise DoneGoofedError
        self.experiment = experiments[0]
        self.subject = subjects[0]
        self.inclusion_time_before = inclusion_time_before
        self.inclusion_time_after = inclusion_time_after
        self.verbose = verbose

        # So here we need to set the time of the recall, which varies by experiment
        self.rectime = 45000 if self.experiment == 'pyFR' else 30000
        if self.experiment == 'ltpFR2':  # Add this for scalp-functionality
            self.rectime = 75000

        # Initialize attributes we'll want to construct initially as None
        self.events = events
        self.sample_rate = None
        self.montage = None
        self.possible_sessions = None
        self.trials = None
        self.included_recalls = None
        return

    # ----------> FUNCTION TO CONSTRUCT STUFF
    def initialize_recall_events(self):
        """Main code to run through the steps in the code"""
        self.set_events()
        self.events = self.add_fields_timebefore_and_timeafter(self.events)
        self.set_samplerate_params()
        self.set_valid_trials()

        # ----------> Check the formatting of the events to ensure that they have correct info
        has_recall_start_events = self.check_events_for_recall_starts(self.events)
        if not has_recall_start_events and self.verbose:
            print('Could not find REC_START events, creating REC_START events')
            self.events = self.create_REC_START_events()

        has_recall_end_events = self.check_events_for_recall_ends(self.events)
        if not has_recall_end_events:
            if self.verbose:
                print('Could not find REC_END events, creating REC_END events')
            self.events = self.create_REC_END_events(events=self.events,
                                                     time_of_the_recall_period=self.rectime)
            if self.verbose:
                print('Warning, Only set valid mstime for REC_END events')

        self.set_included_recalls()
        return

    # -------------> Methods to operate upon the instance
    def set_possible_sessions(self):
        """sets the values of all possible session values (array) to attribute possible_sessions

        Sets
        ------
        Attributes self.possible_sessions
        """

        # If ltpFR2
        if self.experiment in self.jr_scalp.experiments():
            sessions = self.jr_scalp.aggregate_values('sessions',
                                                      subject=self.subject,
                                                      experiment=self.experiment)
        # If RAM:
        if self.experiment in self.jr.experiments():
            # find out the montage for this session
            montage = list(self.jr.aggregate_values('montage',
                                                    subject=self.subject,
                                                    experiment=self.experiment,
                                                    session=self.session))[0]
            self.montage = montage
            # Find out all possible sessions with the montage
            sessions = self.jr.aggregate_values('sessions',
                                                subject=self.subject,
                                                experiment=self.experiment,
                                                montage=montage)
        # if pyFR
        if self.experiment == 'pyFR':
            evs = self.get_pyFR_events(self.subject)
            sessions = np.unique(evs['session'])

        # Replaced np.array((map(int, (sessions)))) for py3 functionality
        self.possible_sessions = np.array(list(map(int, (sessions))))

        # If the user chose a session not in the possible sessions then by default
        # We will defer to the first session of the possible sessions
        """5/18/18: Changing default behavior to raise an error if session inputted is not valid"""
        if self.session not in self.possible_sessions:
            raise DoneGoofedError(self.session, self.possible_sessions)
            #if self.verbose:
                #print('Could not find session {} in possible sessions: {}'.format(self.session, self.possible_sessions))
                #print('Over-riding attribute session input {} with {}'.format(self.session,
                                                                              #self.possible_sessions[0]))

            #self.session = self.possible_sessions[0]

        if self.verbose:
            print('Set Attribute possible_sessions')
        return

    def set_behavioral_event_path(self):
        """Sets the behavioral events path to attribute event_path

        Sets
        -------
        Attributes event_path
        """
        # -----------> Set the behavioral event path

        # If they want to do scalp reference scalp json protocol [ltpFR2]
        if self.experiment in self.jr_scalp.experiments():
            event_path = list(self.jr_scalp.aggregate_values('task_events',
                                                             subject=self.subject,
                                                             experiment=self.experiment,
                                                             session=self.session))[0]

        # If they're doing RAM data (catFR1, FR1)
        elif self.experiment in self.jr.experiments():
            event_path = list(self.jr.aggregate_values('task_events',
                                                       subject=self.subject,
                                                       experiment=self.experiment,
                                                       session=self.session))[0]
        # If they're doing pyFR
        elif self.experiment == 'pyFR':  # THIS WILL INITIALLY HAVE ALL SESSIONS!
            event_path = '/data/events/{}/{}_events.mat'.format(self.experiment, self.subject)

        # TO DO: Add in ltpFR1 functionality
        # event_path = '/data/eeg/scalp/ltp/ltpFR/behavioral/events/events_all_LTP065.mat'


        # If Logan is unsure where the data is
        else:
            if self.verbose:
                print('Unclear where the path of the data is is...')
                print('Is {} a valid experiment?'.format(self.experiment))
                print('Not creating attribute event_path')
            return

        # Set the event path to the attribute event_path
        self.event_path = event_path

        if self.verbose:
            print('Set attribute event_path')

        return

    def set_events(self):
        """sets all behavioral events to attributes events

        Sets
        -------
        Attributes self.events
        """

        if self.verbose:
            print('Set attribute events')

        # Remove practice events if there are any
        trial_field = 'trial' if 'trial' in self.events.dtype.names else 'list'
        self.events = self.events[self.events[trial_field] >= 0]

        if self.verbose:
            print('Removing practice events')

        self.events = append_fields(self.events, [('match', '<i8')])
        if self.verbose:
            print('Adding field "match" to attribute events')

        return

    def set_valid_trials(self):
        """Sets the attribute trials, an array of the unique number of trials for this session

        Sets
        -------
        Attribute self.trials
        """
        trial_field = 'trial' if self.experiment in self.jr_scalp.experiments() else 'list'
        self.trials = np.unique(self.events[trial_field])

        if self.verbose:
            print('Set attribute trials')
        return

    def set_samplerate_params(self):
        """Sets the attribute sample_rate to the instance

        Sets
        ------
        Attribute self.sample_rate

        Notes
        -------
        Codes assumes that for ltpFR2 subjects with IDs < 331, the sample rate is 500, for subjects >= 331
        it assumes a sampling rate of 2048.
        For ieeg subjects it will load the eeg can check manually what the sample_rate is

        This code also will not work on ltpFR1, this functionality needs to be updated in.
        """
        scalp = True if self.experiment in self.jr_scalp.experiments() else False

        if scalp:
            if int(self.subject.split('LTP')[-1]) < 331:
                self.sample_rate = 500.

            elif int(self.subject.split('LTP')[-1]) >= 331:
                self.sample_rate = 2048.

        if not scalp:
            if self.events is None:
                self.set_behavioral_events()

            # Use recall's mstime and eegoffset to determine the sample rate
            # Seriously, use the recall event not an arbitary type event or errors
            recalls = self.events[self.events['type'] == 'REC_WORD']
            diff_in_seconds = (recalls['mstime'][4] - recalls['mstime'][3]) / 1000.
            diff_in_samples = recalls['eegoffset'][4] - recalls['eegoffset'][3]
            # Round is because we want 499.997 etc. to be 500
            self.sample_rate = np.round(diff_in_samples / diff_in_seconds)

        if self.verbose:
            print('Set attribute sample_rate')
        return

    def set_included_recalls(self, events=None):
        """Sets all included recalls to attribute self.recalls

        Parameters
        ----------
        events: np.array, by default None and self.events is used, behavioral events of the subject

        Sets
        -------
        Attributes self.recalls
        """
        if events is None:
            events = self.events

        recalls = events[(events['type'] == 'REC_WORD')
                         & (events['intrusion'] == 0)
                         & (events['timebefore'] > self.inclusion_time_before)
                         & (events['timeafter'] > self.inclusion_time_after)]

        recalls['match'] = np.arange(len(recalls))
        self.included_recalls = recalls

        if self.verbose:
            print('Set attribute included_recalls')

        if len(self.included_recalls) == 0:
            print('No recalls detected for {} session {}'.format(self.subject, self.session))

        return

    # ----------> Staticmethods
    @staticmethod
    def add_fields_timebefore_and_timeafter(events):
        """Adds fields timebefore and timeafter to behavioral events

        Parameters
        ----------
        events: np.array, behavioral events to add the field to

        Returns
        -------
        events: np.array, behavioral events now with the fields timebefore and timeafter added
        """
        events = append_fields(events, [('timebefore', '<i8'), ('timeafter', '<i8')])

        # For the timebefore the first point, set zero, otherwise it's the difference
        events['timebefore'] = np.append(np.array([0]), np.diff(events['mstime']))
        # For the timeafter event set difference of next point, for last it's zero
        events['timeafter'] = np.append(np.diff(events['mstime']), np.array([0]))

        return events

    # -------> Utility methods to allow data that isn't formatted like RAM to work with this code
    @staticmethod
    def check_events_for_recall_ends(events):
        """Check the inputted behavior events to see if they have events corresponding to the end of recall
        Returns True if they have it, False if they don't

        Parameters
        ----------
        events: np.array, behavioral events

        Returns
        -------
        True if there are REC_END type events in events otherwise returns False
        """

        return True if len(events[events['type'] == 'REC_END']) > 0 else False

    @staticmethod
    def check_events_for_recall_starts(events):
        """Check the inputted behavior events to see if they have events corresponding to the start of recall
        Returns True if they have it, False if they don't

        Parameters
        ----------
        events: np.array, behavioral events

        Returns
        -------
        True if there are REC_END type events in events otherwise returns False
        """

        return True if len(events[events['type'] == 'REC_START']) > 0 else False

    @staticmethod
    def create_REC_END_events(events, time_of_the_recall_period=45000):
        """Creates events that corresponding to the end of the recall field

        Parameters
        ----------
        events: np.array, behavioral events of a subject
        time_of_the_recall_period: int, by default 45000, duration in ms of the recall period

        Returns
        -------
        all_events: np.array, behavioral events of a subject with added events for the end of recall


        Notes
        ------
        This will only work on events with REC_START fields, also assumes that there's no break taken
        once the recall period starts...
        Also only correctly sets mstime field everything else is copied from rec_start events
        """
        # To avoid altering the events make a copy first
        rec_stops = deepcopy(events[events['type'] == 'REC_START'])
        # Reset the fields
        rec_stops['type'] = 'REC_END'
        rec_stops['mstime'] += time_of_the_recall_period
        all_events = np.concatenate((events, rec_stops)).view(np.recarray)
        all_events.sort(order='mstime')

        return all_events

    def create_REC_START_events(self):
        """Creates events that corresponding to the start of the recall field

        Parameters
        ----------
        self: the instance of the object

        Returns
        -------
        events: np.array, behavioral events of a subject with added events for the start of recall


        Notes
        ------
        This will work on behavioral events with REC_START events
        This part was very quickly remade to further improve pyFR functionality...
        """
        # Set fields that vary from experiment to experiment
        trial_field = 'trial' if 'trial' in self.events.dtype.names else 'list'
        item_field = 'item_name' if 'item_name' in self.events.dtype.names else 'item'
        item_number_field = 'item_num' if 'item_num' in self.events.dtype.names else 'itemno'

        if self.sample_rate is None:
            self.set_samplerate_params()

        events = self.events
        self.set_valid_trials()

        all_rec_starts = []

        for i, trial in enumerate(self.trials):
            recs = events[(events['type'] == 'REC_WORD') & (events[trial_field] == trial)]
            if len(recs) > 0:
                recs = recs[0]
                rec_starts = deepcopy(recs)
            else:
                continue

            # -------> Set the correct behavioral fields

            # Take the absolute time of the first recall in the trial and minus the relative time since rec start
            rec_starts['mstime'] = recs['mstime'] - recs['rectime']

            # This is the difference in time between the recall start and first recall in seconds
            diff_time_seconds = recs['rectime'] / 1000.

            # Find the number of samples between the first recall and the start of the recall
            n_samples = diff_time_seconds * self.sample_rate

            # The int/round non-sense is because n_samples can be something like 3006.5
            rec_starts['eegoffset'] = int(recs['eegoffset'] - round(n_samples))
            rec_starts[trial_field] = recs[trial_field]
            rec_starts['intrusion'] = -999
            rec_starts['rectime'] = -999
            rec_starts['type'] = 'REC_START'
            rec_starts['serialpos'] = -999
            rec_starts[item_field] = 'X'
            rec_starts[item_number_field] = -999
            # In case we run it on FR1...
            rec_starts['recalled'] = -999
            try:
                rec_starts['timebefore'] = 0
                rec_starts['timeafter'] = 0
                # rec_starts['recserialpos'] = 0
            except:
                pass

            all_rec_starts.append(rec_starts)

        all_rec_starts = np.array(all_rec_starts).view(np.recarray)

        events = np.concatenate((all_rec_starts, events)).view(np.recarray)
        events.sort(order='mstime')

        return events

    @staticmethod
    def get_pyFR_events(subject, experiment='pyFR'):
        """ Utility to get pyFR events

        Parameters
        ----------
        subject: str, subject ID, e.g. 'BW001'
        experiment: str, by default pyFR, experiment ID

        Returns
        -------
        behavioral_events: np.array, behavioral events OF ALL SESSIONS
        """
        event_path = '/data/events/{}/{}_events.mat'.format(experiment, subject)
        base_e_reader = BaseEventReader(filename=event_path, eliminate_events_with_no_eeg=True)
        behavioral_events = base_e_reader.read()
        return behavioral_events

# ------> Create an object to create matched in time periods of deliberation
class DeliberationEventCreator(RetrievalEventCreator):
    """An object used to create recall behavioral events that are formatted in a consistent way regardless of the CML
       experiment.

       PARAMETERS
       -------
       INPUTS:
           :subject:
               str; subject id, e.g. 'R1111M'
           :experiment:
               str, experiment id, e.g. 'FR1'
               valid intracranial experiments:
                   ['FR1', 'FR2', 'FR3', 'FR5', 'FR6', 'PAL1', 'PAL2', 'PAL3', 'PAL5',
                    'PS1', 'PS2', 'PS2.1', 'PS3', 'PS4_FR', 'PS4_catFR', 'PS5_catFR',
                    'TH1', 'TH3', 'THR', 'THR1', 'YC1', 'YC2', 'catFR1', 'catFR2',
                    'catFR3', 'catFR5', 'catFR6', 'pyFR']
               valid scalp experiments:
                   ['ltpFR2']
           :session:
               int, session to analyze
           :rec_inclusion_before:
               int, time in ms before each recall that must be free from other events
               (vocalizations, recalls, stimuli, etc.) to count as a valid recall
           :rec_inclusion_after:
               int, time in ms after each recall that must be free from other events
               (vocalizations, recalls, stimuli, etc.) to count as a valid recall
           :recall_eeg_start:
               int, time in ms prior to each recall's vocalization onset that we want
               to start looking at the eeg at
           :match_tolerance:
               int, time in ms relative to a recall's retrieval period (e.g. from
               recall_eeg_start up until vocalization onset) that a match is allowed
               to vary by while still counting as a match
           :remove_before_recall:
               int, time in ms to remove before a recall (or vocalization) from being
               a valid deliberation period
           :remove_after_recall:
               int, time in ms to remove after a recall (or vocalization) from being
               a valid deliberation period
           ####### TODO: UPDATE #######
           :desired_duration:
               int, by default set as None and assumed to be the total eeg length,
               Not currently implemented besides default setting. Do not use until then
           ############################
           :verbose:
               bool, by default False, whether or not to print out steps along the way,


               Not true anymore:#if True then the code will also generate a report of the goodness of fit of
               Not true anymore:#the matching process


       EXAMPLE USAGE
       --------------
       # ------> Example RAM data
       subject = DeliberationEventCreator(subject='R1111M', experiment='FR1', session=0,
                                          rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                                          remove_before_recall = 2000, remove_after_recall = 2000,
                                          recall_eeg_start = -1250, recall_eeg_end = 250,
                                          match_tolerance = 2000, verbose=True)
       events = subject.create_matched_recarray()


       # -------> Example ltpFR2
       scalp_sub = DeliberationEventCreator(subject='LTP093', experiment='ltpFR2', session=1,
                                            rec_inclusion_before = 2000, rec_inclusion_after = 1000,
                                            remove_before_recall = 3000, remove_after_recall = 3000,
                                            recall_eeg_start = -1250, recall_eeg_end = 250,
                                            match_tolerance = 2000, verbose=True)
       scalp_events = scalp_sub.create_matched_recarray()
       """
    def __init__(self, events,
                 rec_inclusion_before, rec_inclusion_after,
                 recall_eeg_start, recall_eeg_end, match_tolerance,
                 remove_before_recall, remove_after_recall,
                 desired_duration=None, verbose=False):
        # Inheritance (Sets all attributes of RetrievalEventCreator)
        super(DeliberationEventCreator, self).__init__(events=events,
                                                       inclusion_time_before=rec_inclusion_before,
                                                       inclusion_time_after=rec_inclusion_after,
                                                       verbose=verbose)
        # Initialize the relevant RetrievalEventCreator attributes
        self.initialize_recall_events()
        # This seems silly to have four attributes refer to two things?
        self.rec_inclusion_before = rec_inclusion_before
        self.rec_inclusion_after = rec_inclusion_after
        # Intialize relevant passed arguments and relevant DeliberationEventCreator attributes
        # Just to handle everything consistently regardless of user input...
        if np.sign(recall_eeg_start) == -1:
            self.recall_eeg_start = -1 * recall_eeg_start
        else:
            self.recall_eeg_start = recall_eeg_start

        self.recall_eeg_end = recall_eeg_end
        self.match_tolerance = match_tolerance
        self.remove_before_recall = remove_before_recall
        self.remove_after_recall = remove_after_recall
        self.desired_duration = desired_duration
        if self.desired_duration is None:
            self.desired_duration = np.abs(self.recall_eeg_start) + np.abs(self.recall_eeg_end)
        self.trial_field = 'trial' if 'trial' in self.events.dtype.names else 'list'
        # Create attributes we'll set later initially as None
        self.baseline_array = None
        self.matches = None
        self.ordered_recalls = None
        self.matched_events = None

    def set_valid_baseline_intervals(self):
        """Sets to attribute baseline_array an array of 1 and 0 (num_unique_trials x 30000) where 1 is a valid time point

        Parameters
        -----------
        INPUTS EXTRACTED FROM INSTANCE:
            behavioral_events: np.array, behavioral events of a subject for one session of data.
            recall_period: int, by default 30000,
                time in ms of recall period (scalp = 750000, pyFR = 450000, RAM = 300000)
            desired_bl_duration: int, by default 3000,
                the desired time in ms we want to match over (e.g. if looking at recall from -2000 to 0, this
                should be 2000)
            remove_before_recall: int, by default 1500,
                time in ms to exclude before each recall/vocalization as invalid
            remove_after_recall: int, by default 1500,
                time in ms to exclude after each recall/vocalization as invalid
        Sets
        -------
        Attributes self.baseline_array

        """
        # Remove any practice events
        behavioral_events = self.events[self.events[self.trial_field] >= 0]
        # trials = np.unique(behavioral_events[self.trial_field])

        # Create an array of ones of shape trials X recall_period (in ms)
        baseline_array = np.ones((self.rectime * len(self.trials))).reshape(len(self.trials), self.rectime)
        valid, invalid = 1, 0

        # Convert any invalid point in the baseline to zero
        for index, trial in enumerate(self.trials):
            # Get the events of the trial
            trial_events = behavioral_events[behavioral_events[self.trial_field] == trial]

            # Get recall period start and stop points
            starts = trial_events[trial_events['type'] == 'REC_START']
            stops = trial_events[trial_events['type'] == 'REC_END']

            # If we don't have any recall periods then we will invalide the whole thing
            if ((starts.shape[0] == 0) & (stops.shape[0] == 0)):
                print('No recall period detected, trial: ', trial)
                baseline_array[index] = invalid
                continue
            # --------> Find Recalls or vocalizations
            possible_recalls = trial_events[
                (trial_events['type'] == 'REC_WORD') | (trial_events['type'] == 'REC_WORD_VV')]

            # -----> Use Recall rectimes to construct ranges of invalid times before and after them
            if len(possible_recalls['rectime']) == 1:  # If only one recall in the list
                invalid_points = np.arange(possible_recalls['rectime'] - self.remove_before_recall,
                                           possible_recalls['rectime'] + self.remove_after_recall)
            elif len(possible_recalls['rectime']) > 1:  # If multiple recalls in the list
                # TODO: Replace with np.apply or broadcasting of some kind?
                invalid_points = np.concatenate([np.arange(x - self.remove_before_recall, x + self.remove_after_recall)
                                                 for x in possible_recalls['rectime']])
            else:  # Get rid of any trials where we can't find any invalid points.
                baseline_array[index] = invalid
                continue

            # Ensure the points to be invalidated are within the boundaries of the recall period
            invalid_points = invalid_points[np.where(invalid_points >= 0)]
            invalid_points = invalid_points[np.where(invalid_points < self.rectime)]
            invalid_points = (np.unique(invalid_points),)  # ((),) similiar to np.where output

            # Removes initial recall contamination (-remove_before_recall,+remove_after_recall)
            baseline_array[index][invalid_points] = invalid

        self.baseline_array = baseline_array
        return

    def order_recalls_by_num_exact_matches(self):
        """Orders included_recalls array by least to most number of exact matches to create attribute ordered_recalls
        Creates
        -------
        Attribute ordered_recalls

        Notes
        ---------
        ordered_recalls[0] has the least number of matches and ordered_recalls[-1] has the most
        """
        # Desired start and stop points of each included recall
        recs_desired_starts = self.included_recalls['rectime'] - self.recall_eeg_start
        recs_desired_stops = self.included_recalls['rectime'] + self.recall_eeg_end

        # Store matches here
        exactly_matched = []

        # Go through each recall, find perfect matches
        for (start, stop) in zip(recs_desired_starts, recs_desired_stops):
            has_match_in_trial = np.all(self.baseline_array[:, start:stop] == 1, 1)
            exactly_matched.append(self.trials[has_match_in_trial])

        # Sort the recalls by ordering events from least to most number of exact matches
        index_match = np.array([[i, len(x)] for i, x in enumerate(np.array(exactly_matched))])
        sorted_order = pd.DataFrame(index_match, columns=['rec_index', 'num_matches']).sort_values('num_matches')
        ordered_recalls = np.array([self.included_recalls[i] for i in sorted_order['rec_index'].index]).view(
            np.recarray)

        self.ordered_recalls = ordered_recalls
        return

    def match_accumulator(self):
        """Accumulates matches between included recalls and baseline array, upon selection of a match invalidates it for other recalls

        Code will first go through each recall (ordered from least to most number of matches)  and try to select an exact match in time
        in another trial/list. If it cannot, after completeion of all exact matches the code will go through and try to find a tolerated
        match, that is a period in time that is within the instance's match_tolrance relative to the retrieval phase (eeg_rec_start up
        until vocalization onset)

        Modifies
        --------
        Attribute baseline_array

        Sets
        ----
        Attribute matches
        """
        if self.baseline_array is None:
            self.set_valid_baseline_intervals()
        if self.ordered_recalls is None:
            self.order_recalls_by_num_exact_matches()

        # This is used to keep track of the trial vs row indexing issues
        trial_to_row_mapper = dict(zip(self.trials, np.arange(len(self.trials))))

        recs_desired_starts = self.ordered_recalls['rectime'] - self.recall_eeg_start
        recs_desired_stops = self.ordered_recalls['rectime'] + self.recall_eeg_end

        ################ EXACT MATCH ACCUMULATION ################
        # -----> Go through each recall, find perfect matches, if multiple select one closest to trial

        valid, invalid = 1, 0  # Explicit > implicit
        matches = OrderedDict()  # Store matches here
        for index, (start, stop) in enumerate(zip(recs_desired_starts, recs_desired_stops)):
            if index not in matches:
                matches[index] = []
            # Match across all trials from -recall_eeg_start to + recall_eeg_end
            ## TODO: Add in here to modify an option for putting in a different duration
            has_match_in_trial = np.all(self.baseline_array[:, start:stop] == valid, 1)
            trial_matches = self.trials[has_match_in_trial]

            # If there aren't any perfect matches just continue
            if len(trial_matches) == 0:
                continue

            # Select the match that's closest to the recall's trial
            recalls_trial_num = self.ordered_recalls[index][self.trial_field]
            idx_closest_trial = np.abs(trial_matches - recalls_trial_num).argmin()
            selection = trial_matches[idx_closest_trial]

            if index in matches:
                matches[index].append((selection, start, stop))

            # Void the selection so other recalls cannot use it as valid
            self.baseline_array[trial_to_row_mapper[selection], start:stop] = invalid

        ################ TOLERATED MATCH ACCUMULATION ################
        # -------> Go through each recall, find tolerated matches
        if self.verbose:
            print('Starting Tolerated Matching Procedure...')
        for index, (start, stop) in enumerate(zip(recs_desired_starts, recs_desired_stops)):
            if matches[index] != []:
                continue  # Don't redo already matched recalls
            # Tolerance is defined around recall_eeg_start up until volcalization onset
            before_start_within_tol = start - self.match_tolerance
            after_start_within_tol = start + self.match_tolerance + self.recall_eeg_start

            # -----> Sanity check: cannot be before or after recall period
            if before_start_within_tol < 0:
                before_start_within_tol = 0
            if after_start_within_tol > self.baseline_array.shape[-1]:
                after_start_within_tol = self.baseline_array.shape[-1]

            # ------> Find out where there are valid tolerated points
            recalls_trial_num = self.ordered_recalls[index][self.trial_field]
            # Only need to check between tolerated points
            relevant_bl_times = self.baseline_array[:, before_start_within_tol:after_start_within_tol]
            # Use convolution of a kernel of ones for the desired duration to figure out where there are valid periods
            kernel = np.ones(self.desired_duration, dtype=int)
            sliding_sum = np.apply_along_axis(np.convolve, axis=1, arr=relevant_bl_times,
                                              v=kernel, mode='valid')

            valid_rows, valid_time_sliding_sum = np.where(sliding_sum == self.desired_duration)
            # Convert row to trial number through indexing the valid rows
            valid_trials = self.trials[(np.unique(valid_rows),)]
            if len(valid_trials) == 0:
                if self.verbose:
                    print('Could not match recall index {}'.format(index))
                continue

            # Find the closest trial
            idx_closest_trial = np.abs(valid_trials - recalls_trial_num).argmin()
            # Row in baseline_array vs actually recording the correct trial number
            selected_row = valid_rows[idx_closest_trial]
            selected_trial = valid_trials[idx_closest_trial]
            valid_first_point = valid_time_sliding_sum[0]
            # Essentially a conversion between convolution window and mstime
            valid_start = before_start_within_tol + valid_first_point
            valid_stop = valid_start + self.desired_duration  # b/c sliding mean slides to the right
            if index in matches:
                matches[index].append((valid_trials[idx_closest_trial], valid_start, valid_stop))
                # Void the selection so other recalls cannot use it is valid
                self.baseline_array[selected_row, valid_start:valid_stop] = invalid
        self.matches = matches

        return

    def create_matched_recarray(self):
        """Constructs a recarray of ordered_recalls and their matched deliberation points

        Returns
        -------
        behavioral_events: np.recarray, array of included recalls and matched deliberation periods
        """
        # if np.sign(self.recall_eeg_start) == -1:
        # self.recall_eeg_start *= -1

        if self.matched_events is None:
            self.match_accumulator()

        rec_start = self.events[self.events['type'] == 'REC_START']
        # rec_end = self.events[self.events['type']=='REC_END']
        # Non-sense is due to inconsistent fields
        trial_field = 'trial' if 'trial' in self.ordered_recalls.dtype.names else 'list'
        item_field = 'item_name' if 'item_name' in self.ordered_recalls.dtype.names else 'item'
        item_number_field = 'item_num' if 'item_num' in self.ordered_recalls.dtype.names else 'itemno'

        valid_recalls, valid_deliberation = [], []
        # Use the matches dictionary to construct a recarray
        for k, v in enumerate(self.matches):
            if self.matches[v] == [] and self.verbose:
                print('Code could not successfully match recall index {}, dropping recall index {}'.format(k, k))
                continue

            valid_recalls.append(self.ordered_recalls[k])

            trial, rel_start, rel_stop = self.matches[v][0]  # [0] b/c tuple
            trial_rec_start_events = rec_start[rec_start[trial_field] == trial]

            bl = deepcopy(self.ordered_recalls[k])
            bl['type'] = 'REC_BASE'
            bl[trial_field] = trial
            bl[item_field] = 'N/A'
            bl[item_number_field] = -999
            bl['timebefore'] = -999
            bl['timeafter'] = -999
            bl['eegoffset'] = -999
            # Since we'll want to use EEGReader for both at once the below adjustment should work to allow it to do so
            # Changed from - to + because of *= -1 at start of code; bl['rectime'] = rel_start - recall_eeg_start
            bl['rectime'] = rel_start + self.recall_eeg_start
            bl['mstime'] = trial_rec_start_events['mstime'] + bl['rectime']
            bl['eegoffset'] = trial_rec_start_events['eegoffset'] + (self.sample_rate * (bl['rectime'] / 1000.))
            valid_deliberation.append(bl)

        valid_recalls = np.array(valid_recalls).view(np.recarray)
        valid_deliberation = np.array(valid_deliberation).view(np.recarray)

        behavioral_events = np.concatenate((valid_deliberation, valid_recalls)).view(np.recarray)
        behavioral_events.sort(order='match')
        self.matched_events = behavioral_events

        if self.verbose:
            print('Set attribute matched_events')

        return behavioral_events


def append_fields(old_array, list_of_tuples_field_type):
    """Return a new array that is like "old_array", but has additional fields.

    The contents of "old_array" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    *This is necessary to do use than using the np.lib.recfunction.append_fields
    function b/c the json loaded events use a dictionary for stim_params in the events*
    -----
    INPUTS
    old_array: a structured numpy array, the behavioral events from ptsa
    list_of_tuples_field_type: a numpy type description of the new fields
    -----
    OUTPUTS
    new_array: a structured numpy array, a copy of old_array with the new fields
    """
    if old_array.dtype.fields is None:
        raise ValueError("'old_array' must be a structured numpy array")

    new_dtype = old_array.dtype.descr + list_of_tuples_field_type

    # Try to add the new field to the array, should work if it's not already a field
    try:
        new_array = np.empty(old_array.shape, dtype=new_dtype).view(np.recarray)
        for name in old_array.dtype.names:
            new_array[name] = old_array[name]
        return new_array
    # If user accidentally tried to add a field already there, then return the old array
    except ValueError:
        return old_array


class DoneGoofedError(Exception):
    """Exception raised for errors in the input,

    Attributes:
        session -- input session of thes
        possible_sessions -- valid sessions of the subject
    """
    pass
