import pandas as pd
import numpy as np
import scipy as sp
from scipy.spatial import distance, distance_matrix
from pybeh.make_recalls_matrix import make_recalls_matrix
from pybeh.crp import crp
# from pybeh.sem_crp import sem_crp
from pybeh.temp_fact import temp_fact
from pybeh.dist_fact import dist_fact, dist_percentile_rank
from pybeh.mask_maker import make_clean_recalls_mask2d

def get_itemno_matrices(evs, itemno_column='itemno', list_index=['subject', 'session', 'list']):
    """Expects as input a dataframe (df) for one subject"""
    evs.loc[:, itemno_column] = evs.loc[:, itemno_column].astype(int)
    evs['pos'] = evs.groupby(list_index).cumcount()
    itemnos_df = pd.pivot_table(evs, values=itemno_column, 
                                 index=list_index, 
                                 columns='pos', fill_value=0)
    itemnos = itemnos_df.values
    return itemnos

def get_all_matrices(df, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    types = [pres_type, rec_type]
    #only include lists if both presentations and recalls are present (i.e. ntypes == 2)
    df = df.query(type_column + ' in @types')
    ntypes_df = df[list_index + [type_column]].groupby(list_index).agg({type_column: 'nunique'}).reset_index().rename(columns={type_column: 'ntypes'})
    df = df.merge(ntypes_df).query('ntypes == 2')

    pres_itemnos = get_itemno_matrices(df.query(type_column + ' == @pres_type'), 
                                       itemno_column=itemno_column, 
                                       list_index=list_index)
    rec_itemnos = get_itemno_matrices(df.query(type_column + ' == @rec_type'), 
                                       itemno_column=itemno_column, 
                                       list_index=list_index)
    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos)
    return pres_itemnos, rec_itemnos, recalls


def pd_crp(df, lag_num=5, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    """Expects as input a dataframe (df) for one subject"""
    print(df)
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    print(recalls)
    kill = 1/0
    lag_num = min(pres_itemnos.shape[1], lag_num)
    if lag_num != 0:
        prob = crp(recalls=recalls,
                    subjects=['_'] * recalls.shape[0],
                    listLength=pres_itemnos.shape[1],
                    lag_num=lag_num)[0]
    else:
        prob = np.empty((lag_num*2)+1)
    crp_dict = {'prob': prob, 
                'lag': np.arange(-lag_num, (lag_num+1))}
    return pd.DataFrame(crp_dict, index=np.arange(-lag_num, (lag_num+1)))

# def get_sim_mat_old(df, itemno_column, sim_columns, index_cols, method=distance.euclidean):
#     sem_sim_df = df.pivot_table(values=sim_columns, columns=itemno_column, 
#                                               index=index_cols)
#     # https://stackoverflow.com/questions/29723560/distance-matrix-for-rows-in-pandas-dataframe
#     sem_sims = sem_sim_df.apply(lambda col1: sem_sim_df.apply(
#         lambda col2: method(col1, col2))).values
#     return sem_sims

def get_sim_mat(df, sim_cols, itemno_col='itemno', word_val_type="WORD_VALS", p=2, type_column='type'):
    word_val_df = df.query(type_column + ' == @word_val_type').drop_duplicates().sort_values(itemno_col)
    sem_sims = distance_matrix(word_val_df[sim_cols].values, word_val_df[sim_cols].values, p=p)
    return sem_sims

def dist_fact(rec_itemnos=None, pres_itemnos=None, subjects=None, dist_mat=None, is_similarity=False, skip_first_n=0, ret_counts=False):
    """
    Returns a clustering factor score for each subject, based on the provided distance metric (Polyn, Norman, & Kahana,
    2009). Can also be used with a similarity matrix (e.g. LSA, word2vec) if is_similarity is set to True.
    :param rec_itemnos: A trials x recalls matrix containing the ID numbers (between 1 and N) of the items recalled on
        each trial. Extra-list intrusions should appear as -1, and the matrix should be padded with zeros if the number
        of recalls differs by trial.
    :param pres_itemnos: A trials x items matrix containing the ID numbers (between 1 and N) of the items presented on
        each trial.
    :param subjects: A list/array containing identifiers (e.g. subject number) indicating which subject completed each
        trial.
    :param dist_mat: An NxN matrix (where N is the number of words in the wordpool) defining either the distance or
        similarity between every pair of words in the wordpool. Whether dist_mat defines distance or similarity can be
        specified with the is_similarity parameter.
    :param is_similarity: If False, dist_mat is assumed to be a distance matrix. If True, dist_mat is instead treated as
        a similarity matrix (i.e. larger values correspond to smaller distances). (DEFAULT = False)
    :param skip_first_n: An integer indicating the number of recall transitions to ignore from the start of each recall
        period, for the purposes of calculating the clustering factor. This can be useful to avoid biasing your results,
        as early transitions often differ from later transition in terms of their clustering. Note that the first n
        recalls will still count as already recalled words for the purposes of determining which transitions are
        possible. (DEFAULT = 0)
    :return: An array containing the clustering factor score for each subject (sorted by alphabetical order).
    """

    if rec_itemnos is None:
        raise Exception('You must pass a recall_itemnos matrix.')
    if pres_itemnos is None:
        raise Exception('You must pass a pres_itemnos matrix.')
    if subjects is None:
        raise Exception('You must pass a subjects vector.')
    if dist_mat is None:
        raise Exception('You must pass either a similarity matrix or a distance matrix.')
    if len(rec_itemnos) != len(subjects) or len(pres_itemnos) != len(subjects):
        raise Exception('The rec_itemnos and pres_itemnos matrices must have the same number of rows as the list of'
                        'subjects.')
    if not isinstance(skip_first_n, int) or skip_first_n < 0:
        raise ValueError('skip_first_n must be a nonnegative integer.')

    # Convert inputs to numpy arrays if they are not arrays already
    rec_itemnos = np.array(rec_itemnos).astype(int)
    pres_itemnos = np.array(pres_itemnos).astype(int)
    subjects = np.array(subjects)
    dist_mat = np.array(dist_mat)

    # Provide a warning if the user inputs a dist_mat that looks like a similarity matrix (scores on diagonal are
    # large), but has left is_similarity as False
    if (not is_similarity) and np.nanmean(np.diagonal(dist_mat)) > np.nanmean(dist_mat):
        warnings.warn('It looks like you might be using a similarity matrix (e.g. LSA, word2vec) instead of a distance'
                      ' matrix, but you currently have is_similarity set to False. If you are using a similarity'
                      ' matrix, make sure to set is_similarity to True when running dist_fact().')

    # Initialize arrays to store each participant's results
    usub = np.unique(subjects)
    total = np.zeros_like(usub, dtype=float)
    count = np.zeros_like(usub, dtype=float)

    # Identify locations of all correct recalls (not PLI, ELI, or repetition)
    clean_recalls_mask = np.array(make_clean_recalls_mask2d(make_recalls_matrix(pres_itemnos, rec_itemnos)))

    # Calculate distance factor score for each trial
    for i, trial_data in enumerate(rec_itemnos):
        seen = set()
        # Identify the current subject's index in usub to determine their position in the total and count arrays
        subj_ind = np.where(usub == subjects[i])[0][0]
        # Loop over the recalls on the current trial
        for j, rec in enumerate(trial_data[:-1]):
            seen.add(rec)
            # Only count transition if both the current and next recalls are valid
            if clean_recalls_mask[i, j] and clean_recalls_mask[i, j+1] and j >= skip_first_n:
                # Identify the distance between the current recall and all valid recalls that could follow it
                possibles = np.array([dist_mat[rec - 1, poss_rec - 1] for poss_rec in pres_itemnos[i] if poss_rec not in seen])
                # Identify the distance between the current recall and the next
                actual = dist_mat[rec - 1, trial_data[j + 1] - 1]
                # Find the proportion of possible transitions that were larger than the actual transition
                ptile_rank = dist_percentile_rank(actual, possibles, is_similarity)
                # Add transition to the appropriate participant's score
                if ptile_rank is not None:
                    total[subj_ind] += ptile_rank
                    count[subj_ind] += 1

    # Find temporal factor scores as the participants' average transition scores
    count[count == 0] = np.nan
    final_data = total / count
    
    if ret_counts:
        return final_data, total, count
    return final_data

def pd_sem_crp_list(df, sim_columns=None, bins=None, pres_type="WORD", 
               rec_type="REC_WORD", type_column='type', serialpos_col='serialpos', ret_counts=False, p=2):
    """Expects as input a dataframe (df) for one list.
    Doesn't require separate word_vals, expects them to be next to item. 
    Requires bins to be entered from elsewhere"""
    pres_df = df.query(type_column+' == @pres_type').sort_values(serialpos_col)
    pres_itemnos = pres_df[serialpos_col].values[np.newaxis, :]
    rec_itemnos = df.query(type_column+' == @rec_type')[serialpos_col].values[np.newaxis, :]
    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos)

    sem_sims = distance_matrix(pres_df[sim_columns].values, pres_df[sim_columns].values, p=p)
    n_bins = len(bins)

    out = sem_crp(recalls=recalls, 
                       recalls_itemnos=rec_itemnos, 
                       pres_itemnos=pres_itemnos, 
                       subjects=['_'] * recalls.shape[0], 
                       sem_sims=sem_sims, 
                       n_bins=n_bins, 
                       bins=bins,
                       listLength=pres_itemnos.shape[1],
                       ret_counts=ret_counts)
    if ret_counts:
        bin_means, crp, actual, poss = out
    else:
        bin_means, crp = out
    crp_dict = {'prob': crp[0], 
                'sem_bin_mean': bin_means[0],
                'sem_bin': np.arange(n_bins)
               }
    if ret_counts:
        crp_dict['actual'] = actual
        crp_dict['poss'] = poss

    return pd.DataFrame(crp_dict).query('prob == prob')

def pd_dist_fact_list(df, sim_columns=None,
                 skip_first_n=0, serialpos_col='serialpos',
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", type_column='type', ret_counts=False, p=2
                ):
    pres_df = df.query(type_column+' == @pres_type').sort_values(serialpos_col)
    pres_itemnos = pres_df[serialpos_col].values[np.newaxis, :]
 
    rec_itemnos = df.query(type_column+' == @rec_type')[serialpos_col].values[np.newaxis, :]
    #no recalls
#     if rec_itemnos.shape[1] == 0:
#         return np.nan  

    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos)

    dist_mat = distance_matrix(pres_df[sim_columns].values, pres_df[sim_columns].values, p=p)
    
    final_data, total, count = dist_fact(rec_itemnos=rec_itemnos, 
              pres_itemnos=pres_itemnos, 
              subjects=['_'] * recalls.shape[0],
              dist_mat=dist_mat, is_similarity=False, 
              skip_first_n=skip_first_n, ret_counts=True)
#     print(final_data, total, count)
    if count == np.nan:
        print('nan')
        return np.nan
    dist_fact_dict = {'dist_fact': final_data, 
                'total': total,
                'count': count
               }
    return pd.DataFrame(dist_fact_dict)

def pd_dist_fact_list_sub(df, sim_columns=None,
                 skip_first_n=0, list_index=['subject', 'session', 'trial'],
                          sub_index=['subject'], serialpos_col='serialpos',
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    
    sub_dist_fact_df = df.groupby(list_index).apply(
        pd_dist_fact_list, sim_columns=sim_columns, serialpos_col=serialpos_col).reset_index().groupby(
        sub_index).agg({'total': 'sum', 'count': 'sum'}).reset_index()
    
    sub_dist_fact_df['dist_fact'] = sub_dist_fact_df['total'] / sub_dist_fact_df['count']
    sub_dist_fact_df.drop(columns=sub_index, inplace=True)
    return sub_dist_fact_df

def pd_sem_crp_list_sub(df, sim_columns=None,
                 skip_first_n=0, list_index=['subject', 'session', 'trial'],
                          sub_index=['subject'], bins=None, serialpos_col='serialpos',
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    
    sub_sem_crp_df = df.groupby(list_index).apply(
        pd_sem_crp_list, sim_columns=sim_columns, bins=bins, ret_counts=True, serialpos_col=serialpos_col).reset_index().groupby(
        sub_index + ['sem_bin']).agg({'actual': 'sum', 'poss': 'sum'}).reset_index()
    
    sub_sem_crp_df['prob'] = sub_sem_crp_df['actual'] / sub_sem_crp_df['poss']
    sub_sem_crp_df.drop(columns=sub_index, inplace=True)
    return sub_sem_crp_df

def pd_sem_crp(df, itemno_column='itemno', 
                list_index=['subject', 'session', 'list'], sim_columns=None,
                sem_sims=None, n_bins=10, bins=None, pres_type="WORD", 
               rec_type="REC_WORD", word_val_type="WORD_VALS", type_column='type', ret_counts=False):
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    
    if sem_sims is None:
        if df.query('type == @word_val_type')[itemno_column].min() != 1:
            print('expects itemnos to start at 1')
            print('current word val itemnos start at ' + str(df.query('type == @word_val_type')[itemno_column].min()))
        sem_sims = get_sim_mat(df, sim_columns, itemno_col=itemno_column, word_val_type=word_val_type,
                               type_column=type_column)
    if bins is not None:
        n_bins = len(bins)
        
    out = sem_crp(recalls=recalls, 
                   recalls_itemnos=rec_itemnos, 
                   pres_itemnos=pres_itemnos, 
                   subjects=['_'] * recalls.shape[0], 
                   sem_sims=sem_sims, 
                   n_bins=n_bins, 
                   bins=bins,
                   listLength=pres_itemnos.shape[1],
                   ret_counts=ret_counts)
    if ret_counts:
        bin_means, crp, actual, poss = out
    else:
        bin_means, crp = out
    crp_dict = {'prob': crp[0], 
                'sem_bin_mean': bin_means[0],
                'sem_bin': np.arange(n_bins)
               }
    if ret_counts:
        crp_dict['actual'] = actual
        crp_dict['poss'] = poss
        
    return pd.DataFrame(crp_dict).query('prob == prob') #remove bins with no data

def pd_temp_fact(df, skip_first_n=0, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    
    #no recalls
    if pres_itemnos.shape[1] == 0:
        return np.nan
    
    temp_fact_arr = temp_fact(recalls=recalls, 
                  subjects=['_']*recalls.shape[0],
                  listLength=pres_itemnos.shape[1],
                  skip_first_n=skip_first_n)
    
    return temp_fact_arr[0]

def pd_dist_fact(df, rec_itemnos=None, itemno_column='itemno', 
                 list_index=['subject', 'session', 'list'], 
                 dist_mat=None, sim_columns=None, is_similarity=False, 
                 dist_columns=None,
                 skip_first_n=0,
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", word_val_type="WORD_VALS", type_column='type', ret_counts=False
                ):
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    #no recalls
    if pres_itemnos.shape[1] == 0:
        return np.nan
    
    if dist_mat is None:
        if df.query('type == @word_val_type')[itemno_column].min() != 1:
            print('expects itemnos to start at 1')
            print('current word val itemnos start at ' + str(df.query('type == @word_val_type')[itemno_column].min()))
        dist_mat = get_sim_mat(df, sim_columns, itemno_col=itemno_column, 
                               type_column=type_column, word_val_type=word_val_type)
    
    dist_fact_arr = dist_fact(rec_itemnos=rec_itemnos, 
              pres_itemnos=pres_itemnos, 
              subjects=['_'] * recalls.shape[0],
              dist_mat=dist_mat, is_similarity=is_similarity, 
              skip_first_n=skip_first_n)
    return dist_fact_arr[0]

def sem_crp(recalls=None, recalls_itemnos=None, pres_itemnos=None, subjects=None, sem_sims=None, n_bins=10, bins=None, listLength=None, ret_counts=False):
    """bins should not include an upper bin"""
    if recalls_itemnos is None:
        raise Exception('You must pass a recalls-by-item-numbers matrix.')
    elif pres_itemnos is None:
        raise Exception('You must pass a presentations-by-item-numbers matrix.')
    elif sem_sims is None:
        raise Exception('You must pass a semantic similarity matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a listLength')
    elif len(recalls_itemnos) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    # Make sure that all input arrays and matrices are numpy arrays
    recalls = np.array(recalls, dtype=int)
    recalls_itemnos = np.array(recalls_itemnos, dtype=int)
    pres_itemnos = np.array(pres_itemnos, dtype=int)
    subjects = np.array(subjects)
    sem_sims = np.array(sem_sims, dtype=float)

    # Set diagonal of the similarity matrix to nan
    np.fill_diagonal(sem_sims, np.nan)
    # Sort and split all similarities into equally sized bins
    all_sim = sem_sims.flatten()
    all_sim = np.sort(all_sim[~np.isnan(all_sim)])
    if bins is None:
        bins = np.array_split(all_sim, n_bins)
        bins = [b[0] for b in bins]
    else:
        n_bins = len(bins)
    # Convert the similarity matrix to bin numbers for easy bin lookup later
    bin_sims = np.digitize(sem_sims, bins) - 1

    # Convert recalled item numbers to the corresponding indices of the similarity matrix by subtracting 1
    recalls_itemnos -= 1
    pres_itemnos -= 1

    usub = np.unique(subjects)
    bin_means = np.zeros((len(usub), n_bins))
    crp = np.zeros((len(usub), n_bins))
    # For each subject
    for i, subj in enumerate(usub):
        # Create a filter to select only the current subject's data
        subj_mask = subjects == subj
        subj_recalls = recalls[subj_mask]
        subj_rec_itemnos = recalls_itemnos[subj_mask]
        subj_pres_itemnos = pres_itemnos[subj_mask]

        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(subj_recalls))

        # Setup counts for number of possible and actual transitions, as well as the sim value of actual transitions
        actual = np.zeros(n_bins)
        poss = np.zeros(n_bins)
        val = np.zeros(n_bins)

        # For each of the current subject's trials
        for j, trial_recs in enumerate(subj_recalls):
            seen = set()
            # For each recall on the current trial
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j, k] and clean_recalls_mask[j, k+1]:
                    this_recno = subj_rec_itemnos[j, k]
                    next_recno = subj_rec_itemnos[j, k+1]
                    # Lookup semantic similarity and its bin between current recall and next recall
                    sim = sem_sims[this_recno, next_recno]
                    b = bin_sims[this_recno, next_recno]
                    actual[b] += 1
                    val[b] += sim

                    # Get a list of not-yet-recalled word numbers
                    poss_rec = [subj_pres_itemnos[j][x] for x in range(listLength) if x+1 not in seen]
                    # Lookup the similarity bins between the current recall and all possible correct recalls
                    poss_trans = np.unique([bin_sims[this_recno, itemno] for itemno in poss_rec])
                    for b in poss_trans:
                        poss[b] += 1

        crp[i, :] = actual / poss  # CRP is calculated as number of actual transitions / number of possible ones
        bin_means[i, :] = val / actual  # Bin means are defined as the average similarity of actual transitions per bin
    if ret_counts:
        return bin_means, crp, actual, poss
    else:
        return bin_means, crp