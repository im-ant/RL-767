# ==
#
# Requires:
#   tensorflow 1.8+
#   pandas
# ==

import argparse
import csv
import glob
import os

import pandas as pd
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator


def extract_single_hparamCSV(csv_path):
    """
    Read a csv file of hyperparameters, assuming two comma-delim columns
    containing [hpara_name, hparam_value]

    Return a dictionary containing {hparam_name: hparam_value}
    """
    hparam_dict = {}

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            hparam_name = row[0]
            hparam_val = row[1]

            # Try to convert value to int, float, or keep as string
            try:
                hparam_val = int(hparam_val)
            except ValueError:
                try:
                    hparam_val = float(hparam_val)
                except ValueError:
                    hparam_val = str(hparam_val)

            # Save
            hparam_dict[hparam_name] = hparam_val

    return hparam_dict

def extract_single_tfEvent(tfevent_path):
    """
    Given path to a single tfevent file, extract all logged metrics from it

    Output a dictionary containing:
        episode: episode number, also the 'step' index in tfevent
        metric_name: name of the logged metric
        metric_value: value of the logged metric
    """

    epis_list = []
    metric_list = []
    val_list = []

    # Iterate through each event
    for event in summary_iterator(tfevent_path):
        episode_idx = event.step

        # Get the name and value which was stored
        if hasattr(event.summary, 'value'):
            rec_container = event.summary.value
            if len(rec_container) > 0:
                rec_obj = rec_container.pop()
                metric_name = rec_obj.tag
                metric_val = rec_obj.simple_value

                # NOTE maybe TODO: unsure what happens if metriv_val is
                # supposedly a saved video (i.e. 4-D tensor), but seems to be
                # ok so far?

                # Save
                epis_list.append(episode_idx)
                metric_list.append(metric_name)
                val_list.append(metric_val)

    # Creat metric dictionary
    metric_dict = {
        'episode': epis_list,
        'metric_name': metric_list,
        'metric_value': val_list
    }

    return metric_dict

def run2df(dir_path, hparams):
    """
    Parse the tfevent and hyperaparameter (.csv) file from a single run into
    a pandas.DataFrame object

    :param dir_path: str, the path to the directory containing that run
    :param hparams: List, containin the name of the wanted hyperparameters
    :return: pandas.DataFrame
    """

    # preprocess just in case
    if dir_path[-1] == '/':
        dir_path = dir_path[:-1]

    # ===
    # Find tf event file, assuming there is only one
    tfEvent_files = glob.glob(f'{dir_path}/events*tfevents*')
    tfEvent_path = tfEvent_files[0]

    # ===
    # Find hyperparameter file, assuming there is only one (or None?)
    hparam_files = glob.glob(f'{dir_path}/*csv')
    if len(hparam_files) > 0:
        hparam_path = hparam_files[0]
    else:
        hparam_path = None

    # ===
    # Extract
    if hparam_path is not None:
        hparam_dict = extract_single_hparamCSV(hparam_path)
    metric_dict = extract_single_tfEvent(tfEvent_path)

    # ===
    # Combine the metric with wanted hyparameter values
    copy_number = len(metric_dict['episode'])

    if hparam_path is not None:
        for hparam_name in hparams:
            if hparam_name in hparam_dict:
                hparam_val = hparam_dict[hparam_name]
                hparam_list = [hparam_val] * copy_number

                metric_dict[hparam_name] = hparam_list

    # Add the "run name", assuming it is the directory name
    run_name = dir_path.split('/')[-1]
    run_name_list = [run_name] * copy_number
    metric_dict['run_name'] = run_name_list

    # ==
    # Construct dataframe
    df = pd.DataFrame(metric_dict)
    return df

def parse_experiment(experiment_dir, hparams):
    """
    :param experiment_dir: path to the parent direction containing all the runs
                           for this experiment
    """

    # ==
    # Find all the path to directories containing experimental runs

    # Find all full paths to *tfevent* files
    tfevent_all_paths = glob.glob(f'{experiment_dir}/*/*tfevents*')
    # Remove the tf events for only the directory
    exp_run_paths = ['/'.join(p.split('/')[:-1]) for p in tfevent_all_paths]

    # ==
    # For each run get relevant df
    run_df_list = []
    for run_path in exp_run_paths:
        cur_run_df = run2df(run_path, hparams)
        run_df_list.append(cur_run_df)

    # ==
    # Conbine df
    allruns_df = pd.concat(run_df_list)
    allruns_df = allruns_df.reset_index(drop=True)

    return allruns_df

def main():
    parser = argparse.ArgumentParser(description='Parsing training logs')

    parser.add_argument('--in_dir', type=str, default='/Users/anthony/Google_Drive/McGill/Masters/Courses/COMP767-RL/project/testing/2020-04-22_exp3/',
                        help='path to the experiment directory containing runs')
    parser.add_argument('--out_file', type=str, default=None,
                        help='path to save the output pd.Dataframe')

    args = parser.parse_args()
    print(args)

    # ==
    # Define the hyperparameters I want to include in the df's
    wanted_hparams = ['env_name', 'agent_type', 'seed']

    exp_df = parse_experiment(args.in_dir, wanted_hparams)

    if args.out_file is None:
        print(exp_df)
    else:
        print('Writing DataFrame to:', args.out_file)
        exp_df.to_pickle(args.out_file)


    """
    Example paths

    experiment file:
    '/Users/anthony/Google_Drive/McGill/Masters/Courses/COMP767-RL/project/testing/2020-04-22_exp3/'
    """


main()
