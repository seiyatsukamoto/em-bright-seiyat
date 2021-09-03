# Copyright (C) 2018 Shaon Ghosh, Deep Chatterjee, Shasvath Kapadia
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import os
import re
import shutil

from argparse import ArgumentParser
from configparser import ConfigParser
from glob import glob


def writeSubfile(executable, arguments, workdir, subfilename, accounting_group):
    line = 'universe = vanilla\n'
    line += 'executable = {}\n'.format(executable)
    line += 'arguments = "{}"\n'.format(arguments)
    line += 'output = $(executable).stdout\n'
    line += 'error = $(executable).stderr\n'
    line += 'log = $(executable).log\n'
    line += 'accounting_group = {}\n'.format(accounting_group)
    line += 'queue 1\n'

    with open(os.path.join(workdir, subfilename), 'w') as f:
        f.writelines(line)


def main():
    parser = ArgumentParser("Script to write the DAG for source_classification")
    parser.add_argument("-d", "--dagname", required=True,
                        help="Name of the dag file. Placed under --work-dir")
    parser.add_argument("-w", "--work-dir", required=True,
                        help="Working directory to store data outputs")
    parser.add_argument("-i", "--file-dir", required=True,
                        help="File containing input injection sqlite databases")
    parser.add_argument("-c", "--config", required=True,
                        help="Name of the config file")
    parser.add_argument("-e", "--executables-dir", required=True,
                        help="Directory containing executables")
    parser.add_argument("-f", "--data-history", required=True,
                        help="Bayes factor and weights data from previous runs")
    parser.add_argument("-t", "--trigger-db", required=True,
                        help="Trigger database required for p_astro")
    parser.add_argument("-r", "--ranking-data", required=True,
                        help="ranking_data.xml.gz required by p_astro")
    args = parser.parse_args()

    # The binaries that needs to be copied for condor jobs to run
    #bin_path = os.path.join(args.executables_dir, 'bin')
    #assert path.exists(bin_path), \
    #    "{} does not have a bin".format(args.executables_dir)

    config = ConfigParser()
    config.read(args.config)
    abs_config_file = os.path.abspath(args.config)
    abs_trigger_db = os.path.abspath(args.trigger_db)
    abs_ranking_data = os.path.abspath(args.ranking_data)
    abs_data_history = os.path.abspath(args.data_history)

    # Get path for the work directory
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    abs_work_dir = os.path.abspath(args.work_dir)

    # Get injection directory
    inj_file_pattern = config.get('core',
                                  'inj_file_pattern')
    sqlite_run_tag = config.get('core',
                                'sqlite_run_tag')
    accounting_group = config.get('core',
                                  'accounting_group')

    # Get subfile names
    em_bright_extract_sub = config.get('sub_names',
                                       'em_bright_extract')
    em_bright_join_sub = config.get('sub_names',
                                    'em_bright_join')
    em_bright_categorize_sub = config.get('sub_names',
                                          'em_bright_categorize')
    em_bright_train_sub = config.get('sub_names',
                                     'em_bright_train')
    p_astro_histogram_by_bin_sub = \
        config.get('sub_names',
                   'p_astro_histogram_by_bin')
    p_astro_compute_means_sub = \
        config.get('sub_names',
                   'p_astro_compute_means')
    # Get input and output file names prefixes
    em_bright_extract_prefix = config.get('output_filenames',
                                          'em_bright_extract_prefix')
    em_bright_extract_suffix = config.get('output_filenames',
                                          'em_bright_extract_suffix')
    em_bright_join_prefix = config.get('output_filenames',
                                       'em_bright_join_prefix')
    em_bright_join_suffix = config.get('output_filenames',
                                       'em_bright_join_suffix')
    em_bright_categorize_prefix = config.get('output_filenames',
                                             'em_bright_categorize_prefix')
    em_bright_categorize_suffix = config.get('output_filenames',
                                             'em_bright_categorize_suffix')
    em_bright_train_prefix = config.get('output_filenames',
                                        'em_bright_train_prefix')
    em_bright_train_suffix = config.get('output_filenames',
                                        'em_bright_train_suffix')
    p_astro_histogram_by_bin_prefix = config.get('output_filenames',
                                                 'p_astro_histogram_by_bin_prefix')
    p_astro_histogram_by_bin_suffix = config.get('output_filenames',
                                                 'p_astro_histogram_by_bin_suffix')
    p_astro_compute_means_prefix = config.get('output_filenames',
                                              'p_astro_compute_means_prefix')
    p_astro_compute_means_suffix = config.get('output_filenames',
                                              'p_astro_compute_means_suffix')

    # Get node names
    em_bright_extract_nodename = config.get('node_names',
                                            'em_bright_extract')
    em_bright_join_nodename = config.get('node_names',
                                         'em_bright_join')
    em_bright_categorize_nodename = config.get('node_names',
                                               'em_bright_categorize')
    em_bright_train_nodename = config.get('node_names',
                                          'em_bright_train')
    p_astro_histogram_by_bin_nodename = \
        config.get('node_names',
                   'p_astro_histogram_by_bin')
    p_astro_compute_means_nodename = \
        config.get('node_names',
                   'p_astro_compute_means')
    # Get executable names
    em_bright_extract_executable = config.get('executables',
                                              'em_bright_extract')
    em_bright_join_executable = config.get('executables',
                                           'em_bright_join')
    em_bright_categorize_executable = config.get('executables',
                                                 'em_bright_categorize')
    em_bright_train_executable = config.get('executables',
                                            'em_bright_train')
    p_astro_histogram_by_bin_executable = \
        config.get('executables',
                   'p_astro_histogram_by_bin')
    p_astro_compute_means_executable = \
        config.get('executables',
                   'p_astro_compute_means')
    # Get the EoS to be used for EM bright categorization
    em_bright_eos = config.get('em_bright', 'eos_name')
    # Define the executable arguments association
    # FIXME the assoc should potentially be a part of the config
    exec_arg_assoc = {
        em_bright_extract_executable: (
            " --input $(macroinput) --output $(macrooutput)",
            em_bright_extract_sub
        ),
        em_bright_join_executable: (
            " --input $(macroinput) --config $(macroconfig)"
            " --output $(macrooutput)",
            em_bright_join_sub
        ),
        em_bright_categorize_executable: (
            " --input $(macroinput) --output $(macrooutput) --eosname $(macroeos)",
            em_bright_categorize_sub
        ),
        em_bright_train_executable: (
            " --input $(macroinput) --config $(macroconfig) --output $(macrooutput) --param-sweep-plot",
            em_bright_train_sub
        ),
        p_astro_histogram_by_bin_executable: (
            " --input $(macroinput) --output $(macrooutput) --config $(macroconfig)",
            p_astro_histogram_by_bin_sub
        ),
        p_astro_compute_means_executable: (
            " --input $(macroinput)"
            " --ranking-data $(macrorankingdata)"
            " --config $(macroconfig)"
            " --output $(macrooutput)"
            " --trigger-db $(macrotriggerdb)"
            " --data-history $(macrodatahistory)",
            p_astro_compute_means_sub
        )
    }
    # write all sub files
    for exect, arg_sub in exec_arg_assoc.items():
        writeSubfile(exect, arg_sub[0], abs_work_dir, arg_sub[1], accounting_group)

    # Creating list of sqlite files to create the dag
    inj_list = glob(os.path.join(os.path.abspath(args.file_dir), inj_file_pattern))
    # Writing the DAG
    daglines = ''
    parentchildline = ''
    extracted_data_names = []
    for idx, injFilename in enumerate(inj_list):
        match = re.search(sqlite_run_tag, os.path.basename(injFilename))
        if match is None:
            continue
        prefix, startT, duration = match.groups()
        # filename for INJCOINC files
        output_fname = \
            em_bright_extract_prefix + \
            prefix + '-' + startT + '-' + duration + em_bright_extract_suffix

        line = "JOB {}{} {}\n".format(em_bright_extract_nodename,
                                      idx,
                                      em_bright_extract_sub)
        line += 'VARS {}{} macroinput="{}"\n'.format(em_bright_extract_nodename,
                                                     idx,
                                                     injFilename)
        line += 'VARS {}{} macrooutput="{}"\n\n'.format(em_bright_extract_nodename,
                                                        idx,
                                                        output_fname)
        extracted_data_names.append(output_fname)
        parentchildline += 'PARENT {}{} CHILD {}\n'.format(em_bright_extract_nodename,
                                                           idx,
                                                           em_bright_join_nodename)
        daglines += line

    # output for JOIN
    join_output = \
        em_bright_join_prefix + '-' + startT + '-' + \
        duration + em_bright_join_suffix
    join_output = os.path.join(abs_work_dir, join_output)

    line = 'JOB {} {}\n'.format(em_bright_join_nodename,
                                em_bright_join_sub)
    line += 'VARS {} macroinput="{}"\n'.format(em_bright_join_nodename,
                                               abs_work_dir)
    line += 'VARS {} macrooutput="{}"\n'.format(em_bright_join_nodename,
                                                  join_output)
    line += 'VARS {} macroconfig="{}"\n\n'.format(em_bright_join_nodename,
                                                abs_config_file)
    parentchildline += 'PARENT {} CHILD {}\n'.format(em_bright_join_nodename,
                                                     em_bright_categorize_nodename)
    daglines += line

    # input and output for CATEGORIZE
    categorize_input = join_output
    categorize_output = \
        em_bright_categorize_prefix + '-' + startT + '-' + duration + \
        em_bright_categorize_suffix
    categorize_output = os.path.join(abs_work_dir, categorize_output)

    line = 'JOB {} {}\n'.format(em_bright_categorize_nodename,
                                em_bright_categorize_sub)
    line += 'VARS {} macroinput="{}"\n'.format(em_bright_categorize_nodename,
                                               categorize_input)
    line += 'VARS {} macrooutput="{}"\n'.format(em_bright_categorize_nodename,
                                                  categorize_output)
    line += 'VARS {} macroeos="{}"\n\n'.format(em_bright_categorize_nodename,
                                               em_bright_eos)

    parentchildline += 'PARENT {} CHILD {}\n'.format(em_bright_categorize_nodename,
                                                     em_bright_train_nodename)
    daglines += line

    # Training
    train_input = categorize_output
    train_output = \
        em_bright_train_prefix + '-' + startT + '-' + duration + \
        em_bright_train_suffix
    train_output = os.path.join(abs_work_dir, train_output)

    line = 'JOB {} {}\n'.format(em_bright_train_nodename,
                                em_bright_train_sub)
    line += 'VARS {} macroinput="{}"\n'.format(em_bright_train_nodename,
                                               train_input)
    line += 'VARS {} macrooutput="{}"\n'.format(em_bright_train_nodename,
                                                  train_output)
    line += 'VARS {} macroconfig="{}"\n\n'.format(em_bright_train_nodename,
                                                abs_config_file)

    # BINNING
    binning_input = join_output
    binning_output = p_astro_histogram_by_bin_prefix + '-' + startT + '-' + \
        duration + p_astro_histogram_by_bin_suffix
    binning_output = os.path.join(abs_work_dir, binning_output)

    line += 'JOB {} {}\n'.format(p_astro_histogram_by_bin_nodename,
                                 p_astro_histogram_by_bin_sub)
    line += 'VARS {} macroinput="{}"\n'.format(p_astro_histogram_by_bin_nodename,
                                               binning_input)
    line += 'VARS {} macrooutput="{}"\n'.format(p_astro_histogram_by_bin_nodename,
                                                  binning_output)
    line += 'VARS {} macroconfig="{}"\n\n'.format(p_astro_histogram_by_bin_nodename,
                                                abs_config_file)

    parentchildline += 'PARENT {} CHILD {}\n'.format(em_bright_join_nodename,
                                                     p_astro_histogram_by_bin_nodename)

    # compute the means
    means_input = binning_output
    means_output = p_astro_compute_means_prefix + '-' + startT + '-' + duration + \
        p_astro_compute_means_suffix
    means_output = os.path.join(abs_work_dir, means_output)

    line += 'JOB {} {}\n'.format(p_astro_compute_means_nodename,
                                 p_astro_compute_means_sub)
    line += 'VARS {} macroinput="{}"\n'.format(p_astro_compute_means_nodename,
                                               means_input)
    line += 'VARS {} macrooutput="{}"\n'.format(p_astro_compute_means_nodename,
                                                means_output)
    line += 'VARS {} macroconfig="{}"\n'.format(p_astro_compute_means_nodename,
                                                abs_config_file)
    line += 'VARS {} macrorankingdata="{}"\n'.format(p_astro_compute_means_nodename,
                                                     abs_ranking_data)
    line += 'VARS {} macrodatahistory="{}"\n'.format(p_astro_compute_means_nodename,
                                                     abs_data_history)
    line += 'VARS {} macrotriggerdb="{}"\n\n'.format(p_astro_compute_means_nodename,
                                                     abs_trigger_db)
    parentchildline += 'PARENT {} CHILD {}\n'.format(p_astro_histogram_by_bin_nodename,
                                                     p_astro_compute_means_nodename)
    daglines += line
    daglines += parentchildline

    dagname = os.path.join(abs_work_dir, args.dagname)
    with open(dagname, 'w') as f:
        f.writelines(daglines)

    # FIXME Condor refuses to carry over env vars
    # remove when solution is found
    for exe in config['executables']:
        src = os.path.join(args.executables_dir, exe)
        dst = os.path.join(abs_work_dir, exe)
        if os.path.lexists(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    print('DAG is written in {}'.format(os.path.abspath(dagname)))
