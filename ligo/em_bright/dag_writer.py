# Copyright (C) 2018 Shaon Ghosh, Deep Chatterjee
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


def writeSubfile(executable, arguments, workdir, subfilename,
                 accounting_group):
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
    parser = ArgumentParser(
        "Script to write the DAG for source_classification")
    parser.add_argument("-d", "--dagname", required=True,
                        help="Name of the dag file. Placed under --work-dir")
    parser.add_argument("-w", "--work-dir", required=True,
                        help="Working directory to store data outputs")
    parser.add_argument(
        "-i", "--file-dir", required=True,
        help="File containing input injection sqlite databases"
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Name of the config file")
    parser.add_argument("-e", "--executables-dir", required=True,
                        help="Directory containing executables")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    abs_config_file = os.path.abspath(args.config)

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

    # Get node names
    em_bright_extract_nodename = config.get('node_names',
                                            'em_bright_extract')
    em_bright_join_nodename = config.get('node_names',
                                         'em_bright_join')
    em_bright_categorize_nodename = config.get('node_names',
                                               'em_bright_categorize')
    em_bright_train_nodename = config.get('node_names',
                                          'em_bright_train')
    # Get executable names
    em_bright_extract_executable = config.get('executables',
                                              'em_bright_extract')
    em_bright_join_executable = config.get('executables',
                                           'em_bright_join')
    em_bright_categorize_executable = config.get('executables',
                                                 'em_bright_categorize')
    em_bright_train_executable = config.get('executables',
                                            'em_bright_train')
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
            " --input $(macroinput) --output $(macrooutput) "
            "--eosname $(macroeos)",
            em_bright_categorize_sub
        ),
        em_bright_train_executable: (
            " --input $(macroinput) --config $(macroconfig) "
            "--output $(macrooutput) --param-sweep-plot",
            em_bright_train_sub
        )
    }
    # write all sub files
    for exect, arg_sub in exec_arg_assoc.items():
        writeSubfile(
            exect, arg_sub[0], abs_work_dir,
            arg_sub[1], accounting_group
        )

    # Creating list of sqlite files to create the dag
    inj_list = glob(
        os.path.join(
            os.path.abspath(args.file_dir), inj_file_pattern
        )
    )
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
        line += 'VARS {}{} macroinput="{}"\n'.format(
            em_bright_extract_nodename,
            idx, injFilename
        )
        line += 'VARS {}{} macrooutput="{}"\n\n'.format(
            em_bright_extract_nodename, idx, output_fname
        )
        extracted_data_names.append(output_fname)
        parentchildline += 'PARENT {}{} CHILD {}\n'.format(
            em_bright_extract_nodename, idx,
            em_bright_join_nodename
        )
        daglines += line

    # output for JOIN
    join_output = \
        em_bright_join_prefix + '-' + startT + '-' + \
        duration + em_bright_join_suffix
    join_output = os.path.join(abs_work_dir, join_output)

    line = 'JOB {} {}\n'.format(em_bright_join_nodename,
                                em_bright_join_sub)
    line += 'VARS {} macroinput="{}"\n'.format(
        em_bright_join_nodename,
        abs_work_dir
    )
    line += 'VARS {} macrooutput="{}"\n'.format(
        em_bright_join_nodename,
        join_output
    )
    line += 'VARS {} macroconfig="{}"\n\n'.format(
        em_bright_join_nodename,
        abs_config_file
    )
    parentchildline += 'PARENT {} CHILD {}\n'.format(
        em_bright_join_nodename,
        em_bright_categorize_nodename
    )
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

    parentchildline += 'PARENT {} CHILD {}\n'.format(
        em_bright_categorize_nodename,
        em_bright_train_nodename
    )
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
