# Copyright (C) 2018-2021 Shaon Ghosh, Deep Chatterjee
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

import htcondor
from htcondor import dags


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
        em_bright_extract_executable:
            " --input $(macroinput) --output $(macrooutput)",
        em_bright_join_executable:
            " --input $(macroinput) --config $(macroconfig)"
            " --output $(macrooutput)",
        em_bright_categorize_executable:
            " --input $(macroinput) --output $(macrooutput) "
            "--eosname $(macroeos)",
        em_bright_train_executable:
            " --input $(macroinput) --config $(macroconfig) "
            "--output $(macrooutput) --param-sweep-plot",
    }
    # create a dictionary of Submit instances
    condor_sub_dict = dict.fromkeys(exec_arg_assoc)
    # create all sub files
    for exect, arg_sub in exec_arg_assoc.items():
        condor_sub_dict[exect] = htcondor.Submit(
            universe='vanilla',
            executable=exect,
            arguments=arg_sub,
            output='$(executable).stdout',
            error='$(executable).stderr',
            log='$(executable).log',
            accounting_group=accounting_group
        )

    # Creating list of sqlite files to create the dag
    inj_list = glob(
        os.path.join(
            os.path.abspath(args.file_dir), inj_file_pattern
        )
    )
    # Instantiate the DAG
    condor_dag = dags.DAG()
    # get vars for em_bright_extract layer
    em_bright_extract_vars = []
    for idx, injFilename in enumerate(inj_list):
        match = re.search(sqlite_run_tag, os.path.basename(injFilename))
        if match is None:
            continue
        prefix, startT, duration = match.groups()
        # filename for INJCOINC files
        output_fname = \
            em_bright_extract_prefix + \
            prefix + '-' + startT + '-' + duration + em_bright_extract_suffix
        em_bright_extract_vars.append(
            dict(
                macroinput=injFilename,
                macrooutput=output_fname
            )
        )

    em_bright_extract_layer = condor_dag.layer(
        name=em_bright_extract_executable,
        submit_description=condor_sub_dict[em_bright_extract_executable],
        vars=em_bright_extract_vars
    )
    # output for JOIN
    join_output = \
        em_bright_join_prefix + '-' + startT + '-' + \
        duration + em_bright_join_suffix
    join_output = os.path.join(abs_work_dir, join_output)

    em_bright_join_vars = [
        dict(
            macroinput=abs_work_dir,
            macrooutput=join_output,
            macroconfig=abs_config_file
        )
    ]
    em_bright_join_layer = em_bright_extract_layer.child_layer(
        name=em_bright_join_executable,
        submit_description=condor_sub_dict[em_bright_join_executable],
        vars=em_bright_join_vars
    )

    # input and output for CATEGORIZE
    categorize_input = join_output
    categorize_output = \
        em_bright_categorize_prefix + '-' + startT + '-' + duration + \
        em_bright_categorize_suffix
    categorize_output = os.path.join(abs_work_dir, categorize_output)

    em_bright_categorize_vars = [
        dict(
            macroinput=categorize_input,
            macrooutput=categorize_output,
            macroeos=em_bright_eos
        )
    ]
    em_bright_categorize_layer = em_bright_join_layer.child_layer(
        name=em_bright_categorize_executable,
        submit_description=condor_sub_dict[em_bright_categorize_executable],
        vars=em_bright_categorize_vars
    )
    # Training
    train_input = categorize_output
    train_output = \
        em_bright_train_prefix + '-' + startT + '-' + duration + \
        em_bright_train_suffix
    train_output = os.path.join(abs_work_dir, train_output)

    em_bright_train_vars = [
        dict(
            macroinput=train_input,
            macrooutput=train_output,
            macroconfig=abs_config_file
        )
    ]

    em_bright_train_layer = em_bright_categorize_layer.child_layer(  # noqa: F841,E501
        name=em_bright_train_executable,
        submit_description=condor_sub_dict[em_bright_train_executable],
        vars=em_bright_train_vars
    )

    print("DAG desription:\n", condor_dag.describe())
    dags.write_dag(condor_dag, abs_work_dir,
                   dag_file_name=args.dagname)

    # FIXME Condor refuses to carry over env vars
    # remove when solution is found
    for exe in config['executables']:
        src = os.path.join(args.executables_dir, exe)
        dst = os.path.join(abs_work_dir, exe)
        if os.path.lexists(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    print('{} is written in {}'.format(args.dagname, abs_work_dir))
