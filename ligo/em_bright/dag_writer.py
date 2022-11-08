# Copyright(C) 2018-2022 Shaon Ghosh, Deep Chatterjee, Sushant Sharma Chaudhary
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

from . import EOS_BAYES_FACTORS

try:
    import htcondor
    from htcondor import dags
    _HTCONDOR_INSTALLED = True
except ModuleNotFoundError:
    _HTCONDOR_INSTALLED = False
    import warnings
    warnings.warn(
        "HTCondor python bindings need to be installed to use this script. "
        "This script is only used in training. When training new classifiers, "
        "create a developmental version by cloning the git repository, "
        "and installing from the lock file."
    )


def _add_common_workflow(condor_dag, args, common_submit_dict):
    config = ConfigParser()
    config.read(args.config)
    config_path = os.path.abspath(args.config)
    abs_work_dir = os.path.abspath(args.work_dir)
    file_dir = args.file_dir
    # Get injection directory
    inj_file_pattern = config.get('core',
                                  'inj_file_pattern')
    sqlite_run_tag = config.get('core',
                                'sqlite_run_tag')

    # Get input and output file names prefixes
    em_bright_extract_prefix = config.get('output_filenames',
                                          'em_bright_extract_prefix')
    em_bright_extract_suffix = config.get('output_filenames',
                                          'em_bright_extract_suffix')
    em_bright_join_prefix = config.get('output_filenames',
                                       'em_bright_join_prefix')
    em_bright_join_suffix = config.get('output_filenames',
                                       'em_bright_join_suffix')
    # Get executable names
    em_bright_extract_executable = config.get('executables',
                                              'em_bright_extract')
    em_bright_join_executable = config.get('executables',
                                           'em_bright_join')

    # Define the executable arguments association
    exec_arg_assoc = {
        em_bright_extract_executable:
            " --input $(macroinput) --output $(macrooutput)",
        em_bright_join_executable:
            " --input $(macroinput) --config $(macroconfig)"
            " --output $(macrooutput)"
    }
    # create a dictionary of Submit instances
    condor_sub_dict = dict.fromkeys(exec_arg_assoc)
    # create all sub files

    for exect, arg_sub in exec_arg_assoc.items():
        common_submit_dict["executable"] = exect
        common_submit_dict["arguments"] = arg_sub
        condor_sub_dict[exect] = htcondor.Submit(
            common_submit_dict)
    # Creating list of sqlite files to create the dag
    inj_list = glob(
        os.path.join(
            os.path.abspath(file_dir), inj_file_pattern
        )
    )

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
            macroconfig=config_path
        )
    ]
    em_bright_join_layer = em_bright_extract_layer.child_layer(
        name=em_bright_join_executable,
        submit_description=condor_sub_dict[em_bright_join_executable],
        vars=em_bright_join_vars
    )
    join_output_dict = {
        "output": join_output,
        "startT": startT,
        "duration": duration
    }
    return em_bright_join_layer, join_output_dict


def _add_knn_workflow(em_bright_join_layer, args,
                      common_submit_dict,
                      join_output_dict):
    config = ConfigParser()
    config.read(args.config)
    config_path = os.path.abspath(args.config)
    abs_work_dir = os.path.abspath(args.work_dir)

    em_bright_categorize_prefix = config.get('output_filenames',
                                             'em_bright_categorize_prefix')
    em_bright_categorize_suffix = config.get('output_filenames',
                                             'em_bright_categorize_suffix')
    em_bright_train_prefix = config.get('output_filenames',
                                        'em_bright_train_prefix')
    em_bright_train_suffix = config.get('output_filenames',
                                        'em_bright_train_suffix')
    em_bright_categorize_executable = config.get('executables',
                                                 'em_bright_categorize')
    em_bright_train_executable = config.get('executables',
                                            'em_bright_train')
    em_bright_paramater_sweep_plot_executable = config.get(
            'executables', 'em_bright_create_param_sweep_plot'
    )
    exec_arg_assoc = {
        em_bright_train_executable:
            " --input $(macroinput) --config $(macroconfig) "
            "--output $(macrooutput) --param-sweep-plot-prefix $(macroeos)",
        em_bright_paramater_sweep_plot_executable:
            " --input $(macroinput) --config $(macroconfig)"
            " --verbose",
        em_bright_categorize_executable:
            " --input $(macroinput) --output $(macrooutput)"
            " --eosname $(macroeos) "
    }
    condor_sub_dict = dict.fromkeys(exec_arg_assoc)

    for exect, arg_sub in exec_arg_assoc.items():
        common_submit_dict["executable"] = exect
        common_submit_dict["arguments"] = arg_sub
        condor_sub_dict[exect] = htcondor.Submit(
            common_submit_dict)

    # input and output for CATEGORIZE/TRAIN
    em_bright_categorize_vars = list()
    em_bright_train_vars = list()
    categorize_input = join_output_dict["output"]

    for em_bright_eos in EOS_BAYES_FACTORS:
        categorize_output = \
            em_bright_categorize_prefix + '-' + join_output_dict["startT"] + \
            '-' + join_output_dict["duration"] + '-' + em_bright_eos + \
            em_bright_categorize_suffix
        categorize_output = os.path.join(abs_work_dir, categorize_output)

        train_output = \
            em_bright_train_prefix + '-' + join_output_dict["startT"] + \
            '-' + join_output_dict["duration"] + \
            '-' + em_bright_eos + em_bright_train_suffix
        train_output = os.path.join(abs_work_dir, train_output)

        em_bright_categorize_vars.append(
            dict(
                macroinput=categorize_input,
                macrooutput=categorize_output,
                macroeos=em_bright_eos,
            )
        )
        em_bright_train_vars.append(
                dict(
                    macroinput=categorize_output,
                    macrooutput=train_output,
                    macroconfig=config_path,
                    macroeos=em_bright_eos
                )
            )

    em_bright_categorize_layer = em_bright_join_layer.child_layer(
        name=em_bright_categorize_executable,
        submit_description=condor_sub_dict[em_bright_categorize_executable],
        vars=em_bright_categorize_vars
    )

    em_bright_train_layer = em_bright_categorize_layer.child_layer(
            name=em_bright_train_executable,
            submit_description=condor_sub_dict[em_bright_train_executable],
            vars=em_bright_train_vars
        )

    em_bright_param_sweep_layer = em_bright_train_layer.child_layer(  # noqa: F841,E501
            name=em_bright_paramater_sweep_plot_executable,
            submit_description=condor_sub_dict[
                em_bright_paramater_sweep_plot_executable],
            vars=[
                dict(
                    macroinput=abs_work_dir,
                    macroconfig=config_path
                )
            ]
        )


def _add_massgap_workflow(em_bright_join_layer, args,
                          common_submit_dict,
                          join_output_dict):
    config = ConfigParser()
    config.read(args.config)
    config_path = os.path.abspath(args.config)
    abs_work_dir = os.path.abspath(args.work_dir)

    em_bright_categorize_prefix = config.get('output_filenames',
                                             'em_bright_categorize_prefix')
    em_bright_categorize_suffix = config.get('output_filenames',
                                             'em_bright_categorize_suffix')
    em_bright_train_suffix = config.get('output_filenames',
                                        'em_bright_train_suffix')
    em_bright_categorize_executable = config.get('executables',
                                                 'em_bright_categorize')
    em_bright_train_executable = config.get('executables',
                                            'em_bright_train')
    exec_arg_assoc = {
        em_bright_categorize_executable:
            " --input $(macroinput) --output $(macrooutput)"
            " --mass-gap",
        em_bright_train_executable:
            " --input $(macroinput) --config $(macroconfig)"
            " --output $(macrooutput) --mass-gap"
    }
    condor_sub_dict = dict.fromkeys(exec_arg_assoc)
    for exect, arg_sub in exec_arg_assoc.items():
        common_submit_dict["executable"] = exect
        common_submit_dict["arguments"] = arg_sub
        condor_sub_dict[exect] = htcondor.Submit(
            common_submit_dict)

    categorize_input = join_output_dict["output"]
    categorize_output = \
        em_bright_categorize_prefix + '-' + join_output_dict["startT"] + \
        '-' + join_output_dict["duration"] + \
        em_bright_categorize_suffix
    categorize_output = os.path.join(abs_work_dir, categorize_output)

    em_bright_categorize_vars = [
            dict(
                macroinput=categorize_input,
                macrooutput=categorize_output,
            )
    ]
    em_bright_categorize_layer = em_bright_join_layer.child_layer(
        name=em_bright_categorize_executable,
        submit_description=condor_sub_dict[em_bright_categorize_executable],
        vars=em_bright_categorize_vars
    )

    train_output = "mass_gap" + \
        em_bright_train_suffix
    train_output = os.path.join(abs_work_dir, train_output)

    em_bright_train_vars = [
         dict(
                macroinput=categorize_output,
                macrooutput=train_output,
                macroconfig=config_path,
            )
    ]
    em_bright_categorize_layer.child_layer(
            name=em_bright_train_executable,
            submit_description=condor_sub_dict[em_bright_train_executable],
            vars=em_bright_train_vars
        )


def main():
    parser = ArgumentParser("Condor DAG writer for workflow")
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
    parser.add_argument("--mass-gap", action='store_true',
                        help="use --random_forest for Random Forest Mode")
    args = parser.parse_args()

    assert _HTCONDOR_INSTALLED, "HTCondor python bindings missing."

    config = ConfigParser()
    config.read(args.config)
    # Get path for the work directory
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    abs_work_dir = os.path.abspath(args.work_dir)
    accounting_group = config.get('core',
                                  'accounting_group')
    # changes made here
    common_submit_dict = {
            'universe': 'vanilla',
            'request_disk': '1GB',
            'output': '$(executable).stdout',
            'error': '$(executable).stderr',
            'log': '$(executable).log',
            'accounting_group': accounting_group
    }

    condor_dag = dags.DAG()
    join_layer, join_output_dict = _add_common_workflow(
                                                condor_dag,
                                                args,
                                                common_submit_dict)
    if not args.mass_gap:
        _add_knn_workflow(join_layer, args,
                          common_submit_dict,
                          join_output_dict)
    else:
        _add_massgap_workflow(join_layer, args,
                              common_submit_dict,
                              join_output_dict)

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


if __name__ == "__main__":
    main()
