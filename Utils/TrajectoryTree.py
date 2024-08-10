# Copyright (C) 2024
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Routine to build a hierarchical file structure of trajectory files for rapid searching


"""

import os.path
from RMS.Misc import mkdirP
from Utils.MirrorTrajectories import getHeaders, getFieldsFromRow, createFileWithHeaders
import tqdm as tqdm

start_path = os.path.expanduser("~/RMS_data/trajectory_tree")
all_file = os.path.expanduser("~/RMS_data/traj_summary_data/traj_summary_all.txt")

def buildTree(start_path, traj_file, ignore_line_marker = "#"):


    mkdirP(start_path)
    header_list = getHeaders(traj_file)
    print(header_list)
    hierarchy_list = ['Sol lon deg', 'LAMgeo deg', 'BETgeo deg', 'Vgeo km/s' ]
    abb_list = ['sl', 'lg', 'bg', 'vg']

    with open(traj_file) as input_fh:
        for line in tqdm.tqdm(input_fh):
            if line[0] == "\n" or line[0] == ignore_line_marker:
                continue
            values = getFieldsFromRow(line, hierarchy_list, header_list)

            floats, fixed_values = [], []
            for value in values:
                floats.append(float(value))
                fixed_values.append(round(float(value)))

            hierarchy_values_list = []
            for value in fixed_values:
                if value > 0:
                    value_string = "+{:.0f}".format(value)

                else:
                    value_string = "{:.0f}".format(value)
                hierarchy_values_list.append(value_string)

            hierarchical_path = start_path
            for hierarchy, abb, hierarchy_value in zip(hierarchy_list, abb_list, hierarchy_values_list):
                hierarchical_path = os.path.join(hierarchical_path, "{}_{}".format(abb, hierarchy_value))

            mkdirP(hierarchical_path)

            traj_counter = 0
            trajectory_file_name_without_count = ""
            for abb, value in zip(abb_list, hierarchy_values_list):
                trajectory_file_name_without_count += "{}_{}_".format(abb,value)


            if len(os.listdir(hierarchical_path)) == 0:
                trajectory_file_name = "{:0>8}_{}.txt".format(traj_counter, trajectory_file_name_without_count[:-1])
                trajectory_full_path = os.path.join(hierarchical_path, trajectory_file_name)
                createFileWithHeaders(trajectory_full_path, traj_file)
            else:
                trajectory_file_name = os.listdir(hierarchical_path)[0]
                trajectory_full_path = os.path.join(hierarchical_path, trajectory_file_name)

            new_trajectory = True


            fh_test = open(trajectory_full_path, "r")
            for existing_line in fh_test:
                if existing_line[0] == "\n" or existing_line[0] == ignore_line_marker:
                    continue
                traj_counter += 1
                if existing_line == line:
                    new_trajectory = False
                    break
            fh_test.close()

            if new_trajectory:
                with open(trajectory_full_path, "a") as fh_output:
                    fh_output.write(line)
                    traj_counter +=1
                new_name = "{:0>8}_{}.txt".format(traj_counter,trajectory_file_name_without_count[:-1])
                new_full_path = os.path.join(hierarchical_path, new_name)
                #print("\nRenaming {}".format(trajectory_full_path))
                #print("To       {}".format(new_full_path))
                os.rename(trajectory_full_path, new_full_path)



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Download new or changed files from a web page of links \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-f', '--force_reload', dest='force_reload', default=False, action="store_true",
                            help="Force download of all files")

    arg_parser.add_argument('-a', '--all', dest='all', default=False, action="store_true",
                            help="Create a trajectory_summary_all file")

    arg_parser.add_argument('-m', '--max_downloads', dest='max_downloads', default=7, type=int,
                            help="Maximum number of files to download")

    arg_parser.add_argument('-w', '--webpage', dest='page',
                            default="https://globalmeteornetwork.org/data/traj_summary_data/daily/", type=str,
                            help="Webpage to use")

    arg_parser.add_argument('-n', '--no_download', dest='no_download', default=False, action="store_true",
                            help="Do not download anything")

    arg_parser.add_argument('-d', '--drop_dup', dest='drop_duplicates', default=False, action="store_true",
                            help="Detect duplicates and remove from traj_all file")

    cml_args = arg_parser.parse_args()



    buildTree(start_path, all_file)