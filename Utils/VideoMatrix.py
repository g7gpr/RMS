# RPi Meteor Station
# Copyright (C) 2023
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



from __future__ import print_function, division, absolute_import


import os
import sys
import shutil

import datetime
import time
import dateutil
import glob
import sqlite3
import multiprocessing
import logging
import copy
import uuid
import random
import string


if sys.version_info[0] < 3:

    import urllib2

    # Fix Python 2 SSL certs
    try:
        import os, ssl
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)): 
            ssl._create_default_https_context = ssl._create_unverified_context
    except:
        # Print the error
        print("Error: {}".format(sys.exc_info()[0]))

else:
    import urllib.request


import numpy as np

import RMS.ConfigReader as cr
from RMS.Astrometry.Conversions import datetime2JD, geo2Cartesian, altAz2RADec, vectNorm, raDec2Vector
from RMS.Astrometry.Conversions import latLonAlt2ECEF, AER2LatLonAlt, AEH2Range, ECEF2AltAz, ecef2LatLonAlt
from RMS.Math import angularSeparationVect
from RMS.Formats.FFfile import convertFRNameToFF
from RMS.Formats.Platepar import Platepar
from RMS.UploadManager import uploadSFTP
from Utils.StackFFs import stackFFs
from Utils.FRbinViewer import view
from Utils.BatchFFtoImage import batchFFtoImage
from RMS.CaptureDuration import captureDuration


# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz


log = logging.getLogger("logger")

"""
This script programmatically generates and optionally executes a command to generate a matrix of videos

It is based on the technique at 

https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos

The command template is 

ffmpeg
	-i 1.avi -i 2.avi -i 3.avi -i 4.avi
	-filter_complex "
		nullsrc=size=640x480 [base];
		[0:v] setpts=PTS-STARTPTS, scale=320x240 [upperleft];
		[1:v] setpts=PTS-STARTPTS, scale=320x240 [upperright];
		[2:v] setpts=PTS-STARTPTS, scale=320x240 [lowerleft];
		[3:v] setpts=PTS-STARTPTS, scale=320x240 [lowerright];
		[base][upperleft] overlay=shortest=1 [tmp1];
		[tmp1][upperright] overlay=shortest=1:x=320 [tmp2];
		[tmp2][lowerleft] overlay=shortest=1:y=240 [tmp3];
		[tmp3][lowerright] overlay=shortest=1:x=320:y=240
	"
	-c:v libx264 output.mkv
"""

def generateOutput(output_file, lib="libx264",print_nicely=False):


    output_clause = "-c:v {} {}".format(lib,output_file)
    output_clause += "\n " if print_nicely else output_clause
    return output_clause

def generateInputVideo(input_videos, print_nicely=False):

    """
    Generate the video input line, format like
    
    	-i 1.avi -i 2.avi -i 3.avi -i 4.avi
    
    """

    input = ""
    for video in input_videos:
        input += "-i  {} ".format(video)
    input += "\n " if print_nicely else input


    return input

def generateFilter(video_paths, resolution_list, layout_list,print_nicely = False):

    print("Number of videos {}".format(len(video_paths)))

    null_video = "nullsrc=size={}x{}[tmp_0]; ".format(resolution_list[0],resolution_list[1])
    print(null_video)
    res_tile = []
    res_tile.append(int(resolution_list[0] / layout_list[0]))
    res_tile.append(int(resolution_list[1] / layout_list[1]))


    video_counter,filter = 0, '-filter_complex " '

    for video in video_paths:
        filter += "[{}:v] setpts=PTS-STARTPTS,scale={}x{}[tile_{}]; ".format(video_counter,res_tile[0],res_tile[1],video_counter)
        video_counter += 1
    filter += "\n " if print_nicely else filter

    tile_count,x_pos,y_pos = 0,0,0
    for tile_down in range(layout_list[1]):
        for tile_across in range(layout_list[0]):
            #[tmp3][lowerright]overlay=shortest=1:x=320:y=240[tmp4]
            filter += "[tmp_{}][tile_{}]overlay=shortest=1:x={},y={}".format(tile_count,tile_count,x_pos,y_pos)
            tile_count += 1
            if tile_count != layout_list[0] * layout_list[1]:
                filter += "[tmp_{}]".format(tile_count)
            else:
                filter += '" '
            x_pos += res_tile[0]
            filter += "\n " if print_nicely else filter
        x_pos = 0
        y_pos += res_tile[1]


    return filter



def generateCommand(video_paths, resolution, shape, output_filename = "output.mp4", print_nicely = False):

    ffmpeg_command_string = "ffmpeg "

    ffmpeg_command_string += generateInputVideo(video_paths,print_nicely=print_nicely)
    print(ffmpeg_command_string)
    ffmpeg_command_string += generateFilter(video_paths,resolution,shape,print_nicely=print_nicely)
    ffmpeg_command_string += generateOutput(output_filename, print_nicely=print_nicely)


    return ffmpeg_command_string


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Generate an n x n matrix of videos. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-i', '--inputs', nargs='*', metavar='INPUT_VIDEOS', type=str,
                            help="Path to the input videos.")

    arg_parser.add_argument('-g', '--generate', dest='generate_video', default=False, action="store_true",
                            help="Generate the video")

    arg_parser.add_argument('-r', '--resolution', nargs='*', metavar='RESOLUTION', type=int,
                            help="outputresolution e.g 1024 768")

    arg_parser.add_argument('-s', '--shape', nargs='*', metavar='SHAPE', type=int,
                            help="Number of tiles across, number of tiles down e.g 4 3")

    arg_parser.add_argument('-o', '--output', nargs='*', metavar='SHAPE', type=str,
                            help="Output filename")

    cml_args = arg_parser.parse_args()

    # Load the config file
    # syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Set the web page to monitor
    print("Input file path   {}".format(cml_args.inputs))
    print("Output resolution {}".format(cml_args.resolution))
    print("Output shape      {}".format(cml_args.shape))
    print("Output file       {}".format(cml_args.output))

    print_nicely = False
    if cml_args.output is None:
        output_filename = "output.mp4"
    else:
        output_filename = cml_args.output[0]

    if len(cml_args.inputs) > 1:
        ffmpeg_command_string = generateCommand(cml_args.inputs, cml_args.resolution, cml_args.shape, cml_args.output)
    elif len(cml_args.inputs) == 1:
        #possibly been passed a directory of videos
        input_videos = os.listdir(cml_args.inputs[0])
        path_input_videos = cml_args.inputs[0]
        input_videos.sort()
        input_video_paths = []
        for video in input_videos:
            input_video_paths.append(os.path.join(path_input_videos,video))
        ffmpeg_command_string = generateCommand(input_video_paths, cml_args.resolution, cml_args.shape, cml_args.output[0],print_nicely)
    elif len(cml_args.inputs) == 0:
        print("No videos found to process")
        quit()


    print("Returned command string \n {}".format(ffmpeg_command_string))