# RPi Meteor Station
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




from __future__ import print_function, division, absolute_import


import os
import sys
import logging
import subprocess




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
import requests
import tempfile

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

log = logging.getLogger("logger")


def downloadFilesToTmp(urls, stationIDs):

    if cml_args.folder is None:
        working_dir = tempfile.mkdtemp()
        delete_at_end = True
    else:
        working_dir = cml_args.folder
        delete_at_end = False
    print("Folder for downloaded files {}".format(working_dir))


    if len(urls) > 0:
        temp_dir = working_dir
        print("Working in {:s}".format(temp_dir))

    video_paths = []
    for video_url, stationID in zip(urls, stationIDs):

        print("Downloading from URL {:s}".format(video_url))
        video = requests.get(video_url, allow_redirects=True)
        destination_file = os.path.join(temp_dir, "{:s}.mp4".format(stationID.lower()))
        print("Saving into {}".format(destination_file))
        open(destination_file,"wb").write(video.content)
        video_paths.append(destination_file)

    return working_dir, video_paths


def convertListOfStationIDsToListOfUrls(station_ids):


    """
    
    Args:
        station_id: a list of stationIDs


    Returns:
        list of urls pointing to the latest static video for that station
    
    
    e.g converts ["AU000A", "AU000C"] to 
    ["https://globalmeteornetwork.org/weblog/AU/AU000A/static/AU000A_timelapse_static.mp4",
    "https://globalmeteornetwork.org/weblog/AU/AU000C/static/AU000C_timelapse_static.mp4"]
    
    
    """



    video_urls = []
    video_url_template = "https://globalmeteornetwork.org/weblog/{:s}/{:s}/static/{:s}_timelapse_static.mp4"


    for station in station_ids:
        country_code = station[0:2]
        print("Station id {}, Country Code {}".format(station,country_code))
        video_url = video_url_template.format(country_code,station,station)
        video_urls.append(video_url)



    return video_urls

def generateOutput(output_file, lib="libx264",print_nicely=False):


    output_clause = " -c:v {} {}".format(lib,output_file)
    output_clause += "\n " if print_nicely else " "
    return output_clause

def generateInputVideo(input_videos, tile_count, print_nicely=False):

    """
    Generate the video input line, format like
    
    	-i 1.avi -i 2.avi -i 3.avi -i 4.avi
    
    """

    input,vid_count = "", 0
    for video in input_videos:
        input += "-i  {} ".format(video)
        vid_count += 1
        input += "\n " if print_nicely else " "
        if vid_count > tile_count:
            break
        print(input)

    print(input)
    return input

def generateFilter(video_paths, resolution_list, layout_list,print_nicely = False):

    print("Number of videos {}".format(len(video_paths)))

    null_video = "nullsrc=size={}x{}[tmp_0]; ".format(resolution_list[0],resolution_list[1])
    print(null_video)
    res_tile = []
    res_tile.append(int(resolution_list[0] / layout_list[0]))
    res_tile.append(int(resolution_list[1] / layout_list[1]))


    video_counter,filter = 0, '-filter_complex " '
    filter += null_video
    filter += "\n " if print_nicely else " "
    for video in video_paths:
        filter += "[{}:v] setpts=PTS-STARTPTS,scale={}x{}[tile_{}]; ".format(video_counter,res_tile[0],res_tile[1],video_counter)
        video_counter += 1
        if video_counter == layout_list[0] * layout_list[1]:
            break
    filter += "\n " if print_nicely else " "

    tile_count,x_pos,y_pos = 0,0,0
    for tile_down in range(layout_list[1]):
        for tile_across in range(layout_list[0]):

            filter += "[tmp_{}][tile_{}]overlay=shortest=1:x={}:y={}".format(tile_count,tile_count,x_pos,y_pos)
            tile_count += 1
            if tile_count != layout_list[0] * layout_list[1]:
                filter += "[tmp_{}] ; ".format(tile_count)
            else:
                filter += '" '
            x_pos += res_tile[0]
            filter += "\n " if print_nicely else " "
        x_pos = 0
        y_pos += res_tile[1]


    return filter



def generateCommand(video_paths, resolution, shape, output_filename = "output.mp4", print_nicely = False):


    ffmpeg_command_string = "ffmpeg -y -r 30 "
    ffmpeg_command_string += generateInputVideo(video_paths, shape[0] * shape[1],print_nicely=print_nicely)
    ffmpeg_command_string += generateFilter(video_paths,resolution,shape,print_nicely=print_nicely)
    ffmpeg_command_string += generateOutput(output_filename, print_nicely=print_nicely)
    print(ffmpeg_command_string)

    return ffmpeg_command_string


if __name__ == "__main__":

    import argparse

    def list_of_strings(arg):
        return arg.split(',')


    arg_parser = argparse.ArgumentParser(description="Generate an n x n matrix of videos. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-i', '--inputs', nargs=1, metavar='INPUT_VIDEOS', type=str,
                            help="Path to the input videos.")

    arg_parser.add_argument('-c', '--cameras', metavar='CAMERAS', type=list_of_strings,
                            help="Cameras to use.")


    arg_parser.add_argument('-g', '--generate', dest='generate_video', default=False, action="store_true",
                            help="Generate the video, instead of just the command")

    arg_parser.add_argument('-r', '--resolution', nargs=2, metavar='RESOLUTION', type=int,
                            help="outputresolution e.g 1024 768")

    arg_parser.add_argument('-s', '--shape', nargs=2, metavar='SHAPE', type=int,
                            help="Number of tiles across, number of tiles down e.g 4 3")

    arg_parser.add_argument('-o', '--output', nargs=1, metavar='OUTPUT', type=str,
                            help="Output filename")

    arg_parser.add_argument('-w', '--weblog', nargs=1, dest='cameras', metavar='SHAPE', type=str,
                            help="Pull the latest videos from the GMN weblog from the given list of cameras")

    arg_parser.add_argument('-f', '--folder', nargs=1, dest='folder', metavar='SHAPE', type=str,
                            help="Destination for downloaded files")


    cml_args = arg_parser.parse_args()


    # Load the config file
    # syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Set the web page to monitor
    if cml_args.inputs is None and cml_args.cameras is None:
        print("No input arguments given, quitting")
        exit()







    if cml_args.resolution is None:
        cml_args.resolution = [cml_args.shape[0] * 1280, cml_args.shape[1] * 720]

    if not cml_args.cameras is None:
        stationIDs = cml_args.cameras
        url_list = convertListOfStationIDsToListOfUrls(cml_args.cameras)
        print(url_list)
        delete_at_end = True
        video_directory, input_video_paths = downloadFilesToTmp(url_list, stationIDs)


    print("Input file path   {}".format(cml_args.inputs))
    print("Input cameras     {}".format(cml_args.cameras))
    print("Generate video    {}".format(cml_args.generate_video))

    print("Output resolution {}".format(cml_args.resolution))
    print("Output shape      {}".format(cml_args.shape))
    print("Output file       {}".format(cml_args.output))
    print("Weblog list of cameras {}".format(cml_args.cameras))



    print_nicely = True
    ffmpeg_command_string = ""

    if cml_args.output is None:
        output_filename = ["output.mp4"]
    else:
        output_filename = cml_args.output[0]

    if type(cml_args.inputs) == int:

        if len(cml_args.inputs) > 1:
            ffmpeg_command_string = generateCommand(cml_args.inputs, cml_args.resolution, cml_args.shape, cml_args.output)
        elif len(cml_args.inputs) == 1:
            #possibly been passed a directory of videos
            delete_at_end = False
            input_videos = os.listdir(cml_args.inputs[0])
            path_input_videos = cml_args.inputs[0]
            input_videos.sort()
            input_video_paths = []
            for video in input_videos:
                if video.endswith(".mp4"):
                    input_video_paths.append(os.path.join(path_input_videos,video))

        elif len(cml_args.inputs) == 0:
            print("No videos found to process")
            quit()
    else:
        print("Number of files was not an integer")

    print_nicely= False
    ffmpeg_command_string = generateCommand(input_video_paths, cml_args.resolution, cml_args.shape, cml_args.output[0],
                                            print_nicely)

    print("Returned command string \n {}".format(ffmpeg_command_string))
    print()
    print(ffmpeg_command_string.replace("\n"," "))
    if cml_args.generate_video:
        subprocess.call(ffmpeg_command_string.replace("\n"," "), shell=True)
    if delete_at_end and False:
        for input_video in input_video_paths:
            os.unlink(input_video)
        os.rmdir(video_directory)
    print()