# RPi Meteor Station
# Copyright (C) 2016  Denis Vida
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


import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import tqdm


from RMS.Formats.FFfile import filenameToDatetime

def writeCALSTARS(star_list, ff_directory, file_name, cam_code, nrows, ncols, chunk_frames=256):
    """ Writes the star list into the CAMS CALSTARS format. 

    Arguments:
        star_list: [list] a list of star data, entries:
            ff_name, star_data
            star_data entries:
                x, y, bg_level, level
        ff_directory: [str] path to the directory in which the file will be written
        file_name: [str] file name in which the data will be written
        cam_code: [str] camera code
        nrows: [int] number of rows in the image
        ncols: [int] number of columns in the image

    Keyword arguments:
        chunk_frames: [int] Number of frames in the FF file or frame chunk. Default is 256.

    Return:
        None
    """

    with open(os.path.join(ff_directory, file_name), 'w') as star_file:

        # Write the header
        star_file.write("==========================================================================\n")
        star_file.write("RMS star extractor" + "\n")
        star_file.write("Cal time = FF header time plus 255/(2*framerate_Hz) seconds" + "\n")
        star_file.write("      Y       X IntensSum Ampltd  FWHM  BgLvl   SNR NSatPx" + "\n")
        star_file.write("==========================================================================\n")
        star_file.write("FF folder = " + ff_directory + "\n")
        star_file.write("Cam #   = " + str(cam_code) + "\n")
        star_file.write("Nrows   = " + str(nrows) + "\n")
        star_file.write("Ncols   = " + str(ncols) + "\n")
        star_file.write("Nframes = " + str(chunk_frames) + "\n")
        star_file.write("Nstars  = -1" + "\n")

        # Write all stars in the CALSTARS file
        for star in star_list:

            # Skip empty star lists
            if len(star) < 2:
                continue

            # Unpack star data
            ff_name, star_data = star

            # Write star header per image
            star_file.write("==========================================================================\n")
            star_file.write(ff_name + "\n")
            star_file.write("Star area dim = -1" + "\n")
            star_file.write("Integ pixels  = -1" + "\n")

            # Write every star to file
            # CALSTARS format: Y(0) X(1) IntensSum(2) Ampltd(3) FWHM(4) BgLvl(5) SNR(6) NSatPx(7)
            # Input star_data: (y, x, intensity, amplitude, fwhm, background, snr, saturated_count)
            # where intensity=IntensSum (integrated), amplitude=Ampltd (peak)
            for y, x, intensity, amplitude, fwhm, background, snr, saturated_count in list(star_data):

                # Limit the saturation count to 999999
                if saturated_count > 999999:
                    saturated_count = 999999

                # Limit the SNR to 99.99
                if snr > 99.99:
                    snr = 99.99

                star_file.write("{:7.2f} {:7.2f} {:9d} {:6d} {:5.2f} {:6d} {:5.2f} {:6d}".format(
                    round(y, 2), round(x, 2),
                    int(intensity), int(amplitude), fwhm, int(background), snr, int(saturated_count)) + "\n")

        # Write the end separator
        star_file.write("##########################################################################\n")



def readCALSTARS(file_path, file_name, chunk_frames=256):
    """ Reads a list of detected stars from a CAMS CALSTARS format. 

    Arguments:
        file_path: [str] Path to the directory where the CALSTARS file is located.
        file_name: [str] Name of the CALSTARS file.

    Keyword arguments:
        chunk_frames: [int] Number of frames in the FF file or frame chunk. Default is 256.
            Will be overwritten by a number in the CALSTARS file if present.

    Return:
        star_list, chunk_frames: 
            - star_list [list] a list of star data, entries:
                ff_name, star_data
                star_data entries:
                    x, y, bg_level, level, fwhm
            - chunk_frames [int] Number of frames in the FF file or frame chunk.
    """

    
    calstars_path = os.path.join(file_path, file_name)

    # Check if the CALSTARS file exits
    if not os.path.isfile(calstars_path):
        print('The CALSTARS file: {:s} does not exist!'.format(calstars_path))
        return False

    # Open the CALSTARS file for reading
    with open(calstars_path) as star_file:

        calibrationstars_list = []

        ff_name = ''
        star_data = []
        skip_lines = 0
        for line in star_file.readlines()[11:]:

            # Skip lines if necessary
            if skip_lines > 0:
                skip_lines -= 1
                continue

            # Read the number of frames if given (Nframes = ...)
            if "Nframes" in line:
                chunk_frames = int(line.split('=')[-1])
                continue

            # Check for end of star entry
            if (("===" in line) or ("###" in line)) and len(ff_name):

                # Add the star list to the main list
                calibrationstars_list.append([ff_name, star_data])

                # Reset the star list
                star_data = []
                
                continue

            # Remove newline
            line = line.replace('\n', '').replace('\r', '')

            if 'FF' in line:
                ff_name = line
                skip_lines = 2
                continue

            # Split the line
            line = line.split()

            if len(line) < 4:
                continue

            try:
                float(line[0])
                float(line[1])
                int(line[2])
                int(line[3])

            except:
                continue

            # Read the star data
            y, x, level, amplitude = float(line[0]), float(line[1]), int(line[2]), int(line[3])

            # Read FWHM if given
            if len(line) >= 5:
                fwhm = float(line[4])
            else:
                fwhm = -1.0

            # Read the background level
            if len(line) >= 6:
                background = int(line[5])
            else:
                background = -1

            # Read the SNR
            if len(line) >= 7:
                snr = float(line[6])
            else:
                snr = -1.0

            # Read the number of saturated pixels
            if len(line) >= 8:
                saturated_count = int(line[7])
            else:
                saturated_count = -1

            # Save star data
            star_data.append([y, x, level, amplitude, fwhm, background, snr, saturated_count])

    
    return calibrationstars_list, chunk_frames

def maxCALSTARS(file_path, file_name, chunk_frames=256):

    if not os.path.exists(os.path.join(file_path, file_name)):
        return [], None

    calstars_list, chunk  = readCALSTARS(file_path, file_name, chunk_frames)
    calstars_dict = {ff_file: star_data for ff_file, star_data in calstars_list}

    try:
        max_len_ff = max(calstars_dict, key=lambda k: len(calstars_dict[k]))
    except:
        print(f"{file_name} had an empty calstars")
        return [], None


    return calstars_dict[max_len_ff], max_len_ff


def calstarEntrytoArray(calstars_entry, max_intensity=None):

    calstars_arr = np.array(calstars_entry[1])
    coords = np.array((calstars_arr[:, 1], calstars_arr[:, 0], calstars_arr[:, 2] + 50, calstars_arr[:,4])).T
    bitmap = renderStars(coords, (720,1280), gaussian=True)


    max_val = bitmap.max()
    if max_intensity is not None and max_intensity >  0:
        max_intensity = max_intensity * 0.8
        bitmap = np.minimum(bitmap, max_intensity)
        grey = ((bitmap * 255 / max_intensity)).astype(np.uint8)
    elif max_val > 0 and max_intensity is None:
        grey = ((bitmap * 255 / max_val)).astype(np.uint8)
    else:
        grey = np.minimum(bitmap,255).astype(np.uint8)



    return np.minimum(grey,255)

def renderStars(coords, shape, gaussian=True):
    """
    Render circular blobs with per-star radii into a bitmap.

    coords: array of [y, x, intensity, radius]
    shape: (height, width)
    gaussian: if True, use Gaussian falloff instead of flat circle
    """
    H, W = shape
    bitmap = np.zeros((H, W), dtype=np.float32)

    # Cache stencils so repeated radii don't recompute
    stencil_cache = {}

    for x, y, I, R in coords:
        y = int(y)
        x = int(x)
        I = float(I)
        if not gaussian:
            R = int(R * 0.8)
        else:
            R = int(R * 0.8)

        if R <= 0:
            continue

        # Build or retrieve stencil
        if R not in stencil_cache:
            yy, xx = np.ogrid[-R:R+1, -R:R+1]

            if gaussian:
                sigma = R / 2
                stencil = np.exp(-(xx*xx + yy*yy) / (2 * sigma * sigma))
            else:
                stencil = (xx*xx + yy*yy) <= R*R
                stencil = stencil.astype(np.float32)

            stencil_cache[R] = stencil

        stencil = stencil_cache[R]

        # Bounds in output image
        y0 = max(0, y - R)
        y1 = min(H, y + R + 1)
        x0 = max(0, x - R)
        x1 = min(W, x + R + 1)

        # Bounds in stencil
        sy0 = y0 - (y - R)
        sy1 = sy0 + (y1 - y0)
        sx0 = x0 - (x - R)
        sx1 = sx0 + (x1 - x0)

        # Add scaled stencil
        bitmap[y0:y1, x0:x1] += 3 * I * stencil[sy0:sy1, sx0:sx1]

    return bitmap



def calstarEntryToPNG(calstars_list, file_path, ff_name, save_images=False, save_path=None):

    if not len(calstars_list):
        return None
    grey = calstarEntrytoArray(calstars_list)
    grey = annotateImage(grey, calstars_list, intensity=125, rescale=2)

    save_path_name = createSavePathName(file_path, ff_name ,save_path)
    Image.fromarray(grey).save(save_path_name)

    return grey

def createSavePathName(file_path, file_name, save_path, extension='png'):



    if save_path is None:
        save_path_name = os.path.join(file_path, f"{file_name.split('.')[0]}.{extension}")

    else:
        save_path = os.path.expanduser(save_path)
        if os.path.exists(save_path):

            if os.path.isdir(save_path):
                file_name = f"{file_name.split('.')[0]}.png"
                save_path_name = os.path.join(save_path, file_name)
            elif os.path.isfile(save_path):
                save_path_name = save_path

    return save_path_name



def maxCalstarsToPNG(file_path, file_name, save_path=None, chunk_frames=256):

        # Extract img coordinates
        calstars_list, ff_max = maxCALSTARS(os.path.expanduser(file_path), file_name, chunk_frames)

        return calstarEntryToPNG([[ff_max],calstars_list], file_path, ff_max, save_path)


def calstarsToMP4(file_path, file_name, save_path=None, chunk_frames=256):


    file_path = os.path.expanduser(file_path)

    save_path_name = createSavePathName(file_path, file_name, save_path, extension='mp4')

    if not os.path.exists(os.path.join(file_path, file_name)):
        return False



    calstar_list, _ = readCALSTARS(file_path, file_name)

    writer = imageio.get_writer(save_path_name, fps=2)


    text_x, text_y = 10, 720 - 20
    font_scale = 0.4
    thickness = 1


    max_intensity = max([coordinates[2]
                      for calstar in calstar_list
                      for coordinates in calstar[1]])

    for frame in calstar_list:

        ff_name = frame[0]
        grey = calstarEntrytoArray(frame, max_intensity)
        stationID = frame[0].split("_")[1]
        timestamp = filenameToDatetime(ff_name).strftime("%Y-%m-%d %H:%M:%S")
        text = f"{stationID} {timestamp} UTC {len(frame[1])}"
        grey = drawTextOnBitmap(grey, text, text_x, text_y)

        grey = annotateImage(grey, frame, intensity=255)

        g = grey.astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)  # (H, W, 3)


        writer.append_data(rgb)

    writer.close()
    pass

def annotateImage(bitmap, frame, intensity=255, rescale=2):


    img = Image.fromarray(bitmap.astype(np.uint8), mode='L')
    w, h = img.size
    img = img.resize((w * rescale, h * rescale), Image.NEAREST)
    draw = ImageDraw.Draw(img)


    for s in frame[1]:

        annotation_text = f"x:{int(s[1])} y:{int(s[0])}\nAmp:{s[3]:4} FWHM:{s[4]:4.2} SNR:{s[6]:5.2} "
        fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        fontSize = 12
        font = ImageFont.truetype(fontPath, fontSize)

        draw.text((int(s[1] + 5) * rescale, int(s[0] + 5) * rescale), annotation_text, fill=intensity, font=font)

    return np.array(img, dtype=np.uint8)


def drawTextOnBitmap(bitmap, text, x, y, intensity=255):
    """
    Draw text onto a grayscale NumPy bitmap using Pillow.
    """
    img = Image.fromarray(bitmap.astype(np.uint8), mode='L')
    draw = ImageDraw.Draw(img)

    fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fontSize = 12
    font = ImageFont.truetype(fontPath, fontSize)


    draw.text((x, y), text, fill=intensity, font=font)

    return np.array(img, dtype=np.uint8)

