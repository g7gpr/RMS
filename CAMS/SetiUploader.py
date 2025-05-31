import os
from ftplib import FTP
import RMS.ConfigReader as cr
import RMS.Formats.CAL as cal
import RMS.Formats.Platepar as pp
import logging
from RMS.Logger import initLogging
from RMS.Formats.Platepar import Platepar
from zipfile import ZipFile
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo, writeFTPdetectinfo


# Credentials

setisite = 'camsftp.seti.org'
setiuser = 'anonymous@seti.org'
setipass = 'anonymous'
setidir = '/incoming/'


# Camera codes to directory mappings

mapping = [
            [[900, 999], "LOCAMS"],
            [[11001, 11999], "LOCAMS"],
            [[4000, 4999], "AR"],
            [[7000, 7999], "AUS"],
            [[300, 399], "BeNeLux"],
            [[800, 899], "BeNeLux"],
            [[3000, 3999], "BeNeLux"],
            [[1000, 1099], "EXOSS"],
            [[1110, 1199], "EXOSS"],
            [[26000, 26999], "EXOSS"],
            [[000, 199], "cams"],
            [[500, 573], "cams"],
            [[23001, 23999], "cams"],
            [[8000, 8999], "CHILE"],
            [[25000, 25999], "NMN"],
            [[5000, 5999], "FL"],
            [[24001, 24999], "India"],
            [[9000, 9999], "NAMIBIA"],
            [[700, 749], "NZ"],
            [[2000, 2999], "NZ"],
            [[581, 599], "SA"],
            [[6000, 6999], "SA"],
            [[22001, 22999], "Texas"],
            [[21001, 21999], "TK"],
            [[750, 799], "UAE"]
            ]


def valueToDirectory(camera_code,map):

    """

    Args:
        camera_code: cams camera code
        map: the mapping from camera_codes to remote directory

    Returns:
        remote zip file path

    """

    destination_directory = None
    for mapping in map:
        range_min = mapping[0][0]
        range_max = mapping[0][1]
        directory = mapping[1]
        if range_min <= int(camera_code) <= range_max:
            destination_directory = directory

    if destination_directory is None:
        return setidir
        log.warn("Could not find a remote directory for camera {}".format(camera_code))

    return os.path.join(setidir, destination_directory)

def getNightTime(directory, microseconds=False):

    """

    Args:
        directory: RMS style directory
        microseconds: use microseconds

    Returns:
        the time of the directory
    """

    if directory[-1] == os.sep:
        directory = directory[:-1]

    # Extract time from night name
    _, night_name = os.path.split(directory)
    if microseconds:
        night_time = "_".join(night_name.split('_')[1:4])
    else:
        night_time = "_".join(night_name.split('_')[1:4])[:-3]


    return night_time

def createCALFileName(cams_code, directory):

    """

    Args:
        cams_code: the cams code
        directory: the archived directory worked on

    Returns:
        the name of the CAL file which would be generated
    """

    night_time = getNightTime(directory)
    # Construct the CAL file name
    file_name = "CAL_{:06d}_{:s}.txt".format(cams_code, night_time)
    return file_name

def createRTPFileName(cams_code, directory):

    """

    Args:
        cams_code: the cams code
        directory: the diretory working on

    Returns:
        the RTP file name which would be generated
    """

    night_time = getNightTime(directory, microseconds=True)
    file_name = "RTPdetectinfo_{:06d}_{:s}.txt".format(cams_code, night_time)
    return file_name

def convertFTPtoRTP(night_directory, cal_file_name, config):

    """

    Args:
        night_directory: the RMS night directory being worked on
        cal_file_name: the SETI style cal file name
        config: RMS config file

    Returns:
        the name of the newly generated cams RTP file

    """

    ftpdetectinfo_name = "FTPdetectinfo_{}_{}.txt".format(config.stationID,
                                                          getNightTime(night_directory,microseconds=True))

    cams_code_formatted = "{:06d}".format(int(config.cams_code))
    rtpdetectinfo_name = "RTPdetectinfo_{}_{}.txt".format(cams_code_formatted,
                                                          getNightTime(night_directory, microseconds=True))

    _, fps, meteor_list = readFTPdetectinfo(night_directory, ftpdetectinfo_name, ret_input_format=True)



    platepar = Platepar()
    platepar.read(os.path.join(night_directory,config.platepar_name))
    # Replace the camera code with the CAMS code
    for met in meteor_list:
        # Replace the station name and the FF file format
        ff_name = met[0]
        ff_name = ff_name.replace('.fits', '.bin')
        ff_name = ff_name.replace(config.stationID, cams_code_formatted)
        met[0] = ff_name

    # Write the CAMS compatible FTPdetectinfo file

    writeFTPdetectinfo(meteor_list, night_directory, \
                       rtpdetectinfo_name, night_directory, cams_code_formatted, fps, calibration=cal_file_name, \
                       celestial_coords_given=(platepar is not None))
    return rtpdetectinfo_name

def createSetiZipName(directory, config):

    """

    Args:
        directory: RMS archive directory
        config: RMS config file

    Returns:
        name of the SETI style zip file
    """

    directory = os.path.basename(directory)
    date_code = directory.split("_")[1]
    year, month, day = date_code[0:4], date_code[4:6], date_code[6:8]
    time_code = directory.split("_")[2]
    hour, minute, second = time_code[0:2], time_code[2:4], time_code[4:6]
    name = ("{}_{}_{}_{:06d}_{}_{}_{}.zip".format(year, month, day, config.cams_code, hour, minute, second ))

    return name

def zipFiles(file_list, night_directory, config, log):

    """

    Args:
        file_list: list of files to be zipped
        night_directory: the RMS archive directory
        config: RMS config file

    Returns:
        None if the zip file already existed
        otherwise, the path to the zip file
    """


    zip_name = os.path.join(night_directory, createSetiZipName(night_directory, config))
    if os.path.exists(zip_name):

        return None

    log.info("Zipping to {}".format(zip_name))

    with ZipFile(zip_name, 'w') as zip:
        for file_name in file_list:
            log.info("  Adding {}".format(file_name))
            zip.write(file_name, arcname=os.path.basename(file_name))

    return zip_name

def sendByFTP(zip_name, log):



    """
    Send the zip file by FTP to seti. If this fails, return false, and delete the local copy of the zip file.
    The next day the zip will be remade and sent again.

    Args:
        zip_name:name of the zip file to send

    Returns:
        True if successful, False otherwise
    """

    if zip_name is None:

        return
    cams_code = os.path.basename(zip_name).split("_")[3]
    remote_directory = valueToDirectory(cams_code,mapping)
    try:
        ftp = FTP(setisite, setiuser, setipass)
        ftp.cwd(remote_directory)
        ftp.storbinary("STOR {}".format(os.path.basename(zip_name)), open(zip_name, "rb"))
        ftp.close()
        log.info("Sent {} to {}".format(zip_name, remote_directory))
        return True
    except:
        log.warning("Failed to send {}, deleted zip and will try tomorrow".format(zip_name))
        os.unlink(zip_name)
        return False


def rmsExternal(captured_night_dir, archived_night_dir, config):

    initLogging(config, 'SETI_')
    log = logging.getLogger("logger")
    archived_night_dir = os.path.expanduser(archived_night_dir)
    log.info("SetiUpload started")

    if config.cams_code == 0:
        log.warning("cams_code set to {}, ending".format(config.cams_code))
        return None
    #get list of directories in archived_night_dir

    stationID, cams_code = config.stationID, config.cams_code

    archived_directory_full_path = os.path.join(config.data_dir, config.archived_dir)
    archived_directory_list = os.listdir(archived_directory_full_path)
    for file_object in archived_directory_list:
        if os.path.isdir(os.path.join(archived_night_dir, file_object)):
            if file_object.startswith("{}_".format(stationID)):
                night_directory = os.path.join(archived_directory_full_path, file_object)

                cal_file = createCALFileName(config.cams_code, night_directory)
                cal_file_path = os.path.join(night_directory, cal_file)

                rtp_file = createRTPFileName(config.cams_code, night_directory)
                rtp_file_path = os.path.join(night_directory, rtp_file)

                # create a CAL file if it does not exist



                if os.path.exists(cal_file_path):
                    pass
                else:

                    platepar = pp.Platepar()
                    platepar.read(os.path.join(night_directory, config.platepar_name))
                    cal_file_to_send = cal.writeCAL(night_directory, config, platepar)



                # create an RTP file, if it does not exist

                if os.path.exists(rtp_file_path):
                    pass
                else:
                    log.info("RTP file {} does not exist".format(rtp_file_path))
                    if os.path.exists(cal_file_path):
                        # write the FTP file
                        convertFTPtoRTP(night_directory, cal_file_path, config)
                pass

                if os.path.exists(cal_file_path) and os.path.exists(rtp_file_path):
                    sendByFTP(zipFiles([cal_file_path, rtp_file_path], night_directory, config, log), log)
    log.info("SetiUpload complete")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


if __name__ == '__main__':
    config = cr.loadConfigFromDirectory(".config", os.path.expanduser("~/source/RMS"))
    rmsExternal("~/RMS_data/CapturedFiles", "~/RMS_data/ArchivedFiles", config)
