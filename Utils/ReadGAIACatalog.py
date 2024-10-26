import os
import numpy as np
import argparse
import pickle
import subprocess
import datetime

import urllib.request
import time
from operator import itemgetter,attrgetter
from tqdm import tqdm
import random
import time
from RMS.Misc import mkdirP

import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import equatorialCoordPrecession

"""
To Do list

1.  Write code to generate workspace - done
2.  Automate download of source information
3.  Fix variable and filenames - done
4.  Execute only if pickle not exist - done
5.  Comments
6.  Confirm overlap on imported datasets - done
7.  Confirm BVRI against Sky2000
8.  Confirm BSC5 for missing stars
9.  For stars with imported names, confirm name against remote catalogue
10. 


"""


# Read from http://jvo.nao.ac.jp/portal/gaia/dr3.do
# Photometric calculations at https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html

# Example star name query
# http://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+1137162861578295168

j1900, j1950, j2000 = 2415020, 2433282.5,  2451545.0
b1900, b1950 = 2415020.31352, 2433282.4235



def hms2Degrees(h,m,s):

    return 360 * (((s / 60 ) + m / 60) + h / 24)

def degrees2HMS(d):

    h = d // (360/24)
    m = (d - h  * 360/24) * 24 * 60 / 360 // 1
    s = (d - h * 360/24 - m * 360 / (24*60)) * (24*60*60) / 360


    return h, m, s, "{}h{}m{:.2f}s".format(int(h),int(m),float(s))

def dms2Degrees(d,m,s):

    return ((s / 60 ) + m / 60) + d,"{}°{}m{:.2f}s".format(int(d),int(m),float(s))

def degrees2DMS(deg):

    d = deg // 1
    m = (deg - d)  * 60 // 1
    s = (deg - d - (m / 60)) * 3600

    return d,m,s,"{}°{}'{:.2f}".format(int(d),int(m),float(s))

def seconds2DHMS(s, end_time = False):

    d  = s // (3600 * 24)
    r  = s - d * 3600 * 24
    h  = r // 3600
    r -= h * 3600
    m  = r // 60
    r -= m * 60

    if end_time:

        end_time = datetime.datetime.now() + datetime.timedelta(seconds=s)

        return "{}d:{}h{}m{}s {}".format(int(d),str(int(h)).zfill(2),str(int(m)).zfill(2),str(int(r)).zfill(2), end_time)

        pass

    else:

        return "{}d:{}h{}m{}s".format(int(d),str(int(h)).zfill(2),str(int(m)).zfill(2),str(int(r)).zfill(2))

def besselianPrecession(jd_initial_epoch, ra_initial_epoch, dec_initial_epoch, jd_final_epoch, pm_ra = 0 , pm_dec = 0 ):

    """

    Takes degrees, return degrees
    pm in arcseconds per annum - not in seconds of time

    """


    # from Meeus, Astromonical Algorithms p.134


    # initially working in degrees

    T = (jd_initial_epoch - 2451545.0) /  36525
    t = (jd_final_epoch - jd_initial_epoch) / 36525         # time elapsed in julian centuries


    # calculate proper motion

    pm_ra_initial_2_final, pm_dec_initial_2_final = pm_ra * t * 100, pm_dec * t * 100
    ra_initial_epoch_pm = ra_initial_epoch + (pm_ra_initial_2_final / 3600)
    dec_initial_epoch_pm = dec_initial_epoch + (pm_dec_initial_2_final / 3600)

    # to radians


    zeta     = (2306.2181 + 1.39656 * T + 0.000139 * T ** 2) * t
    zeta    += (0.30188 - 0.000344 * T) * t **2 + 0.017998 * t ** 3

    ZETA     = (2306.2181 + 1.39656 * T + 0.000139 * T ** 2) * t
    ZETA    += (1.09468 + 0.000066 * T) * t ** 2 + 0.018203 * t ** 3

    theta    = (2004.3109 - 0.85330 * T - 0.000217 * T ** 2) * t
    theta   -= (0.42665 + 0.000217 * T) * t ** 2
    theta   -= 0.041833 * t ** 3

    # convert arcseconds to degrees
    zeta, ZETA, theta = zeta / 3600, ZETA / 3600, theta / 3600

    # to radians
    ra, dec = np.radians(ra_initial_epoch_pm), np.radians(dec_initial_epoch_pm)
    zr, Zr, tr = np.radians(zeta), np.radians(ZETA), np.radians(theta)

    A = np.cos(dec) * np.sin(ra + zr)
    B = np.cos(tr) * np.cos(dec) * np.cos(ra + zr) - np.sin(tr) * np.sin(dec)
    C = np.sin(tr) * np.cos(dec) * np.cos(ra + zr) + np.cos(tr) * np.sin(dec)

    #to degrees
    ra_final_epoch, dec_final_epoch = np.degrees(np.arctan2(A,B) + Zr), np.degrees(np.arcsin(C))

    if ra_final_epoch < 0:
        ra_final_epoch += 360

    return ra_final_epoch, dec_final_epoch





def getRaDecHD(name, gdr3):

    with open(os.path.expanduser("~/tmp/catalogueassembly/inputdata/henrydraper/catalog.dat") , 'r') as fh:

        name_match = False
        for star in fh:
            #the catalogue name is a fixed width field - so needs padding out.
            hd_cat_number = star[0:6]

            cat_name = "HD {}".format(star[0:6])

            if cat_name == name:
                name_match = True
                break
        if not name_match:
            #print("Failed to find a match for {}/{}".format(gdr3,name))
            return "Not evaluated", "Not evaluted"
        DM      = star[6:18]
        RAh     = star[18:20]
        RAdm    = star[20:23]
        DEsign  = star[23]
        DEd     = star[24:26]
        DEm     = star[26:28]
        Ptm     = star[29:34]
        Ptg     = star[36:41]



        ra_b1900 = (int(RAh) + int(RAdm) / 600)*(360/24)
        if DEsign == "+":
            dec_b1900 = (int(DEd) + float(DEm) / 60)
        else:
            dec_b1900 = 0 - (int(DEd) + float(DEm) / 60)


        #ra, dec = besselianPrecession(b1900, ra_b1900, dec_b1900, j2000)
        #print(ra,dec)
        ra,dec = equatorialCoordPrecession(b1900, j2000, np.radians(ra_b1900), np.radians(dec_b1900))

        #ra, dec = equatorialCoordPrecession(b1900,j2000, ra_b1900, dec_b1900)


        return np.degrees(ra), np.degrees(dec)



def getRaDecSAO(name, gdr3):

    with open(os.path.expanduser("~/tmp/catalogueassembly/inputdata/smithsonianastrophysicalobservatory/sao.dat") , 'r') as fh:

        name_match = False
        for star in fh:
            cat_name = "SAO {}".format(star[0:6])

            if cat_name == name:
                name_match = True
                break
        if not name_match:
            #print("Failed to find a match for {}/{}".format(gdr3,name))
            return "Not evaluated", "Not evaluated"

        RAh     = star[7:9]
        RAm     = star[9:11]
        RAs     = star[11:17]
        pmRA    = star[17:24]
        DEsign  = star[41]
        DEd     = star[42:44]
        DEm     = star[44:46]
        DEs     = star[46:51]
        pmDE    = star[51:57]



        # working in radians


        ra_j1950 = np.radians(((int(RAh) + int(RAm) / 60) + float(RAs) / 3600) * 360 / 24)
        if DEsign == "+":
            dec_j1950 = np.radians(int(DEd) + float(DEm) / 60 + float(DEs)/3600)
        else:
            dec_j1950 = 0 - np.radians(int(DEd) + float(DEm) / 60 + float(DEs)/3600 )


        ra, dec = np.degrees( equatorialCoordPrecession(j1950,j2000, ra_j1950, dec_j1950))

        # pmRA and pmDEC in arsec per annum
        ra = ra + float(pmRA) * 50 / 3600
        dec = dec + float(pmDE) * 50 /3600



        return ra, dec

def getRaDecTYC(name, gdr3):

    # Name format TYC 3771-224-1 needs padding
    try:
        field_1 = name.split()[1].split("-")[0].zfill(4)
        field_2 = name.split()[1].split("-")[1].zfill(5)
        field_3 = name.split()[1].split("-")[2]
        name = "TYC {}-{}-{}".format(field_1,field_2,field_3)
        name_as_integer = int("{}{}{}".format(field_1, field_2, field_3))
    except:
        print("Failed to parse {} for {}".format(name,gdr3))
        return "Not evaluated", "Not evaluated"


    tycho_2_files_list = sorted(os.listdir(os.path.expanduser("~/tmp/catalogueassembly/inputdata/tycho-2")))

    first_iteration = True
    for search_file in tycho_2_files_list:

        with open(os.path.join(os.path.expanduser("~/tmp/catalogueassembly/inputdata/tycho-2"), search_file), 'r') as search_fh:
            if first_iteration:
                last_search_file = search_file
                first_iteration = False
            first_line_name = search_fh.readline().split("|")[0]
            first_line_name_as_integer = int(first_line_name.replace(" ",""))
            if first_line_name_as_integer > name_as_integer:
                search_file = last_search_file
                break
            last_search_file = search_file



    #for file in tycho_2_files_list:

    with open(os.path.join(os.path.expanduser("~/tmp/catalogueassembly/inputdata/tycho-2"), search_file ), 'r') as fh:

        name_match = False

        for star in fh:
            cat_name = "TYC {}-{}-{}".format(str((star[0:4].strip())),str((star[5:10].strip())),str((star[11].strip())))
            if cat_name == name:
                name_match = True
                break


        if not name_match:
            # print("Failed to find a match for {}/{}".format(gdr3,name, cat_name))
            return "Not evaluated", "Not evaluated"

        RAmdeg  = star[15:27]
        DEcmdeg = star[28:40]

    return RAmdeg, DEcmdeg


def download2MASS(input_data):

    names = ['aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aaj', 'aak', 'aal',
             'aam', 'aan', 'aao', 'aap', 'aaq', 'aar', 'aas', 'aat', 'aau', 'aav', 'aaw', 'aax', 'aay', 'aaz']

    for name in names:
        download_path = "https://irsa.ipac.caltech.edu/2MASS/download/allsky/psc_{}.gz".format(name)
        print("Downloading {}".format(download_path))
        download(download_path, input_data, "2micronallstarsurvey")

    names = ['aba', 'abb', 'abc', 'abd', 'abe', 'abf', 'abg', 'abh', 'abi', 'abj', 'abk', 'abl',
             'abm', 'abn', 'abo', 'abp', 'abq', 'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz']

    for name in names:
        download_path = "https://irsa.ipac.caltech.edu/2MASS/download/allsky/psc_{}.gz".format(name)
        print("Downloading {}".format(download_path))
        download(download_path, input_data, "2micronallstarsurvey")

    names = ['aca', 'acb', 'acc', 'acd', 'ace']

    for name in names:
        download_path = "https://irsa.ipac.caltech.edu/2MASS/download/allsky/psc_{}.gz".format(name)
        print("Downloading {}".format(download_path))
        download(download_path, input_data, "2micronallstarsurvey")

    names = ['baa', 'bab', 'bac', 'bad', 'bae', 'baf', 'bag', 'bah', 'bai', 'baj', 'bak', 'bal',
             'bam', 'ban', 'bao', 'bap', 'baq', 'bar', 'bas', 'bat', 'bau', 'bav', 'baw', 'bax', 'bay', 'baz']

    for name in names:
        download_path = "https://irsa.ipac.caltech.edu/2MASS/download/allsky/psc_{}.gz".format(name)
        print("Downloading {}".format(download_path))
        download(download_path, input_data, "2micronallstarsurvey")

    names = ['bba', 'bbb', 'bbc', 'bbd', 'bbe', 'bbf', 'bbg', 'bbh', 'bbi']

    for name in names:
        download_path = "https://irsa.ipac.caltech.edu/2MASS/download/allsky/psc_{}.gz".format(name)
        print("Downloading {}".format(download_path))
        download(download_path, input_data, "2micronallstarsurvey")

    download_path = "https://irsa.ipac.caltech.edu/2MASS/download/allsky/scn_aaa.gz".format(name)
    print("Downloading {}".format(download_path))
    download(download_path, input_data, "2micronallstarsurvey")


def downloadTIC(input_data):



    for dec in range(0, 88, 2):
        dec_start_string = str(dec + 2).zfill(2)
        dec_end_string = str(dec).zfill(2)
        if dec == 0:
            download_path = "https://archive.stsci.edu/missions/tess/catalogs/tic_v82/tic_dec{}_00S__{}_00N.csv.gz".format(
                    dec_start_string, dec_end_string)
        else:
            download_path = "https://archive.stsci.edu/missions/tess/catalogs/tic_v82/tic_dec{}_00S__{}_00S.csv.gz".format(
                    dec_start_string, dec_end_string)
        print("Downloading {}".format(download_path))
        download(download_path, input_data, "tessinputcatalogueversion8")

    for dec in range(0, 88, 2):
        dec_start_string = str(dec).zfill(2)
        dec_end_string = str(dec + 2).zfill(2)
        download_path = "https://archive.stsci.edu/missions/tess/catalogs/tic_v82/tic_dec{}_00N__{}_00N.csv.gz".format(
                dec_start_string, dec_end_string)
        print("Downloading {}".format(download_path))
        download(download_path, input_data, "tessinputcatalogueversion8")


def getRaDecTIC(name,gdr3):

    return ["Not evaluated","Not evaluated"]


def getRaDec2MASS(name, gdr3):

    return ["Not evaluated","Not evaluated"]






def getRaDec(name, gdr3):

    #name_preference_list = ["NAME", "HD", "SAO", "TYC", "TIC", "2MASS", "GAIA_DR3"]

    catalogue_coords = ["Null","Null"]

    if name[0:len('HD')] == "HD":
        catalogue_coords = getRaDecHD(name, gdr3)
    elif name[0:len('SAO')] == "SAO":
        catalogue_coords = getRaDecSAO(name, gdr3)
    elif name[0:len('TYC')] == "TYC":
        catalogue_coords = getRaDecTYC(name, gdr3)
    elif name[0:len('TIC')] == "TIC":
        catalogue_coords = getRaDecTIC(name, gdr3)
    elif name[0:len('2MASS')] == "2MASS":
        catalogue_coords = getRaDec2MASS(name, gdr3)

    deviation = 0

    return catalogue_coords


def download(url,directory_location, data_name, force_download=False):


    download_path = os.path.join(directory_location, data_name)
    mkdirP(download_path)
    filename = os.path.join(download_path,os.path.basename(url))
    if (not os.path.exists(filename) and not os.path.exists(os.path.splitext(filename)[0])) or force_download:
        try:
            print("Downloading {} to {}".format(url, filename))
            urllib.request.urlretrieve(url, filename)
            if os.path.splitext(filename)[1] == ".gz":
                subprocess.run(["gunzip", filename])
        except:
            print("Download of {} from {} failed".format(filename, url))


def createWorkArea(base_path=None):

    """
    Create the work area

       name_preference_list = ["NAME", "HD", "SAO", "TYC", "TIC", "2MASS", "GAIA_DR3"]

    """
    if base_path is None:
        base_path = ["~/tmp/"]

    working_path = os.path.expanduser(os.path.expanduser(base_path[0]))
    working_path = os.path.join(working_path,"catalogueassembly")
    mkdirP(os.path.expanduser(working_path))


    input_data = (os.path.join(working_path,"inputdata"))
    #http://tdc-www.harvard.edu/catalogs/bsc5.dat.gz




    download("http://tdc-www.harvard.edu/catalogs/bsc5.dat.gz",input_data, "BSC5")
    download("https://github.com/CroatianMeteorNetwork/RMS/raw/master/Catalogs/STARS9TH_VBVRI.txt", input_data, "stars9th")
    download("https://cdsarc.cds.unistra.fr/ftp/III/135A/catalog.dat.gz", input_data,"henrydraper")

    download("http://tdc-www.harvard.edu/catalogs/sky2kv4n.dat.gz", input_data, "Sky2000")
    download("http://tdc-www.harvard.edu/catalogs/sky2kv4.readme", input_data, "Sky2000")
    mkdirP(os.path.join(input_data, "simbad"))
    download("https://cdsarc.cds.unistra.fr/ftp/I/131A/sao.dat.gz", input_data, "smithsonianastrophysicalobservatory")
    mkdirP(os.path.join(input_data, "henrydraper"))

    for i in range(0,19):
        download_path = "https://cdsarc.cds.unistra.fr/ftp/I/259/tyc2.dat.{:0>{}}.gz".format(i,2)
        download(download_path, input_data, "tycho-2")

    download("https://cdsarc.cds.unistra.fr/ftp/I/131A/sao.dat.gz", input_data, "smithsonianastrophysicalobservatory")
    download("https://cdsarc.cds.unistra.fr/ftp/I/131A/sao.dat.gz", input_data, "smithsonianastrophysicalobservatory")
    download("http://rvrgm.asuscomm.com:8243/data/gaia_dr3/result_gaiadr3_20240107230522822_107_78_1.psv.gz", input_data,"gaia")

    for i in range(1,37):
        download_path = "http://rvrgm.asuscomm.com:8243/data/simbad/IDENT_{:0>{}}.txt.gz".format(i,2)
        print("Downloading from {}".format(download_path))
        download(download_path, input_data, os.path.join("simbad","identtable"))
    download_path = "http://rvrgm.asuscomm.com:8243/data/simbad/IDENT_18_1.txt.gz".format(i, 2)
    download(download_path, input_data, os.path.join("simbad", "identtable"))
    mkdirP(os.path.join(input_data, "TessInputCatalogVersion8"))
    mkdirP(os.path.join(input_data, "2micronallstarsurvey"))
    mkdirP(os.path.join(input_data, "gaia_dr3"))
    mkdirP(os.path.join(working_path, "workingfiles"))
    mkdirP(os.path.join(working_path, "pickles"))
    mkdirP(os.path.join(working_path, "outputdata"))

    #downloadTIC(input_data)
    #download2MASS(input_data)

    return working_path







def findInSorted(target,dataset, field_no=0, search_range=None,
                 interpolate=True, interpolate_factor=0.05, get_nearest=True,
                 recursion_count=0):

    """
    High speed recursive searching in fairly evenly distributed sorted lists of lists
    Very efficient for long lists - several orders of magnitude faster than built ins

    parameters:

    target: value to be found
    field_no: field to be searched
    search_range: [start,end], generally no need to pass in a value, used only for recursion
    interpolate: guess the range for the next search by interpolation
    interpolate factor: lower numbers will close the search range down more quickly, but risk falling back to bifurcation
    get_nearest: [bool] if true return the index of the nearest value to the target, else return the indices either side
    recursion_count: the number of recursions required to find the element - generally no reason to pass a value here

    returns: [low,high], recursion_count

    """

    # initialisation
    if search_range is None:
        recursion_count = 0
        search_range = [0,len(dataset) -1]


        # if the target is out of range
        if target > dataset[search_range[1]][field_no]:
            return [search_range[1],search_range[1]+1], recursion_count

        if target < dataset[search_range[0]][field_no]:
            return [search_range[0]-1,search_range[0]], recursion_count

    # find the middle of the search range
    midpoint = round ((search_range[0] + search_range[1]) /2)

    if search_range[0] + 1 == search_range[1]:

        if dataset[search_range[0]] == target:
            return[search_range[0], search_range[0]]
        elif dataset[search_range[1]] == target:
            return[search_range[1], search_range[1]]
        elif dataset[search_range[0]][field_no] < target < dataset[search_range[1]][field_no]:
            return search_range, recursion_count +1



    if interpolate:
        span = search_range[1] - search_range[0]
        fraction = (target - dataset[search_range[0]][field_no]) / (dataset[search_range[1]][field_no] - dataset[search_range[0]][field_no])
        estimation = search_range[0] + fraction * span
        interpolated_range = [round(estimation - span * 0.05), round(estimation+span*0.05)]
        interpolated_range[0] = 0 if interpolated_range[0] < 0 else interpolated_range[0]
        interpolated_range[1] = len(dataset)-1 if interpolated_range[1] > len(dataset)-1 else interpolated_range[1]
        #print(interpolated_range[0], interpolated_range[1], len(dataset))
        if dataset[interpolated_range[0]][field_no] < target < dataset[interpolated_range[1]][field_no]:
            search_range = interpolated_range
            search_range, recursion_count = findInSorted(target,dataset,field_no,search_range=search_range,
                                                            interpolate=interpolate,
                                                            interpolate_factor = interpolate_factor,
                                                            recursion_count = recursion_count + 1,
                                                            get_nearest = get_nearest)
        else:
            search_range, recursion_count = findInSorted(target,dataset,field_no,search_range=search_range,
                                                            interpolate=False,
                                                            interpolate_factor= interpolate_factor,
                                                            recursion_count= recursion_count +1,
                                                            get_nearest = get_nearest)

    else:
        if target < dataset[midpoint][field_no]:
            search_range, recursion_count =  findInSorted(target,dataset,field_no,
                                                            search_range = [search_range[0],midpoint],
                                                            interpolate=False,
                                                            interpolate_factor = interpolate_factor,
                                                            recursion_count = recursion_count + 1,
                                                            get_nearest=get_nearest)
        else:
            search_range, recursion_count = findInSorted(target, dataset, field_no,
                                                            search_range=[midpoint, search_range[1]],
                                                            interpolate=False,
                                                            interpolate_factor=interpolate_factor,
                                                            recursion_count=recursion_count + 1,
                                                            get_nearest=get_nearest)

    if search_range[1] - search_range[0] == 1 and get_nearest:
        if target-dataset[search_range[0]][field_no] < dataset[search_range[1]][field_no] - target and search_range[0] < search_range[1]:
            search_range[1] -= 1
        else:
            search_range[0] += 1



    return search_range, recursion_count

def handleGaia3DRAuxiliaryInfo(line):

    """

    Reads the Gaia 3D Auxiliary information and returns two parameters,

    example auxiliary information

    # condition: phot_g_mean_mag < 12
    # columns: designation, ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, classprob_dsc_specmod_star, classprob_dsc_specmod_binarystar, spectraltype_esphs

    -----
    data
    -----


    # totalCount: 3087821


    The information and the name of the field.

    """




    print("Handling {}".format(line))
    if "condition" in line:
        condition = line.split(':')[1]
        print("Condition {}".format(condition))
        return "Condition", condition
    elif "columns:" in line:
        columns_list_raw = line.split(':')[1].split(",")
        columns_list = []
        for col_name in columns_list_raw:
            columns_list.append(col_name.strip())
        return "Columns", columns_list
    elif "totalCount" in line:
        object_count = line.split(':')[1]
        return "Object_count", object_count

def readGaiaCatalogTxt(file_path, lim_mag=None, header_designator = "#", max_objects = None):


    """

    Read star data from the GAIA catalog in the .psv format - works with any column order.

    returns:
        catalogue: Nested list of catalogue information
        columns: List of column headings
    """


    results = []
    print(file_path)
    # intialise

    condition, columns, object_count = None, None, None

    if max_objects is None:
        max_objects = np.inf
    else:
        max_objects = max_objects[0]
        print("Only reading {} objects".format(max_objects))

    with open(file_path) as f:


        # Initialise list to hold the catalogue
        catalogue, objects = [],0

        # Read Gaia catalog

        for line in tqdm(f):

            if line[0]==header_designator:
                info_type, data_returned = handleGaia3DRAuxiliaryInfo(line[1:])
                condition = data_returned if info_type == "Condition" else condition
                columns = data_returned if info_type == "Columns" else columns
                object_count = data_returned if info_type == "Object_count" else object_count
                continue

            catalogue.append(line.split('|'))
            objects += 1
            if objects > max_objects:
                break

        # sort by designation - also known is gaia id

        designator_column_number = columns.index("designation")

        catalogue_sorted_by_gaia_id = sorted(catalogue,key=lambda object_line: object_line[designator_column_number])
        print("Object_count {}".format(object_count))
        print("Catalogue_length {}".format(len(catalogue)))


        return [catalogue_sorted_by_gaia_id, columns]


def generateOID2Main_ID(input_file_path, output_file_path, max_objects = None):



    results = []
    print(input_file_path)
    # intialise

    columns, object_count = None, None

    if max_objects is None:
        max_objects = np.inf
    else:
        max_objects = max_objects[0]
        print("Only reading {} objects".format(max_objects))

    with open(input_file_path) as f:

        # Initialise list to hold the catalogue
        catalogue, objects = [],0

        # Read catalog

        #no header marker, just trust that the first line is a list of columns
        line = f.readline()
        columns_raw = line.split('|')
        columns = []
        for col in columns_raw:
            columns.append(col.strip())


        # then a horizontal separator

        line = f.readline()

        for line in tqdm(f):


            line_list_txt = line.split('|')
            if len(line_list_txt)  != 67:
                print("Skipping line {}".format(line_list_txt))
                continue
            line_list_typed = []
            col_counter = 0
            for value in line_list_txt:
                if col_counter == columns.index("oid"):
                    line_list_typed.append(int(value))
                else:
                    line_list_typed.append(value.strip())
                col_counter += 1
            catalogue.append(line_list_typed)
            objects += 1
            if objects > max_objects:
                break

        # sort by designation - the Gaia3 code

        sort_col = columns.index("oid")

        len_catalogue_line = len(catalogue[25])
        print("Normal line length is {}".format(len_catalogue_line))
        for line in catalogue:
            if len(line) != len_catalogue_line:
                print("Line length is only {}".format(len(line)))


        print("Sorting on column {}:{}".format(sort_col, columns[sort_col]))
        catalogue_sorted = sorted(catalogue,key=lambda object_line: object_line[sort_col])
        print("Object_count {}".format(object_count))
        print("Catalogue_length {}".format(len(catalogue)))

        oid_column, main_id_column = columns.index("oid"), columns.index("main_id")

        oid_list, main_id_list = [],[]
        with open(output_file_path, 'w') as fh:

            for line in catalogue_sorted:
                line_string = "|{}|{}|\n".format(line[oid_column],line[main_id_column].replace('"',""))
                fh.write(line_string)
                oid_list.append(line[oid_column])
                main_id_list.append(line[main_id_column])

        return catalogue_sorted, columns, oid_list, main_id_list


def generateGaia2SimbadCodeFromIDSTables(catalogue, columns):


    """
    This function is probably too slow, and I cannot find a way to optimise it

    """
    star_list, designator_list = [],[]
    for star in catalogue:

        (star[columns.index("designation")].replace(" ", "+"))
        designator = star[columns.index("designation")]
        designator_list.append(designator)


    cross_reference_list = []

    table_files = sorted(os.listdir(os.path.expanduser("~/tmp/IDS_TABLE/")))


    for table_file in table_files:
        path_table_file =  os.path.join(os.path.expanduser("~/tmp/IDS_TABLE/", table_file))
        print(path_table_file)
        with open(path_table_file,'r') as fh:

            for line in tqdm(fh):
                ident_list = line.replace('"','').split("|")

                for ident in ident_list:
                    if "Gaia DR3" in ident:
                        gaia_dr3_ident = ident
                        if ident in designator_list:
                            cross_reference_list.append([gaia_dr3_ident, ident_list[-1].strip()])


    # sort by gaia ident to speed up the future join
    cross_reference_list_sorted_by_id = sorted(cross_reference_list,key=lambda gaia2simbadcode: gaia2simbadcode[0])
    return cross_reference_list_sorted_by_id


def generateGaia2SimbadCodeFromIdentTables(input_directory, working_path,catalogue, columns):
    """
    Replacement for IDS function

    """

    cross_reference_list, cross_reference_list_DR3_only = [],[]

    table_files = sorted(os.listdir(input_directory))

    last_files_last_oid_ref = 1
    for table_file in table_files:
        path_table_file = os.path.join(input_directory, table_file)
        print(path_table_file)
        first_line = True
        with open(path_table_file, 'r') as fh:

            for line in tqdm(fh):
                line_list = line.replace("\n","").split("|")
                if len(line_list) != 2:
                    continue

                id = line_list[0].replace('"','').strip()
                oid_ref = line_list[1]

                if oid_ref != "oidref" and id !="id" and "-" not in oid_ref:
                    cross_reference_list.append([id,int(oid_ref)])
                    if first_line:
                        if int(last_files_last_oid_ref) < int(oid_ref):
                            print("Gap in oid_refs")
                            print("Last files last_oid_ref {}, {} first_oid_ref {}".format(last_files_last_oid_ref,table_file, oid_ref))
                            quit()
                        else:
                            print("Last files last_oid_ref {}, this files first_oid_ref {}".format(last_files_last_oid_ref, oid_ref))
                            first_line = False

                    if "Gaia DR3" in id:

                        cross_reference_list_DR3_only.append([id, int(oid_ref)])
                else:
                    pass
            last_files_last_oid_ref = oid_ref


    print("Sorting")
    # sort by oid to speed up the future join
    cross_reference_list_sorted_by_id = sorted(cross_reference_list, key=lambda gaia2simbadcode: gaia2simbadcode[1])
    # sort by Gaia DR3 reference to speed up the join
    cross_reference_list_DR3_sorted_by_GaiaReference = sorted(cross_reference_list_DR3_only, key=lambda gaia2simbadcode: gaia2simbadcode[0])
    print("Sorted")


    print("Starting write of name2oid.txt")
    with open(os.path.join(working_path,"name2oid.txt"),"w") as fh:
        for line in cross_reference_list_sorted_by_id:
            line_string = "|"
            for value in line:
                line_string += str(value)
                line_string += "|"
            line_string += "\n"
            fh.write(line_string)
    print("Finished write")

    print("Starting write of name2oid_dr3_only.txt")
    with open(os.path.join(working_path,"name2oid_dr3_only.txt"),"w") as fh:
        for line in cross_reference_list_DR3_sorted_by_GaiaReference:
            line_string = "|"
            for value in line:
                line_string += str(value)
                line_string += "|"
            line_string += "\n"
            fh.write(line_string)
    print("Finished write")




    id_list, oid_list = [],[]

    for relationship in cross_reference_list_sorted_by_id:
        id_list.append(relationship[0])
        oid_list.append(relationship[1])

    id_list_gaia_dr3_only, oid_list_dr3_only = [],[]
    for relationship in cross_reference_list_DR3_sorted_by_GaiaReference:
        id_list_gaia_dr3_only.append(relationship[0])
        oid_list_dr3_only.append(relationship[1])



    print("Largest oid {}".format(max(oid_list)))
    return [id_list, oid_list], [id_list_gaia_dr3_only, oid_list_dr3_only]

def generatePreferredNameLookUpList(input_filename,output_filename):

    """
    This produces a list of oid references against all the preferred names.
    It will be ordered by oid, the same as the catalogue


    """

    name_preference_list = ["NAME","HD","SAO","TYC","TIC","2MASS","Gaia DR3"]

    with open(input_filename, 'r') as fh:
        num_lines = sum(1 for line in fh)

    with open(input_filename, "r") as fh:
        first_iteration = True
        look_up_list = []
        line_no = 0
        start_time = datetime.datetime.utcnow()
        for line in fh:
            line_no += 1
            if line_no % 1000 == 0:
                elapsed_time = (datetime.datetime.utcnow() - start_time).total_seconds()
                processing_rate = elapsed_time / line_no # in seconds per line
                time_to_completion, per_cent_done = num_lines * processing_rate, 100 * line_no / num_lines
                print("{} {:.2f}% {}/{}".format(seconds2DHMS(time_to_completion, end_time=True), per_cent_done, line_no, num_lines), end="\r")

            line_list = line.split("|")

            # first time through do the initialisation
            if first_iteration:
                first_iteration = False
                reference_names_list = []
                contains_DR3 = False
                last_line_oid_ref = int(line_list[2])

            if last_line_oid_ref == int(line_list[2]):
                reference_names_list.append(line_list[1])

                if line_list[1][0:len("Gaia DR3")] == "Gaia DR3":
                    contains_DR3 = True
                    Gaia_DR3_code = line_list[1]

            else:
                # this is a new oid

                # first handle the previous collection
                if contains_DR3:
                    best_name_score = len(name_preference_list) - 1
                    best_name = Gaia_DR3_code #name_preference_list[best_name_score]
                    for name in reference_names_list:
                        for name_preference in name_preference_list:
                            if name[0:len(name_preference)] == name_preference and name_preference_list.index(name_preference) < best_name_score:
                                best_name_score = name_preference_list.index(name_preference)
                                #remove problematic last letters in HD catalogue
                                if name[-1].isalpha() and name[0:2] == "HD":
                                    original = name
                                    name = name[:-1]


                                best_name = name



                    ra, dec = getRaDec(best_name, Gaia_DR3_code)
                    look_up_list.append([Gaia_DR3_code, best_name, line_list[2], ra, dec])

                #then reinitialise
                contains_DR3 = False
                Gaia_DR3_code = ""
                reference_names_list = []

                #record this line
                reference_names_list.append(line_list[2])


            last_line_oid_ref = int(line_list[2])

    #Sort by Gaia_DR3_ident

    print("Sorting")
    look_up_list_sorted_by_DR3 = sorted(look_up_list, key=lambda DR3: DR3[0])
    print("Sorted")

    print("Storing")
    with open(output_filename, "w") as fh:
        for line in look_up_list_sorted_by_DR3:
            output_line = "|"
            for value in line:
                output_line += str(value)
                output_line += "|"
            output_line += "\n"
            fh.write(output_line)

    look_up_list_sorted_by_DR3_DR3, look_up_list_sorted_by_DR3_best_name = [],[]
    for relationship in look_up_list_sorted_by_DR3:
        look_up_list_sorted_by_DR3_DR3.append(relationship[0])
        look_up_list_sorted_by_DR3_best_name.append(relationship[1])


    return [look_up_list_sorted_by_DR3_DR3,look_up_list_sorted_by_DR3_best_name]

def calculatePhotometryold(G,Gbp,Grp,c1,c2,c3,c4,c5):

    value = (c1 + c2 * (Gbp-Grp) + c3 * (Gbp-Grp) ** 2 + c4 * (Gbp-Grp) ** 3 + c5 * (Gbp-Grp) ** 4 - G) * -1

    return value


def calculatePhotometry(inputs,coefficients):

    value = 0 - inputs[0]
    Gbp_Grp = inputs[1] - inputs[2]
    for index in range(0,len(coefficients)):
        value += coefficients[index] * (Gbp_Grp ** index)
    return - value


def johnsonCousins(inputs):

    coefficients_list = [[ 0.01448, -0.68740, -0.3604,  0.06718, -0.006061],    #G-B
                         [-0.02275,  0.39610, -0.1243, -0.01396,  0.003775],    #G-R
                         [-0.02704,  0.01424, -0.2156,  0.01426,  0       ],    #G-V
                         [ 0.01753,  0.76000, -0.0991,  0      ,  0       ]]    #G-Ic

    results = []

    for coefficients_index in range(0,len(coefficients_list)):
        coefficients = (coefficients_list[coefficients_index])
        result = (calculatePhotometry(inputs,coefficients))
        results.append(result)


    return results




def generateDR3CatalogueWithSimbadCode(gaia_catalogue, gaia_columns, name_list, oid_list, name_list_dr3_only, oid_list_dr3_only, gaia_dr3_2_preferred_name_gaia_dr3, gaia_dr3_2_preferred_name_name, output_filename):

        gaia_columns.append("B")
        gaia_columns.append("V")
        gaia_columns.append("R")

        gaia_columns.append("Ic")
        gaia_columns.append("oid")
        gaia_columns.append("preferred_name")

        fh = open(output_filename, 'w')
        line_string = "|"

        for column_name in gaia_columns:
            line_string += column_name
            line_string += "|"
        line_string += "\n"
        fh.write(line_string)

        # initialisation
        catalogue_with_oid, last_oid_index, last_gaia_dr3_2_preferred_name_index = [],0,0
        len_of_list = len(name_list)
        len_of_gaia_dr3_2_preferred_name = len(gaia_dr3_2_preferred_name_gaia_dr3)
        last_checked_oid_index = 0
        last_checked_preferred_name_index = 0

        line_no,num_lines = 0, len(gaia_catalogue)
        start_time = datetime.datetime.utcnow()
        for catalogue_line in gaia_catalogue:
            line_no += 1
            if line_no % 100 == 0:
                elapsed_time = (datetime.datetime.utcnow() - start_time).total_seconds()
                processing_rate = elapsed_time / line_no  # in seconds per line
                time_to_completion, per_cent_done = (num_lines * processing_rate) - elapsed_time, 100 * line_no / num_lines
                print("{} {:.2f}% {}/{}".format(seconds2DHMS(time_to_completion, end_time=True), per_cent_done,
                                                line_no, num_lines), end="\r")

            gaia_dr3_ident = catalogue_line[0]

            main_id = ""
            if gaia_dr3_ident in name_list_dr3_only:

                #oid_index is being used as a pointer so that we keep recursion to a minimum

                oid_dr3_only="-1"
                for oid_index in range(last_checked_oid_index,len_of_list):
                    name_dr3_only, oid_dr3_only = name_list_dr3_only[oid_index], oid_list_dr3_only[oid_index]
                    if name_dr3_only == gaia_dr3_ident:
                        last_checked_oid_index = oid_index
                        break
                oid = oid_dr3_only

                gaia_dr3_ident_lookup = ""

                for preferred_name_index in range(last_checked_preferred_name_index,len_of_gaia_dr3_2_preferred_name):
                    gaia_dr3_ident_lookup, preferred_name = gaia_dr3_2_preferred_name_gaia_dr3[preferred_name_index], gaia_dr3_2_preferred_name_name[preferred_name_index]
                    if gaia_dr3_ident_lookup == gaia_dr3_ident:
                        last_checked_preferred_name_index = preferred_name_index
                        break

                if gaia_dr3_ident_lookup == gaia_dr3_ident:

                    main_id = preferred_name
                else:
                    last_checked_preferred_name_index = 0
                    oid, main_id = "-1", gaia_dr3_ident

                # don't do this here. Will be much more efficient to do it on a reverse ordered oid list


            else:

                oid, main_id = "-1", gaia_dr3_ident
                # Missing Gaia DR3 ident {} in oid list
                # This occurs for an object which is in the GaiaDR3 catalogue, but which does not have a Simbad OID code
                pass


            try:
                BRVIc = johnsonCousins([float(catalogue_line[5]),float(catalogue_line[6]),float(catalogue_line[7])])
                # column order is V, B-V, R, Ic
                # catalogue_line.append(str(BRVIc[2]))            #V
                # catalogue_line.append(str(BRVIc[0]-BRVIc[2]))   #B-V
                # catalogue_line.append(str(BRVIc[1]))            #R
                # catalogue_line.append(str(BRVIc[3]))            #Ic

                # column order is B,V,R,Ic
                catalogue_line.append(str(BRVIc[0]))            #B
                catalogue_line.append(str(BRVIc[1]))            #V
                catalogue_line.append(str(BRVIc[2]))            #R
                catalogue_line.append(str(BRVIc[3]))            #Ic
            except:

                catalogue_line.append(str("NOT CALCULATED"))
                catalogue_line.append(str("NOT CALCULATED"))
                catalogue_line.append(str("NOT CALCULATED"))
                catalogue_line.append(str("NOT CALCULATED"))

            catalogue_line.append(oid)
            catalogue_line.append(main_id)

            line_string = "|"
            line_with_oid = []

            for value in catalogue_line:
                line_string += str(value).replace("\n", "").strip()
                line_string += "|"
                line_with_oid.append(value)
            line_with_oid.append(str(oid))
            catalogue_with_oid.append(line_with_oid)
            line_string += "\n"
            fh = open(output_filename, 'a')
            fh.write(line_string)

        fh.close()

        #this needs to be sorted by oid
        oid_col_no = gaia_columns.index('oid')
        catalogue_with_oid_sorted_oid = sorted(catalogue_with_oid, key=lambda oid_sort: oid_sort[oid_col_no])

        return catalogue_with_oid_sorted_oid,gaia_columns


def testFind():

    accumulated_time_recursion, accumulated_time_built_in = 0 ,0
    for test_count in range(0,10):
        random_list, random_list2 = [], []
        for i in range(0,1000):
            random_number =  random.randint(0,360000000) / 1000000
            random_list.append([random_number])
            random_list2.append(random_number)
        target = random.random() * 360

        random_list = sorted(random_list)
        start_time = time.time()
        search_range, recursion_count = findInSorted(target, random_list, interpolate=True, get_nearest=True, interpolate_factor=0.05)
        end_time = time.time()
        accumulated_time_recursion += end_time - start_time
        recursion_result = random_list[search_range[0]][0]

        start_time = time.time()
        built_in_result = min(random_list2, key=lambda x:abs(x-target))
        end_time = time.time()
        accumulated_time_built_in += end_time - start_time
        if recursion_result != built_in_result:
            print("Search Error")
            quit()

        print(accumulated_time_recursion, accumulated_time_built_in)
        if search_range[0] < 0 or search_range[1] > len(random_list) -1:
            print("Result out of range")
        else:
            print("Target {}, {},{}".format(target, random_list[search_range[0]][0],random_list[search_range[1]][0]))
        print("delta recursion {}, delta_built_in {}".format(target-recursion_result, target-built_in_result))
        print(recursion_result, built_in_result)
        print("Accumulated times: Recursion {}, Built in {}".format(accumulated_time_built_in, accumulated_time_recursion))





if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS



    # Init the command line arguments parser

    arg_parser = argparse.ArgumentParser(description="Tool for encoding GAIA DR3 information into RMS format.  \n+" +
                                                     "python -m Utils.ReadGAIACatalog /home/david/tmp/catalogueassembly/inputdata/gaia/result_gaiadr3_20240107230522822_107_78_1.psv \n" +
                                                     "will start processing")


    arg_parser.add_argument('input_path', metavar='INPUT_FILE', type=str,
                            help='Path to the input file.')

    arg_parser.add_argument('-m', '--maxobjects', nargs=1, metavar='MAX_OBJECTS', type=int,
                            help="The maximum number of objects to read, useful for debugging.")

    arg_parser.add_argument('-w', '--workarea', nargs=1, metavar='MAX_OBJECTS', type=str,
                            help="Directory for working.")


    cml_args = arg_parser.parse_args()

    ### test besselian precession

    if False:
        print(getRaDecTYC("TYC   85-1075-1","Blah"))

        print(degrees2HMS(41.054063))
        print(degrees2DMS(49.348483))

        print("one", end="\r")
        print("two", end="\r")
        start_time = datetime.datetime.utcnow()
        time.sleep(10)
        end_time = datetime.datetime.utcnow()
        elapsed_time = (end_time - start_time).total_seconds()
        print(elapsed_time)
        print(seconds2DHMS((elapsed_time)))
        print(seconds2DHMS(7652))

        ra, dec = besselianPrecession(j2000, 360 * (2/24 + 44 / (60*24) + 11.986 / (24 * 3600)), (49 + 13/60 + 42.48 / 3600), 2462088.69 ,  pm_ra = 0.03425 * 3600 * 360/(60*60*24), pm_dec = -0.0895)
        print(ra,dec)

    working_path = createWorkArea(cml_args.workarea)
    pickle_path = os.path.join(working_path,"pickles")


    #This provides a lookup table to go from Simbad oid key to main_id, which I think is the name that GMN wishes to use
    #print("Reading Simbad Basic ")



    print(getRaDecHD("HD    333","Gaia DR3 2861084531426930944" ))


    #Read in the Gaia catalogue

    gaia_catalogue_pickle_file_path = os.path.join(pickle_path, 'gaia_dr3.pickle')
    name_2_oid_list_pickle_file_path = os.path.join(pickle_path, 'name_2_oid_list.pickle')
    name_2_oid_list_dr3_only_pickle_file_path = os.path.join(pickle_path, 'name_2_oid_list_dr3_only.pickle')
    preferred_name_look_up_list_pickle_file_path = os.path.join(pickle_path,'gaiaDR3_2_preferred_name_look_up_list.pickle')

    if os.path.exists(gaia_catalogue_pickle_file_path):
        print("Unpickling gaia_list data                                1/8")
        with open(gaia_catalogue_pickle_file_path, 'rb') as fh:
            gaia_list = pickle.load(fh)
    else:
        print("Generating gaia_list pickle")
        gaia_list = readGaiaCatalogTxt(os.path.expanduser(cml_args.input_path), max_objects=cml_args.maxobjects)
        print("Gaia catalogue read complete")
        print("Pickling gaia_catalogue data                             1/8")
        with open(gaia_catalogue_pickle_file_path, 'wb') as fh:
            pickle.dump(gaia_list, fh)




        #From the Gaia catalogue produce a relationship between Gaia ident and Simbad Code
    if os.path.exists(name_2_oid_list_pickle_file_path) and os.path.exists(name_2_oid_list_dr3_only_pickle_file_path):

        print("Unpickling name_2_oid_list                               2/8")
        with open(name_2_oid_list_pickle_file_path, 'rb') as fh:
            name_2_oid_list = pickle.load(fh)

        print("Unpickling name_2_oid_list_dr3_only                      3/8")
        with open(name_2_oid_list_dr3_only_pickle_file_path, 'rb') as fh:
            name_2_oid_list_dr3_only = pickle.load(fh)

    else:
        print("Pickling name_2_oid_list and name_2_oid_list_dr3_only")
        name_2_oid_list, name_2_oid_list_dr3_only = generateGaia2SimbadCodeFromIdentTables(os.path.expanduser("~/tmp/catalogueassembly/inputdata/simbad/identtable"), working_path, gaia_list[0], gaia_list[1])

        print("Pickling name_2_oid_list                                 2/8")
        with open(name_2_oid_list_pickle_file_path, 'wb') as fh:
            pickle.dump(name_2_oid_list, fh)

        print("Pickling name_2_oid_list_dr3_only                        3/8")
        with open(name_2_oid_list_dr3_only_pickle_file_path, 'wb') as fh:
            pickle.dump(name_2_oid_list_dr3_only, fh)

    print("Preparing preferred name lookup table")




    if not os.path.exists(preferred_name_look_up_list_pickle_file_path):

        preferred_name_look_up_list = generatePreferredNameLookUpList(os.path.expanduser("~/tmp/catalogueassembly/name2oid.txt"),
                                                                      os.path.expanduser("~/tmp/oid2preferredname.txt"))

        print("Pickling preferred_name_look_up_list                     4/8")
        with open(preferred_name_look_up_list_pickle_file_path,'wb') as fh:
            pickle.dump(preferred_name_look_up_list,fh)


    else:

        print("Reading preferred_name_look_up_list                      4/8")
        with open(preferred_name_look_up_list_pickle_file_path, 'rb') as fh:
            gaiaDR3_2_preferred_name_DR3 = pickle.load(fh)

    print("Produce Gaia catalogue sorted by simbad code")
    catalogue_with_oid,gaia_columns = generateDR3CatalogueWithSimbadCode(gaia_list[0], gaia_list[1], name_2_oid_list[0], name_2_oid_list[1],
                                                            name_2_oid_list_dr3_only[0], name_2_oid_list_dr3_only[1],
                                                            gaiaDR3_2_preferred_name_DR3[0], gaiaDR3_2_preferred_name_DR3[1],
                                                    os.path.expanduser("~/tmp/gaiacatalogue_with_simbad_code.txt"))

    print("Gaia catalogue sorted by simbad code produced")

    with open (os.path.expanduser("~/tmp/catalogue_with_oid_sorted_by_oid.txt", 'wb')) as f:

        output_line = "|"
        for value in gaia_columns:
            output_line += str(value)
            output_line += "|"
        output_line += "\n"
        f.write(output_line)

        for line in catalogue_with_oid:
            output_line = "|"
            for value in line:
                output_line += str(value)
                output_line += "|"
            output_line += "\n"
            f.write(output_line)

