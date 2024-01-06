import os
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import urllib.request
import time

# Read from http://jvo.nao.ac.jp/portal/gaia/dr3.do
# Photometric calculations at https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html

# Example star name query
# http://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+1137162861578295168


def handleHeader(line):

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
    """ Read star data from the GAIA catalog in the .psv format - works with any column order. """


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
                info_type, data_returned = handleHeader(line[1:])
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


        return catalogue_sorted_by_gaia_id, columns

def getStarNames(catalogue):

    star_list, designator_list = [],[]
    for star in catalogue:

        (star[columns.index("designation")].replace(" ", "+"))
        designator = star[columns.index("designation")]
        designator_list.append(designator)


    cross_reference_list = []
    with open("/home/david/tmp/IDS.txt",'r') as fh:
        print(designator_list)
        for line in fh:
            ident_list = line.replace('"','').split("|")

            for ident in ident_list:
                if "DR3" in ident:
                    gaia_dr3_ident = ident
                    if ident in designator_list:
                        cross_reference_list.append([gaia_dr3_ident, ident_list[-1].strip()])


    # sort by gaia ident to speed up the future join
    cross_reference_list_sorted_by_id = sorted(cross_reference_list,key=lambda gaia2sinbadcode: gaia2sinbadcode[0])
    return cross_reference_list_sorted_by_id


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS



    # Init the command line arguments parser

    arg_parser = argparse.ArgumentParser(description="Tool for encoding GAIA DR3 information into RMS format.")

    arg_parser.add_argument('input_path', metavar='INPUT_FILE', type=str,
                            help='Path to the input file.')

    arg_parser.add_argument('-m', '--maxobjects', nargs=1, metavar='MAX_OBJECTS', type=int,
                            help="The maximum number of objects to read, useful for debugging.")

    cml_args = arg_parser.parse_args()



    print("Reading file from {}".format(cml_args.input_path))




    catalogue, columns = readGaiaCatalogTxt(os.path.expanduser(cml_args.input_path),max_objects=cml_args.maxobjects)

    with open('/home/david/tmp/catalogue.pickle', 'wb') as fh:
        pickle.dump(catalogue, fh)

    print(columns)

    gaia2name_list = getStarNames(catalogue)

    with open("/home/david/tmp/gaia2id.pickle", 'wb') as fh:

        pickle.dump(gaia2name_list, fh)

    gaia_dr3_catalogue_with_sinbad_code = []

    # optimise this code - both lists are sorted so can be merged more efficiently
    for catalogue_line in catalogue:
        gaia_dr3_ident = catalogue_line[0]

        for relation in gaia2name_list:
            if relation[0] == gaia_dr3_ident:
                sinbad_code = relation[1]
                print("{}|{}".format(gaia_dr3_ident, sinbad_code))
                catalogue_line.append(sinbad_code)
                gaia_dr3_catalogue_with_sinbad_code.append(catalogue_line)

    with open("/home/david/tmp/gaiacatalogue_with_sinbad_code.txt", 'w') as fh:

        line_string = "|"
        for column_name in columns:
            line_string += column_name
            line_string += "|"
        line_string += "sinbad_code|"
        line_string += "\n"
        fh.write(line_string)

        for line in gaia_dr3_catalogue_with_sinbad_code:
            line_string = "|"
            print(line)
            for value in line:
                line_string += value.replace("\n","")
                line_string += "|"
            line_string += "\n"
            fh.write(line_string)




    #ra, dec, mag = results.T


    #plt.scatter(ra, dec, s=0.1)
    #plt.show()