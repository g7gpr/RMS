import os
import numpy as np
import argparse
import pickle
import urllib.request
import time
from operator import itemgetter,attrgetter
from tqdm import tqdm

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

        # sort by designation - also known is gaia id

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

    table_files = sorted(os.listdir("/home/david/tmp/IDS_TABLE/"))


    for table_file in table_files:
        path_table_file =  os.path.join("/home/david/tmp/IDS_TABLE/", table_file)
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


def generateGaia2SimbadCodeFromIdentTables(catalogue, columns):
    """
    Replacement for IDS function

    """
    star_list, designator_list = [], []
    for star in catalogue:
        (star[columns.index("designation")].replace(" ", "+"))
        designator = star[columns.index("designation")]
        designator_list.append(designator)

    cross_reference_list = []

    table_files = sorted(os.listdir("/home/david/tmp/IDENT_TABLE/"))

    for table_file in table_files:
        path_table_file = os.path.join("/home/david/tmp/IDENT_TABLE/", table_file)
        print(path_table_file)
        with open(path_table_file, 'r') as fh:

            for line in tqdm(fh):
                line_list = line.replace("\n","").split("|")
                if len(line_list) != 2:
                    continue
                id = line_list[0].replace('"','').strip()
                oidref=line_list[1]

                if "Gaia DR3" == id[0:8]:

                    cross_reference_list.append([id,int(oidref)])
            print(len(cross_reference_list))

    # sort by gaia ident to speed up the future join
    print("Sorting")
    cross_reference_list_sorted_by_id = sorted(cross_reference_list, key=lambda gaia2simbadcode: gaia2simbadcode[0])
    print("Sorted")
    gaia_id_list, oid_list = [],[]

    for relationship in cross_reference_list_sorted_by_id:
        gaia_id_list.append(relationship[0])
        oid_list.append(relationship[1])


    return gaia_id_list, oid_list


def generateDR3CatalogueWithSimbadCode(catalogue, columns, gaia_id_list, oid_list, main_id_list_simbad, oid_list_simbad, output_filename):


    with open(output_filename, 'w') as fh:

        line_string = "|"
        for column_name in columns:
            line_string += column_name
            line_string += "|"
        line_string += "simbad_code|main_id|"
        line_string += "\n"
        fh.write(line_string)




    # optimise this code - both lists are sorted so can be merged more efficiently
        for catalogue_line in tqdm(catalogue):
            gaia_dr3_ident = catalogue_line[0]
        #add the simbad oid
            if gaia_dr3_ident in gaia_id_list:
                oid_index = gaia_id_list.index(gaia_dr3_ident)
                oid = oid_list[oid_index]
                # since we had a valid simbad oid, try and find the name
                if oid in oid_list_simbad:
                    oid_list_simbad_index = oid_list_simbad.index(oid)
                    main_id = main_id_list_simbad[oid_list_simbad_index]
                else:
                    main_id = gaia_dr3_ident
            else:
                oid, main_id = "-1", gaia_dr3_ident

            catalogue_line.append(oid)
            catalogue_line.append(main_id)

            for value in catalogue_line:
                line_string += str(value).replace("\n", "").strip()
                line_string += "|"
            line_string += "\n"
            fh.write(line_string)






if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS



    # Init the command line arguments parser

    arg_parser = argparse.ArgumentParser(description="Tool for encoding GAIA DR3 information into RMS format.")

    arg_parser.add_argument('input_path', metavar='INPUT_FILE', type=str,
                            help='Path to the input file.')

    arg_parser.add_argument('-m', '--maxobjects', nargs=1, metavar='MAX_OBJECTS', type=int,
                            help="The maximum number of objects to read, useful for debugging.")

    cml_args = arg_parser.parse_args()



    #This provides a lookup table to go from Simbad oid key to main_id, which I think is the name that GMN wishes to use
    print("Reading Simbad Basic ")
    simbadBasicSortedByOID, columns, main_id_list_simbad, oid_list_simbad = generateOID2Main_ID("/home/david/tmp/simbad_basic.txt", "/home/david/tmp/oid2main_id.txt", max_objects=cml_args.maxobjects)
    print("Simbad Basic read completed - oid2main_id file written")


    #Read in the Gaia catalogue
    print("Reading in the Gaia Catalogue")
    catalogue, columns = readGaiaCatalogTxt(os.path.expanduser(cml_args.input_path),max_objects=cml_args.maxobjects)
    print("Gaia Catalogue read complete")


    #From the Gaia catalogue produce a relationship between Gaia ident and Simbad Code

    print("Producing relationship from Gaia DR3 ident to Simbad Code")
    gaia_id_list, oid_list = generateGaia2SimbadCodeFromIdentTables(catalogue, columns)

    print("Pickling work so far")
    #with open('gaia_id_list','wb') as fh:
    #    pickle.dump(gaia_id_list, fh)

    #with open('oid_list','wb') as fh:
    #    pickle.dump(oid_list, fh)

    #with open('gaia_id_list', 'rb') as fh:
    #    gaia_id_list = pickle.load(fh)

    #with open('oid_list', 'rb') as fh:
    #    oid_list = pickle.load(fh)

    print("Relationship from Gaia DR3 ident to Simbad code produced")

    print("Produce Gaia catalogue sorted by simbad code")
    generateDR3CatalogueWithSimbadCode(catalogue, columns, gaia_id_list, oid_list, main_id_list_simbad, oid_list_simbad,"/home/david/tmp/gaiacatalogue_with_simbad_code.txt")
    print("Gaia catalogue sored by simbad code produced")




    #ra, dec, mag = results.T


    #plt.scatter(ra, dec, s=0.1)
    #plt.show()