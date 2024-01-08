import os
import numpy as np
import argparse
import pickle
import urllib.request
import time
from operator import itemgetter,attrgetter
from tqdm import tqdm
import random
import time

# Read from http://jvo.nao.ac.jp/portal/gaia/dr3.do
# Photometric calculations at https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html

# Example star name query
# http://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+1137162861578295168


def findInSorted(target,dataset, field_no=0, search_range=None,
                 interpolate=True, interpolate_factor=0.05, get_nearest=True,
                 recursion_count=0):

    """
    High speed recursive searching in fairly evenly distributed sorted lists of lists
    Very efficient for long lists

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

    cross_reference_list, cross_reference_list_DR3_only = [],[]

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
                if oidref != "oidref" and id !="id" and "-" not in oidref:
                    cross_reference_list.append([id,int(oidref)])
                    if "Gaia DR3" in id:
                        cross_reference_list_DR3_only.append([id, int(oidref)])
                else:
                    pass
            print(len(cross_reference_list))


    print("Sorting")
    # sort by oid to speed up the future join
    cross_reference_list_sorted_by_id = sorted(cross_reference_list, key=lambda gaia2simbadcode: gaia2simbadcode[1])
    # sort by Gaia DR3 reference to speed up the join
    cross_reference_list_DR3_sorted_by_GaiaReference = sorted(cross_reference_list_DR3_only, key=lambda gaia2simbadcode: gaia2simbadcode[0])
    print("Sorted")


    print("Starting write of name2oid.txt")
    with open("/home/david/tmp/name2oid.txt","w") as fh:
        for line in cross_reference_list_sorted_by_id:
            line_string = "|"
            for value in line:
                line_string += str(value)
                line_string += "|"
            line_string += "\n"
            fh.write(line_string)
    print("Finished write")

    print("Starting write of name2oid_dr3_only.txt")
    with open("/home/david/tmp/name2oid_dr3_only.txt","w") as fh:
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
    return id_list, oid_list, id_list_gaia_dr3_only, oid_list_dr3_only

def generateNameLookUpList(input_filename,output_filename):

    """
    This produces a list of oid references against all the preferred names.
    It will be ordered by oid, the same as the catalogue


    """

    name_preference_list = ["NAME","HD","SAO","TYC","TIC","2MASS","GAIA_DR3"]

    with open(input_filename, "r") as fh:
        first_iteration = True
        look_up_list = []

        for line in tqdm(fh):
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
                    best_name_score = np.inf
                    for name in reference_names_list:
                        for name_preference in name_preference_list:
                            if name[0:len(name_preference)] == name_preference and name_preference_list.index(name_preference) < best_name_score:
                                best_name_score = name_preference_list.index(name_preference)
                                best_name = name
                    look_up_list.append([Gaia_DR3_code, best_name,line_list[2]])

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


    return look_up_list_sorted_by_DR3_DR3,look_up_list_sorted_by_DR3_best_name

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

        gaia_columns.append("V")
        gaia_columns.append("B-V")
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

        for catalogue_line in tqdm(gaia_catalogue):
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
                catalogue_line.append(str(BRVIc[2]))            #V
                catalogue_line.append(str(BRVIc[0]-BRVIc[2]))   #B-V
                catalogue_line.append(str(BRVIc[1]))            #R
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




if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS



    # Init the command line arguments parser

    arg_parser = argparse.ArgumentParser(description="Tool for encoding GAIA DR3 information into RMS format.")

    arg_parser.add_argument('input_path', metavar='INPUT_FILE', type=str,
                            help='Path to the input file.')

    arg_parser.add_argument('-m', '--maxobjects', nargs=1, metavar='MAX_OBJECTS', type=int,
                            help="The maximum number of objects to read, useful for debugging.")

    cml_args = arg_parser.parse_args()

    print(johnsonCousins([11.342619,12.138336,10.472202]))


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




    if False:
        #This provides a lookup table to go from Simbad oid key to main_id, which I think is the name that GMN wishes to use
        #print("Reading Simbad Basic ")
        simbadBasicSortedByOID, simbad_columns, main_id_list_simbad, oid_list_simbad = generateOID2Main_ID("/home/david/tmp/simbad_basic.txt", "/home/david/tmp/oid2main_id.txt", max_objects=cml_args.maxobjects)
        #print("Simbad Basic read completed - oid2main_id file written")


        #Read in the Gaia catalogue
        print("Reading in the Gaia Catalogue")
        gaia_catalogue, gaia_columns = readGaiaCatalogTxt(os.path.expanduser(cml_args.input_path), max_objects=cml_args.maxobjects)
        print("Gaia Catalogue read complete")

        print("Pickling gaia_catalogue data             1/8")
        with open('/home/david/tmp/pickles/gaia_catalogue.pickle','wb') as fh:
            pickle.dump(gaia_catalogue, fh)

        print("Pickling gaia_catalogue columns          2/8")
        with open('/home/david/tmp/pickles/gaia_columns.pickle','wb') as fh:
            pickle.dump(gaia_columns, fh)

        #From the Gaia catalogue produce a relationship between Gaia ident and Simbad Code

        print("Producing relationship from Gaia DR3 ident to Simbad Code")
        name_list, oid_list,id_list_gaia_dr3_only, oid_list_gaia_dr3_only = generateGaia2SimbadCodeFromIdentTables(gaia_catalogue, gaia_columns)

        print("Pickling name_list                       3/8")
        with open('/home/david/tmp/pickles/name_list.pickle','wb') as fh:
            pickle.dump(name_list, fh)

        print("Pickling oid_list                        4/8")
        with open('/home/david/tmp/pickles/oid_list.pickle','wb') as fh:
            pickle.dump(oid_list, fh)

        print("Pickling id_list_gaia_dr3_only           5/8")
        with open('/home/david/tmp/pickles/id_list_gaia_dr3_only.pickle','wb') as fh:
            pickle.dump(id_list_gaia_dr3_only, fh)

        print("Pickling oid_list_gaia_dr3_only          6/8")
        with open('/home/david/tmp/pickles/oid_list_gaia_dr3_only.pickle','wb') as fh:
            pickle.dump(oid_list_gaia_dr3_only, fh)

        print("Preparing lookup table")

        gaiaDR3_2_preferred_name_DR3, gaiaDR3_2_preferred_name_name = generateNameLookUpList("/home/david/tmp/name2oid.txt","/home/david/tmp/oid2preferredname.txt")

        print("Pickling gaiaDR_2_preferred_name_DR3     7/8")
        with open('/home/david/tmp/pickles/gaiaDR3_2_preferred_name_DR3.pickle','wb') as fh:
            pickle.dump(gaiaDR3_2_preferred_name_DR3,fh)

        print("Pickling gaia_DR3_2_preferred_name_name  8/8")
        with open('/home/david/tmp/pickles/gaiaDR3_2_preferred_name_name.pickle','wb') as fh:
            pickle.dump(gaiaDR3_2_preferred_name_name, fh)

    else:
        print("Unpickling")

        print("Gaia_catalogue data             1/8")
        with open('/home/david/tmp/pickles/gaia_catalogue.pickle', 'rb') as fh:
            gaia_catalogue = pickle.load(fh)

        print("Gaia_catalogue columns          2/8")
        with open('/home/david/tmp/pickles/gaia_columns.pickle', 'rb') as fh:
            gaia_columns = pickle.load(fh)

        print("name_list                       3/8")
        with open('/home/david/tmp/pickles/name_list.pickle', 'rb') as fh:
            name_list = pickle.load(fh)

        print("oid_list                        4/8")
        with open('/home/david/tmp/pickles/oid_list.pickle', 'rb') as fh:
            oid_list = pickle.load(fh)

        print("id_list_gaia_dr3_only           5/8")
        with open('/home/david/tmp/pickles/id_list_gaia_dr3_only.pickle', 'rb') as fh:
            id_list_gaia_dr3_only = pickle.load(fh)

        print("oid_list_gaia_dr3_only          6/8")
        with open('/home/david/tmp/pickles/oid_list_gaia_dr3_only.pickle', 'rb') as fh:
            oid_list_gaia_dr3_only = pickle.load(fh)


        print("gaiaDR_2_preferred_name_DR3     7/8")
        with open('/home/david/tmp/pickles/gaiaDR3_2_preferred_name_DR3.pickle', 'rb') as fh:
            gaiaDR3_2_preferred_name_DR3 = pickle.load(fh)

        print("gaia_DR3_2_preferred_name_name  8/8")
        with open('/home/david/tmp/pickles/gaiaDR3_2_preferred_name_name.pickle', 'rb') as fh:
            gaiaDR3_2_preferred_name_name = pickle.load(fh)

    print("Produce Gaia catalogue sorted by simbad code")
    catalogue_with_oid,gaia_columns = generateDR3CatalogueWithSimbadCode(gaia_catalogue, gaia_columns, name_list, oid_list,
                                                            id_list_gaia_dr3_only, oid_list_gaia_dr3_only,
                                                            gaiaDR3_2_preferred_name_DR3,gaiaDR3_2_preferred_name_name,
                                                    "/home/david/tmp/gaiacatalogue_with_simbad_code.txt")
    print("Gaia catalogue sorted by simbad code produced")

    with open ("/home/david/tmp/catalogue_with_oid_sorted_by_oid.txt", 'wb') as f:

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

