import os.path

import psycopg
import numpy as np
import argparse


from constellations import loadConstellationLines, filterSouthernConstellations, filterNorthernConstellations

from atlas import plotAtlasTwoPanel, readIAUCSN

DB_SCALE_FACTOR = 1e6


def fetchHemisphereRadec(conn, hemisphere="south", limit_rows=5000, mag_limit=3, test_mode=False):



    mag_scaled_limit = int(mag_limit * DB_SCALE_FACTOR)

    if hemisphere == "south":
        dec_filter = "dec < 0"
    else:
        dec_filter = "dec >= 0"

    if test_mode:
        trusted_stars = [
            ("Sirius", 101.2875, -16.7161),
            ("Canopus", 95.9879, -52.6957),
            ("Arcturus", 213.9153, 19.1824),
            ("Vega", 279.2347, 38.7837),
            ("Capella", 79.1723, 45.9979),
            ("Rigel", 78.6345, -8.2016),
            ("Betelgeuse", 88.7929, 7.4071),
            ("Fomalhaut", 344.4128, -29.6222),
            ("Antares", 247.3519, -26.4320),
            ("Deneb", 310.3579, 45.2803),
            ("Altair", 297.6958, 8.8683),
            ("Aldebaran", 68.9800, 16.5093),
        ]


        r_list, d_list, m_list, n_list = [], [], [], []
        for n, r, d in trusted_stars:
            if hemisphere == "south":
                if d < 0:
                    r_list.append(r)
                    d_list.append(d)
                    m_list.append(3)
                    n_list.append(n)
            else:
                if d > 0:
                    r_list.append(r)
                    d_list.append(d)
                    m_list.append(3)
                    n_list.append(n)

        ra_list, dec_list, names_list, mag_list = r_list, d_list, n_list, m_list

        print(hemisphere, ra_list, dec_list)

        star_data, _, _, extra_values_dict = readIAUCSN(os.path.expanduser("~/source/RMS/Catalogs"),"IAU-CSN.txt", additional_fields=['Name/ASCII'])

        ra_list, dec_list, mag_list, names_list = [], [], [], []
        for r, n in zip(star_data, extra_values_dict['Name/ASCII']):
            if hemisphere == "south":
                if r[1] < 0:
                    ra_list.append(r[0])
                    dec_list.append(r[1])
                    mag_list.append(r[2])
                    names_list.append(n.decode().split()[0])
            else:
                if r[1] > 0:
                    ra_list.append(r[0])
                    dec_list.append(r[1])
                    mag_list.append(r[2])
                    names_list.append(n.decode().split()[0])

        pass

    else:

        names_list = None
        query = f"""
            SELECT ra, dec, mag
            FROM observation
            WHERE ra IS NOT NULL
              AND dec IS NOT NULL
              AND mag < {mag_scaled_limit}
              AND station_name = 'AU000K' or station_name = 'AU0007'
              AND {dec_filter}
            LIMIT {limit_rows};
        """

        ra_list = []
        dec_list = []
        mag_list = []
        names_list = []

        with conn.cursor() as cur:
            cur.execute(query)
            for ra_scaled, dec_scaled, mag_scaled in cur:
                ra_list.append(ra_scaled / DB_SCALE_FACTOR)
                dec_list.append(dec_scaled / DB_SCALE_FACTOR)
                mag_list.append(mag_scaled / DB_SCALE_FACTOR)

    ra_list, dec_list = np.array(ra_list), np.array(dec_list)



    return ra_list, dec_list, mag_list, names_list



if __name__ == "__main__":
    ### COMMAND LINE ARGUMENTS




    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting capture and compression.
        """)

    # Add a mutually exclusive for the parser (the arguments in the group can't be given at the same)
    arg_group = arg_parser.add_mutually_exclusive_group()


    arg_parser.add_argument('-t', '--test_mode', action="store_true", help="""Run in test mode """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    test_mode = cml_args.test_mode

    constellations_all = loadConstellationLines(
        "~/source/RMS/share/constellation_lines.csv"
    )
    constellations_south = filterSouthernConstellations(constellations_all)
    constellations_north = filterNorthernConstellations(constellations_all)

    for c in constellations_north:
        print(c)


    with psycopg.connect(
        host="192.168.1.190",
        dbname="star_data",
        user="ingest_user",
    ) as conn:
        ra_south_deg, dec_south_deg, mag_south, names_list_south = fetchHemisphereRadec(conn, "south", test_mode=test_mode)
        ra_north_deg, dec_north_deg, mag_north, names_list_north = fetchHemisphereRadec(conn, "north", test_mode=test_mode)


    fig = plotAtlasTwoPanel(
        ra_south_deg,
        dec_south_deg,
        mag_south,
        ra_north_deg,
        dec_north_deg,
        mag_north,
        constellations_south,
        constellations_north,
        names_list_north,
        names_list_south    )



    fig.savefig(
        "GMNSkySurvey.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.2,
    )
