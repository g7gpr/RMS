
import os
import numpy as np


def readGaiaCatalogTxt(file_path, lim_mag=None):
    """ Read star data from the GAIA catalog in the .dat format. """


    results = []

    with open(file_path) as f:

        # Skip the header
        f.next()
        f.next()

        # Read Gaia catalog
        for line in f:

            line = line.replace('\n', '').replace('\r', '')
            line = line.split('|')

            ra, dec, mag = list(map(float, line))

            # Skip magnitudes below the limiting magnitude
            if lim_mag is not None:
                if mag < lim_mag:
                    continue


            results.append([ra, dec, mag])



        return np.array(results)



def loadGaiaCatalog(dir_path, file_name, lim_mag=None):
    """ Read star data from the GAIA catalog in the .npy format. 
    
    Arguments:
        dir_path: [str] Path to the directory where the catalog file is located.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        lim_mag: [float] Faintest magnitude to return. None by default, which will return all stars.

    Return:
        results: [2d ndarray] Rows of (ra, dec, mag), angular values are in degrees.
    """

    file_path = os.path.join(dir_path, file_name)

    # Read the catalog
    results = np.load(file_path, allow_pickle=False)


    # Filter by limiting magnitude
    if lim_mag is not None:

        results = results[results[:, 2] < lim_mag]


    return results


            


    


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # file_path = 'gaia_dr2.dat'
    # results = readGaiaCatalogTxt(file_path)


    file_name = 'gaia_dr2_mag_11.5.npy'

    results = loadGaiaCatalog('.', file_name, lim_mag=7)

    #np.save('gaia_dr2_mag_11.5', results, allow_pickle=False)

    ra, dec, mag = results.T


    plt.scatter(ra, dec, s=0.1)
    plt.show()