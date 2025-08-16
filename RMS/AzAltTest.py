import os

import numpy as np

from RMS.Astrometry.Conversions import AER2ECEF, latLonAlt2ECEF, ECEF2AltAz, ecef2LatLonAlt
from RMS.Formats.Platepar import Platepar
from RMS.Math import angularSeparationDeg



pp = Platepar()
pp.read(os.path.expanduser("~/source/RMS/platepar_cmn2010.cal"))
az, alt = pp.az_centre + 5, pp.alt_centre + 5

station_ecef = latLonAlt2ECEF(np.radians(pp.lat), np.radians(pp.lon), pp.elev)
lat_check, lon_check, alt_check = ecef2LatLonAlt(station_ecef[0], station_ecef[1], station_ecef[2])

print("Original lat, lon, alt {:.6f},{:.6f},{:.2f}".format(pp.lat, pp.lon, pp.elev))
print("Check lat, lon, alt    {:.6f},{:.6f},{:.2f}".format(np.degrees(lat_check), np.degrees(lon_check), (alt_check)))

# Calculate a point at range 1000km at given az alt from station
ecef_point_x, ecef_point_y, ecef_point_z = AER2ECEF(az, alt, 1000, pp.lat, pp.lon, pp.elev)

# Take the reverse of this function
check_az, check_alt = ECEF2AltAz(station_ecef, (ecef_point_x, ecef_point_y, ecef_point_z))

print("Orginal az, alt {:.3f} , {:.3f}".format(az,alt))
print("Check   az, alt {:.3f},  {:.3f}".format(check_az, check_alt))

# Compute the differences in az alt
az_alt_angle_error = angularSeparationDeg(az, alt, check_az, check_alt)
print("Az alt angle error {:.3f}".format(az_alt_angle_error))

