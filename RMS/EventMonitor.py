# RPi Meteor Station
# Copyright (C) 2016
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



""""" Automatically uploads data files based on time and trajectory information given on a website. """

from __future__ import print_function, division, absolute_import

import sqlite3
import multiprocessing
import RMS.ConfigReader as cr
import urllib.request
import os
import shutil
import time
import copy
import uuid
import numpy as np
import datetime
import argparse
import math
import random, string
from glob import glob


from RMS.Astrometry.Conversions import datetime2JD, geo2Cartesian, altAz2RADec, vectNorm
from RMS.Astrometry.Conversions import latLonAlt2ECEF, ecef2LatLonAlt, AER2LatLonAlt, AEH2Range, ECEF2AltAz
from RMS.Math import angularSeparationVect, polarToCartesian
from RMS.Formats.Platepar import Platepar
from datetime import datetime
from dateutil import parser
from Utils.StackFFs import stackFFs
from Utils.BatchFFtoImage import batchFFtoImage
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz
from RMS.UploadManager import uploadSFTP
import logging


log = logging.getLogger("logger")


class EventContainer(object):

    def __init__(self, dt, lat, lon, ht):

        # Required parameters

        self.dt,self.time_tolerance = dt, 0

        self.lat, self.lat_std, self.lon, self.lon_std, self.ht, self.ht_std = lat, 0, lon, 0, ht,0
        self.lat2, self.lat2_std, self.lon2, self.lon2_std, self.ht2, self.ht2_std = 0,0,0,0,0,0
        self.close_radius, self.far_radius = 0,0

        # Or trajectory information from the first point
        self.azim, self.azim_std, self.elev, self.elev_std, self.elev_is_max = 0,0,0,0,False

        self.stations_required = ""
        self.respond_to = ""

        # These are internal control properties

        self.uuid = ""
        self.event_spec_type = 0
        self.files_uploaded = []
        self.time_completed = None
        self.observed_status = None
        self.processed_status = False

        self.start_distance, self.start_angle, self.end_distance, self.end_angle = 0,0,0,0
        self.fovra, self.fovdec = 0 , 0


    def setValue(self, variable_name, value):

        """ Receive a name and value pair, and put them into this event

        Arguments:
            variable_name: Name of the variable
            value        : Value to be assigned

        Return:
            Nothing
        """
        # Extract the variable name, truncate before any '(' used for units
        variable_name = variable_name.strip().split('(')[0].strip()

        if value == "":
            return

        # Mandatory parameters

        self.dt = value if "EventTime" == variable_name else self.dt
        self.time_tolerance = value if "TimeTolerance" == variable_name else self.time_tolerance
        self.lat = float(value) if "EventLat" == variable_name else self.lat
        self.lat_std = float(value) if "EventLatStd" == variable_name else self.lat_std
        self.lon = float(value) if "EventLon" == variable_name else self.lon
        self.lon_std = float(value) if "EventLonStd" == variable_name else self.lon_std
        self.ht = float(value) if "EventHt" == variable_name else self.ht
        self.ht_std = float(value) if "EventHtStd" == variable_name else self.ht_std

        # radii

        self.close_radius = float(value) if "CloseRadius" == variable_name else self.close_radius
        self.far_radius = float(value) if "FarRadius" == variable_name else self.far_radius

        # Optional parameters, if trajectory is set by a start and an end

        self.lat2 = float(value) if "EventLat2" == variable_name else self.lat2
        self.lat2_std = float(value) if "EventLat2Std" == variable_name else self.lat2_std
        self.lon2 = float(value) if "EventLon2" == variable_name else self.lon2
        self.lon2_std = float(value) if "EventLon2Std" == variable_name else self.lon2_std
        self.ht2 = float(value) if "EventHt2" == variable_name else self.ht2
        self.ht2_std = float(value) if "EventHt2Std" == variable_name else self.ht2_std

        # Optional parameters for defining trajectory by a start point, and a direction

        if "EventAzim" == variable_name:
            if value is None:
                self.azim = 0
            else:
                self.azim = float(value)

        if "EventAzimStd" == variable_name:
            if value is None:
                self.azim_std = 0
            else:
                self.azim_std = float(value)

        if "EventElev" == variable_name:
            if value is None:
                self.elev = 0
            else:
                self.elev = float(value)

        if "EventElevStd" == variable_name:
            if value is None:
                self.elev_std = 0
            else:
                self.elev_std = float(value)

        if "EventElevIsMax" == variable_name:
            if value == "True":
                self.elev_is_max = True
            else:
                self.elev_is_max = False

        # Control information

        self.stations_required = str(value) if "StationsRequired" == variable_name else self.stations_required
        self.uuid = str(value) if "uuid" == variable_name else self.uuid
        self.respond_to = str(value) if "RespondTo" == variable_name else self.respond_to

    def eventToString(self):

        """ Turn an event into a string

        Arguments:

        Return:
            String representation of an event
        """

        output = "# Required \n"
        output += ("EventTime                : {}\n".format(self.dt))
        output += ("TimeTolerance (s)        : {:.0f}\n".format(self.time_tolerance))
        output += ("EventLat (deg +N)        : {:3.2f}\n".format(self.lat))
        output += ("EventLatStd (deg)        : {:3.2f}\n".format(self.lat_std))
        output += ("EventLon (deg +E)        : {:3.2f}\n".format(self.lon))
        output += ("EventLonStd (deg)        : {:3.2f}\n".format(self.lon_std))
        output += ("EventHt (km)             : {:3.2f}\n".format(self.ht))
        output += ("EventHtStd (km)          : {:3.2f}\n".format(self.ht_std))
        output += ("CloseRadius(km)          : {:3.2f}\n".format(self.close_radius))
        output += ("FarRadius (km)           : {:3.2f}\n".format(self.far_radius))
        output += "\n"
        output += "# Optional second point      \n"
        output += ("EventLat2 (deg +N)       : {:3.2f}\n".format(self.lat2))
        output += ("EventLat2Std (deg)       : {:3.2f}\n".format(self.lat2_std))
        output += ("EventLon2 (deg +E)       : {:3.2f}\n".format(self.lon2))
        output += ("EventLon2Std (deg)       : {:3.2f}\n".format(self.lon2_std))
        output += ("EventHt2 (km)            : {:3.2f}\n".format(self.ht2))
        output += ("EventHtStd2 (km)         : {:3.2f}\n".format(self.ht2_std))
        output += "\n"
        output += "# Or a trajectory instead    \n"
        output += ("EventAzim (deg +E of N)  : {:3.2f}\n".format(self.azim))
        output += ("EventAzimStd (deg)       : {:3.2f}\n".format(self.azim_std))
        output += ("EventElev (deg)          : {:3.2f}\n".format(self.elev))
        output += ("EventElevStd (deg):      : {:3.2f}\n".format(self.elev_std))
        output += ("EventElevIsMax           : {:3.2f}\n".format(self.elev_is_max))
        output += "\n"
        output += "# Control information        \n"
        output += ("StationsRequired         : {}\n".format(self.stations_required))
        output += ("uuid                     : {}\n".format(self.uuid))
        output += ("RespondTo                : {}\n".format(self.respond_to))

        output += "# Trajectory information     \n"
        output += ("Start Distance (km)      : {:3.2f}\n".format(self.start_distance / 1000))
        output += ("Start Angle              : {:3.2f}\n".format(self.start_angle))
        output += ("End Distance (km)        : {:3.2f}\n".format(self.end_distance / 1000))
        output += ("End Angle                : {:3.2f}\n".format(self.end_angle))
        output += "# Station information        \n"
        output += ("Field of view RA         : {:3.2f}\n".format(self.fovra))
        output += ("Field of view Dec        : {:3.2f}\n".format(self.fovdec))

        output += "\n"
        output += "END"
        output += "\n"

        return output

    def checkReasonable(self):

        """ Receive an event, check if it is reasonable, and optionally try to fix it up
            Crucially, this function prevents any excessive requests being made that may compromise capture

        Arguments:


        Return:
            reasonable: [bool] The event is reasonable
        """


        reasonable = True

        reasonable = False if self.lat == "" else reasonable
        reasonable = False if self.lat is None else reasonable
        reasonable = False if self.lon == "" else reasonable
        reasonable = False if self.lon is None else reasonable
        reasonable = False if float(self.time_tolerance) > 300 else reasonable
        reasonable = False if self.close_radius > self.far_radius else reasonable

        return reasonable



    def transformToLatLon(self):

        """Take an event, establish how it has been defined, and convert to representation as
        a pair of Lat,Lon,Ht parameters.

        """

        # Work out if this is defined by point and azimuth and elevation
        azim_elev_definition = True
        azim_elev_definition = False if self.lon2 != 0 else azim_elev_definition
        azim_elev_definition = False if self.lat2 != 0 else azim_elev_definition
        azim_elev_definition = False if self.ht2 != 0 else azim_elev_definition

        if not azim_elev_definition:
            return

        # Copy observed lat, lon and height local variables for ease of comprehension and convert to meters
        obsvd_lat, obsvd_lon, obsvd_ht = self.lat, self.lon, self.ht * 1000

        # Set minimum and maximum luminous flight heights
        min_lum_flt_ht, max_lum_flt_ht = 20000, 100000

        # Elevation is always relative to intersection of trajectory with a horizontal plane below
        # Therefore elevation is always positive

        # For this routine elevation must always be within 10 - 90 degrees
        min_elev_hard, min_elev, prob_elev, max_elev = 0, 10, 45, 90

        # Detect, fix and log elevations outside range
        if min_elev < self.elev < max_elev:
            pass
        else:
            log.info("Elevation {} is not within range of {} - {} degrees.".format(self.elev, min_elev, max_elev))

            # If elevation is not within min_elev_hard and max_elev degrees set to prob_elev
            self.elev = self.elev if min_elev_hard < self.elev < max_elev else prob_elev

            # If elevation is min_elev_hard - min_elev degrees set to min_elev
            self.elev = min_elev if min_elev_hard < self.elev < min_elev else self.elev
            log.info("Elevation set to {} degrees.".format(self.elev))


        # Handle estimated start heights are outside normal range of luminous flight
        # Need to add gap so that angles can be calculated for consistency checks
        gap = 1000
        max_lum_flt_ht = obsvd_ht + gap if obsvd_ht >= (max_lum_flt_ht - gap) else max_lum_flt_ht
        min_lum_flt_ht = obsvd_ht - gap if obsvd_ht <= (min_lum_flt_ht + gap) else min_lum_flt_ht



        # Find range to maximum heights in reverse trajectory direction
        bwd_range = AEH2Range(self.azim, self.elev, max_lum_flt_ht, obsvd_lat, obsvd_lon, obsvd_ht)

        # Find range to minimum height in forward trajectory direction.
        # This is done by reflecting the trajectory in a horizontal plane midway between obsvd_ht and min_lum_flt_ht
        # This simplifies the calculation, but introduces a small imprecision
        fwd_range = AEH2Range(self.azim, self.elev, obsvd_ht, obsvd_lat, obsvd_lon, min_lum_flt_ht)

        # Iterate to find accurate solution - limit iterations to 100
        for n in range(100):
         self.lat2, self.lon2, ht2_m = AER2LatLonAlt(self.azim, 0 - self.elev, fwd_range, obsvd_lat, obsvd_lon, obsvd_ht)
         error =  (ht2_m - min_lum_flt_ht) / max_lum_flt_ht  # use max to avoid any div zero errors
         fwd_range = fwd_range + fwd_range * error * 0.1
         if error < 0.000005:
             break

        # Backwards azimuth
        azim_rev = self.azim + 180 if self.azim < 180 else self.azim - 180

        # Move event start point back to intersection with max_lum_flt_ht
        self.lat, self.lon, ht_m =  AER2LatLonAlt(azim_rev, self.elev, bwd_range,obsvd_lat, obsvd_lon,obsvd_ht)
        self.ht = ht_m / 1000

        # Calculate end point of trajectory and convert to km
        self.lat2, self.lon2, ht2_m = AER2LatLonAlt(self.azim, 0 - self.elev, fwd_range, obsvd_lat, obsvd_lon,obsvd_ht)
        self.ht2 = ht2_m / 1000

        # Post calculation checks - not required for operation

        # Convert to ECEF
        x1, y1, z1 = latLonAlt2ECEF(np.radians(self.lat), np.radians(self.lon), self.ht * 1000)
        x2, y2, z2 = latLonAlt2ECEF(np.radians(self.lat2), np.radians(self.lon2), self.ht2 * 1000)
        x_obs, y_obs, z_obs = latLonAlt2ECEF(np.radians(obsvd_lat), np.radians(obsvd_lon), obsvd_ht)

        # Calculate vectors of three points on trajectory
        maximum_point = np.array([x1, y1, z1])
        minimum_point = np.array([x2, y2, z2])
        observed_point = np.array([x_obs, y_obs, z_obs])

        # Calculate Alt Az between three points
        min_obs_az, min_obs_el = ECEF2AltAz(observed_point, minimum_point)
        min_max_az, min_max_el = ECEF2AltAz(maximum_point, minimum_point)
        obs_max_az, obs_max_el = ECEF2AltAz(maximum_point, observed_point)

        if angdf(min_obs_az,min_max_az) > 1 or angdf(min_obs_az,obs_max_az) > 1 or \
                               angdf(min_obs_el,min_max_el) > 1 or angdf(min_obs_el,obs_max_el) > 1:
            print("Observation at lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(obsvd_lat, obsvd_lon, obsvd_ht))
            print("Propagate fwds, bwds {:.0f},{:.0f} metres".format(fwd_range, bwd_range))
            print("At az, az_rev, el {:.4f} ,{:.4f} , {:.4f}".format(self.azim, azim_rev, self.elev))
            print("Start lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat, self.lon, self.ht * 1000))
            print("End   lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat2, self.lon2, self.ht2 * 1000))
            print("Minimum height to Observed height az,el {:.4f},{:.4f}".format(min_obs_az, min_obs_el))
            print("Minimum height to Maximum height az,el {:.4f},{:.4f}".format(min_max_az, min_max_el))
            print("Observed height to Maximum height az,el {:4f},{:.4f}".format(obs_max_az, obs_max_el))

        # Log any errors
        # Check that az from the minimum to the observation height as the same as the minimum to the maximum height
        # And the minimum to the observation height is the same as the observation to the maximum height
        if angdf(min_obs_az,min_max_az) > 1 or angdf(min_obs_az,obs_max_az) > 1:
            log.info("Error in Azimuth calculations")
            log.info("Observation at lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(obsvd_lat,obsvd_lon,obsvd_ht))
            log.info("Propagate fwds, bwds {:.0f},{:.0f} metres".format(fwd_range, bwd_range))
            log.info("At az, az_rev, el {:.4f} ,{:.4f} , {:.4f}".format(self.azim, azim_rev, self.elev))
            log.info("Start lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat, self.lon, self.ht * 1000))
            log.info("End   lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat2, self.lon2, self.ht2 * 1000))
            log.info("Minimum height to Observed height az,el {},{}".format(min_obs_az, min_obs_el))
            log.info("Minimum height to Maximum height az,el {},{}".format(min_max_az, min_max_el))
            log.info("Observed height to Maximum height az,el {},{}".format(obs_max_az, obs_max_el))

        # Check that el from the minimum to the observation height as the same as the minimum to the maximum height
        # And the minimum to the observation height is the same as the observation to the maximum height
        if angdf(min_obs_el,min_max_el) > 1 or angdf(min_obs_el,obs_max_el) > 1:
            log.info("Error in Elevation calculations")
            log.info("Trajectory created from observation at lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(obsvd_lat,obsvd_lon,obsvd_ht))
            log.info("Propagate fwds, bwds {:.0f},{:.0f} metres".format(fwd_range, bwd_range))
            log.info("At az, az_rev, el {:.4f} ,{:.4f} , {:.4f}".format(self.azim,azim_rev, self.elev))
            log.info("Start lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat, self.lon,self.ht * 1000))
            log.info("End   lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat2, self.lon2,self.ht2 * 1000))
            log.info("Minimum height to Observed height az,el {},{}".format(min_obs_az, min_obs_el))
            log.info("Minimum height to Maximum height az,el {},{}".format(min_max_az, min_max_el))
            log.info("Observed height to Maximum height az,el {},{}".format(obs_max_az, obs_max_el))

        # End of post calculation checks

class EventMonitor(multiprocessing.Process):

    def __init__(self, config):
        """ Automatically uploads data files of an event (e.g. fireball) as given on the website.
        Arguments:
            config: [Config] Configuration object.
        """



        super(EventMonitor, self).__init__()
        # Hold two configs - one for the locations of folders - syscon, and one for the lats and lons etc. - config
        self.config = config        #the config that will be used for all data processing - lats, lons etc.
        self.syscon = config        #the config that describes where the folders are
        # The path to the event monitor database
        self.event_monitor_db_path = os.path.join(os.path.abspath(self.config.data_dir),
                                                  self.config.event_monitor_db_name)
        self.conn = self.createEventMonitorDB()

        # Load the event monitor database. Any problems, delete and recreate.
        self.db_conn = self.getConnectionToEventMonitorDB()
        self.exit = multiprocessing.Event()
        self.event_monitor_db_name = "event_monitor.db"

        self.check_interval = self.syscon.event_monitor_check_interval


        log.info("Started EventMonitor")
        log.info("Monitoring {} at {:3.2f} minute intervals".format(self.syscon.event_monitor_webpage,self.syscon.event_monitor_check_interval))
        log.info("Local db path name {}".format(self.syscon.event_monitor_db_name))
        log.info("Reporting data to {}/{}".format(self.syscon.hostname, self.syscon.event_monitor_remote_dir))


    def createEventMonitorDB(self, test_mode = False):
        """ Creates the event monitor database. """

        # Create the event monitor database
        if test_mode:
            self.event_monitor_db_path = os.path.expanduser(os.path.join(self.syscon.data_dir, self.event_monitor_db_name))
            if os.path.exists(self.event_monitor_db_path):
                os.unlink(self.event_monitor_db_path)

        if not os.path.exists(os.path.dirname(self.event_monitor_db_path)):
            # Handle the very rare case where this could run before any observation sessions
            # and RMS_data does not exist
            os.makedirs(os.path.dirname(self.event_monitor_db_path))

        conn = sqlite3.connect(self.event_monitor_db_path)
        log.info("Created a database at {}".format(self.event_monitor_db_path))

        # Returns true if the table event_monitor exists in the database
        tables = conn.cursor().execute(
            """SELECT name FROM sqlite_master WHERE type = 'table' and name = 'event_monitor';""").fetchall()

        if tables:
            return conn

        conn.execute("""CREATE TABLE event_monitor (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,   
                            EventTime TEXT NOT NULL,
                            TimeTolerance REAL NOT NULL,
                            EventLat REAL NOT NULL,
                            EventLatStd REAL NOT NULL,
                            EventLon REAL NOT NULL,
                            EventLonStd REAL NOT NULL,
                            EventHt REAL NOT NULL,
                            EventHtStd REAL NOT NULL,
                            CloseRadius REAL NOT NULL,
                            FarRadius REAL NOT NULL,
                            EventLat2 REAL NOT NULL,
                            EventLat2Std REAL NOT NULL,
                            EventLon2 REAL NOT NULL,
                            EventLon2Std REAL NOT NULL,
                            EventHt2 REAL NOT NULL,
                            EventHt2Std REAL NOT NULL,
                            EventAzim REAL NOT NULL,
                            EventAzimStd REAL NOT NULL,
                            EventElev REAL NOT NULL,
                            EventElevStd REAL NOT NULL,
                            EventElevIsMax BOOL,
                            filesuploaded TEXT,
                            timeadded TEXT,
                            timecompleted TEXT,
                            observedstatus BOOL,
                            processedstatus BOOL,
                            receivedbyserver BOOL,
                            uuid TEXT,              
                            RespondTo TEXT
                            )""")

        # Commit the changes
        conn.commit()

        # Close the connection
        self.db_conn = conn

    def delEventMonitorDB(self):

        """ Delete the event monitor database.


        Arguments:


        Return:
            Status: [bool] True if a db was found at that location, otherwise false

        """

        # This check is to prevent accidental deletion of the working directory

        if os.path.isfile(self.event_monitor_db_path):
            os.remove(self.event_monitor_db_path)
            return True
        else:
            return False

    def addDBcol(self, column, coltype):
        """ Add a new column to the database

        Arguments:
            column: [string] Name of column to add
            coltype: [string] type of columnd to add

        Return:
            Status: [bool] True if successful otherwise false

        """

        sql_command = ""
        sql_command += "ALTER TABLE event_monitor  \n"
        sql_command += "ADD {} {}; ".format(column, coltype)

        try:
            conn = sqlite3.connect(self.event_monitor_db_path)
            conn.execute(sql_command)
            conn.close()
            return True
        except:
            return False

    def deleteDBoldrecords(self):


        """

        Remove old record from the database, notional time of 14 days selected.
        The deletion is made on the criteria of when the record was added to the database, not the event date
        If the event is is still listed on the website, then it will be added, and uploaded.

        """

        sql_statement = ""
        sql_statement += "DELETE from event_monitor \n"
        sql_statement += "WHERE                     \n"
        sql_statement += "timeadded < date('now', '-1 day')"

        try:
         cursor = self.db_conn.cursor()
         cursor.execute(sql_statement)
         self.db_conn.commit()

        except:
         log.info("Database purge failed")
         self.delEventMonitorDB()
         self.createEventMonitorDB()
        return None


    def getConnectionToEventMonitorDB(self):
        """ Loads the event monitor database

            Arguments:


            Return:
                connection: [connection] A connection to the database

            """

        # Create the event monitor database if it does not exist
        if not os.path.isfile(self.event_monitor_db_path):
            self.createEventMonitorDB()

        # Load the event monitor database - only gets done here
        try:
         self.conn = sqlite3.connect(self.event_monitor_db_path)
        except:
         os.unlink(self.event_monitor_db_path)
         self.createEventMonitorDB()

        return self.conn

    def eventExists(self, event):

        sql_statement = ""
        sql_statement += "SELECT COUNT(*) FROM event_monitor \n"
        sql_statement += "WHERE \n"
        sql_statement += "EventTime = '{}'          AND \n".format(event.dt)
        sql_statement += "EventLat = '{}'               AND \n".format(event.lat)
        sql_statement += "EventLon = '{}'               AND \n".format(event.lon)
        sql_statement += "EventHt = '{}'                AND \n".format(event.ht)
        sql_statement += "EventLatStd = '{}'            AND \n".format(event.lat_std)
        sql_statement += "EventLonStd = '{}'            AND \n".format(event.lon_std)
        sql_statement += "EventHtStd = '{}'             AND \n".format(event.ht_std)
        sql_statement += "EventLat2 = '{}'              AND \n".format(event.lat2)
        sql_statement += "EventLon2 = '{}'              AND \n".format(event.lon2)
        sql_statement += "EventHt2 = '{}'               AND \n".format(event.ht2)
        sql_statement += "EventLat2Std = '{}'           AND \n".format(event.lat2_std)
        sql_statement += "EventLon2Std = '{}'           AND \n".format(event.lon2_std)
        sql_statement += "EventHt2Std = '{}'            AND \n".format(event.ht2_std)
        sql_statement += "FarRadius = '{}'              AND \n".format(event.far_radius)
        sql_statement += "CloseRadius = '{}'            AND \n".format(event.close_radius)
        sql_statement += "TimeTolerance = '{}'          AND \n".format(event.time_tolerance)
        sql_statement += "RespondTo = '{}'                  \n".format(event.respond_to)

        # does a similar event exist
        # query gets the number of rows matching, not the actual rows

        try:
         return (self.db_conn.cursor().execute(sql_statement).fetchone())[0] != 0
        except:
         log.info("Check for event exists failed")
         return False


    def delOldRecords(self):

        """

        Remove old record from the database, notional time of 14 days selected.
        The deletion is made on the criteria of when the record was added to the database, not the event date
        If the event is is still listed on the website, then it will be added, and uploaded.

        """



        sql_statement = ""
        sql_statement += "DELETE from event_monitor \n"
        sql_statement += "WHERE                     \n"
        sql_statement += "timeadded < date('now', '-14 day')"



        try:
            cursor = self.db_conn.cursor()
            cursor.execute(sql_statement)
            self.db_conn.commit()

        except:
            log.info("Database purge failed")
            self.delEventMonitorDB()
            self.createEventMonitorDB()
        return None

    def addEvent(self, event):

        """

        Checks to see if an event exists, if not then add to the database

            Arguments:
                event: [event] Event to be added to the database

            Return:
                added: [bool] True if added, else false

            """

        self.delOldRecords()

        if not self.eventExists(event):
            sql_statement = ""
            sql_statement += "INSERT INTO event_monitor \n"
            sql_statement += "("
            sql_statement += "EventTime, TimeTolerance,                   \n"
            sql_statement += "EventLat ,EventLatStd ,EventLon , EventLonStd , EventHt ,EventHtStd,        \n"
            sql_statement += "CloseRadius, FarRadius,                     \n"
            sql_statement += "EventLat2, EventLat2Std, EventLon2, EventLon2Std,EventHt2, EventHt2Std,      \n"
            sql_statement += "EventAzim, EventAzimStd, EventElev, EventElevStd, EventElevIsMax,    \n"
            sql_statement += "processedstatus, uuid, RespondTo, timeadded \n"
            sql_statement += ")                                           \n"

            sql_statement += "VALUES "
            sql_statement += "(                            \n"
            sql_statement += "'{}',{},                     \n".format(event.dt, event.time_tolerance)
            sql_statement += "{},  {}, {}, {}, {}, {},     \n".format(event.lat, event.lat_std, event.lon, event.lon_std,
                                                                      event.ht, event.ht_std)
            sql_statement += "{},  {},                     \n".format(event.close_radius, event.far_radius)
            sql_statement += "{},  {}, {}, {}, {}, {},     \n".format(event.lat2, event.lat2_std, event.lon2,
                                                                      event.lon2_std, event.ht2, event.ht2_std)
            sql_statement += "{},  {}, {}, {}, {} ,        \n".format(event.azim, event.azim_std, event.elev,
                                                                      event.elev_std,
                                                                      event.elev_is_max)
            sql_statement += "{}, '{}', '{}',              \n".format(0, uuid.uuid4(), event.respond_to)
            sql_statement += "CURRENT_TIMESTAMP ) \n"

            try:
                cursor = self.db_conn.cursor()
                cursor.execute(sql_statement)
                self.db_conn.commit()

            except:
                log.info("Add event failed")

            log.info("Added event at {} to the database".format(event.dt))
            return True
        else:
            #log.info("Event at {} already in the database".format(event.dt))
            return False

    def markEventAsProcessed(self, event):

        """ Marks an event as having been processed

        Arguments:
            event: [event] Event to be marked as processed

        Return:
        """

        sql_statement = ""
        sql_statement += "UPDATE event_monitor                 \n"
        sql_statement += "SET                                  \n"
        sql_statement += "processedstatus = 1,                 \n"
        sql_statement += "timecompleted   = CURRENT_TIMESTAMP  \n".format(datetime.now())
        sql_statement += "                                     \n"
        sql_statement += "WHERE                                \n"
        sql_statement += "uuid = '{}'                          \n".format(event.uuid)
        try:
         self.db_conn.cursor().execute(sql_statement)
         self.db_conn.commit()
         log.info("Event at {} marked as processed".format(event.dt))
        except:
         log.info("Database error")


    def markEventAsUploaded(self, event, file_list):

        """ Checks to see if an event exists, if not then add to the database

            Arguments:
                event: [event] Event to be marked as uploaded
                file_list: [list of strings] Files uploaded

            Return:
        """

        files_uploaded = ""
        for file in file_list:
            files_uploaded += os.path.basename(file) + " "

        sql_statement = ""
        sql_statement += "UPDATE event_monitor \n"
        sql_statement += "SET                  \n"
        sql_statement += "filesuploaded  = '{}'\n".format(files_uploaded)
        sql_statement += "                     \n"
        sql_statement += "WHERE                \n"
        sql_statement += "uuid = '{}'          \n".format(event.uuid)

        try:
         cursor = self.db_conn.cursor()
         cursor.execute(sql_statement)
         self.db_conn.commit()
         log.info("Event at {} marked as uploaded".format(event.dt))
        except:
         log.info("Database error")

    def markEventAsReceivedByServer(self, uuid):

        """ Updates table when server publishes UUID of an event which has been sent
            This allows public acknowledgement of a stations transmission to be obfuscated

            Arguments:
                   uuid: [string] uuid of event received by server

            Return:
                   Nothing
        """

        sql_statement = ""
        sql_statement += "UPDATE event_monitor     \n"
        sql_statement += "SET                      \n"
        sql_statement += "receivedbyserver =   '{}'\n".format("1")
        sql_statement += "                         \n"
        sql_statement += "WHERE                    \n"
        sql_statement += "uuid = '{}'              \n".format(uuid)

        cursor = self.db_conn.cursor()
        cursor.execute(sql_statement)
        self.db_conn.commit()

    def getEventsfromWebPage(self, testmode=False):

        """ Reads a webpage, and generates a list of events

            Arguments:

            Return:
                events : [list of events]
        """

        event = EventContainer(0, 0, 0, 0)  # Initialise it empty
        events = []

        if not testmode:
            try:
                web_page = urllib.request.urlopen(self.syscon.event_monitor_webpage).read().decode("utf-8").splitlines()

            except:
                # Return an empty list
                log.info("Event monitor found no page at {}".format(self.syscon.event_monitor_webpage))
                return events
        else:
            f = open(os.path.expanduser("~/RMS_data/event_watchlist.txt"), "r")
            web_page = f.read().splitlines()
            f.close()

        for line in web_page:

            line = line.split('#')[0]  # remove anything to the right of comments

            if ":" in line:  # then it is a value pair

                try:
                    variable_name = line.split(":")[0].strip()  # get variable name
                    value = line.split(":")[1].strip()  # get value
                    event.setValue(variable_name, value)  # and put into this event container
                except:
                    pass

            else:
                if "END" in line:
                    events.append(copy.copy(event))  # copy, because these are references, not objects
                    event = EventContainer(0, 0, 0, 0)  # Initialise it empty
        #log.info("Read {} events from {}".format(len(events), self.syscon.event_monitor_webpage))

        return events

    def getUnprocessedEventsfromDB(self):

        """ Get the unprocessed events from the database

            Arguments:

            Return:
                events : [list of events]
        """

        sql_statement = ""
        sql_statement += "SELECT "
        sql_statement += ""
        sql_query_cols = ""
        sql_query_cols += "EventTime,TimeTolerance,EventLat,EventLatStd,EventLon, EventLonStd, EventHt, EventHtStd, "
        sql_query_cols += "FarRadius,CloseRadius, uuid,"
        sql_query_cols += "EventLat2, EventLat2Std, EventLon2, EventLon2Std,EventHt2, EventHt2Std, "
        sql_query_cols += "EventAzim, EventAzimStd, EventElev, EventElevStd, EventElevIsMax, RespondTo"
        sql_statement += sql_query_cols
        sql_statement += " \n"
        sql_statement += "FROM event_monitor "
        sql_statement += "WHERE processedstatus = 0"

        cursor = self.db_conn.cursor().execute(sql_statement)
        events = []

        # iterate through the rows, one row to an event

        for row in cursor:
            event = EventContainer(0, 0, 0, 0)
            col_count = 0
            cols_list = sql_query_cols.split(',')
            # iterate through the columns, one column to a value
            for col in row:
                event.setValue(cols_list[col_count].strip(), col)
                col_count += 1
                # this is the end of an event
            events.append(copy.copy(event))

        # iterated through all events
        return events

    def getFile(self, file_name, directory):

        """ Get the path to the file in the directory if it exists.
            If not, then return the path to ~/source/RMS


            Arguments:
                file_name: [string] name of file
                directory: [string] path to preferred directory

            Return:
                 file: [string] Path to platepar
        """

        file_list = []
        if os.path.isfile(os.path.join(directory, file_name)):
            file_list.append(str(os.path.join(directory, file_name)))
            return file_list
        else:

            if os.path.isfile(os.path.join(os.path.expanduser("~/source/RMS"), file_name)):
                file_list.append(str(os.path.join(os.path.expanduser("~/source/RMS"), file_name)))
                return file_list
        return []

    def getPlateparFilePath(self, event):

        """ Get the path to the best platepar from the directory matching the event time


            Arguments:
                event: [event]

            Return:
                file: [string] Path to platepar
        """

        file_list = []

        if len(self.getDirectoryList(event)) > 0:
            file_list += self.getFile("platepar_cmn2010.cal", self.getDirectoryList(event)[0])
        if len(file_list) != 0:
            return file_list[0]
        else:
            return False

    def getDirectoryList(self, event):

        """ Get the paths of directories which may contain files associated with an event

             Arguments:
                 event: [event]

             Return:
                 directorylist: [list of paths] List of directories
        """

        directory_list = []
        event_time = convertGMNTimeToPOSIX(event.dt)

        # iterate across the folders in CapturedFiles and convert the directory time to posix time
        if os.path.exists(os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir)):
            for night_directory in os.listdir(
                    os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir)):
                directory_POSIX_time = convertGMNTimeToPOSIX(night_directory[7:22])

                # if the POSIX time representation is before the event, and within 16 hours add to the list of directories
                # most unlikely that a single event could be split across two directories, unless there was large time uncertainty
                if directory_POSIX_time < event_time and (event_time - directory_POSIX_time).total_seconds() < 16 * 3600:
                    directory_list.append(
                        os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir,
                                     night_directory))
        return directory_list

    def findEventFiles(self, event, directory_list, file_extension_list):

        """Take an event, directory list and an extension list and return paths to files

           Arguments:
                event: [event] Event of interest
                directory_list: [list of paths] List of directories which may contain the files sought
                file_extension_list: [list of extensions] List of file extensions such as .fits, .bin

           Return:
                file_list: [list of paths] List of paths to files
        """
        try:
            event_time = parser.parse(event.dt)
        except:
            event_time = convertGMNTimeToPOSIX(event.dt)

        file_list = []
        # Iterate through the directory list, appending files with the correct extension
        for directory in directory_list:
            for file_extension in file_extension_list:
                for file in os.listdir(directory):
                    if file.endswith(file_extension):
                        file_POSIX_time = convertGMNTimeToPOSIX(file[10:25])
                        if abs((file_POSIX_time - event_time).total_seconds()) < event.time_tolerance:
                            file_list.append(os.path.join(directory, file))

        return file_list

    def getfilelist(self, event):

        """Take an event, return paths to files

           Arguments:
               event: [event] Event of interest


           Return:
               file_list: [list of paths] List of paths to files
        """

        file_list = []

        file_list += self.findEventFiles(event, self.getDirectoryList(event), [".fits", ".bin"])
        if len(self.getDirectoryList(event)) > 0:
            file_list += self.getFile(".config", self.getDirectoryList(event)[0])
            file_list += self.getFile("platepar_cmn2010.cal", self.getDirectoryList(event)[0])

        return file_list

    def calculateclosestpoint(self, beg_lat, beg_lon, beg_ele, end_lat, end_lon, end_ele, ref_lat, ref_lon, ref_ele):

        """
        Calculate the closest approach of a trajectory to a reference point

        refer to https://globalmeteornetwork.groups.io/g/main/topic/96374390#8250


        Args:
            beg_lat: [float] Starting latitude of the trajectory
            beg_lon: [float] Starting longitude of the trajectory
            beg_ele: [float] Beginning height of the trajectory
            end_lat: [float] Ending latitude of the trajectory
            end_lon: [float] Ending longitude of the trajectory
            end_ele: [float] Ending height of the trajectory
            ref_lat: [float] Station latitude
            ref_lon: [float] Station longitude
            ref_ele: [float] Station height

        Returns:
            start_dis: Distance from station to start of trajectory
            end_dist: Distance from station to end of trajectory
            closest_dist: Distance at the closest point (possibly outside the start and end)

        """

        # Convert coordinates to ECEF
        beg_ecef = np.array(latLonAlt2ECEF(np.radians(beg_lat), np.radians(beg_lon), beg_ele))
        end_ecef = np.array(latLonAlt2ECEF(np.radians(end_lat), np.radians(end_lon), end_ele))
        ref_ecef = np.array(latLonAlt2ECEF(np.radians(ref_lat), np.radians(ref_lon), ref_ele))

        traj_vec = vectNorm(end_ecef - beg_ecef)
        start_vec, end_vec = (ref_ecef - beg_ecef), (ref_ecef - end_ecef)
        start_dist, end_dist = (np.sqrt((np.sum(start_vec ** 2)))), (np.sqrt((np.sum(end_vec ** 2))))

        # Consider whether vector is zero length by looking at start and end
        if [beg_lat, beg_lon, beg_ele] != [end_lat, end_lon, end_ele]:
         # Vector start and end points are different, so possible to
         # calculate the projection of the reference vector onto the trajectory vector
         proj_vec = beg_ecef + np.dot(start_vec, traj_vec) * traj_vec

         # Hence, calculate the vector at the nearest point, and the closest distance
         closest_vec = ref_ecef - proj_vec
         closest_dist = (np.sqrt(np.sum(closest_vec ** 2)))

        else:

         # Vector has zero length, do not try to calculate projection
         closest_dist = start_dist

        return start_dist, end_dist, closest_dist

    def trajectoryVisible(self, rp, event):

        """
        Given a platepar and an event, calculate the centiles of the trajectory which would be in the FoV.
        Working is in ECI, relative to the station coordinates.

        Args:
            rp: [platepar] reference platepar
            event: [event] event of interest

        Returns:
            points_in_fov: [integer] the number of points out of 100 in the field of view
            start_distance: [float] the distance in metres from the station to the trajectory start
            start_angle: [float] the angle between the vector from the station to start of the trajectory
                        and the vector of the centre of the FOV
            end_distance: [float] the distance in metres from the station to the trajectort end
            end_angle: [float] the angle between the vector from the station to end of the trajectory
                        and the vector of the centre of the FOV
            fov_ra: [float]  field of view Ra (degrees)
            fov_dec: [float] fov_dec of view Dec (degrees)

        """
        # Calculate diagonal FoV of camera
        diagonal_fov = np.sqrt(rp.fov_v ** 2 + rp.fov_h ** 2)

        # Calculation origin will be the ECI of the station taken from the platepar
        jul_date = datetime2JD(convertGMNTimeToPOSIX(event.dt))
        origin = np.array(geo2Cartesian(rp.lat, rp.lon, rp.elev, jul_date))

        # Convert trajectory start and end point coordinates to cartesian ECI at JD of event
        traj_sta_pt = np.array(geo2Cartesian(event.lat, event.lon, event.ht * 1000, jul_date))
        traj_end_pt = np.array(geo2Cartesian(event.lat2, event.lon2, event.ht2 * 1000, jul_date))

        # Make relative (_rel) to station coordinates
        stapt_rel, endpt_rel = traj_sta_pt - origin, traj_end_pt - origin

        # trajectory vector, and vector for traverse
        traj_vec = traj_end_pt - traj_sta_pt
        traj_inc = traj_vec / 100

        # the az_centre, alt_centre of the camera
        az_centre, alt_centre = platepar2AltAz(rp)

        # calculate Field of View RA and Dec at event time, and
        fov_ra, fov_dec = altAz2RADec(az_centre, alt_centre, jul_date, rp.lat, rp.lon)

        fov_vec = np.array(raDec2ECI(fov_ra, fov_dec))

        # iterate along the trajectory counting points in the field of view
        points_in_fov = 0
        for i in range(0, 100):
            point = (stapt_rel + i * traj_inc)
            point_fov = np.degrees(angularSeparationVect(vectNorm(point), vectNorm(fov_vec)))
            if point_fov < diagonal_fov / 2:
                points_in_fov += 1

        # calculate some additional information for confidence
        start_distance = (np.sqrt(np.sum(stapt_rel ** 2)))
        start_angle = math.degrees(angularSeparationVect(vectNorm(stapt_rel), vectNorm(fov_vec)))
        end_distance = (np.sqrt(np.sum(endpt_rel ** 2)))
        end_angle = math.degrees(angularSeparationVect(vectNorm(endpt_rel), vectNorm(fov_vec)))

        return points_in_fov, start_distance, start_angle, end_distance, end_angle, fov_ra, fov_dec

    def trajectoryThroughFOV(self, event):

        """
        For the trajectory contained in the event, calculate if it passed through the FoV defined by the
        of the time of the event

        Args:
            event: [event] Calculate if the trajectory of this event passed through the field of view

        Returns:
            pts_in_FOV: [integer] Number of points of the trajectory split into 100 parts
                                   apparently in the FOV of the camera
            sta_dist: [float] Distance from station to the start of the trajectory
            sta_ang: [float] Angle from the centre of the FoV to the start of the trajectory
            end_dist: [float] Distance from station to the end of the trajectory
            end_ang: [float] Angle from the centre of the FoV to the end of the trajectory
        """

        # Read in the platepar for the event
        rp = Platepar()
        if not rp.read(self.getPlateparFilePath(event)):
            rp.read(os.path.abspath('.'))

        pts_in_FOV, sta_dist, sta_ang, end_dist, end_ang, fov_RA, fov_DEC = self.trajectoryVisible(rp, event)
        return pts_in_FOV, sta_dist, sta_ang, end_dist, end_ang, fov_RA, fov_DEC

    def doUpload(self, event, evcon, file_list, keep_files = False, no_upload = False, test_mode = False):

        """Move all the files to a single directory. Make MP4s, stacks and jpgs
           Archive into a bz2 file and upload, using rsync. Delete all working folders.

        Args:
            event: [event] the event to be uploaded
            evcon: [path] path to the config file for the event
            file_list: [list of paths] the files to be uploaded
            keep_files: [bool] keep the files after upload
            no_upload: [bool] if True do everything apart from uploading
            test_mode: [bool] if True prevents upload

        Returns:

        """

        event_monitor_directory = os.path.expanduser(os.path.join(self.syscon.data_dir, "EventMonitor"))
        upload_filename = "{}_{}".format(evcon.stationID, event.dt)
        this_event_directory = os.path.join(event_monitor_directory, upload_filename)

        # get rid of the eventdirectory, should never be needed
        if not keep_files:
            if os.path.exists(this_event_directory) and event_monitor_directory != "" and upload_filename != "":
                shutil.rmtree(this_event_directory)

        # create a new event directory
        if not os.path.exists(this_event_directory):
            os.makedirs(this_event_directory)

        # put all the files from the filelist into the event directory
        for file in file_list:
            shutil.copy(file, this_event_directory)

        # make a stack
        stackFFs(this_event_directory, "jpg", captured_stack=True)

        if True:
            batchFFtoImage(os.path.join(event_monitor_directory, upload_filename), "jpg", add_timestamp=True,
                           ff_component='maxpixel')

        with open(os.path.join(event_monitor_directory, upload_filename, "event_report.txt"), "w") as info:
            info.write(event.eventToString())

        # remove any leftover .bz2 files
        if not keep_files:
            if os.path.isfile(os.path.join(event_monitor_directory, "{}.tar.bz2".format(upload_filename))):
                os.remove(os.path.join(event_monitor_directory, "{}.tar.bz2".format(upload_filename)))

        if not test_mode:
            if os.path.isdir(event_monitor_directory) and upload_filename != "":
             log.info("Making archive of {}".format(os.path.join(event_monitor_directory, upload_filename)))
             base_name = os.path.join(event_monitor_directory,upload_filename)
             root_dir = os.path.join(event_monitor_directory,upload_filename)
             base_dir = os.path.join(event_monitor_directory,upload_filename)
             log.info("Base name : {}".format(upload_filename))
             log.info("Root dir  : {}".format(root_dir))
             log.info("Base dir  : {}".format(base_dir))
             archive_name = shutil.make_archive(base_name, 'bztar', root_dir, base_dir)
            else:
             log.info("Not making an archive of {}, not sensible.".format(os.path.join(event_monitor_directory, upload_filename)))

        # Remove the directory where the files were assembled
        if not keep_files:
            if os.path.exists(this_event_directory) and this_event_directory != "":
                shutil.rmtree(this_event_directory)

        if not no_upload and not test_mode:
         archives = glob(os.path.join(event_monitor_directory,"*.bz2"))
         upload_status = uploadSFTP(self.syscon.hostname, self.syscon.stationID.lower(),event_monitor_directory,self.syscon.event_monitor_remote_dir,archives,rsa_private_key=self.config.rsa_private_key)
         # set to the fast check rate after an upload
         self.check_interval = self.syscon.event_monitor_check_interval_fast
         log.info("Now checking at {:2.2f} minute intervals".format(self.check_interval))
         pass
        else:
         upload_status = False


        # Remove the directory
        if not keep_files and upload_status:
            shutil.rmtree(event_monitor_directory)

    def checkEvents(self, ev_con, test_mode = False):

        """
        Args:
            ev_con: configuration object at the time of this event

        Returns:
            Nothing
        """

        # Get the work to be done
        unprocessed = self.getUnprocessedEventsfromDB()

        # Iterate through the work
        for event in unprocessed:

            # Events can be specified in different ways, make sure converted to LatLon
            event.transformToLatLon()
            # Get the files
            file_list = self.getfilelist(event)

            # If there are no files, then mark as processed and continue
            if (len(file_list) == 0 or file_list == [None]) and not test_mode:
                log.info("No files for event at {}".format(event.dt))
                self.markEventAsProcessed(event)
                continue

            # If there is a .config file then parse it as evcon - not the station config
            for file in file_list:
                if file.endswith(".config"):
                    ev_con = cr.parse(file)

            # From the infinitely extended trajectory, work out the closest point to the camera
            start_dist, end_dist, atmos_dist = self.calculateclosestpoint(event.lat, event.lon, event.ht * 1000,
                                                                          event.lat2,
                                                                          event.lon2, event.ht2 * 1000, ev_con.latitude,
                                                                          ev_con.longitude, ev_con.elevation)
            min_dist = min([start_dist, end_dist, atmos_dist])

            # If trajectory outside the farradius, do nothing, and mark as processed
            if min_dist > event.far_radius * 1000 and not test_mode:
                log.info("Event at {} was {:4.1f}km away, outside {:4.1f}km, so was ignored".format(event.dt, min_dist / 1000, event.far_radius))
                self.markEventAsProcessed(event)
                # Do no more work
                continue


            # If trajectory inside the closeradius, then do the upload and mark as processed
            if min_dist < event.close_radius * 1000 and not test_mode:
                # this is just for info
                log.info("Event at {} was {:4.1f}km away, inside {:4.1f}km so is uploaded with no further checks.".format(event.dt, min_dist / 1000, event.close_radius))
                count, event.start_distance, event.start_angle, event.end_distance, event.end_angle, event.fovra, event.fovdec = self.trajectoryThroughFOV(
                    event)
                self.doUpload(event, ev_con, file_list, test_mode)
                self.markEventAsProcessed(event)
                if len(file_list) > 0:
                    self.markEventAsUploaded(event, file_list)
                # Do no more work
                continue


            # If trajectory inside the farradius, then check if the trajectory went through the FoV
            # The returned count is the number of 100th parts of the trajectory observed through the FoV
            if min_dist < event.far_radius * 1000 or test_mode:
                log.info("Event at {} was {:4.1f}km away, inside {:4.1f}km, consider FOV.".format(event.dt, min_dist / 1000, event.far_radius))
                count, event.start_distance, event.start_angle, event.end_distance, event.end_angle, event.fovra, event.fovdec = self.trajectoryThroughFOV(event)
                if count != 0:
                    log.info("Event at {} had {} points out of 100 in the trajectory in the FOV. Uploading.".format(event.dt, count))
                    self.doUpload(event, ev_con, file_list, test_mode=test_mode)
                    self.markEventAsUploaded(event, file_list)
                    if test_mode:
                        rp = Platepar()
                        rp.read(self.getPlateparFilePath(event))
                        with open(os.path.expanduser(os.path.join(self.syscon.data_dir, "testlog")), 'at') as logfile:
                            logfile.write(
                                "{} LOC {} Az:{:3.1f} El:{:3.1f} sta_lat:{:3.4f} sta_lon:{:3.4f} sta_dist:{:3.0f} end_dist:{:3.0f} fov_h:{:3.1f} fov_v:{:3.1f} sa:{:3.1f} ea::{:3.1f} \n".format(
                                    convertGMNTimeToPOSIX(event.dt), ev_con.stationID, rp.az_centre, rp.alt_centre,
                                    rp.lat, rp.lon, event.start_distance / 1000, event.end_distance / 1000, rp.fov_h,
                                    rp.fov_v, event.start_angle, event.end_angle))
                else:
                    log.info("Event at {} did not pass through FOV.".format(event.dt))
                if not test_mode:
                    self.markEventAsProcessed(event)
                # Do no more work
                continue
        return None

    def start(self):
        """ Starts the event monitor. """

        super(EventMonitor, self).start()
        log.info("EventMonitor was started")

    def stop(self):
        """ Stops the event monitor. """

        self.db_conn.close()
        time.sleep(2)
        self.exit.set()
        self.join()
        log.info("EventMonitor has stopped")

    def getEventsAndCheck(self, testmode=False):
        """
        Gets event(s) from the webpage, or a local file.
        Calls self.addevent to add them to the database
        Calls self.checkevents to see if the database holds any unprocessed events

        Args:
            testmode: [bool] if set true looks for a local file, rather than a web address

        Returns:
            Nothing
        """

        events = self.getEventsfromWebPage(testmode)
        for event in events:
            if event.checkReasonable():
                self.addEvent(event)

        # Go through all events and check if they need to be uploaded - this iterates through the database
        self.checkEvents(self.config, test_mode=testmode)

    def run(self):


        # Delay to get everything else done first
        time.sleep(20)
        while not self.exit.is_set():

            self.getEventsAndCheck()
            # Wait for the next check
            self.exit.wait(60 * self.check_interval)
            #We are running fast, but have not made an upload, then check more slowly next time
            if self.check_interval < self.syscon.event_monitor_check_interval:
                self.check_interval = self.check_interval * 1.1
                log.info("Check interval now set to {:2.2f} minutes".format(self.check_interval))


def angdf(a1,a2):
    normalised = a1-a2 % 360
    return min(360-normalised, normalised)

def convertGMNTimeToPOSIX(timestring):
    """
    Converts the filenaming time convention used by GMN into posix

    Args:
        timestring: [string] time represented as a string e.g. 20230527_032115

    Returns:
        posix compatible time
    """

    dt_object = datetime.strptime(timestring.strip(), "%Y%m%d_%H%M%S")
    return dt_object

#https://stackoverflow.com/questions/2030053/how-to-generate-random-strings-in-python
def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def platepar2AltAz(rp):

    """

    Args:
        rp: Platepar

    Returns:
        Ra_d : [degrees] Ra of the platepar at its creation date
        dec_d : [degrees] Dec of the platepar at its creation date
        JD : [float] JD of the platepar creation
        lat : [float] lat of the station
        lon : [float] lon of the station

    """

    RA_d = np.radians(rp.RA_d)
    dec_d = np.radians(rp.dec_d)
    JD = rp.JD
    lat = np.radians(rp.lat)
    lon = np.radians(rp.lon)

    return np.degrees(cyTrueRaDec2ApparentAltAz(RA_d, dec_d, JD, lat, lon))


def raDec2ECI(ra, dec):

    """

    Convert right ascension and declination to Earth-centered inertial vector.

    Arguments:
        ra: [float] right ascension in degrees
        dec: [float] declination in degrees

    Return:
        (x, y, z): [tuple of floats] Earth-centered inertial coordinates in metres

    """

    x = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z = np.sin(np.radians(dec))

    return x, y, z


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="""Check a web page for trajectories, and upload relevant data. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-o', '--oneshot', dest='one_shot', default=False, action="store_true",
                            help="Run once, and terminate.")

    arg_parser.add_argument('-d', '--deletedb', dest='delete_db', default=False, action="store_true",
                            help="Delete the event_monitor database at initialisation.")

    arg_parser.add_argument('-k', '--keepfiles', dest='keepfiles', default=False, action="store_true",
                            help="Keep working files")

    arg_parser.add_argument('-n', '--noupload', dest='noupload', default=False, action="store_true",
                            help="Do not upload")


    cml_args = arg_parser.parse_args()

    # Load the config file
    syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Set the web page to monitor


    try:
        # Add a random string after the URL to defeat caching
        print(syscon.event_monitor_webpage)
        web_page = urllib.request.urlopen(syscon.event_monitor_webpage + "?" + randomword(6)).read().decode("utf-8").splitlines()
    except:

        log.info("Nothing found at {}".format(syscon.event_monitor_webpage))


    if cml_args.delete_db and os.path.isfile(os.path.expanduser("~/RMS_data/event_monitor.db")):
        os.unlink(os.path.expanduser("~/RMS_data/event_monitor.db"))

    em = EventMonitor(syscon)



    if cml_args.one_shot:
        print("EventMonitor running once")
        em.getEventsAndCheck()

    else:
        print("EventMonitor running indefinitely")
        em.start()

