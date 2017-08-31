#   Class to hold spherical earth coordinates
#   Author: Michael Pantic, michael.pantic@gmail.com
#   License: TBD

import numpy as np
import datetime

class SphericalEarth:

	def __init__(self):
		self.lon = 0	# longitude
		self.lat = 0	# latitude
		self.alt = 0	# altitude

	def toDMS(self):
		# Returns longitude/latitude in Degree - Minutes- Seconds
		lon_deg = np.rad2deg(self.lon)
		lat_deg = np.rad2deg(self.lat)

		lon_d = int(lon_deg)
		lon_md = abs(lon_deg-lon_d)*60
		lon_m = int(lon_md)
		lon_s = (lon_md-lon_m)*60

		lat_d = int(lat_deg)
		lat_md = abs(lat_deg-lat_d)*60
		lat_m = int(lat_md)
		lat_s = (lat_md-lat_m)*60

		return [[lon_d,lon_m,lon_s],[lat_d,lat_m,lat_s]]
