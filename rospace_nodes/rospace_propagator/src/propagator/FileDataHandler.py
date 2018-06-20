# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import orekit

from datetime import datetime, timedelta

from org.orekit.models.earth import GeoMagneticModelLoader, GeoMagneticField
from org.orekit.data import DataProvidersManager
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.forces.drag.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.forces.drag.atmosphere.data import CelesTrackWeather
from org.orekit.utils import IERSConventions as IERS


def _get_name_of_loaded_files(folder_name):
    '''
    Gets names of files in defined folder loaded by the data provider.

    Args:
        folder_name: string of folder name containing files

    Returns:
        List<String>: all file names loaded by data provider in folder
    '''
    file_names = []
    manager = DataProvidersManager.getInstance()
    string_set = manager.getLoadedDataNames()
    for i in string_set:
        if folder_name in i:
            file_names.append(i.rsplit('/', 1)[1])

    return file_names


def to_orekit_date(epoch):
    """
    Method to convert UTC simulation time from python's datetime object to
    orekits AbsoluteDate object

    Args:
        epoch: UTC simulation time as datetime object

    Returns:
        AbsoluteDate: simulation time in UTC
    """
    seconds = float(epoch.second) + float(epoch.microsecond) / 1e6
    orekit_date = AbsoluteDate(epoch.year,
                               epoch.month,
                               epoch.day,
                               epoch.hour,
                               epoch.minute,
                               seconds,
                               TimeScalesFactory.getUTC())

    return orekit_date


class FileDataHandler(object):
    _data_checklist = dict()
    """Holds dates for which data from orekit-data folder is loaded"""
    _mag_field_coll = None
    """Java Collection holding all loaded magnetic field models"""
    mag_field_model = None
    """Currently used magnetic field model, transformed to correct year"""

    @staticmethod
    def load_magnetic_field_models(epoch):
        curr_year = GeoMagneticField.getDecimalYear(epoch.day,
                                                    epoch.month,
                                                    epoch.year)
        gmLoader = GeoMagneticModelLoader()
        manager = DataProvidersManager.getInstance()
        loaded = manager.feed('(?:IGRF|igrf)\\p{Digit}\\p{Digit}\\.(?:cof|COF)', gmLoader)
        # igrf better, because also old data available..
        if not loaded:
            loaded = manager.feed('(?:WMM|wmm)\\p{Digit}\\p{Digit}\\.(?:cof|COF)', gmLoader)
            if loaded:
                mesg = "\033[93m  [WARN] [load_magnetic_field_models] Could " \
                   + "not load IGRF model. Using WMM instead.\033[0m"
                print mesg
            else:
                raise ValueError("No magnetic field model found!")

        # get correct model from Collection and transform to correct year
        GMM_coll = gmLoader.getModels()
        valid_mod = None
        for GMM in GMM_coll:
            if GMM.validTo() >= curr_year and GMM.validFrom() < curr_year:
                valid_mod = GMM
                break

        if valid_mod is None:
            mesg = "No magnetic field model found by data provider for year " \
                    + str(curr_year) + "."
            raise ValueError(mesg)

        FileDataHandler._mag_field_coll = GMM_coll
        if valid_mod.supportsTimeTransform():
            # only prediction model supports time transformation
            FileDataHandler.mag_field_model = valid_mod.transformModel(curr_year)
        else:
            FileDataHandler.mag_field_model = valid_mod

    @staticmethod
    def create_data_validity_checklist():
        """Get files loader by DataProvider and create dict() with valid dates for
        loaded data.

        Creates a list with valid start and ending date for data loaded by the
        DataProvider during building. The method looks for follwing folders
        holding the correspoding files:

            Earth-Orientation-Parameters: EOP files using IERS2010 convetions
            MSAFE: NASA Marshall Solar Activity Future Estimation files
            Magnetic-Field-Models: magentic field model data files

        The list should be used before every propagation step, to check if data
        is loaded/still exists for current simulation time. Otherwise
        simulation could return results with coarse accuracy. If e.g. no
        EOP data is available, null correction is used, which could worsen the
        propagators accuracy.
        """
        checklist = dict()
        start_dates = []
        end_dates = []

        EOP_file = _get_name_of_loaded_files('Earth-Orientation-Parameters')
        if EOP_file:
            EOPHist = FramesFactory.getEOPHistory(IERS.IERS_2010, False)
            EOP_start_date = EOPHist.getStartDate()
            EOP_end_date = EOPHist.getEndDate()
            checklist['EOP_dates'] = [EOP_start_date, EOP_end_date]
            start_dates.append(EOP_start_date)
            end_dates.append(EOP_end_date)

        CelesTrack_file = _get_name_of_loaded_files('CELESTRACK')
        if CelesTrack_file:
            ctw = CelesTrackWeather("(?:sw|SW)\\p{Digit}+\\.(?:txt|TXT)")
            ctw_start_date = ctw.getMinDate()
            ctw_end_date = ctw.getMaxDate()
            checklist['CTW_dates'] = [ctw_start_date, ctw_end_date]
            start_dates.append(ctw_start_date)
            end_dates.append(ctw_end_date)

        MSAFE_file = _get_name_of_loaded_files('MSAFE')
        if MSAFE_file:
            msafe = \
                MarshallSolarActivityFutureEstimation(
                 "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)" +
                 "\\p{Digit}\\p{Digit}\\p{Digit}\\p{Digit}F10\\" +
                 ".(?:txt|TXT)",
                 MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
            MS_start_date = msafe.getMinDate()
            MS_end_date = msafe.getMaxDate()
            checklist['MSAFE_dates'] = [MS_start_date, MS_end_date]
            start_dates.append(MS_start_date)
            end_dates.append(MS_end_date)

        if FileDataHandler._mag_field_coll:
            coll_iterator = FileDataHandler._mag_field_coll.iterator()
            first = coll_iterator.next()
            GM_start_date = first.validFrom()
            GM_end_date = first.validTo()
            for GM in coll_iterator:
                if GM_start_date > GM.validFrom():
                    GM_start_date = GM.validFrom()
                if GM_end_date < GM.validTo():
                    GM_end_date = GM.validTo()

            # convert to absolute date for later comparison
            dec_year = GM_start_date
            base = datetime(int(dec_year), 1, 1)
            dec = timedelta(seconds=(base.replace(year=base.year + 1) -
                            base).total_seconds() * (dec_year-int(dec_year)))
            GM_start_date = to_orekit_date(base + dec)
            dec_year = GM_end_date
            base = datetime(int(dec_year), 1, 1)
            dec = timedelta(seconds=(base.replace(year=base.year + 1) -
                            base).total_seconds() * (dec_year-int(dec_year)))
            GM_end_date = to_orekit_date(base + dec)
            checklist['MagField_dates'] = [GM_start_date, GM_end_date]
            start_dates.append(GM_start_date)
            end_dates.append(GM_end_date)

        if checklist:  # if any data loaded define first and last date
            # using as date zero 01.01.1850. Assuming no data before.
            absolute_zero = AbsoluteDate(1850, 1, 1, TimeScalesFactory.getUTC())
            first_date = min(start_dates, key=lambda p: p.durationFrom(absolute_zero))
            last_date = min(end_dates, key=lambda p: p.durationFrom(absolute_zero))

            checklist['MinMax_dates'] = [first_date, last_date]

        mesg = "[INFO]: Simulation can run between epochs: " + \
               str(first_date) + " & " + str(last_date) + \
               " (based on loaded files)."
        print mesg

        FileDataHandler._data_checklist = checklist

    @staticmethod
    def check_data_availability(epoch):
        """
        Checks if loaded files from orekit-data have data for current epoch.

        Also checks if loaded magnetic model is valid for current epoch and
        updates the model's coefficients to current epoch using secular
        variation coefficients.

        Args:
            oDate: AbsoluteDate object of current epoch

        Raises:
            ValueError: if no data loaded for current epoch
        """
        min_max = FileDataHandler._data_checklist['MinMax_dates']
        oDate = to_orekit_date(epoch)
        if oDate.compareTo(min_max[0]) < 0:
            err_msg = "No file loaded with valid data for current epoch " + \
                      str(oDate) + "! Earliest possible epoch: " + min_max[0]
            raise ValueError(err_msg)
        if oDate.compareTo(min_max[1]) > 0:
            err_msg = "No file loaded with valid data for current epoch " + \
                      str(oDate) + "! Latest possible epoch: " + min_max[1]
            raise ValueError(err_msg)

        d_year = GeoMagneticField.getDecimalYear(epoch.day,
                                                 epoch.month,
                                                 epoch.year)

        # transform model to current date if inside valid epoch. Else load new
        # model (should be available -> checking for out-of-bounds date above)
        mdl = FileDataHandler.mag_field_model
        if mdl.validFrom() < d_year and mdl.validTo() >= d_year:
            # still using correct model. Check if time transform necessary
            if mdl.supportsTimeTransform() and mdl.getEpoch != d_year:
                FileDataHandler.mag_field_model = mdl.transformModel(d_year)
        else:
            # need to load new 5-year model
            mdl_iterator = FileDataHandler._mag_field_coll.iterator()
            for GMM in mdl_iterator:
                if GMM.validTo() >= d_year and GMM.validFrom() < d_year:
                    break
            if mdl.supportsTimeTransform():
                FileDataHandler.mag_field_model = GMM.transformModel(d_year)
            else:
                FileDataHandler.mag_field_model = GMM
