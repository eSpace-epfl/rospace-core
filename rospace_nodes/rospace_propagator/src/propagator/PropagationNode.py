#!/usr/bin/env python

# Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

import rospy
import time
import threading

from rospace_lib.misc.PropagatorParser import parse_configuration_files
from rospace_lib.misc.Spacecrafts import Simulator_Spacecraft
from rospace_lib.misc.FileDataHandler import FileDataHandler
from rospace_lib.clock import SimTimePublisher

from OrekitPropagator import OrekitPropagator

from std_srvs.srv import Empty


class ExitServer(threading.Thread):
    """Server which shuts down node correctly when called.

    Rospy currently has a bug which doesn't shutdown the node correctly.
    This causes a problem when a profiler is used, as the results are not output
    if shut down is not performed in the right way.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.exiting = False
        self.start()

    def exit_node(self, req):
        self.exiting = True
        return []

    def run(self):
        rospy.Service('/exit_node', Empty, self.exit_node)
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('propagation_node', anonymous=True)

    ExitServer = ExitServer()
    SimTime = SimTimePublisher()
    SimTime.set_up_simulation_time()

    OrekitPropagator.init_jvm()

    # Initialize Data handlers, loading data in orekit.zip file
    FileDataHandler.load_magnetic_field_models(SimTime.datetime_oe_epoch)

    spacecrafts = []  # List of to be propagated spacecrafts
    sc_settings = rospy.get_param("scenario/init_coords")
    for ns_spacecraft, init_coords in sc_settings.items():
        # Parse settings for every spacecraft independently
        spc = parse_configuration_files(Simulator_Spacecraft(ns_spacecraft),
                                        init_coords,
                                        SimTime.datetime_oe_epoch)

        # Build propagator object from settings
        spc.build_propagator(SimTime.datetime_oe_epoch)

        # Set up publishers and subscribers
        spc.build_communication()

        spacecrafts.append(spc)

    FileDataHandler.create_data_validity_checklist()

    rospy.loginfo("Propagators initialized!")

    while not rospy.is_shutdown() and not ExitServer.exiting:
        comp_time = time.clock()

        epoch_now = SimTime.update_simulation_time()
        if SimTime.time_shift_passed:
            # check if data still loaded
            FileDataHandler.check_data_availability(epoch_now)

            for spc in spacecrafts:
                try:
                    spc.propagate(epoch_now)  # propagate to epoch_now
                except Exception as e:
                    print "ERROR in propagation of: [", spc.namespace, "]"
                    print e.message, e.args
                    print "Shutting down Propagator!"
                    ExitServer.exiting = True

            for spc in spacecrafts:
                # Publish messages
                spc.publish()

        # Maintain correct frequency
        SimTime.sleep_to_keep_frequency()
