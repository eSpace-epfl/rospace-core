# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import rospy
import threading

from rospace_msgs.srv import SyncNodeService, ClockService


class SimTimeService(threading.Thread):
    """
    Class setting up and communicating with Clock service called by GUI.

    Can start/pause simulation and change simulation parameters
    like publishing frequency and step size.
    """

    def __init__(self,
                 realtime_factor,
                 frequency,
                 step_size,
                 start_running=False):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.start()
        self.SimRunning = start_running
        self.realtime_factor = realtime_factor
        self.frequency = frequency
        self.step_size = step_size

        self.syncSubscribers = 0
        self.readyCount = 0

    def handle_start_stop_clock(self, req):
        """
        Method called when service requested.

        Start/Pauses simulation or changes siulation parameter depending
        on input given through GUI

        Args:
            req: ClockService srv message

        Return:
            bool: SimulationRunning
            float: Step size
            float: Publish frequency .
        """

        if req.trigger:  # start/stop simulation
            if self.SimRunning:
                self.SimRunning = False
            else:
                self.SimRunning = True
        elif req.dt_size > 0 and req.p_freq > 0:
            self.frequency = req.p_freq
            self.step_size = req.dt_size
            self.realtime_factor = req.p_freq * req.dt_size

        return [self.SimRunning, self.step_size, self.frequency]

    def handle_sync_nodes(self, req):
        """
        Very basic Service to sync nodes.

        Every node has to subscribe and after each time step
        call service.

        Args:
            req: SyncNodeService srv message

        Returns:
            bool : True if adding node to subscribtion list,
                   False if node reports ready
        """

        if req.subscribe is True and req.ready is False:
            self.syncSubscribers += 1
            return True
        elif req.subscribe is False and req.ready is True:
            self.updateReadyCount(reset=False)
            return False

    def updateReadyCount(self, reset):
        """
        Method to count nodes which reported to be ready after one time step.

        Args:
            reset: if true resets ready count if false increases count by 1

        """

        self.lock.acquire()
        if reset:
            self.readyCount = 0
        else:
            self.readyCount += 1
        self.lock.release()

    def run(self):
        """
        Rospy service for synchronizing nodes and simulation time.
        Method is running/spinning on seperate thread.
        """
        rospy.Service('/sync_nodes',
                      SyncNodeService,
                      self.handle_sync_nodes)
        rospy.loginfo("Node-Synchronisation Service ready.")

        rospy.Service('/start_stop_clock',
                      ClockService,
                      self.handle_start_stop_clock)
        rospy.loginfo("Clock Service ready. Can start simulation now.")

        rospy.spin()  # wait for node to shutdown.
        self.root.quit()
