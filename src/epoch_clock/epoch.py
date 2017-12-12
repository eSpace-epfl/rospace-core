import rospy
from datetime import datetime, timedelta
import time


class Epoch:
    def __init__(self):
        i = 0
        while not rospy.has_param("/epoch") and i < 10 and not rospy.is_shutdown():
            rospy.logwarn("Epoch not avalailable yet. Waiting...")
            i = i + 1
            time.sleep(1)

        self.epoch_string = rospy.get_param('/epoch')
        self.epoch_datetime = datetime.strptime(self.epoch_string, "%Y-%m-%d %H:%M:%S")
        self.publish_frequency = rospy.get_param('/publish_freq')
        self.time_step_size = rospy.get_param('/time_step_size')

    def now(self):
        time_since_epoch = rospy.Time.now()
        time_delta = timedelta(0, time_since_epoch.secs, time_since_epoch.nsecs / 1e3)
        return self.epoch_datetime + time_delta

    def now_jd(self):
        now_utc = self.now()
        J2000 = 2451545.0
        J2000_date = datetime(2000, 1, 1, 12, 00, 00)  # UTC time of J2000
        delta = now_utc - J2000_date
        return J2000 + delta.total_seconds() / (60.0 * 60 * 24)

    def get_frequency(self):
        return self.publish_frequency

    def get_stepSize(self):
        return self.stepSize

    def changeRate(self, new_rate):
        # Only propagator node is allowed to change this! Do not change!
        self.publish_rate = new_rate

    def changeStep(self, new_step):
        # Only propagator node is allowed to change this! Do not change!
        self.time_step_size = new_step
