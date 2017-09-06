import rospy
from datetime import datetime, timedelta
import time


class Epoch:

    def __init__(self):
        i = 0
        while not rospy.has_param("/epoch") and i < 10 and not rospy.is_shutdown():
            rospy.logwarn("Epoch not avalailable yet. Waiting...")
            i = i+1
            time.sleep(1)

        self.epoch_string = rospy.get_param('/epoch')
        self.epoch_datetime = datetime.strptime(self.epoch_string, "%Y-%m-%d %H:%M:%S")

    def now(self):
        time_since_epoch = rospy.Time.now()
        time_delta = timedelta(0, time_since_epoch.secs, time_since_epoch.nsecs/1e3)
        return self.epoch_datetime + time_delta

