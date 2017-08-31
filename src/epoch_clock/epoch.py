import rospy
from jdcal import gcal2jd, jd2gcal
from datetime import datetime, timedelta

class Epoch:

    def __init__(self):
        self.epoch_string = rospy.get_param('/epoch')
        self.epoch_datetime = datetime.strptime(self.epoch_string, "%Y-%m-%d %H:%M:%S")

    def now(self):
        time_since_epoch = rospy.Time.now()
        time_delta = timedelta(0,time_since_epoch.secs,time_since_epoch.nsecs/1e3)
        return self.epoch_datetime + time_delta
