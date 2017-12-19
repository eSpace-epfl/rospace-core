#!/usr/bin/env python
import roslib
roslib.load_manifest('space_vision')
import sys, time
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import vision_algo.vision_lib as lib

#from __future__ import print_function

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image, queue_size=10)
    self.cube_pos_pub = rospy.Publisher("target_pos", String, queue_size=10)
    self.cube_quat_pub = rospy.Publisher("cube_quaternion", String, queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)

    self.last_cube_pos = (0.0, 0.0, 0.0)

  def callback(self,data):
    t_start = time.time()
    print("start timing")
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    rospy.loginfo("received image with format {} x {} x {}".format(rows,cols,channels))
    
    d, azim, elev, quat = lib.img_analysis(cv_image, self.last_cube_pos, debug=False)
    cube_pos = [d, azim, elev]

    self.last_cube_pos=cube_pos
    try:
      self.cube_pos_pub.publish("cube position : {} ".format(cube_pos))
      rospy.loginfo("publish cube position : {}".format(cube_pos))
      self.cube_quat_pub.publish("cube quaternion : {} ".format(quat))
      rospy.loginfo("publish cube quaternion : {}".format(quat))
    except e:
      print(e)

    t_end = time.time()
    try:
      rospy.loginfo("image processed in {} seconds".format(t_end-t_start))
    except(e):
      print(e)

def main(args):
  while not rospy.is_shutdown():
    ic = image_converter()

    rospy.init_node('space_vision', anonymous=True)

    pub = rospy.Publisher("image_topic", Image, queue_size=10)
    time.sleep(1)

    print(len(args))
    print(args)
    for i in range(1,len(args)):

      img = cv2.imread(args[i],1)

      img_msg = ic.bridge.cv2_to_imgmsg(img, encoding='bgr8')
      #for j in range(20):
      pub.publish(img_msg)
      print("published on image_topic")
      #time.sleep(1)

    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)