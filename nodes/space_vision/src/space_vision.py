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
import vision_algo.hist_eq as lib

#from __future__ import print_function

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image, queue_size=10)
    self.cube_pos_pub = rospy.Publisher("target_pos", String, queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    rospy.loginfo("received image with format {} x {} x {}".format(rows,cols,channels))

    #if cols > 200 and rows > 200 :
    #  cv2.circle(cv_image, (100,100), 50, 255)
    #  rospy.loginfo("drawing circle")  aa

    #call function for cube detection + analysis
    # d, azim, elev = analysis(cv_image)

    d, azim, elev = lib.img_analysis(cv_image)
    data = [d, azim, elev ]

    try:
      self.cube_pos_pub.publish("cube position : {} ".format(data))
      rospy.loginfo("publish cube position : {}".format(data))
    except e:
      print(e)
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(0)

    #try:
    #  self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    #except CvBridgeError as e:
    #  print(e)


def main(args):
  while not rospy.is_shutdown():
    ic = image_converter()

    rospy.init_node('space_vision', anonymous=True)

    pub = rospy.Publisher("image_topic", Image, queue_size=1)
    time.sleep(1)

    img = cv2.imread(args[1],1)
    #cv2.imshow("image",img)
    #cv2.waitKey(1000)
    img_msg = ic.bridge.cv2_to_imgmsg(img, encoding='bgr8' )
    pub.publish(img_msg)
    print("published on image_topic")
    #print(sys.path)

    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)