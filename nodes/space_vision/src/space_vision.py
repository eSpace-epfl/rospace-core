#!/usr/bin/env python
import roslib
roslib.load_manifest('space_vision')
import os, sys, time
import rospy
import cv2
import scipy.misc
import numpy as np
import argparse
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import vision_algo.vision_lib as lib


class image_converter:

  def __init__(self, path):
    self.image_pub = rospy.Publisher("image_topic_2",Image, queue_size=10)
    self.cube_pos_pub = rospy.Publisher("target_pos", String, queue_size=10)
    self.cube_quat_pub = rospy.Publisher("cube_quaternion", String, queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("camera/image_raw",Image,self.callback)

    self.last_cube_pos = (0, 0)

    if path is not None:
      self.save_file = os.path.join(path,'data.txt')

      if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path,'original'))
        os.makedirs(os.path.join(path,'processed'))

      self.image_path = path
      open(self.save_file,'w')
      with open(self.save_file, 'w') as save_file:
        save_file.write("Format : timestamp; range; azim; elev; quat\n")
    else:
      self.save_file = None
      self.image_path = None

  def callback(self,data):
    t_start = time.time()
    print("start timing")
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    rospy.loginfo("received image with format {} x {} x {}".format(rows,cols,channels))
    
    d, azim, elev, quat, cm_coo, processed_image,  cube_found = lib.img_analysis(cv_image, self.last_cube_pos, mode='test')

    if cube_found:
      cube_pos = [d, azim, elev]

      self.last_cube_pos=cm_coo

      self.cube_pos_pub.publish("cube position : {} ".format(cube_pos))
      rospy.loginfo("publish cube position : {}".format(cube_pos))
      self.cube_quat_pub.publish("cube quaternion : {} ".format(quat))
      rospy.loginfo("publish cube quaternion : {}".format(quat))

      if self.save_file is not None:
        current_time = time.time()
        with open(self.save_file, 'a') as save_file:
          save_file.write("{}; {} ; {} ; {} ; {}\n".format(current_time, d,azim,elev,quat))

        scipy.misc.imsave(os.path.join(self.image_path,'processed','img{}.png'.format(current_time)),processed_image)
        scipy.misc.imsave(os.path.join(self.image_path,'original','img{}.png'.format(current_time)),cv_image)



    else:
      rospy.loginfo("Cube not found on image")
      self.last_cube_pos = (0, 0)

    t_end = time.time()

    try:
      rospy.loginfo("image processed in {} seconds".format(t_end-t_start))

    except(e):
      print(e)

def main(args):
  parser = argparse.ArgumentParser()

  parser.add_argument('-s', '--save_path', default=None)
  parser.add_argument('-i', '--image_path', nargs='+', default=None)

  args2 = vars(parser.parse_args())

  while not rospy.is_shutdown():
    ic = image_converter(args2['save_path'])

    rospy.init_node('space_vision', anonymous=True)

    pub = rospy.Publisher("camera/image_raw", Image, queue_size=10)
    time.sleep(1)

    if args2['image_path'] is not None:
      for i in range(0,len(args2['image_path'])):

        img = cv2.imread(args2['image_path'][i],1)

        img_msg = ic.bridge.cv2_to_imgmsg(img, encoding='bgr8')

        #for j in range(20):
        pub.publish(img_msg)
        print("published on image_topic")
        time.sleep(1)

    print('waiting')
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)