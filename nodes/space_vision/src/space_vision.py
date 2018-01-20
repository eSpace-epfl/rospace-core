#!/usr/bin/env python

"""
    This Class implemented an image subscriber, performs the target detection using the vision_lib library and publishes
    the aziuth, elevation, range and pose of the target. If the module is called, it creates an instance of the class, and
    processes the images given in -i (if any), saving the results in the path provided in -s (if any)

    Author: Gaetan Ramet
    License: TBD
"""
import roslib
roslib.load_manifest('space_vision')
import os, sys, time
import rospy
import cv2
import scipy.misc
import numpy as np
import argparse
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import vision_algo.vision_lib as lib


class ImageAnalyser:
    """
    use this class to create a subscriber/publisher which performs image analysis on received images
    """

    def __init__(self, path):
        self.image_pub = rospy.Publisher("image_topic_2", Image, queue_size=10)
        '''Image publisher for processed images'''

        self.cube_pos_pub = rospy.Publisher("target_pos", String, queue_size=10)
        '''Cube position publisher'''

        self.cube_quat_pub = rospy.Publisher("cube_quaternion", String, queue_size=10)
        '''Cube pose publisher'''

        self.bridge = CvBridge()
        '''Bridge between ROS and OpenCV'''

        self.image_sub = rospy.Subscriber("camera/image_raw",Image,self.callback)
        '''Image subscriber for the images to be analyzed'''

        self.last_cube_pos = (0, 0)
        '''Last known position of the cube in the image. (0,0) means unknown'''

        self.save_file = None
        '''Path to save file'''

        self.image_path = None
        '''Path to saving folder'''

        self.write_signal_pub = rospy.Publisher("writer_trigger", Bool, queue_size=1)
        '''Publisher for triggering the wirting of files with real data'''

        if path is not None:
            self.save_file = os.path.join(path, 'data.txt')

            if not os.path.exists(path):
                os.makedirs(path)
                os.makedirs(os.path.join(path, 'original'))
                os.makedirs(os.path.join(path, 'processed'))

            self.image_path = path

            with open(self.save_file, 'w') as save_file:
                save_file.write("Format : timestamp; range; azim; elev; quat\n")

    def callback(self, data):
        """Callback of the image subscriber for the image analysis routine

            Args:
                data: The received data
        """

        t_start = time.time()
        print("start timing")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        rospy.loginfo("received image with format {} x {} x {}".format(rows,cols,channels))
    
        d, azim, elev, quat, cm_coo, processed_image, cube_found = lib.img_analysis(cv_image, self.last_cube_pos, mode='test')

        if cube_found:
            cube_pos = [d, azim, elev]

            self.last_cube_pos = cm_coo

            self.cube_pos_pub.publish("cube position : {} ".format(cube_pos))
            rospy.loginfo("publish cube position : {}".format(cube_pos))
            self.cube_quat_pub.publish("cube quaternion : {} ".format(quat))
            rospy.loginfo("publish cube quaternion : {}".format(quat))

            if self.save_file is not None:

                self.write_signal_pub.publish(True)
                current_time = time.time()
                with open(self.save_file, 'a') as save_file:
                    save_file.write("{}; {} ; {} ; {} ; {}\n".format(current_time, d, azim, elev, quat))

                scipy.misc.imsave(os.path.join(self.image_path, 'processed', 'img{}.png'.format(current_time)), processed_image)
                scipy.misc.imsave(os.path.join(self.image_path, 'original', 'img{}.png'.format(current_time)), cv_image)

        else:
            rospy.loginfo("Cube not found on image")
            self.last_cube_pos = (0, 0)

        t_end = time.time()

        try:
            rospy.loginfo("image processed in {} seconds".format(t_end-t_start))

        except(e):
            print(e)


def main(args):
    """Main function. Parses arguments given as input ad generates an instance of ImageAnalyser
    to process the given images (if any)"""

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save_path', default=None)
    parser.add_argument('-i', '--image_path', nargs='+', default=None)

    args2 = vars(parser.parse_args())

    while not rospy.is_shutdown():
        ic = ImageAnalyser(args2['save_path'])

        rospy.init_node('space_vision', anonymous=True)

        pub = rospy.Publisher("camera/image_raw", Image, queue_size=10)
        time.sleep(1)

        if args2['image_path'] is not None:
            print(args2['image_path'][0])
            for i in range(0,len(args2['image_path'])):

                #for img_number in range(120,320):
                    #print('img{:d}.png'.format(img_number))
                    #img = cv2.imread(os.path.join(args2['image_path'][0],'img{:d}.png'.format(img_number)) ,1)
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