#!/usr/bin/env python
import numpy as np
import sys
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import math

class DisparityExtender:

    CAR_WIDTH = 0.45
    # the min difference between adjacent LiDAR points for us to call them disparate
    DIFFERENCE_THRESHOLD = 2.
    STRAIGHTS_SPEED = 4.0
    CORNERS_SPEED = 3.0
    DRAG_SPEED = 4.0
    # the extra safety room we plan for along walls (as a percentage of car_width/2)
    SAFETY_PERCENTAGE = 900.
    def __init__(self):
        self.STEERING_SENSITIVITY = 3.
        self.COEFFICIENT = 2.5
        self.EXP_COEFFICIENT = 0.02
        self.X_POWER = 1.8
        self.QUADRANT_FACTOR = 3.5

        self.time = rospy.get_time()
        self.speed = 4. # Initial Speed?
        
        lidarscan_topic = '/scan'
        odom_topic = '/vesc/odom'
        drive_topic = '/vesc/low_level/ackermann_cmd_mux/input/teleop' # come back to me later
        
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                self.odom_cb, queue_size=1, buff_size=2**24)
        self.lidar_sub = rospy.Subscriber(lidarscan_topic, LaserScan,
                self.process_lidar, queue_size=1, buff_size=2**24)
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped,
                queue_size=1)

    def odom_cb(self, data):
        self.speed = data.twist.twist.linear.x

    def preprocess_lidar(self, ranges):
        """ Any preprocessing of the LiDAR data can be done in this function.
            Possible Improvements: smoothing of outliers in the data and placing
            a cap on the maximum distance a point can be.
        """
        # remove quadrant of LiDAR directly behind us
        ranges = np.clip(ranges, 0, 16)
        eighth = int(len(ranges)/self.QUADRANT_FACTOR)

        return np.array(ranges[eighth:-eighth])


    def get_differences(self, ranges):
        """ Gets the absolute difference between adjacent elements in
            in the LiDAR data and returns them in an array.
            Possible Improvements: replace for loop with numpy array arithmetic
        """
        differences = [0.] # set first element to 0
        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i]-ranges[i-1]))
        return differences

    def get_disparities(self, differences, threshold):
        """ Gets the indexes of the LiDAR points that were greatly
            different to their adjacent point.
            Possible Improvements: replace for loop with numpy array arithmetic
        """
        disparities = []
        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)
        return disparities

    def get_num_points_to_cover(self, dist, width):
        """ Returns the number of LiDAR points that correspond to a width at
            a given distance.
            We calculate the angle that would span the width at this distance,
            then convert this angle to the number of LiDAR points that
            span this angle.
            Current math for angle:
                sin(angle/2) = (w/2)/d) = w/2d
                angle/2 = sininv(w/2d)
                angle = 2sininv(w/2d)
                where w is the width to cover, and d is the distance to the close
                point.
            Possible Improvements: use a different method to calculate the angle
        """
        angle = 1.5*np.arctan(width/(2*dist))
        num_points = int(np.ceil(angle / self.radians_per_point))
        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """ 'covers' a number of LiDAR points with the distance of a closer
            LiDAR point, to avoid us crashing with the corner of the car.
            num_points: the number of points to cover
            start_idx: the LiDAR point we are using as our distance
            cover_right: True/False, decides whether we cover the points to
                         right or to the left of start_idx
            ranges: the LiDAR points
            Possible improvements: reduce this function to fewer lines
        """
        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx+1+i
                if next_idx >= len(ranges): break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx-1-i
                if next_idx < 0: break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        return ranges

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        """ For each pair of points we have decided have a large difference
            between them, we choose which side to cover (the opposite to
            the closer point), call the cover function, and return the
            resultant covered array.
            Possible Improvements: reduce to fewer lines
        """
        width_to_cover = 0.155 * (1+extra_pct/100)
        for index in disparities:
            first_idx = index-1
            points = ranges[first_idx:first_idx+2]
            close_idx = first_idx+np.argmin(points)
            far_idx = first_idx+np.argmax(points)
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(close_dist,
                    width_to_cover)
            cover_right = close_idx < far_idx
            ranges = self.cover_points(num_points_to_cover, close_idx,
                cover_right, ranges)
        return ranges

    def get_steering_angle(self, range_index, range_len):
        """ Calculate the angle that corresponds to a given LiDAR point and
            process it into a steering angle.
            Possible improvements: smoothing of aggressive steering angles
        """
        lidar_angle = (range_index - (range_len/2)) * self.radians_per_point
        steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))/self.STEERING_SENSITIVITY
        return steering_angle

    def process_lidar(self, data):
        """ Run the disparity extender algorithm!
            Possible improvements: varying the speed based on the
            steering angle or the distance to the farthest point.
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()

        ranges = data.ranges
        self.radians_per_point = (2*np.pi)/len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(disparities, proc_ranges,
                self.CAR_WIDTH, self.SAFETY_PERCENTAGE)
        steering_angle = self.get_steering_angle(proc_ranges.argmax(),
                len(proc_ranges))

        x = max(proc_ranges[227:237])
        speed = self.COEFFICIENT*math.exp(self.EXP_COEFFICIENT*(x**self.X_POWER))
        rospy.loginfo('x: {}, speed: {}'.format(x, speed))

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

if __name__ == '__main__':
    rospy.init_node("disparity_extender_node", anonymous=True)
    disparity = DisparityExtender()
    rospy.sleep(0.1)
    rospy.spin()
