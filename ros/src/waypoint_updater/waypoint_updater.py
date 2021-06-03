#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import numpy as np
import logging

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
STATE_DRIVE = 0
STATE_STOP = 1


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        sub1 = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.traffic_wp_idx = -1
        self.setpoint_speed = 12
        self.traffic_stop_offset = 3
        self.num_of_wp = 0
        self.min_traffic_distance = 12

        # States
        self.stop_trajectory = []
        self.car_state = STATE_DRIVE
        self.current_traffic_stop = -1

        self.loop()

    def loop(self):
        # Waypoint publishing frequency 50Hz
        rate = rospy.Rate(50)
        while  not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.publish_waypoints()

            rate.sleep()
        

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Waypoints are arranged as ordered points of the road
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        closest_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(closest_vect - prev_vect, pos_vect - closest_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx


    def publish_waypoints(self):

        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header
        closest_idx = self.get_closest_waypoint_idx()

        # Form a list of waypoints taking into account that it is circular
        waypoints = []
        base_waypoints = self.base_waypoints.waypoints

        if closest_idx + LOOKAHEAD_WPS > self.num_of_wp:
            waypoints = base_waypoints[closest_idx:] + base_waypoints[:(closest_idx + LOOKAHEAD_WPS) % self.num_of_wp]
        else:
            waypoints = base_waypoints[closest_idx: closest_idx + LOOKAHEAD_WPS]

        # Transition states
        # =================
        if self.car_state == STATE_DRIVE:           

            red_light = self.traffic_wp_idx != -1
            transition_to_stop = False
            
            # If we are to close to stop position when it turns red, we would not 
            # be able to stop before stepping over the stop position. That could cause
            # illogical behavior since traffic_wp_idx would no longer represent the state
            # of the current traffic lights. Given these limitations, if the stop position
            # is closer then min_traffic_distance then we would simply ignore it (it would
            # be better if traffic_cb would contain complete state of the traffic lights,
            # then we could stop as soon it is yellow).
            if red_light:
                dist_traffic_idx = self.get_idx_distance(closest_idx, self.traffic_wp_idx)
                transition_to_stop = (dist_traffic_idx >= self.min_traffic_distance) and \
                                     (dist_traffic_idx <= LOOKAHEAD_WPS)
            
            if transition_to_stop:
                self.car_state = STATE_STOP
                self.current_traffic_stop = self.traffic_wp_idx
            
        elif self.car_state == STATE_STOP:

            transition_to_drive = self.traffic_wp_idx == -1

            if transition_to_drive:
                self.stop_trajectory = []
                self.current_traffic_stop = -1
                self.car_state = STATE_DRIVE

        # Execute states
        # ==============
        if self.car_state == STATE_DRIVE:
            lane.waypoints = waypoints

        if self.car_state == STATE_STOP:

            stop_trajectory = self.stop_trajectory

            # Stop trajectory exists
            if stop_trajectory:
                found_idx = [i for i, x in enumerate(stop_trajectory) if x[0] == closest_idx]

                if found_idx:
                    lane.waypoints = [x[1] for x in stop_trajectory[found_idx[0]:]]
                else:
                    lane.waypoints = [stop_trajectory[-1][1]]
            else:
                lane.waypoints = self.decelerate_waypoints(waypoints, closest_idx, self.current_traffic_stop - closest_idx)        

        return lane

    def decelerate_waypoints(self, waypoints, start_idx, lookahead_wp):
        
        # Calculate constant deceleration rate to stop the car (a = v**2 / (2*s))
        stop_idx = lookahead_wp - self.traffic_stop_offset
        dist = self.distance(waypoints, 0, stop_idx)
        deceleration = self.setpoint_speed ** 2.0 / (2.0 * dist)

        self.stop_trajectory = []
        dist_remaining = dist        

        for i in range(0, stop_idx):

            wp = waypoints[i]
            # Create new wp instance, and keep pose of the original waypoints
            p = Waypoint()
            p.pose = wp.pose

            # Modify speed by assuming constant deceleration until we stop (a = v^2 / (2*s))
            vel = math.sqrt(max(2 * deceleration * dist_remaining, 0))      

            if vel < 1.0:
                vel = 0

            p.twist.twist.linear.x = vel
            dist_remaining -= self.distance(waypoints, i, i + 1)
            self.stop_trajectory.append(((start_idx + i) % self.num_of_wp, p))

        for i in range(stop_idx, stop_idx + self.traffic_stop_offset):
            wp = waypoints[i]
            # Create new wp instance, and keep pose of the original waypoints
            p = Waypoint()
            p.pose = wp.pose
            p.twist.twist.linear.x = 0
            self.stop_trajectory.append(((start_idx + i) % self.num_of_wp, p))

        return [x[1] for x in self.stop_trajectory]


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg
        # pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # The project is setup such that this method is only called once, enabling us to
        # store waypoint for future use (hence topic name is /base_waypoints).
        self.base_waypoints = waypoints

        # Finding closest waypoint to the car
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] 
                for waypoint in waypoints.waypoints]
            # Tree structure to enable efficient lookup of nearest waypoints
            self.waypoint_tree = KDTree(self.waypoints_2d)
            self.num_of_wp = len(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, wp_idx, velocity):
        waypoints[wp_idx].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp_start_idx, wp_end_idx):
        """
        Calculates cumulative distance over waypoints by summing the distance between
        every two adjacent waypoint pairs.
        """
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp_start_idx, wp_end_idx):            
            dist += dl(waypoints[i].pose.pose.position, waypoints[i + 1].pose.pose.position)
        return dist

    def get_idx_distance(self, from_idx, to_idx):
        """
        Positive distance between (from) and (to) waypoints in the direction of
        increasing indices. 
        """
        dist_to_start = self.num_of_wp - from_idx

        return (to_idx + dist_to_start) % self.num_of_wp
        

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
