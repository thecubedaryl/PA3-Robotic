#!/usr/bin/env python3
"""
ROS navigator: AprilTag-only localization, A* path following, car_cmd_switch control.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

import rospy
from duckietown_msgs.msg import AprilTagDetectionArray, Twist2DStamped

# Duckietown expects Twist2DStamped on car_cmd_switch (v ~ cmd_vel.linear.x, omega ~ cmd_vel.angular.z).

# -----------------------------------------------------------------------------
# Defaults (overridden by ROS params under ~)
# -----------------------------------------------------------------------------
ROBOT_NAME_DEFAULT = "autobot01"
LINEAR_SPEED_DEFAULT = 0.2
ANGULAR_GAIN_DEFAULT = 1.5
PROXIMITY_THRESHOLD_DEFAULT = 0.3

# Align before driving forward (rad)
ALIGN_ANGLE_MAX_DEFAULT = 0.18
# Slow search rotation when target tag is lost (rad/s)
SEARCH_ANGULAR_SPEED_DEFAULT = 0.35
# Cap angular rate (rad/s)
MAX_ANGULAR_SPEED_DEFAULT = 1.8
# Drop tag measurement if detections are older than this (seconds)
DETECTION_STALE_SEC_DEFAULT = 0.5

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import astar  # noqa: E402


def _norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _bearing_xz(tx: float, ty: float, tz: float) -> float:
    """Horizontal bearing in camera frame with +z forward, +x right (typical optical)."""
    return math.atan2(tx, tz)


class Assignment3Navigator:
    def __init__(self) -> None:
        self.robot_name = rospy.get_param("~robot_name", ROBOT_NAME_DEFAULT)
        self.linear_speed = float(rospy.get_param("~linear_speed", LINEAR_SPEED_DEFAULT))
        self.angular_gain = float(rospy.get_param("~angular_gain", ANGULAR_GAIN_DEFAULT))
        self.proximity_threshold = float(
            rospy.get_param("~proximity_threshold", PROXIMITY_THRESHOLD_DEFAULT)
        )
        self.align_angle_max = float(
            rospy.get_param("~align_angle_max", ALIGN_ANGLE_MAX_DEFAULT)
        )
        self.search_omega = float(
            rospy.get_param("~search_angular_speed", SEARCH_ANGULAR_SPEED_DEFAULT)
        )
        self.max_omega = float(rospy.get_param("~max_angular_speed", MAX_ANGULAR_SPEED_DEFAULT))
        self.detection_stale_sec = float(
            rospy.get_param("~detection_stale_sec", DETECTION_STALE_SEC_DEFAULT)
        )

        path, cost = astar.astar_search(0, 15)
        if path is None:
            rospy.logfatal("A* found no path from N0 to N15.")
            raise RuntimeError("A* found no path from N0 to N15.")
        self.path: List[int] = path
        rospy.loginfo("A* path: %s", astar.format_path(self.path))
        rospy.loginfo("A* total cost: %s", cost)
        print("Path sequence: %s" % astar.format_path(self.path))
        print("Total cost: %s" % cost)

        # Next index in self.path we are navigating toward (1 .. len-1)
        self._leg = 1
        self._goal_done = False

        # Latest per-tag metrics from detections (continuously updated)
        self._tag_metrics: Dict[int, Tuple[float, float, rospy.Time]] = {}
        # Default matches dt-core daffy apriltag_detector_node (private ~detections).
        # Verify on-robot: rostopic list | grep -i april
        # Override with ~apriltag_detections_topic if your stack differs.
        det_topic = rospy.get_param(
            "~apriltag_detections_topic",
            "/%s/apriltag_detector_node/detections" % self.robot_name,
        )
        self._sub = rospy.Subscriber(det_topic, AprilTagDetectionArray, self._on_detections, queue_size=1)

        cmd_topic = "/%s/car_cmd_switch_node/cmd" % self.robot_name
        self._pub = rospy.Publisher(cmd_topic, Twist2DStamped, queue_size=1)

        self._rate = rospy.Rate(30.0)
        rospy.sleep(0.3)
        rospy.loginfo("Subscribing: %s", det_topic)
        rospy.loginfo("Publishing:  %s (Twist2DStamped: v, omega)", cmd_topic)

    def _on_detections(self, msg: AprilTagDetectionArray) -> None:
        now = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        metrics: Dict[int, Tuple[float, float, rospy.Time]] = {}
        for det in msg.detections:
            tid = int(det.tag_id)
            # duckietown_msgs/AprilTagDetection defines `geometry_msgs/Transform transform`
            # (Vector3 translation + Quaternion rotation) — not TransformStamped, so
            # use det.transform.translation, not det.transform.transform.translation.
            t = det.transform.translation
            d = _norm3(t.x, t.y, t.z)
            yaw_off = _bearing_xz(t.x, t.y, t.z)
            metrics[tid] = (d, yaw_off, now)
        self._tag_metrics = metrics

    def _publish_cmd(self, v: float, omega: float) -> None:
        omega = max(-self.max_omega, min(self.max_omega, omega))
        m = Twist2DStamped()
        m.header.stamp = rospy.Time.now()
        m.v = float(v)
        m.omega = float(omega)
        self._pub.publish(m)

    def _stop(self) -> None:
        self._publish_cmd(0.0, 0.0)

    def run(self) -> None:
        while not rospy.is_shutdown() and not self._goal_done:
            if self._leg >= len(self.path):
                self._finish_goal()
                break

            target_node = self.path[self._leg]
            info = self._tag_metrics.get(target_node)
            if info is not None:
                _d, _y, st = info
                age = (rospy.Time.now() - st).to_sec()
                if age > self.detection_stale_sec:
                    info = None

            if info is None:
                # Stop translation, rotate slowly to reacquire next tag
                self._publish_cmd(0.0, self.search_omega)
            else:
                dist, yaw_err, _stamp = info
                if dist < self.proximity_threshold:
                    rospy.loginfo("Reached node N%d (tag distance %.3f m).", target_node, dist)
                    self._leg += 1
                    if self._leg >= len(self.path):
                        self._finish_goal()
                    continue
                omega = self.angular_gain * yaw_err
                if abs(yaw_err) > self.align_angle_max:
                    self._publish_cmd(0.0, omega)
                else:
                    self._publish_cmd(self.linear_speed, omega)

            self._rate.sleep()

    def _finish_goal(self) -> None:
        if self._goal_done:
            return
        self._goal_done = True
        self._stop()
        print("Goal Reached")
        rospy.loginfo("Goal Reached")
        rospy.signal_shutdown("goal_reached")


def main() -> None:
    rospy.init_node("assignment3_navigator", anonymous=False)
    nav = Assignment3Navigator()
    nav.run()


if __name__ == "__main__":
    main()
