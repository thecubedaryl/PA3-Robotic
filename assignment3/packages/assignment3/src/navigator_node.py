#!/usr/bin/env python3
"""
ROS navigator: ArUco-only localization, A* path following, car_cmd_switch control
with debug image publishing.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rospy
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage

# -----------------------------------------------------------------------------
# Defaults (overridden by ROS params under ~)
# -----------------------------------------------------------------------------
ROBOT_NAME_DEFAULT = "bear"

LINEAR_SPEED_DEFAULT = 0.18
ANGULAR_GAIN_DEFAULT = 0.7
PROXIMITY_THRESHOLD_DEFAULT = 0.45
ALIGN_ANGLE_MAX_DEFAULT = 0.20      # 0.35 → 0.20: hareket öncesi daha iyi hizalama
SEARCH_ANGULAR_SPEED_DEFAULT = 0.3
SEARCH_LINEAR_SPEED_DEFAULT = 0.02
ALIGN_LINEAR_SPEED_DEFAULT = 0.10
MAX_ANGULAR_SPEED_DEFAULT = 0.60
DETECTION_STALE_SEC_DEFAULT = 1.2   # 0.8 → 1.2: geçici kayıplarda stop azalır
WAIT_LOG_PERIOD_SEC_DEFAULT = 5.0
YAW_BIAS_DEFAULT = 0.05
CMD_LOG_PERIOD_SEC_DEFAULT = 1.0
HANDOFF_LINEAR_SPEED_DEFAULT = 0.06
HANDOFF_DURATION_SEC_DEFAULT = 1.0

ARUCO_TAG_SIZE_METERS_DEFAULT = 0.065
ARUCO_DICTIONARY_DEFAULT = "DICT_5X5_50"

# Gerçek kalibrasyon değerleri (bear.yaml)
CAMERA_FX_DEFAULT = 270.4563591302591
CAMERA_FY_DEFAULT = 269.2951665378049
CAMERA_CX_DEFAULT = 314.1813567017415
CAMERA_CY_DEFAULT = 218.88618596346137

# Distortion katsayıları (bear.yaml) - plumb_bob modeli
DIST_K1_DEFAULT = -0.19162991260105328
DIST_K2_DEFAULT =  0.026384790215657535
DIST_P1_DEFAULT =  0.005682129590129115
DIST_P2_DEFAULT =  0.0006647376545041703
DIST_K3_DEFAULT =  0.0

# Detection confirmation: hedef tag gorulur gorulmez takip baslasin.
CONFIRM_FRAMES = 1
# Reach kararini vermeden once hedef tag yakinliginin kac frame korunmasi gerektigi
REACH_CONFIRM_FRAMES = 3

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import astar  # noqa: E402


def _norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _bearing_xz(tx: float, ty: float, tz: float) -> float:
    """Horizontal bearing in camera frame with +z forward, +x right."""
    return math.atan2(tx, tz)


def _build_aruco_dictionary(name: str):
    attr = getattr(cv2.aruco, name, None)
    if attr is None:
        raise ValueError("Unsupported ArUco dictionary '%s'" % name)
    return cv2.aruco.getPredefinedDictionary(attr)


def _build_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def _detect_markers(image, dictionary, parameters):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(image)
    return cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)


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
        self.search_linear_speed = float(
            rospy.get_param("~search_linear_speed", SEARCH_LINEAR_SPEED_DEFAULT)
        )
        self.align_linear_speed = float(
            rospy.get_param("~align_linear_speed", ALIGN_LINEAR_SPEED_DEFAULT)
        )
        self.max_omega = float(
            rospy.get_param("~max_angular_speed", MAX_ANGULAR_SPEED_DEFAULT)
        )
        self.detection_stale_sec = float(
            rospy.get_param("~detection_stale_sec", DETECTION_STALE_SEC_DEFAULT)
        )
        self.wait_log_period_sec = float(
            rospy.get_param("~wait_log_period_sec", WAIT_LOG_PERIOD_SEC_DEFAULT)
        )
        self.cmd_log_period_sec = float(
            rospy.get_param("~cmd_log_period_sec", CMD_LOG_PERIOD_SEC_DEFAULT)
        )
        self.handoff_linear_speed = float(
            rospy.get_param("~handoff_linear_speed", HANDOFF_LINEAR_SPEED_DEFAULT)
        )
        self.handoff_duration_sec = float(
            rospy.get_param("~handoff_duration_sec", HANDOFF_DURATION_SEC_DEFAULT)
        )
        self.yaw_bias = float(rospy.get_param("~yaw_bias", YAW_BIAS_DEFAULT))
        self.aruco_tag_size = float(
            rospy.get_param("~aruco_tag_size_meters", ARUCO_TAG_SIZE_METERS_DEFAULT)
        )
        self.aruco_dictionary_name = str(
            rospy.get_param("~aruco_dictionary", ARUCO_DICTIONARY_DEFAULT)
        )

        # Gerçek kalibrasyon matrisi (bear.yaml)
        self.camera_matrix = np.array(
            [
                [
                    float(rospy.get_param("~camera_fx", CAMERA_FX_DEFAULT)),
                    0.0,
                    float(rospy.get_param("~camera_cx", CAMERA_CX_DEFAULT)),
                ],
                [
                    0.0,
                    float(rospy.get_param("~camera_fy", CAMERA_FY_DEFAULT)),
                    float(rospy.get_param("~camera_cy", CAMERA_CY_DEFAULT)),
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Gerçek distortion katsayıları (bear.yaml) - artık sıfır değil
        self.dist_coeffs = np.array(
            [
                [float(rospy.get_param("~dist_k1", DIST_K1_DEFAULT))],
                [float(rospy.get_param("~dist_k2", DIST_K2_DEFAULT))],
                [float(rospy.get_param("~dist_p1", DIST_P1_DEFAULT))],
                [float(rospy.get_param("~dist_p2", DIST_P2_DEFAULT))],
                [float(rospy.get_param("~dist_k3", DIST_K3_DEFAULT))],
            ],
            dtype=np.float32,
        )

        self.aruco_dictionary = _build_aruco_dictionary(self.aruco_dictionary_name)
        self.aruco_parameters = _build_detector_parameters()

        # Run A* with verbose expansion log (prints g, h, f for each expanded node)
        path, cost = astar.astar_search(0, 15, verbose=True)
        if path is None:
            rospy.logfatal("A* found no path from N0 to N15.")
            raise RuntimeError("A* found no path from N0 to N15.")
        self.path: List[int] = path
        self._path_cost: float = cost

        rospy.loginfo("A* path: %s", astar.format_path(self.path))
        rospy.loginfo("A* total cost: %.4f", cost)
        print("\nPath sequence : %s" % astar.format_path(self.path))
        print("Total cost    : %.4f\n" % cost)

        self._leg = 1
        self._goal_done = False

        # Latest per-tag metrics: tag_id -> (distance, yaw_off, stamp)
        self._tag_metrics: Dict[int, Tuple[float, float, rospy.Time]] = {}
        self._last_camera_msg_time: rospy.Time | None = None

        # Detection confirmation buffer: tag_id -> ardışık görülme sayısı
        self._detection_buffer: Dict[int, int] = {}
        self._reach_buffer: Dict[int, int] = {}
        self._handoff_until = rospy.Time(0)

        image_topic = rospy.get_param(
            "~camera_image_topic",
            "/%s/camera_node/image/compressed" % self.robot_name,
        )
        self._sub = rospy.Subscriber(
            image_topic,
            CompressedImage,
            self._on_camera_image,
            queue_size=1,
            buff_size=2 ** 24,
        )

        cmd_topic = rospy.get_param(
            "~cmd_topic",
            "/%s/car_cmd_switch_node/cmd" % self.robot_name,
        )
        self.cmd_topic = cmd_topic
        self._pub = rospy.Publisher(cmd_topic, Twist2DStamped, queue_size=1)

        self.debug_pub = rospy.Publisher(
            "/%s/debug_image/compressed" % self.robot_name,
            CompressedImage,
            queue_size=1,
        )

        self._rate = rospy.Rate(20.0)
        rospy.on_shutdown(self._on_shutdown)
        rospy.sleep(0.3)

        rospy.loginfo("Subscribing: %s", image_topic)
        rospy.loginfo(
            "Using ArUco dictionary %s, tag size %.3f m",
            self.aruco_dictionary_name,
            self.aruco_tag_size,
        )
        rospy.loginfo("Publishing: %s (Twist2DStamped: v, omega)", cmd_topic)
        rospy.loginfo(
            "Publishing debug image: /%s/debug_image/compressed",
            self.robot_name,
        )
        rospy.loginfo(
            "Camera matrix: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            self.camera_matrix[0, 0],
            self.camera_matrix[1, 1],
            self.camera_matrix[0, 2],
            self.camera_matrix[1, 2],
        )

    def _draw_debug_overlay(
        self,
        frame: np.ndarray,
        corners,
        ids,
        tvecs,
    ) -> np.ndarray:
        debug = frame.copy()
        target_node = self.path[self._leg] if self._leg < len(self.path) else -1

        cv2.putText(
            debug,
            f"TARGET: N{target_node}" if target_node >= 0 else "TARGET: DONE",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            debug,
            f"LEG: {self._leg}/{len(self.path)-1}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        if ids is None or len(ids) == 0:
            return debug

        flat_ids = ids.reshape(-1)
        for i, tag_id in enumerate(flat_ids):
            pts = corners[i][0].astype(int)
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            tx, ty, tz = [float(v) for v in tvecs[i][0]]
            dist = _norm3(tx, ty, tz)
            yaw = _bearing_xz(tx, ty, tz)

            color = (0, 255, 0)
            if int(tag_id) == target_node:
                color = (0, 0, 255)

            cv2.polylines(debug, [pts], True, color, 2)

            # Confirmation count göster
            conf = self._detection_buffer.get(int(tag_id), 0)
            label = f"ID:{int(tag_id)} d:{dist:.2f} y:{yaw:.2f} c:{conf}"
            cv2.putText(
                debug,
                label,
                (center_x - 60, max(20, center_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                2,
            )

            arrow_len = 40
            end_x = int(center_x + arrow_len * math.sin(yaw))
            end_y = center_y
            cv2.arrowedLine(
                debug,
                (center_x, center_y),
                (end_x, end_y),
                (255, 255, 255),
                2,
                tipLength=0.25,
            )

        return debug

    def _publish_debug_image(self, frame: np.ndarray) -> None:
        try:
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                return
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = encoded.tobytes()
            self.debug_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(
                self.wait_log_period_sec,
                "Failed to publish debug image: %s",
                str(e),
            )

    def _on_camera_image(self, msg: CompressedImage) -> None:
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        self._last_camera_msg_time = stamp

        np_image = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn_throttle(
                self.wait_log_period_sec,
                "Failed to decode compressed camera image for robot '%s'.",
                self.robot_name,
            )
            return

        # Kalibrasyon 640x480 için yapıldı - aynı çözünürlükte tut
        frame = cv2.resize(frame, (640, 480))

        corners, ids, _rejected = _detect_markers(
            frame, self.aruco_dictionary, self.aruco_parameters
        )

        if ids is not None and len(ids) > 0:
            rospy.loginfo_throttle(1.0, "Detected IDs: %s", ids.reshape(-1).tolist())

        metrics: Dict[int, Tuple[float, float, rospy.Time]] = {}
        tvecs = None

        if ids is not None and len(ids) > 0:
            try:
                _rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    self.aruco_tag_size,
                    self.camera_matrix,
                    self.dist_coeffs,
                )

                flat_ids = ids.reshape(-1)
                for idx, tag_id in enumerate(flat_ids):
                    tx, ty, tz = [float(v) for v in tvecs[idx][0]]
                    dist = _norm3(tx, ty, tz)
                    yaw_off = _bearing_xz(tx, ty, tz)

                    if 0.01 < dist < 1.5:
                        metrics[int(tag_id)] = (dist, yaw_off, stamp)
            except Exception as e:
                rospy.logwarn_throttle(
                    self.wait_log_period_sec,
                    "Pose estimation failed: %s",
                    str(e),
                )

        # Detection confirmation buffer guncelle
        for tag_id in metrics:
            self._detection_buffer[tag_id] = self._detection_buffer.get(tag_id, 0) + 1
            if metrics[tag_id][0] < self.proximity_threshold:
                self._reach_buffer[tag_id] = self._reach_buffer.get(tag_id, 0) + 1
            else:
                self._reach_buffer[tag_id] = 0

        # Kaybolanlarda sifirla
        for tag_id in list(self._detection_buffer.keys()):
            if tag_id not in metrics:
                self._detection_buffer[tag_id] = 0
        for tag_id in list(self._reach_buffer.keys()):
            if tag_id not in metrics:
                self._reach_buffer[tag_id] = 0

        # Son gorulen etiketleri hemen silme; stale kontrolu run() icinde yapiliyor.
        for tag_id, data in metrics.items():
            self._tag_metrics[tag_id] = data

        if tvecs is not None and ids is not None and len(ids) > 0:
            debug_frame = self._draw_debug_overlay(frame, corners, ids, tvecs)
        else:
            debug_frame = frame.copy()
            target_node = self.path[self._leg] if self._leg < len(self.path) else -1
            cv2.putText(
                debug_frame,
                f"TARGET: N{target_node}" if target_node >= 0 else "TARGET: DONE",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        self._publish_debug_image(debug_frame)

    def _publish_cmd(self, v: float, omega: float) -> None:
        omega = max(-self.max_omega, min(self.max_omega, omega))
        m = Twist2DStamped()
        m.header.stamp = rospy.Time.now()
        m.v = float(v)
        m.omega = float(omega)
        self._pub.publish(m)
        rospy.loginfo_throttle(
            self.cmd_log_period_sec,
            "Publishing cmd to %s: v=%.3f omega=%.3f",
            self.cmd_topic,
            m.v,
            m.omega,
        )

    def _stop(self) -> None:
        self._publish_cmd(0.0, 0.0)

    def _hard_stop(self) -> None:
        for _ in range(5):
            self._publish_cmd(0.0, 0.0)
            rospy.sleep(0.05)

    def _on_shutdown(self) -> None:
        try:
            self._hard_stop()
        except Exception:
            pass

    def run(self) -> None:
        while not rospy.is_shutdown() and not self._goal_done:
            if self._leg >= len(self.path):
                self._finish_goal()
                break

            current_node = self.path[self._leg - 1]
            target_node = self.path[self._leg]
            info = self._tag_metrics.get(target_node)
            current_info = self._tag_metrics.get(current_node)
            now = rospy.Time.now()

            # Stale kontrol
            if info is not None:
                _d, _y, st = info
                age = (now - st).to_sec()
                if age > self.detection_stale_sec:
                    info = None
            if current_info is not None:
                _cd, _cy, cst = current_info
                current_age = (now - cst).to_sec()
                if current_age > self.detection_stale_sec:
                    current_info = None

            # Confirmation kontrolü: yeterince ardışık frame'de görülmeli
            if info is not None:
                hit_count = self._detection_buffer.get(target_node, 0)
                if hit_count < CONFIRM_FRAMES:
                    info = None

            if info is None:
                if now < self._handoff_until:
                    self._publish_cmd(self.handoff_linear_speed, 0.0)
                elif current_info is not None:
                    # Hedef tag görünmüyorken mevcut node tag'i hâlâ görünüyorsa
                    # önce kısa düz ilerle; hemen arama dönüşüne girme.
                    self._publish_cmd(self.handoff_linear_speed, 0.0)
                else:
                    self._log_wait_reason(target_node)
                    # Assignment metnine uygun olarak hedef tag kaybolunca dur veya yavasca
                    # yeniden kazan; ileri tarama ile kestirme davranis verme.
                    self._publish_cmd(self.search_linear_speed, self.search_omega)
            else:
                dist, yaw_err, _stamp = info
                yaw_err += self.yaw_bias

                reach_count = self._reach_buffer.get(target_node, 0)
                if dist < self.proximity_threshold and reach_count >= REACH_CONFIRM_FRAMES:
                    rospy.loginfo("Reached node N%d (tag distance %.3f m).", target_node, dist)
                    # Ulaşılan node tag verisini temizle — sonraki leg eski veriyi kullanmasın
                    self._tag_metrics.pop(target_node, None)
                    self._leg += 1
                    # Buffer'lari sifirla
                    self._detection_buffer = {}
                    self._reach_buffer = {}
                    self._handoff_until = rospy.Time.now() + rospy.Duration.from_sec(
                        self.handoff_duration_sec
                    )
                    if self._leg >= len(self.path):
                        self._finish_goal()
                    continue

                omega = self.angular_gain * yaw_err

                if abs(yaw_err) > self.align_angle_max:
                    # Tam kilitlenmesin diye hizalanırken yavas ileri akis ver.
                    self._publish_cmd(self.align_linear_speed, omega)
                else:
                    self._publish_cmd(self.linear_speed, omega)

            self._rate.sleep()

    def _log_wait_reason(self, target_node: int) -> None:
        if self._last_camera_msg_time is None:
            rospy.logwarn_throttle(
                self.wait_log_period_sec,
                "No camera images received yet on robot '%s'. Check camera topic.",
                self.robot_name,
            )
            return

        age = (rospy.Time.now() - self._last_camera_msg_time).to_sec()
        if age > self.detection_stale_sec:
            rospy.logwarn_throttle(
                self.wait_log_period_sec,
                "Camera images are stale (last frame %.2f s ago). Expected next target tag: N%d.",
                age,
                target_node,
            )
            return

        fresh_visible_tags = []
        now = rospy.Time.now()
        for tag_id, (_dist, _yaw, stamp) in self._tag_metrics.items():
            if (now - stamp).to_sec() <= self.detection_stale_sec:
                fresh_visible_tags.append(tag_id)
        fresh_visible_tags.sort()
        rospy.loginfo_throttle(
            self.wait_log_period_sec,
            "ArUco stream is active but target tag N%d is not visible. Currently visible tags: %s",
            target_node,
            fresh_visible_tags if fresh_visible_tags else "none",
        )

    def _finish_goal(self) -> None:
        if self._goal_done:
            return
        self._goal_done = True
        self._hard_stop()

        # Verify N15 ARTag was freshly detected (assignment requirement)
        now = rospy.Time.now()
        n15_info = self._tag_metrics.get(15)
        if n15_info is None:
            rospy.logwarn("Stopping: N15 ARTag not in detection cache.")
        else:
            age = (now - n15_info[2]).to_sec()
            if age > self.detection_stale_sec:
                rospy.logwarn(
                    "Stopping: last N15 detection is %.2f s old (stale).", age
                )

        # Required terminal output (assignment Section 4 & 3.3)
        print("\nPath sequence : %s" % astar.format_path(self.path))
        print("Total cost    : %.4f" % self._path_cost)
        print("Goal Reached")
        rospy.loginfo("Goal Reached")
        rospy.signal_shutdown("goal_reached")


def main() -> None:
    rospy.init_node("assignment3_navigator", anonymous=False)
    nav = Assignment3Navigator()
    nav.run()


if __name__ == "__main__":
    main()
