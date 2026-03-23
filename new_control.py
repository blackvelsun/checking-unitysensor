#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import pandas as pd
import os
from nav_msgs.msg import Path, Odometry
from erp42_msgs.msg import ControlMessage


class StanleyControlNode(Node):
    def __init__(self):
        super().__init__('stanley_control_node')
        self.get_logger().info("--- Stanley Control Node (Unity Linked) Started ---")

        # 1. 파라미터 및 상수 설정
        self.k = 0.5      
        self.k_sc = 1.0   
        self.L = 1.04     
        self.target_v = 5.0 / 3.6 

        # [중요] CSV 파일 경로 (본인 환경에 맞는지 확인하세요)
        self.declare_parameter('path_file', '/home/kdm/ACCA2026/src/erp42/erp42_control/erp42_control/complex_path.csv')
        self.csv_path = self.get_parameter('path_file').get_parameter_value().string_value

        # 2. 데이터 저장 변수
        self.cx, self.cy, self.cyaw = [], [], []
        self.state = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'v': 0.0}

        # [추가] 초기화 시 CSV 데이터 로드
        self.load_path_from_csv()

        # 3. 구독 및 발행 설정
        # 유니티 토픽 리스트에 맞춰 수정함
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/localization/kinematic_state',  # 수정됨! (/odom -> /localization/kinematic_state)
            self.odom_callback, 
            10
        ) 
        
        self.control_pub = self.create_publisher(ControlMessage, '/erp42_ctrl', 10) 

        # 4. 제어 루프 타이머 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.alive_count = 0

    def load_path_from_csv(self):
        """CSV 파일을 읽어 경로 설정"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            self.cx = df['x'].tolist()
            self.cy = df['y'].tolist()
            if 'yaw' in df.columns:
                self.cyaw = df['yaw'].tolist()
            else:
                self.calculate_yaw_from_points()
            self.get_logger().info(f"Loaded {len(self.cx)} points from CSV.")
        else:
            self.get_logger().error(f"CSV NOT FOUND: {self.csv_path}")

    def calculate_yaw_from_points(self):
        self.cyaw = []
        for i in range(len(self.cx)-1):
            self.cyaw.append(math.atan2(self.cy[i+1]-self.cy[i], self.cx[i+1]-self.cx[i]))
        if self.cx: self.cyaw.append(self.cyaw[-1])

    def odom_callback(self, msg):
        # 유니티에서 오는 위치/속도 데이터 업데이트
        self.state['x'] = msg.pose.pose.position.x
        self.state['y'] = msg.pose.pose.position.y
        self.state['v'] = msg.twist.twist.linear.x
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.state['yaw'] = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if not self.cx:
            self.get_logger().warn("Waiting for Path (CSV)...", throttle_duration_sec=2.0)
            return
        
        # Stanley Control 로직
        fx = self.state['x'] + self.L * math.cos(self.state['yaw'])
        fy = self.state['y'] + self.L * math.sin(self.state['yaw'])

        dx = [fx - icx for icx in self.cx]
        dy = [fy - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        error_vec = [dx[target_idx], dy[target_idx]]
        unit_vec_n = [-math.sin(self.cyaw[target_idx]), math.cos(self.cyaw[target_idx])]
        error_front_axle = np.dot(error_vec, unit_vec_n)

        theta_e = self.normalize_angle(self.cyaw[target_idx] - self.state['yaw'])
        theta_d = math.atan2(self.k * error_front_axle, self.state['v'] + self.k_sc)
        delta = theta_e + theta_d

        self.publish_control(delta, self.target_v)

    def publish_control(self, steer, speed):
        msg = ControlMessage()
        msg.mora = 1         
        msg.estop = 0        
        msg.gear = 0         
        steer_deg = np.rad2deg(steer)
        msg.steer = int(np.clip(steer_deg * 71, -2000, 2000))
        msg.speed = int(speed * 3.6 * 10) 
        msg.brake = 0
        self.alive_count = (self.alive_count + 1) % 256
        msg.alive = self.alive_count
        self.control_pub.publish(msg)

    def normalize_angle(self, angle):
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = StanleyControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()