#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose, Twist, Vector3
from std_msgs.msg import Float32MultiArray, String
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.geometry import EulerZXY
import time
import threading
import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpotIsaacAdapter(Node):
    """
    Adapter to convert Boston Dynamics SDK commands to ROS2 messages for Isaac Sim

    This adapter mocks the RobotCommandClient for Spot and publishes ROS2 messages
    that can be consumed by an Isaac Sim script to move a simulated Spot robot.
    """

    # Default topic configurations
    DEFAULT_CONFIG = {
        'topics': {
            'trajectory': '/spot/cmd_trajectory',
            'velocity': '/spot/cmd_velocity',
            'stand': '/spot/cmd_stand',
            'sit': '/spot/cmd_sit',
            'arm_pose': '/spot/cmd_arm_pose',
            'gripper': '/spot/cmd_gripper',
            'arm_joints': '/spot/cmd_arm_joints'
        },
        'qos': {
            'depth': 10,
            'reliability': 'reliable'  # Options: 'reliable', 'best_effort'
        }
    }

    def __init__(self, config_file=None):
        # Get robot namespace from environment variable
        self.robot_name = os.environ.get('ROBOT_NAME', 'spot')

        # Initialize the node with namespace from robot name
        super().__init__(f'{self.robot_name}_isaac_adapter')

        # Load configuration
        self.config = self.load_config(config_file)

        # Apply namespace to topics
        self.topics = self.apply_namespace_to_topics(self.config['topics'])

        # Get QoS settings
        qos_depth = self.config['qos'].get('depth', 10)
        qos_reliability = self.config['qos'].get('reliability', 'reliable')

        # Set QoS profile
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=qos_depth)
        if qos_reliability == 'reliable':
            qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Create ROS2 publishers with configured topics and QoS
        self.trajectory_pub = self.create_publisher(Pose, self.topics['trajectory'], qos)
        self.velocity_pub = self.create_publisher(Twist, self.topics['velocity'], qos)
        self.stand_pub = self.create_publisher(String, self.topics['stand'], qos)
        self.sit_pub = self.create_publisher(String, self.topics['sit'], qos)
        self.arm_pose_pub = self.create_publisher(Pose, self.topics['arm_pose'], qos)
        self.gripper_pub = self.create_publisher(Float32MultiArray, self.topics['gripper'], qos)
        self.arm_joint_pub = self.create_publisher(Float32MultiArray, self.topics['arm_joints'], qos)

        # Command IDs (incremented for each command)
        self.command_id = 0

        # Lock for thread safety
        self.lock = threading.Lock()

        # Mock feedback responses for different command types
        self.feedback_responses = {}

        self.get_logger().info(f'{self.robot_name.upper()} Isaac Adapter started with configuration:')
        self.get_logger().info(f'Topics: {self.topics}')

    def load_config(self, config_file):
        """Load configuration from file or use defaults"""
        config = self.DEFAULT_CONFIG.copy()

        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)

                # Update configuration with file values
                if 'topics' in file_config:
                    config['topics'].update(file_config['topics'])
                if 'qos' in file_config:
                    config['qos'].update(file_config['qos'])

                self.get_logger().info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.get_logger().error(f"Error loading config file: {e}")
                self.get_logger().warning("Using default configuration")
        else:
            self.get_logger().info("No config file provided, using default configuration")

        return config

    def apply_namespace_to_topics(self, topics):
        """Apply robot namespace to all topics"""
        namespaced_topics = {}

        for key, topic in topics.items():
            # Skip topics that already have a namespace
            if topic.startswith('/'):
                # Replace the first part of the topic path with the robot_name
                parts = topic.split('/')
                if len(parts) > 2:  # Has at least one part after the leading '/'
                    parts[1] = self.robot_name
                    namespaced_topics[key] = '/'.join(parts)
                else:
                    # If the topic is just '/', add the robot name
                    namespaced_topics[key] = f'/{self.robot_name}{topic}'
            else:
                # For relative topics, add the namespace
                namespaced_topics[key] = f'/{self.robot_name}/{topic}'

        return namespaced_topics