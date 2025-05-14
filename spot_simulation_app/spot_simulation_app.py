#!/usr/bin/env python
"""
| File: standalone_quadruped_app.py
| Description: A standalone Isaac Sim application that sets up a quadruped robot (Spot) in a scene.
|              This application allows controlling the robot with keyboard inputs.
"""

ROBOT_NAME = "spot"

# Define policy and USD file paths - option 1: local file paths
POLICY_PATH = "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_policy.pt"
ENV_PATH = "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_env.yaml"

# Using Omniverse URL for USD
USD_PATH = "omniverse://airlab-storage.andrew.cmu.edu:8443/Projects/AirStack/robots/spot_with_arm/spot_with_arm/spot_free_mass.usd"

# Flag to skip file existence check for Omniverse URLs (set to True for Omniverse URLs)
SKIP_OMNIVERSE_CHECK = True

# Imports to start Isaac Sim from this script
import carb
import numpy as np
import os
from typing import Optional
from isaacsim import SimulationApp
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import asyncio

# Start Isaac Sim's simulation environment (must be instantiated right after the import)
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
from pxr import Usd, UsdGeom, Gf, Tf
import omni.graph.core as og
import omni.graph.tools as ogt

import omni.timeline
import omni.appwindow
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.policy.examples.controllers import PolicyController
from omni.isaac.core import World
from isaacsim.robot.policy.examples.controllers.config_loader import get_robot_joint_properties
from omni.isaac.core.utils.extensions import disable_extension, enable_extension


# Enable/disable ROS bridge extensions to keep only ROS2 Bridge
disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

class ROS2CommandSubscriber(Node):
    """
    ROS2 node to subscribe to Twist messages and convert them to robot commands
    """

    def __init__(self):
        super().__init__('spot_command_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            f'/{ROBOT_NAME}/cmd_vel',  # Standard topic for velocity commands
            self.twist_callback,
            10)  # QoS profile depth

        # Initialize command vector [forward_vel, lateral_vel, yaw_vel]
        self.command = np.array([0.0, 0.0, 0.0])

        # Add timestamp for tracking last message
        self.last_msg_time = time.time()

        # Timestamp for tracking last keyboard input
        self.last_keyboard_time = 0.0

        # Flag to track if keyboard is currently active
        self.keyboard_active = False

        # Timeout in seconds for ROS messages
        self.msg_timeout = 0.5  # 500ms timeout

        # Timeout for keyboard commands (how long to keep accepting keyboard input)
        self.keyboard_timeout = 0.5  # 500ms timeout for keyboard

        # Create a timer that checks for timeouts
        self.timer = self.create_timer(0.1, self.check_timeouts)  # Check every 100ms

        self.get_logger().info('ROS2 Twist command subscriber initialized')

    def twist_callback(self, msg):
        # Only process ROS messages if keyboard isn't active
        if not self.keyboard_active:
            # Extract linearX, linearY, and angularZ from Twist message
            linear_scale = 1.0  # Adjust based on desired max speed
            angular_scale = 1.0  # Adjust based on desired max turn rate

            self.command[0] = msg.linear.x * linear_scale  # Forward velocity
            self.command[1] = msg.linear.y * linear_scale  # Lateral velocity
            self.command[2] = msg.angular.z * angular_scale  # Yaw velocity

            # Apply low-pass filter or limits if needed
            self.command = np.clip(self.command, -2.0, 2.0)  # Limit command range

            # Update the timestamp of the last received message
            self.last_msg_time = time.time()

            self.get_logger().debug(f'Received ROS command: {self.command}')

    def check_timeouts(self):
        """Check for timeouts on both ROS and keyboard inputs"""
        current_time = time.time()

        # First check keyboard timeout
        time_since_keyboard = current_time - self.last_keyboard_time
        if self.keyboard_active and time_since_keyboard > self.keyboard_timeout:
            self.get_logger().info('Keyboard inactive - releasing control')
            self.keyboard_active = False
            # Zero the command when keyboard becomes inactive
            self.command = np.array([0.0, 0.0, 0.0])

        # If keyboard is not active, check ROS timeout
        if not self.keyboard_active:
            time_since_last_msg = current_time - self.last_msg_time
            if time_since_last_msg > self.msg_timeout and not np.array_equal(self.command, np.zeros(3)):
                # If no message received for timeout period and command is not already zero
                self.command = np.array([0.0, 0.0, 0.0])
                self.get_logger().info('ROS command timeout - setting velocity to zero')

    def get_command(self):
        return self.command

    def set_keyboard_command(self, command):
        """Set command from keyboard input"""
        # Update the command
        self.command = command
        # Mark keyboard as active and update timestamp
        self.keyboard_active = True
        self.last_keyboard_time = time.time()


class SpotWithArmFlatTerrainPolicy(PolicyController):
    """The Spot quadruped with arm, using body-only policy"""

    # Define the joint name patterns
    LEG_JOINT_PATTERNS = ["fl_", "fr_", "hl_", "hr_"]
    ARM_JOINT_PATTERNS = ["arm0_"]

    # Correct joint order based on spot policy
    JOINT_ORDER = ['fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn', 'hl_kn',
                   'hr_kn']
    DEFAULT_POSITIONS = [0.1, -0.1, 0.1, -0.1, 0.9, 0.9, 1.1, 1.1, -1.5, -1.5, -1.5, -1.5]

    # Arm joint names and desired positions
    ARM_JOINT_NAMES = ['arm0_sh0', 'arm0_sh1', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']
    ARM_DEFAULT_POSITIONS = [0.0, -3.10843, 3.05258, 0.0, 0.0, 0.0, 0.0]

    def __init__(
            self,
            prim_path: str,
            root_path: Optional[str] = None,
            name: str = "spot_with_arm",
            usd_path: Optional[str] = None,
            position: Optional[np.ndarray] = None,
            orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.
        """
        if usd_path is None:
            # Use the default USD path defined at the top of the file
            usd_path = USD_PATH
            carb.log_info(f"Using default USD path: {usd_path}")

        try:
            # Call the base PolicyController initialization
            super().__init__(name, prim_path, root_path, usd_path, position, orientation)

            # Set decimation and timing parameters - will be overwritten when policy is loaded
            self._decimation = 4
            self._dt = 0.02

            # Action-related settings
            self._action_scale = 0.2
            self._previous_action = np.zeros(12)
            self._policy_counter = 0

            # These will be identified during initialize
            self.default_pos = np.array(self.DEFAULT_POSITIONS)  # Use known default positions
            self._full_default_pos = None
            self._leg_joint_indices = []
            self._arm_joint_indices = []
            self._joint_mapping = {}  # Maps policy joint index to robot joint index
            self._arm_joint_mapping = {}  # Maps arm joint name to robot joint index

            # Add stage event subscription for Action Graph handling
            self._stage_event_sub = None



            carb.log_info(f"Successfully created SpotWithArmFlatTerrainPolicy object for {prim_path}")
        except Exception as e:
            carb.log_error(f"Error initializing SpotWithArmFlatTerrainPolicy: {str(e)}")
            raise

    def initialize(self, **kwargs):
        """
        Initialize the robot and identify leg and arm joints
        """
        # Initialize the robot
        self.robot.initialize(physics_sim_view=kwargs.get("physics_sim_view", None))
        self.robot.get_articulation_controller().set_effort_modes(kwargs.get("effort_modes", "force"))
        self.robot.get_articulation_controller().switch_control_mode(kwargs.get("control_mode", "position"))

        # Print joint names for debugging
        print("Spot with Arm Joint Names:", self.robot.dof_names)

        # Identify joint indices by name pattern and create mapping
        self._leg_joint_indices = []
        self._arm_joint_indices = []
        self._joint_mapping = {}
        self._arm_joint_mapping = {}

        # Create mapping from policy joint indices to robot joint indices
        for policy_idx, joint_name in enumerate(self.JOINT_ORDER):
            found = False
            for robot_idx, robot_joint_name in enumerate(self.robot.dof_names):
                if joint_name in robot_joint_name:
                    self._joint_mapping[policy_idx] = robot_idx
                    self._leg_joint_indices.append(robot_idx)
                    found = True
                    break
            if not found:
                print(f"WARNING: Could not find joint {joint_name} in robot")

        # Create mapping for arm joints
        for i, joint_name in enumerate(self.ARM_JOINT_NAMES):
            found = False
            for robot_idx, robot_joint_name in enumerate(self.robot.dof_names):
                if joint_name in robot_joint_name:
                    self._arm_joint_mapping[joint_name] = robot_idx
                    self._arm_joint_indices.append(robot_idx)
                    found = True
                    break
            if not found:
                print(f"WARNING: Could not find arm joint {joint_name} in robot")

        print(f"Identified {len(self._leg_joint_indices)} leg joints: {self._leg_joint_indices}")
        print(f"Leg joint mapping: {self._joint_mapping}")
        print(f"Identified {len(self._arm_joint_indices)} arm joints: {self._arm_joint_indices}")
        print(f"Arm joint mapping: {self._arm_joint_mapping}")

        # Get properties from the policy environment
        max_effort, max_vel, stiffness, damping, policy_default_pos, default_vel = get_robot_joint_properties(
            self.policy_env_params, self.robot.dof_names
        )

        # Get number of joints using the num_dof property
        num_joints = self.robot.num_dof

        # Create properly sized arrays for the full robot
        full_stiffness = np.ones(num_joints) * 20.0
        full_damping = np.ones(num_joints) * 0.5
        full_max_effort = np.ones(num_joints) * 100.0
        full_max_vel = np.ones(num_joints) * 100.0

        # Apply the policy values to the leg joints using the mapping
        for policy_idx, robot_idx in self._joint_mapping.items():
            if policy_idx < len(stiffness) and robot_idx < num_joints:
                full_stiffness[robot_idx] = stiffness[policy_idx]
                full_damping[robot_idx] = damping[policy_idx]
                full_max_effort[robot_idx] = max_effort[policy_idx]
                full_max_vel[robot_idx] = max_vel[policy_idx]

        # Increase stiffness for arm joints for more stability
        for idx in self._arm_joint_indices:
            if idx < num_joints:
                full_stiffness[idx] = 100.0
                full_damping[idx] = 5.0

        # Set the properly sized arrays
        self.robot._articulation_view.set_gains(full_stiffness, full_damping)
        self.robot._articulation_view.set_max_efforts(full_max_effort)
        self.robot._articulation_view.set_max_joint_velocities(full_max_vel)

        # Set articulation properties if requested
        if kwargs.get("set_articulation_props", True):
            self._set_articulation_props()

        # Get the current joint positions and store them
        self._full_default_pos = self.robot.get_joint_positions().copy()

        # Set leg joints to their default positions
        for policy_idx, robot_idx in self._joint_mapping.items():
            if policy_idx < len(self.default_pos) and robot_idx < len(self._full_default_pos):
                self._full_default_pos[robot_idx] = self.default_pos[policy_idx]

        # Set arm joints to their default positions
        for i, joint_name in enumerate(self.ARM_JOINT_NAMES):
            if joint_name in self._arm_joint_mapping and i < len(self.ARM_DEFAULT_POSITIONS):
                robot_idx = self._arm_joint_mapping[joint_name]
                if robot_idx < len(self._full_default_pos):
                    self._full_default_pos[robot_idx] = self.ARM_DEFAULT_POSITIONS[i]

        # Apply the default positions
        action = ArticulationAction(joint_positions=self._full_default_pos)
        self.robot.apply_action(action)

        print(f"Initialization complete. Robot has {num_joints} joints.")
        print(f"Full default positions: {self._full_default_pos}")

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy
        """
        # Initialize if needed
        if self._full_default_pos is None:
            self.initialize()
            return np.zeros(48)  # Return zeros during initialization

        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command

        # Joint states - get all joint positions and velocities
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()

        # Create policy observation with mapped joints
        leg_positions = np.zeros(12)
        leg_velocities = np.zeros(12)

        # Map robot joint values to policy joint order
        for policy_idx, robot_idx in self._joint_mapping.items():
            if policy_idx < 12 and robot_idx < len(current_joint_pos):
                leg_positions[policy_idx] = current_joint_pos[robot_idx]
                leg_velocities[policy_idx] = current_joint_vel[robot_idx]

        # Use only the leg joints for the policy
        obs[12:24] = leg_positions - self.default_pos
        obs[24:36] = leg_velocities

        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired joint positions and apply them to the articulation
        """
        # Initialize if needed
        if self._full_default_pos is None:
            self.initialize()
            return

        # Only compute new action at decimated rate
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # Create a full joint position array for all joints
        full_joint_positions = self._full_default_pos.copy()

        # Update leg joint positions using the mapping
        for policy_idx, robot_idx in self._joint_mapping.items():
            if policy_idx < len(self.action) and robot_idx < len(full_joint_positions):
                full_joint_positions[robot_idx] = self.default_pos[policy_idx] + (
                        self.action[policy_idx] * self._action_scale)

        # Keep arm in fixed position using the arm joint mapping
        for i, joint_name in enumerate(self.ARM_JOINT_NAMES):
            if joint_name in self._arm_joint_mapping and i < len(self.ARM_DEFAULT_POSITIONS):
                robot_idx = self._arm_joint_mapping[joint_name]
                if robot_idx < len(full_joint_positions):
                    full_joint_positions[robot_idx] = self.ARM_DEFAULT_POSITIONS[i]

        # Apply the action to the full articulation
        action = ArticulationAction(joint_positions=full_joint_positions)
        self.robot.apply_action(action)

        self._policy_counter += 1

    def post_reset(self):
        """
        Reset the robot's state
        """
        self.robot.post_reset()

        # Reset our internal state to force re-initialization
        self._full_default_pos = None
        self._leg_joint_indices = []
        self._arm_joint_indices = []
        self._joint_mapping = {}
        self._arm_joint_mapping = {}
        self._policy_counter = 0

        # Re-initialize
        self.initialize()


class StandaloneQuadrupedApp:
    """
    A standalone application for running a quadruped robot simulation.
    """

    def __init__(self):
        """
        Method that initializes the app and sets up the simulation environment.
        """
        # Initialize ROS2
        rclpy.init()
        self.ros2_node = ROS2CommandSubscriber()

        # Define paths to policy and USD files
        self.policy_path = POLICY_PATH
        self.env_path = ENV_PATH
        self.usd_path = USD_PATH

        carb.log_info(f"Using Policy Path: {self.policy_path}")
        carb.log_info(f"Using Env Path: {self.env_path}")
        carb.log_info(f"Using USD Path: {self.usd_path}")

        # Verify files exist if needed
        if not SKIP_OMNIVERSE_CHECK:
            self._check_files_exist()
        else:
            carb.log_info("Skipping file existence check for Omniverse URLs")

        # World settings
        self._world_settings = {
            "stage_units_in_meters": 1.0,
            "physics_dt": 1.0 / 500.0,
            "rendering_dt": 10.0 / 500.0
        }

        # Initialize command vector [forward_vel, lateral_vel, yaw_vel]
        self._base_command = np.array([0.0, 0.0, 0.0])

        # Keyboard mappings for robot control
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [2.0, 0.0, 0.0],
            "UP": [2.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-2.0, 0.0, 0.0],
            "DOWN": [-2.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -2.0, 0.0],
            "RIGHT": [0.0, -2.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 2.0, 0.0],
            "LEFT": [0.0, 2.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 2.0],
            "N": [0.0, 0.0, 2.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -2.0],
            "M": [0.0, 0.0, -2.0],
        }

        # Acquire the timeline for simulation control
        self.timeline = omni.timeline.get_timeline_interface()

        # Create the simulation world with the specified settings
        self.world = World(
            physics_dt=self._world_settings["physics_dt"],
            rendering_dt=self._world_settings["rendering_dt"],
            stage_units_in_meters=self._world_settings["stage_units_in_meters"]
        )

        # Setup scene - ground plane and robot
        self.setup_scene()

        # Setup input callbacks for keyboard control
        self.setup_callbacks()

        # Physics flag to track initialization
        self._physics_ready = False
        self._action_graphs_initialized = False

        # Flag to track simulation state
        self.stop_sim = False

        # Action graph related variables
        self._action_graph = None
        self._on_tick_nodes = []  # Store all OnTick/OnPlaybackTick nodes

        # Reset the simulation environment
        self.world.reset()

    def setup_scene(self):
        """
        Sets up the simulation scene with ground plane and robot.
        """
        # Add ground plane
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )

        # For Omniverse URLs, use a more reliable approach
        if self.usd_path.startswith("omniverse://"):
            carb.log_info(f"Loading Spot robot from Omniverse URL: {self.usd_path}")

            # Create the robot prim path
            robot_prim_path = "/World/Spot"

            # Directly create a reference to the USD file
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            from pxr import Usd, UsdGeom, Gf

            # Create the Xform for the robot if it doesn't exist
            if not stage.GetPrimAtPath(robot_prim_path).IsValid():
                robot_prim = UsdGeom.Xform.Define(stage, robot_prim_path)

                # Add the reference to the USD file
                robot_prim.GetPrim().GetReferences().AddReference(self.usd_path)

                # Set the initial position using a transform matrix
                xform = UsdGeom.Xformable(robot_prim)

                # Check if we need to add a transform op or if one already exists
                try:
                    # Try to set translation through the API safely
                    if not xform.GetXformOpOrderAttr().HasAuthoredValue():
                        # Only add if no transform ops exist yet
                        translate_op = xform.AddTranslateOp()
                        translate_op.Set(Gf.Vec3d(0, 0, 0.8))
                    else:
                        carb.log_info("Transform ops already exist for the robot prim")
                except Exception as e:
                    carb.log_warn(f"Failed to set transform for robot: {str(e)}")
                    carb.log_info("Will rely on position parameter from controller instead")
            else:
                carb.log_info(f"Robot prim already exists at {robot_prim_path}")

            # Now create the robot controller
            self.spot = SpotWithArmFlatTerrainPolicy(
                prim_path=robot_prim_path,
                name="Spot",
                position=np.array([0, 0, 0.8]),
            )

            carb.log_info("Created Spot robot controller for Omniverse USD")
        else:
            # Local file approach
            try:
                carb.log_info(f"Attempting to load Spot robot from: {self.usd_path}")
                self.spot = SpotWithArmFlatTerrainPolicy(
                    prim_path="/World/Spot",
                    name="Spot",
                    usd_path=self.usd_path,
                    position=np.array([0, 0, 0.8]),
                )

                carb.log_info("Successfully created Spot robot instance")
            except Exception as e:
                # If the specified USD fails, try to use a default USD from Isaac Sim
                carb.log_error(f"Failed to load robot with error: {str(e)}")
                carb.log_info("Attempting to load default Spot robot from Omniverse...")

                default_usd = "omniverse://localhost/Isaac/Robots/BostonDynamics/spot/spot.usd"
                carb.log_info(f"Using fallback USD path: {default_usd}")

                self.spot = SpotWithArmFlatTerrainPolicy(
                    prim_path="/World/Spot",
                    name="Spot",
                    usd_path=default_usd,
                    position=np.array([0, 0, 0.8]),
                )
                carb.log_info("Successfully created Spot robot instance with default USD")

        # Subscribe to timeline events to handle simulation stop
        self._event_timer_callback = self.timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._timeline_timer_callback_fn
        )

    def _ensure_robot_loaded(self):
        """
        Ensure the robot is fully loaded before trying to access Action Graphs
        """
        try:
            # Wait for a few physics steps before trying Action Graphs
            if not hasattr(self, '_robot_load_counter'):
                self._robot_load_counter = 0

            self._robot_load_counter += 1

            # Only try after robot has been initialized
            if self._robot_load_counter > 10 and self.spot and self.spot.robot:
                return True

            return False
        except:
            return False

    def setup_callbacks(self):
        """
        Sets up input and physics callbacks.
        """
        # Set up keyboard input
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

        # Add physics step callback
        self.world.add_physics_callback("physics_step", self.on_physics_step)

    def _initialize_ros2_bridge(self):
        """
        Initialize ROS2 bridge if needed
        """
        try:
            carb.log_info("Attempting to initialize ROS2 bridge...")

            # Import and enable the ROS2 bridge extension
            import omni.kit.app
            manager = omni.kit.app.get_app().get_extension_manager()

            # Enable the extension if it's not already enabled
            if not manager.is_extension_enabled("omni.isaac.ros2_bridge"):
                carb.log_info("Enabling ROS2 bridge extension...")
                manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
                time.sleep(1.0)  # Give it time to load

            # Import after enabling
            import omni.isaac.ros2_bridge

            # Initialize if the method exists
            if hasattr(omni.isaac.ros2_bridge, 'initialize'):
                result = omni.isaac.ros2_bridge.initialize()
                carb.log_info(f"ROS2 bridge initialization result: {result}")
            else:
                carb.log_info("ROS2 bridge initialize method not found")

        except Exception as e:
            carb.log_error(f"Error initializing ROS2 bridge: {e}")

    def _find_and_initialize_action_graph(self):
        """
        Find and properly initialize the ROS2_Camera_and_TFs Action Graph using forum solution
        """
        carb.log_info("Searching for ROS2_Camera_and_TFs Action Graph...")

        try:
            # Potential paths for the action graph
            graph_paths = [
                "/World/Spot/ROS2_Camera_and_TFs",
                "/World/Spot/body/ROS2_Camera_and_TFs"
            ]

            found_graph = None
            for path in graph_paths:
                try:
                    graph = og.get_graph_by_path(path)
                    if graph:
                        carb.log_info(f"Found Action Graph at: {path}")
                        found_graph = graph
                        self._action_graph = graph
                        break
                except:
                    continue

            if not found_graph:
                # Search all graphs
                all_graphs = og.get_all_graphs()
                carb.log_info(f"Searching through {len(all_graphs)} graphs...")

                for graph in all_graphs:
                    try:
                        graph_path = graph.get_path_to_graph()
                        if "ROS2_Camera_and_TFs" in graph_path or (
                                "ROS2" in graph_path and "/World/Spot" in graph_path):
                            carb.log_info(f"Found matching graph: {graph_path}")
                            found_graph = graph
                            self._action_graph = graph
                            break
                    except:
                        continue

            if found_graph:
                # Find all OnTick/OnPlaybackTick nodes in the graph
                nodes = found_graph.get_nodes()
                self._on_tick_nodes = []

                for node in nodes:
                    try:
                        node_type = node.get_type_name()
                        node_path = node.get_path()
                        carb.log_info(f"Found node: {node_type} at {node_path}")

                        # Look for OnTick or OnPlaybackTick nodes
                        if "OnPlaybackTick" in node_type or "OnTick" in node_type:
                            self._on_tick_nodes.append(node)
                            carb.log_info(f"Added OnTick node: {node_type} at {node_path}")
                    except:
                        continue

                # Enable the graph
                try:
                    if hasattr(found_graph, 'enable'):
                        found_graph.enable()
                        carb.log_info("Enabled graph using enable()")

                    if hasattr(found_graph, 'set_enabled'):
                        found_graph.set_enabled(True)
                        carb.log_info("Enabled graph using set_enabled()")

                    # Remove the evaluator code that's causing errors
                    # No evaluator configuration needed

                except Exception as e:
                    carb.log_error(f"Error configuring graph: {e}")

                # Mark as initialized
                self._action_graphs_initialized = True
                carb.log_info(f"Action Graph successfully initialized with {len(self._on_tick_nodes)} tick nodes!")
                return True
            else:
                carb.log_warn("No ROS2_Camera_and_TFs Action Graph found!")
                return False

        except Exception as e:
            carb.log_error(f"Error finding action graph: {e}")
            return False

    def _trigger_action_graph_impulse(self):
        """
        Manually trigger the action graph using impulse events as suggested in the forum
        """
        try:
            if self._action_graph and self._on_tick_nodes:
                # Use Controller to set impulse on OnTick nodes
                controller = og.Controller()

                for node in self._on_tick_nodes:
                    try:
                        node_path = node.get_path()
                        # Set enableImpulse to True using the Controller (forum solution)
                        impulse_attr = f"{node_path}.state:enableImpulse"
                        controller.set(impulse_attr, True)
                        carb.log_debug(f"Triggered impulse on: {node_path}")
                    except Exception as e:
                        carb.log_debug(f"Failed to trigger node {node_path}: {e}")
                        continue

        except Exception as e:
            carb.log_error(f"Error triggering action graph impulse: {e}")

    def _check_files_exist(self):
        """
        Verify that required files exist.
        """
        if SKIP_OMNIVERSE_CHECK:
            carb.log_info("Skipping file existence check for Omniverse URLs")
            return

        # Only check local files, not Omniverse URLs
        files_to_check = []
        for file_path, file_desc in [
            (self.policy_path, "Policy file"),
            (self.env_path, "Environment config file"),
            (self.usd_path, "USD file")
        ]:
            # Skip checking Omniverse URLs
            if not file_path.startswith("omniverse://"):
                files_to_check.append((file_path, file_desc))
            else:
                carb.log_info(f"Using Omniverse URL for {file_desc}: {file_path}")

        for file_path, file_desc in files_to_check:
            if not os.path.exists(file_path):
                carb.log_error(f"{file_desc} not found: {file_path}")
                carb.log_info(f"Please verify that the path {file_path} is correct and accessible.")
                carb.log_info("If using Omniverse URLs, set SKIP_OMNIVERSE_CHECK = True at the top of the file.")

                raise FileNotFoundError(f"{file_desc} not found: {file_path}")

    def on_physics_step(self, step_size):
        """
        Physics step callback - updates robot state and ensures Action Graphs run.
        """
        # Process ROS2 callbacks
        rclpy.spin_once(self.ros2_node, timeout_sec=0)

        # Always get the command from the ROS node
        self._base_command = self.ros2_node.get_command()

        # Initialize physics counter if it doesn't exist
        if not hasattr(self, '_physics_counter'):
            self._physics_counter = 0

        # Special handling for first few physics steps
        if self._physics_counter < 10:
            carb.log_info(f"Physics step {self._physics_counter + 1}/10 - Initializing...")

        # Action Graph initialization sequence
        if not self._action_graphs_initialized:
            # First attempt after robot is loaded
            if self._physics_counter == 10 and self._ensure_robot_loaded():
                carb.log_info("First attempt to initialize Action Graphs...")
                # Initialize ROS2 bridge first
                self._initialize_ros2_bridge()
                success = self._find_and_initialize_action_graph()
                if success:
                    carb.log_info("Action Graphs successfully initialized!")

            # Second attempt if first failed
            elif self._physics_counter == 30:
                carb.log_info("Second attempt to initialize Action Graphs...")
                success = self._find_and_initialize_action_graph()
                if success:
                    carb.log_info("Action Graphs successfully initialized!")

            # Third attempt if previous failed
            elif self._physics_counter == 60:
                carb.log_info("Third attempt to initialize Action Graphs...")
                success = self._find_and_initialize_action_graph()
                if not success:
                    carb.log_warn("Failed to initialize Action Graphs after multiple attempts")
                    # Set flag to prevent further attempts
                    self._action_graphs_initialized = True

        # Trigger action graph using the forum solution (KEY CHANGE!)
        if self._action_graphs_initialized and self._action_graph:
            # Use the impulse method from the forum solution
            self._trigger_action_graph_impulse()

        if self._physics_ready:
            # Forward command to robot
            self.spot.forward(step_size, self._base_command)
        else:
            # Initialize robot on first physics step
            self._physics_ready = True

            try:
                # Load and initialize the robot
                carb.log_info(f"Loading policy from: {self.policy_path}")
                self.spot.load_policy(self.policy_path, self.env_path)
                self.spot.initialize()
                self.spot.post_reset()

                # Create full default state for all joints
                full_default_state = np.zeros(self.spot.robot.num_dof)

                # Set leg joints using the mapping
                for policy_idx, robot_idx in self.spot._joint_mapping.items():
                    if policy_idx < len(self.spot.default_pos) and robot_idx < len(full_default_state):
                        full_default_state[robot_idx] = self.spot.default_pos[policy_idx]

                # Set arm joints using the mapping
                for i, joint_name in enumerate(self.spot.ARM_JOINT_NAMES):
                    if joint_name in self.spot._arm_joint_mapping and i < len(self.spot.ARM_DEFAULT_POSITIONS):
                        robot_idx = self.spot._arm_joint_mapping[joint_name]
                        if robot_idx < len(full_default_state):
                            full_default_state[robot_idx] = self.spot.ARM_DEFAULT_POSITIONS[i]

                # Set the default state with all joint positions
                self.spot.robot.set_joints_default_state(full_default_state)
                carb.log_info("Robot initialized successfully")

            except Exception as e:
                carb.log_error(f"Failed to initialize robot: {str(e)}")
                self._physics_ready = False

        self._physics_counter += 1

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """
        Keyboard event callback - handles keyboard input.

        Args:
            event: The keyboard event.

        Returns:
            bool: True to continue processing events.
        """
        # Handle exit key (ESC)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "ESCAPE":
            carb.log_info("ESC pressed - exiting simulation")
            self.exit_simulation()
            return True

        # When a key is pressed or released, adjust the command
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # On pressing, increment the command
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
                carb.log_info(f"Keyboard command: {self._base_command}")

                # Update the command in ROS node
                self.ros2_node.set_keyboard_command(self._base_command)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # On release, decrement the command
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
                carb.log_info(f"Keyboard command: {self._base_command}")

                # Update the command in ROS node
                self.ros2_node.set_keyboard_command(self._base_command)

        return True

    def _timeline_timer_callback_fn(self, event):
        """
        Timeline event callback - handles simulation stop events.
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            carb.log_info("Simulation stopped - preparing to reset...")

            # Reset physics-ready flag so initialization happens again
            self._physics_ready = False
            self._action_graphs_initialized = False

            # Reset action graph references
            self._action_graph = None
            self._on_tick_nodes = []

            # Reset the simulation
            self.reset_simulation()

            # Schedule a delayed restart (give time for cleanup)
            self._schedule_restart()

    def _schedule_restart(self):
        """
        Schedule a delayed restart of the timeline using asyncio
        """

        async def delayed_restart():
            # Wait a short time for complete stop
            await asyncio.sleep(0.5)

            # Restart the timeline
            carb.log_info("Restarting simulation...")
            self.timeline.play()

        # Use asyncio directly
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(delayed_restart())
            else:
                loop.run_until_complete(delayed_restart())
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(delayed_restart())

    def reset_simulation(self):
        """
        Reset the simulation to initial state without shutting down.
        """
        # Reset the base command
        self._base_command = np.array([0.0, 0.0, 0.0])

        # Reset the world state
        self.world.reset()

        # Reset the robot
        if self.spot:
            self.spot.post_reset()

        carb.log_info("Simulation has been reset")

    def run(self):
        """
        Main application loop that runs the simulation.
        """
        carb.log_info("\n" + "-" * 80)
        carb.log_info("Quadruped Simulation Started")
        carb.log_info("Keyboard Controls:")
        carb.log_info("  Forward: UP arrow or NUMPAD 8")
        carb.log_info("  Backward: DOWN arrow or NUMPAD 2")
        carb.log_info("  Strafe Left: LEFT arrow or NUMPAD 4")
        carb.log_info("  Strafe Right: RIGHT arrow or NUMPAD 6")
        carb.log_info("  Turn Left: N or NUMPAD 7")
        carb.log_info("  Turn Right: M or NUMPAD 9")
        carb.log_info("  Stop/Reset: ESC")
        carb.log_info("-" * 80 + "\n")

        # Start the simulation
        self.timeline.play()

        # Wait a bit for everything to load properly
        carb.log_info("Waiting for simulation to initialize...")
        time.sleep(2.0)
        carb.log_info("Starting main loop")

        # The main loop - only exit if the application is closing
        while simulation_app.is_running():
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

            # Check if manual exit requested
            if self.stop_sim:
                break

        # Cleanup and stop
        carb.log_warn("Quadruped Simulation App is closing.")
        self.timeline.stop()

        # Clean up ROS2
        self.ros2_node.destroy_node()
        rclpy.shutdown()

        # Clean up physics callback
        if self.world.physics_callback_exists("physics_step"):
            self.world.remove_physics_callback("physics_step")

        # Clean up timeline callback
        self._event_timer_callback = None

        # Close the application
        simulation_app.close()

    def exit_simulation(self):
        """
        Method to properly exit the simulation.
        """
        self.stop_sim = True


def main():
    # Instantiate the app
    try:
        app = StandaloneQuadrupedApp()
        # Run the application loop
        app.run()
    except Exception as e:
        carb.log_error(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        # Ensure we close the simulation app even if there's an error
        simulation_app.close()


if __name__ == "__main__":
    main()


