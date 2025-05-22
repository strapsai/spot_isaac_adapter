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

# USD / Pixar
from pxr import Usd, UsdGeom, Gf, Tf, Sdf

# Basic Omni and Isaac imports
import omni.graph.core as og
import omni.graph.tools as ogt
import omni.timeline
import omni.appwindow

# Core Isaac Sim functionality
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.robot.policy.examples.controllers.config_loader import get_robot_joint_properties
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.extensions import disable_extension, enable_extension

# Other utilities
import omni.client
import omni.kit.commands
from omni.usd import get_stage_next_free_path
from omni.isaac.nucleus import get_assets_root_path
from scipy.spatial.transform import Rotation


# Enable the required extensions for the simulation
EXTENSIONS_PEOPLE = [
    'omni.anim.graph.core',  # Load core first
    'omni.anim.graph.bundle',  # Then bundle
    'omni.anim.curve.core',  # Curve support
    'omni.anim.timeline',  # Timeline support
    'omni.anim.retarget.core',  # Retargeting core
    'omni.anim.people',  # People (loads schemas)
    'omni.anim.navigation.bundle',  # Navigation
    'omni.anim.graph.ui',  # UI components
    'omni.anim.retarget.bundle',  # Retargeting bundle
    'omni.anim.retarget.ui',  # Retargeting UI
    'omni.kit.scripting',  # Scripting support
    'omni.graph.io',  # Graph I/O
]

# Enable people extensions with proper error handling and delays
carb.log_info("Enabling people extensions...")
for i, ext_people in enumerate(EXTENSIONS_PEOPLE):
    try:
        carb.log_info(f"Enabling extension {i + 1}/{len(EXTENSIONS_PEOPLE)}: {ext_people}")
        enable_extension(ext_people)

        # Small delay between extensions to ensure proper loading
        if i < len(EXTENSIONS_PEOPLE) - 1:
            time.sleep(0.2)

        carb.log_info(f"Successfully enabled {ext_people}")
    except Exception as e:
        carb.log_error(f"Failed to enable {ext_people}: {e}")

# Enable/disable ROS bridge extensions
disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

# NOW import the animation modules after extensions are enabled
try:
    import omni.anim.graph.core as ag
    from omni.anim.people import PeopleSettings
    carb.log_info("Successfully imported animation modules")
except ImportError as e:
    carb.log_error(f"Failed to import animation modules: {e}")
    # Set a flag to disable people functionality if animation import fails
    ANIMATION_AVAILABLE = False
else:
    ANIMATION_AVAILABLE = True


class StandalonePersonController:
    """Base controller class for person behavior"""

    def __init__(self):
        self._person = None

    @property
    def person(self):
        return self._person

    def initialize(self, person):
        self._person = person

    def update_state(self, state):
        pass

    def update(self, dt: float):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass


class RandomWalkPersonController(StandalonePersonController):
    """Controller that makes a person walk randomly"""

    def __init__(self):
        super().__init__()
        self._current_direction = np.random.uniform(0, 2 * np.pi)
        self._speed = 1.0
        self._last_direction_change = 0.0
        self._direction_change_interval = np.random.uniform(2.0, 5.0)
        self._position = np.array([0.0, 0.0, 0.0])

    def update(self, dt: float):
        if not self._person:
            return

        # Update timer
        self._last_direction_change += dt

        # Occasionally change direction
        if self._last_direction_change >= self._direction_change_interval:
            if np.random.random() < 0.2:
                self._current_direction += np.random.uniform(-np.pi / 3, np.pi / 3)

            self._last_direction_change = 0.0
            self._direction_change_interval = np.random.uniform(2.0, 5.0)

        # Calculate new position
        dx = self._speed * np.cos(self._current_direction) * dt
        dy = self._speed * np.sin(self._current_direction) * dt

        self._position[0] += dx
        self._position[1] += dy

        # Update the person's target position
        self._person.update_target_position(self._position)


class PersonState:
    """Simple state class for person"""

    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])


class StandalonePerson:
    """Standalone person implementation that doesn't depend on Pegasus"""

    # Get root assets path from settings
    setting_dict = carb.settings.get_settings()
    people_asset_folder = setting_dict.get(PeopleSettings.CHARACTER_ASSETS_PATH)
    character_root_prim_path = setting_dict.get(PeopleSettings.CHARACTER_PRIM_PATH)
    assets_root_path = None

    if not character_root_prim_path:
        character_root_prim_path = "/World/Characters"

    if people_asset_folder:
        assets_root_path = people_asset_folder
    else:
        try:
            root_path = get_assets_root_path()
            if root_path is not None:
                assets_root_path = "{}/Isaac/People/Characters".format(root_path)
            else:
                assets_root_path = None
                carb.log_warn("Assets root path is None, people functionality will be limited")
        except RuntimeError:
            carb.log_warn("Could not find assets root folder, people functionality will be limited")
            assets_root_path = None


    def __init__(
            self,
            world,
            stage_prefix: str,
            character_name: str = None,
            init_pos=[0.0, 0.0, 0.0],
            init_yaw=0.0,
            controller: StandalonePersonController = None
    ):
        """Initialize the person object"""

        # Reference to world and stage
        self._world = world
        self._current_stage = self._world.stage

        # State management
        self._state = PersonState()
        self._state.position = np.array(init_pos)
        self._state.orientation = Rotation.from_euler('z', init_yaw, degrees=False).as_quat()

        # Target for movement
        self._target_position = np.array(init_pos)
        self._target_speed = 1.0

        # Get unique stage path
        self._stage_prefix = get_stage_next_free_path(
            self._current_stage,
            StandalonePerson.character_root_prim_path + '/' + stage_prefix,
            False
        )

        # Character assets
        self._character_name = character_name
        self.char_usd_file = self.get_path_for_character_prim(character_name)

        # Initialize simulation flags
        self._sim_running = False
        self.character_graph = None
        self.character_skel_root = None
        self.character_skel_root_stage_path = None

        # Spawn the person
        self.spawn_agent(self.char_usd_file, self._stage_prefix, init_pos, init_yaw)
        self.add_animation_graph_to_agent()

        # Add callbacks
        self._world.add_physics_callback(self._stage_prefix + "/state", self.update_state)
        self._world.add_physics_callback(self._stage_prefix + "/update", self.update)
        self._world.add_timeline_callback(self._stage_prefix + "/start_stop_sim", self.sim_start_stop)

    @property
    def state(self):
        return self._state

    def sim_start_stop(self, event):
        """Handle simulation start/stop events"""
        if self._world.is_playing() and not self._sim_running:
            self._sim_running = True
            self.start()
        elif self._world.is_stopped() and self._sim_running:
            self._sim_running = False
            self.stop()

    def start(self):
        if self._controller:
            self._controller.start()

    def stop(self):
        if self._controller:
            self._controller.stop()

    def update(self, dt: float):
        """Update person movement and animation"""

        # Get character graph with retry logic
        if not self.character_graph:
            self.character_graph = ag.get_character(self.character_skel_root_stage_path)
            if not self.character_graph:
                # Try alternative method if first fails
                try:
                    # Sometimes the stage path needs to be the skeleton root path
                    skel_path = str(self.character_skel_root.GetPrimPath())
                    self.character_graph = ag.get_character(skel_path)
                except:
                    pass

        # Only proceed if we have a valid graph
        if not self.character_graph:
            carb.log_info(f"Character graph not ready yet for {self._stage_prefix}")
            return

        # Update controller
        if self._controller:
            self._controller.update(dt)

        # Calculate distance to target
        distance_to_target = np.linalg.norm(self._target_position - self._state.position)

        # Set animation based on distance to target
        try:
            if distance_to_target > 0.1:
                self.character_graph.set_variable("Action", "Walk")

                # Convert numpy arrays to carb.Float3 properly
                current_pos = carb.Float3(float(self._state.position[0]),
                                          float(self._state.position[1]),
                                          float(self._state.position[2]))
                target_pos = carb.Float3(float(self._target_position[0]),
                                         float(self._target_position[1]),
                                         float(self._target_position[2]))

                self.character_graph.set_variable("PathPoints", [current_pos, target_pos])
                self.character_graph.set_variable("Walk", float(self._target_speed))
            else:
                self.character_graph.set_variable("Walk", 0.0)
                self.character_graph.set_variable("Action", "Idle")
        except Exception as e:
            carb.log_error(f"Error setting character graph variables: {e}")
            # Reset character_graph to None so it will be recreated next frame
            self.character_graph = None

    def update_target_position(self, position, walk_speed=1.0):
        """Update the target position and walking speed"""
        self._target_position = np.array(position)
        self._target_speed = walk_speed

    def update_state(self, dt: float):
        """Update the person's current state from simulation"""

        # Get character graph with retry logic
        if not self.character_graph:
            self.character_graph = ag.get_character(self.character_skel_root_stage_path)
            if not self.character_graph:
                try:
                    skel_path = str(self.character_skel_root.GetPrimPath())
                    self.character_graph = ag.get_character(skel_path)
                except:
                    pass

        if not self.character_graph:
            carb.log_info(f"Character graph not ready for state update: {self._stage_prefix}")
            return

        # Get world transform with error handling
        try:
            pos = carb.Float3(0, 0, 0)
            rot = carb.Float4(0, 0, 0, 0)
            self.character_graph.get_world_transform(pos, rot)

            # Update state
            self._state.position = np.array([pos[0], pos[1], pos[2]])
            self._state.orientation = np.array([rot.x, rot.y, rot.z, rot.w])

            # Notify controller
            if self._controller:
                self._controller.update_state(self._state)
        except Exception as e:
            carb.log_error(f"Error getting world transform: {e}")
            # Reset character_graph to None so it will be recreated next frame
            self.character_graph = None

    def spawn_agent(self, usd_file, stage_name, init_pos, init_yaw):
        """Spawn the character in the simulation"""

        # Ensure character root exists
        if not self._current_stage.GetPrimAtPath(StandalonePerson.character_root_prim_path):
            prims.create_prim(StandalonePerson.character_root_prim_path, "Xform")

        # Ensure biped setup exists
        biped_path = StandalonePerson.character_root_prim_path + "/Biped_Setup"
        if not self._current_stage.GetPrimAtPath(biped_path):
            biped_usd = StandalonePerson.assets_root_path + "/Biped_Setup.usd"
            prim = prims.create_prim(biped_path, "Xform", usd_path=biped_usd)
            prim.GetAttribute("visibility").Set("invisible")

        # Create the character prim
        if not usd_file:
            carb.log_error(f"Could not find USD file for character: {self._character_name}")
            return

        self.prim = prims.create_prim(stage_name, "Xform", usd_path=usd_file)

        # Set initial transform
        self.prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))
        )

        # Set initial rotation
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(init_yaw))
        orient_attr = self.prim.GetAttribute("xformOp:orient")
        if orient_attr.Get() is not None and isinstance(orient_attr.Get(), Gf.Quatf):
            orient_attr.Set(Gf.Quatf(rotation.GetQuat()))
        else:
            orient_attr.Set(rotation.GetQuat())

        # Find skeleton root
        self.character_skel_root, self.character_skel_root_stage_path = self._find_skel_root(
            self._current_stage, self._stage_prefix
        )

    def add_animation_graph_to_agent(self):
        """Add animation graph to the character with improved error handling"""

        if not self.character_skel_root:
            carb.log_warn("No skeleton root found, cannot add animation graph")
            return

        try:
            # Get animation graph path
            anim_graph_path = StandalonePerson.character_root_prim_path + "/Biped_Setup/CharacterAnimation/AnimationGraph"
            animation_graph = self._current_stage.GetPrimAtPath(anim_graph_path)

            if not animation_graph.IsValid():
                carb.log_error(f"Animation graph not found at {anim_graph_path}")
                # Try alternative path
                alt_path = StandalonePerson.character_root_prim_path + "/Biped_Setup/AnimationGraph"
                animation_graph = self._current_stage.GetPrimAtPath(alt_path)
                if animation_graph.IsValid():
                    anim_graph_path = alt_path
                else:
                    return

            # Simply attempt to remove any existing animation graph API
            # without checking if it exists first (avoiding HasAPI)
            try:
                # Try newer API first
                try:
                    omni.kit.commands.execute(
                        "RemoveAnimationGraphAPICommand",
                        paths=[self.character_skel_root.GetPrimPath()]
                    )
                    carb.log_info(
                        "Successfully removed existing animation graph API using RemoveAnimationGraphAPICommand")
                except:
                    # Fallback to older API
                    try:
                        omni.kit.commands.execute(
                            "RemoveAnimationGraphAPI",
                            prim_path=self.character_skel_root.GetPrimPath()
                        )
                        carb.log_info("Successfully removed existing animation graph API using RemoveAnimationGraphAPI")
                    except Exception as e:
                        # It's okay if this fails - likely means there was no API to remove
                        carb.log_info(f"No animation graph API to remove: {e}")
            except Exception as e:
                # Non-critical error, can continue
                carb.log_warn(f"Could not remove old animation graph API: {e}")

            # Add new animation graph API - use newer API if available
            try:
                # Try newer API first
                omni.kit.commands.execute(
                    "ApplyAnimationGraphAPICommand",
                    paths=[self.character_skel_root.GetPrimPath()],
                    animation_graph_path=animation_graph.GetPrimPath()
                )
                carb.log_info("Successfully applied animation graph API using ApplyAnimationGraphAPICommand")
            except Exception as e1:
                # Fallback to older API
                try:
                    omni.kit.commands.execute(
                        "ApplyAnimationGraphAPI",
                        prim_path=self.character_skel_root.GetPrimPath(),
                        animation_graph_path=animation_graph.GetPrimPath()
                    )
                    carb.log_info("Successfully applied animation graph API using ApplyAnimationGraphAPI")
                except Exception as e2:
                    carb.log_error(f"Failed to apply animation graph API: {e1}, fallback also failed: {e2}")
                    return

            carb.log_info(f"Successfully applied animation graph to {self.character_skel_root.GetPrimPath()}")

        except Exception as e:
            carb.log_error(f"Error adding animation graph: {e}")


    @staticmethod
    def _find_skel_root(stage, stage_prefix):
        """Find the SkelRoot prim in the hierarchy"""

        prim = stage.GetPrimAtPath(stage_prefix)

        if prim.GetTypeName() == "SkelRoot":
            return prim, stage_prefix

        # Recursively search children
        children = prim.GetAllChildren()
        if not children:
            return None, None

        for child in children:
            child_prim, child_path = StandalonePerson._find_skel_root(
                stage, stage_prefix + "/" + child.GetName()
            )
            if child_prim is not None:
                return child_prim, child_path

        return None, None

    @staticmethod
    def get_character_asset_list():
        """Get list of available character assets"""

        if not StandalonePerson.assets_root_path:
            carb.log_error("No assets root path configured")
            return []

        result, folder_list = omni.client.list("{}/".format(StandalonePerson.assets_root_path))

        if result != omni.client.Result.OK:
            carb.log_error("Unable to get character assets from provided asset root path.")
            return []

        # Filter for directories only
        character_list = [
            folder.relative_path for folder in folder_list
            if (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
               and not folder.relative_path.startswith(".")
        ]

        return character_list

    @staticmethod
    def get_path_for_character_prim(agent_name):
        """Get the USD path for a character"""

        if not agent_name or not StandalonePerson.assets_root_path:
            carb.log_error("Invalid agent name or assets path")
            return None

        agent_folder = "{}/{}".format(StandalonePerson.assets_root_path, agent_name)
        result, properties = omni.client.stat(agent_folder)

        if result != omni.client.Result.OK:
            carb.log_error(f"Character folder does not exist: {agent_folder}")
            return None

        # Get USD file in the folder
        character_usd = StandalonePerson._get_usd_in_folder(agent_folder)
        if not character_usd:
            return None

        return "{}/{}".format(agent_folder, character_usd)

    @staticmethod
    def _get_usd_in_folder(character_folder_path):
        """Find USD file in character folder"""

        result, folder_list = omni.client.list(character_folder_path)

        if result != omni.client.Result.OK:
            carb.log_error(f"Unable to read character folder path at {character_folder_path}")
            return None

        for item in folder_list:
            if item.relative_path.endswith(".usd"):
                return item.relative_path

        carb.log_error(f"Unable to find a .usd file in {character_folder_path}")
        return None


class ROS2CommandSubscriber(Node):
    """
    ROS2 node to subscribe to Twist messages and convert them to robot commands
    """

    def __init__(self):
        super().__init__('spot_command_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            f'/auto_cmd',  # Standard topic for velocity commands
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
            robot_prim_path = "/World/spot"

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

            self.setup_people()

            carb.log_info("Created Spot robot controller for Omniverse USD")
        else:
            # Local file approach
            try:
                carb.log_info(f"Attempting to load Spot robot from: {self.usd_path}")
                self.spot = SpotWithArmFlatTerrainPolicy(
                    prim_path="/World/spot",
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
                    prim_path="/World/spot",
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

    def setup_people(self):
        """
        Set up people with random walking behavior using standalone implementation
        """
        # Check if animation modules are available
        if not globals().get('ANIMATION_AVAILABLE', False):
            carb.log_warn("Animation modules not available, skipping people setup")
            return

        try:
            # Check if we have animation modules available
            if 'ag' not in globals() or 'PeopleSettings' not in globals():
                carb.log_warn("Animation modules not properly imported, skipping people setup")
                return

            # Get available character assets
            people_assets_list = StandalonePerson.get_character_asset_list()
            carb.log_info(f"Available people assets: {people_assets_list}")

            if not people_assets_list:
                carb.log_warn("No people assets found, skipping people setup")
                return

            # Create multiple people with random walking behavior
            self.people = []
            num_people = 3  # Number of people to spawn

            for i in range(num_people):
                try:
                    # Random initial position around the robot
                    init_pos = [
                        np.random.uniform(-5, 5),  # x
                        np.random.uniform(-5, 5),  # y
                        0.0  # z (ground level)
                    ]

                    # Random person asset
                    character_name = np.random.choice(people_assets_list)

                    # Create controller
                    controller = RandomWalkPersonController()

                    # Create person
                    person = StandalonePerson(
                        world=self.world,
                        stage_prefix=f"person_{i}",
                        character_name=character_name,
                        init_pos=init_pos,
                        init_yaw=np.random.uniform(0, 2 * np.pi),
                        controller=controller
                    )

                    self.people.append(person)
                    carb.log_info(f"Created person {i} ({character_name}) at position {init_pos}")
                except Exception as person_error:
                    carb.log_error(f"Failed to create person {i}: {person_error}")
                    continue  # Continue with next person instead of crashing

        except Exception as e:
            carb.log_error(f"Could not create people: {e}")
            # Don't let people creation failure stop the robot simulation
            self.people = []  # Ensure people list exists even if empty

    def _find_and_initialize_action_graph(self):
        """
        Find and properly initialize the ROS2_Camera_and_TFs Action Graph using forum solution
        """
        carb.log_info("Searching for ROS2_Camera_and_TFs Action Graph...")

        try:
            # Potential paths for the action graph
            graph_paths = [
                "/World/spot/ROS2_Camera_and_TFs",
                "/World/spot/body/ROS2_Camera_and_TFs"
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
                                "ROS2" in graph_path and "/World/spot" in graph_path):
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
                        carb.log_info(f"Triggered impulse on: {node_path}")
                    except Exception as e:
                        carb.log_info(f"Failed to trigger node {node_path}: {e}")
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

        # Ensure people are still active after resets
        if hasattr(self, 'people') and self.people:
            for person in self.people:
                if person._world and not person._world.physics_callback_exists(person._stage_prefix + "/update"):
                    # Re-add callbacks if they were removed during reset
                    person._world.add_physics_callback(person._stage_prefix + "/state", person.update_state)
                    person._world.add_physics_callback(person._stage_prefix + "/update", person.update)

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

        # Reset people if they exist
        if hasattr(self, 'people') and self.people:
            for person in self.people:
                # Re-establish callbacks after reset
                person._world.add_physics_callback(person._stage_prefix + "/state", person.update_state)
                person._world.add_physics_callback(person._stage_prefix + "/update", person.update)

            carb.log_info("Simulation has been reset")

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


