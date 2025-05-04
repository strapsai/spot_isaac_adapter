# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional

import numpy as np
import omni
import carb
import omni.appwindow  # Contains handle to keyboard
from isaacsim.examples.interactive.base_sample import BaseSample

import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path

from isaacsim.robot.policy.examples.controllers.config_loader import get_robot_joint_properties


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
        if usd_path == None:
            # Use the Spot with arm USD path
            usd_path = "/simulation/spot_ws/assets/spot_with_arm/spot_free_mass.usd"

        # Call the base PolicyController initialization
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        # Load the policy
        self.load_policy(
            "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_policy.pt",
            "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_env.yaml",
        )

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

class QuadrupedExample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 500.0
        self._world_settings["rendering_dt"] = 10.0 / 500.0
        self._base_command = [0.0, 0.0, 0.0]

        # bindings for keyboard to command
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

    def setup_scene(self) -> None:
        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )
        self.spot = SpotWithArmFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path="omniverse://airlab-storage.andrew.cmu.edu:8443/Projects/AirStack/robots/spot_with_arm/spot_with_arm/spot_free_mass.usd",
            position=np.array([0, 0, 0.8]),
        )
        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._timeline_timer_callback_fn
        )

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._physics_ready = False
        self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        await self._world.play_async()

    def on_physics_step(self, step_size) -> None:
        if self._physics_ready:
            self.spot.forward(step_size, self._base_command)
        else:
            self._physics_ready = True
            self.spot.initialize()
            self.spot.post_reset()
            self.spot.robot.set_joints_default_state(self.spot.default_pos)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Subscriber callback to when kit is updated."""

        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.spot:
            self._physics_ready = False

    def world_cleanup(self):
        self._event_timer_callback = None
        if self._world.physics_callback_exists("physics_step"):
            self._world.remove_physics_callback("physics_step")
