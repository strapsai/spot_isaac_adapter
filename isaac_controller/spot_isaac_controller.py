"""
Script Node for controlling the arm and body of the Spot robot in Isaac Sim.
This script uses the same approach as the working StandaloneQuadrupedApp,
following patterns from BaseSample.
"""

import numpy as np
import torch
import io
import omni.client
import yaml
import time
import os
from typing import Optional
import traceback
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.robot.policy.examples.controllers.config_loader import get_robot_joint_properties
import omni.physx as _physx
import omni.timeline

# Robot
ROBOT_NAME = "spot"

# Define policy and USD file paths
POLICY_PATH = "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_policy.pt"
ENV_PATH = "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_env.yaml"
USD_PATH = "omniverse://airlab-storage.andrew.cmu.edu:8443/Projects/AirStack/robots/spot_with_arm/spot_with_arm/spot_free_mass.usd"

# Skip file existence check for Omniverse URLs
SKIP_OMNIVERSE_CHECK = True

# World settings - defined as in BaseSample derivation
WORLD_SETTINGS = {
    "stage_units_in_meters": 1.0,
    "physics_dt": 1.0 / 500.0,
    "rendering_dt": 10.0 / 500.0
}

# Input keyboard mapping (same as QuadrupedExample)
INPUT_KEYBOARD_MAPPING = {
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

# Initial stabilization time
INITIAL_STABILIZATION_TIME = 100  # Frames to wait before processing commands
# Simulation has started flag (global)
simulation_started = False
# Initial stabilization counter
stabilization_counter = 0


def apply_world_settings():
    """
    Apply the world settings from the StandaloneQuadrupedApp.
    This includes physics timestep, rendering timestep, and units.
    Following the BaseSample pattern.
    """
    try:
        print("Applying world settings...")

        # These settings match QuadrupedExample from BaseSample
        world_settings = {
            "stage_units_in_meters": 1.0,
            "physics_dt": 1.0 / 500.0,
            "rendering_dt": 10.0 / 500.0
        }

        # Get timeline for setting rendering rate
        # Import inside the function to avoid unbound local variable error
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()

        # Correct method name - use set_time_codes_per_second (without the trailing 's')
        # Or if that doesn't exist, try to find the correct method for setting the time codes per second
        try:
            # Try the corrected method name first
            if hasattr(timeline, 'set_time_codes_per_second'):
                timeline.set_time_codes_per_second(1.0 / world_settings["rendering_dt"])
                print(f"Set timeline timestep to {world_settings['rendering_dt']}")
            # If that doesn't work, check other possible methods
            elif hasattr(timeline, 'set_ticks_per_second'):
                timeline.set_ticks_per_second(1.0 / world_settings["rendering_dt"])
                print(f"Set timeline ticks per second to {world_settings['rendering_dt']}")
            # If we can't set it, just print the current value
            else:
                if hasattr(timeline, 'get_time_codes_per_seconds'):
                    current_tps = timeline.get_time_codes_per_seconds()
                    print(f"Current timeline time codes per second: {current_tps}")
                print("Unable to set timeline timestep - method not found")
        except Exception as e:
            print(f"Error setting timeline timestep: {e}")
            traceback.print_exc()

        # Set the physics timestep
        import omni.physx as _physx
        physics_context = _physx.get_physx_interface().get_physics_context()
        physics_context.set_timestep(world_settings["physics_dt"])
        print(f"Set physics timestep to {world_settings['physics_dt']}")

        # Set stage units
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        from pxr import UsdGeom
        UsdGeom.SetStageMetersPerUnit(stage, world_settings["stage_units_in_meters"])
        print(f"Set stage units to {world_settings['stage_units_in_meters']} meters per unit")

        # Get default scene and configure physics properties
        try:
            from omni.physx import get_physx_interface
            scene = get_physx_interface().get_physx_scene()
            if scene:
                scene.set_solver_position_iteration_count(8)  # More iterations for better stability
                scene.set_solver_velocity_iteration_count(2)
                print("Set physics solver iteration counts")
        except Exception as e:
            print(f"Could not set physics scene properties: {e}")

        return True
    except Exception as e:
        print(f"Error applying world settings: {str(e)}")
        traceback.print_exc()
        return False


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
            root_path = None,
            name: str = "spot_with_arm",
            usd_path = None,
            position = None,
            orientation = None,
    ) -> None:
        """
        Initialize robot and load RL policy.
        """
        if usd_path is None:
            # Use the default USD path defined at the top of the file
            usd_path = USD_PATH
            print(f"Using default USD path: {usd_path}")

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

            print(f"Successfully created SpotWithArmFlatTerrainPolicy object for {prim_path}")
        except Exception as e:
            print(f"Error initializing SpotWithArmFlatTerrainPolicy: {str(e)}")
            traceback.print_exc()
            raise

    def initialize(self, **kwargs):
        """
        Initialize the robot and identify leg and arm joints
        """
        try:
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
            return True
        except Exception as e:
            print(f"Error in initialize: {str(e)}")
            traceback.print_exc()
            return False

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
        try:
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
        except Exception as e:
            print(f"Error in forward: {str(e)}")
            traceback.print_exc()

    def post_reset(self):
        """
        Reset the robot's state
        """
        try:
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
        except Exception as e:
            print(f"Error in post_reset: {str(e)}")
            traceback.print_exc()


def set_ground_plane_physics():
    """Set better physics properties for the ground plane"""
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        # Common ground plane paths
        ground_plane_paths = ["/World/defaultGroundPlane", "/defaultGroundPlane", "/FlatGrid"]

        for path in ground_plane_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                print(f"Setting physics properties for ground plane at {path}")

                # Set higher friction values
                from pxr import UsdPhysics, PhysxSchema

                # Apply physics material
                material = UsdPhysics.MaterialAPI.Apply(prim)
                material.CreateStaticFrictionAttr().Set(0.8)  # Higher static friction
                material.CreateDynamicFrictionAttr().Set(0.6)  # Higher dynamic friction
                material.CreateRestitutionAttr().Set(0.01)  # Low restitution

                # Apply PhysX specific material properties for better stability
                physx_material = PhysxSchema.PhysxMaterialAPI.Apply(prim)
                physx_material.CreateFrictionCombineModeAttr().Set("multiply")
                physx_material.CreateRestitutionCombineModeAttr().Set("min")

                # Apply collision API
                collision = UsdPhysics.CollisionAPI.Apply(prim)

                # Apply PhysX collision API for advanced properties
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)

                print(f"Successfully set physics properties for {path}")
                return True

        print("Warning: Could not find ground plane in the scene")
        return False
    except Exception as e:
        print(f"Error setting ground plane physics: {str(e)}")
        traceback.print_exc()
        return False


def reduce_arm_mass(robot_prim_path):
    """Reduce the mass of arm links for better stability"""
    try:
        import omni.usd
        from pxr import UsdPhysics, Usd

        stage = omni.usd.get_context().get_stage()

        # Common arm link patterns
        arm_link_patterns = ["arm0_", "gripper"]

        # Find the robot prim
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim.IsValid():
            print(f"WARNING: Robot prim not found at {robot_prim_path}")
            return False

        # Modify arm links
        arm_links_found = 0
        for prim in Usd.PrimRange(robot_prim):
            prim_name = prim.GetName()

            # Check if this is an arm link
            is_arm_link = any(pattern in prim_name for pattern in arm_link_patterns)
            if is_arm_link:
                # Get the mass API
                massAPI = UsdPhysics.MassAPI.Get(prim)
                if massAPI:
                    # Reduce mass to improve stability
                    massAPI.CreateMassAttr().Set(0.1)  # Very light
                    arm_links_found += 1

        print(f"Successfully modified {arm_links_found} arm links")
        return True
    except Exception as e:
        print(f"Error reducing arm mass: {str(e)}")
        traceback.print_exc()
        return False


def setup(db):
    """
    Setup function for the script node.
    Creates a SpotWithArmFlatTerrainPolicy following pattern from QuadrupedExample.
    """
    try:
        print("Setting up SpotWithArmFlatTerrainPolicy script node")

        # Apply world settings first - using same approach as BaseSample derivative
        apply_world_settings()

        # Create ground plane with same parameters as in QuadrupedExample
        try:
            # Get stage to add ground plane
            import omni.usd
            from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf

            stage = omni.usd.get_context().get_stage()

            # Create ground plane at "/World/defaultGroundPlane"
            ground_path = "/World/defaultGroundPlane"
            if not stage.GetPrimAtPath(ground_path).IsValid():
                print(f"Creating ground plane at {ground_path}")
                ground_prim = UsdGeom.Plane.Define(stage, ground_path)

                # Set transform
                xform = UsdGeom.Xformable(ground_prim)
                xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))  # At origin
                xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))  # No rotation
                ground_prim.GetSizeAttr().Set(100.0)  # Large ground plane

                # Add physics
                rb_api = UsdPhysics.RigidBodyAPI.Apply(ground_prim.GetPrim())
                rb_api.CreateRigidBodyEnabledAttr().Set(True)
                rb_api.CreateKinematicEnabledAttr().Set(True)  # Static ground plane

                # Set collision
                collision_api = UsdPhysics.CollisionAPI.Apply(ground_prim.GetPrim())

                # Set material
                material_api = UsdPhysics.MaterialAPI.Apply(ground_prim.GetPrim())
                material_api.CreateStaticFrictionAttr().Set(0.2)
                material_api.CreateDynamicFrictionAttr().Set(0.2)
                material_api.CreateRestitutionAttr().Set(0.01)

                # Add PhysX specific properties
                physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(ground_prim.GetPrim())
                physx_material_api.CreateFrictionCombineModeAttr().Set("multiply")

                print("Ground plane created successfully")
        except Exception as e:
            print(f"Error creating ground plane: {e}")
            traceback.print_exc()

            # As fallback, try setting physics on existing ground if available
            set_ground_plane_physics()

        # Create persistent state storage
        if not hasattr(db, "per_instance_state"):
            db.per_instance_state = type('', (), {})()
            db.per_instance_state.base_command = np.zeros(3)
            db.per_instance_state.key_state = {}
            db.per_instance_state.prev_command = np.zeros(3)

            # Initialize key state tracking for all mapped keys
            for key in INPUT_KEYBOARD_MAPPING.keys():
                db.per_instance_state.key_state[key] = False

        # Get the robot path
        robot_prim_path = f"/{ROBOT_NAME}"

        # Reduce arm mass for better stability
        reduce_arm_mass(robot_prim_path)

        # Create the policy controller - following QuadrupedExample pattern
        try:
            print(f"Creating robot at {robot_prim_path} with USD {USD_PATH}")

            # This follows the same creation pattern as in QuadrupedExample
            db.spot_policy = SpotWithArmFlatTerrainPolicy(
                prim_path=robot_prim_path,
                name="Spot",
                usd_path=USD_PATH,
                position=np.array([0, 0, 0.8])
            )

            # Load policy
            print(f"Loading policy from {POLICY_PATH}")
            db.spot_policy.load_policy(POLICY_PATH, ENV_PATH)

            # Get timeline for STOP event subscription
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            db.event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP),
                lambda event: setattr(db, 'stop_requested', True)
            )

            print("Robot created and policy loaded successfully")
        except Exception as e:
            print(f"Error creating robot: {e}")
            traceback.print_exc()

        # Initialize the controller
        try:
            success = db.spot_policy.initialize()
            if success:
                print("Successfully initialized SpotWithArmFlatTerrainPolicy")
            else:
                print("WARNING: Policy initialization had issues")
        except Exception as e:
            print(f"Error initializing policy: {e}")
            traceback.print_exc()

        # Reset global states
        global stabilization_counter, simulation_started
        stabilization_counter = 0
        simulation_started = False

        # Start the timeline
        try:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            if not timeline.is_playing():
                timeline.play()
                print("Started timeline playback")
        except Exception as e:
            print(f"Error starting timeline: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"Error in setup: {str(e)}")
        traceback.print_exc()
        db.spot_policy = None


def compute(db):
    """
    Compute function called every frame.
    Follows the QuadrupedExample approach.
    """
    try:
        # Check if we have state initialized
        if not hasattr(db, "spot_policy") or db.spot_policy is None:
            print("No policy controller available, skipping compute")
            return True

        # Global stabilization phase
        global stabilization_counter, simulation_started
        if not simulation_started:
            # First run initialization
            print("First execution of compute - initializing simulation")
            simulation_started = True

            # Apply world settings again to ensure they're properly set
            apply_world_settings()

            # Get timeline
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            if not timeline.is_playing():
                timeline.play()
                print("Started timeline playback from compute")

            return True

        # Check initial stabilization period
        if stabilization_counter < INITIAL_STABILIZATION_TIME:
            stabilization_counter += 1
            if stabilization_counter % 10 == 0:
                print(f"Global stabilization: {stabilization_counter}/{INITIAL_STABILIZATION_TIME}")
            return True

        # Initialize state if needed
        if not hasattr(db, "per_instance_state"):
            db.per_instance_state = type('', (), {})()
            db.per_instance_state.base_command = np.zeros(3)
            db.per_instance_state.key_state = {}
            db.per_instance_state.prev_command = np.zeros(3)

            # Initialize key state tracking for all mapped keys
            for key in INPUT_KEYBOARD_MAPPING.keys():
                db.per_instance_state.key_state[key] = False

        # Handle physics ready state - follows QuadrupedExample approach
        physics_ready = getattr(db, "physics_ready", False)
        if not physics_ready:
            # Initialize the robot on first physics step
            try:
                print("First physics step - initializing robot")
                if hasattr(db.spot_policy, "initialize"):
                    db.spot_policy.initialize()
                    db.spot_policy.post_reset()
                    db.spot_policy.robot.set_joints_default_state(db.spot_policy.default_pos)
                    db.physics_ready = True
                    print("Robot initialized successfully")
                else:
                    print("WARNING: Policy has no initialize method")
            except Exception as e:
                print(f"Error initializing robot: {e}")
                traceback.print_exc()
            return True

        try:
            # Make sure per_instance_state is initialized before accessing it
            if not hasattr(db, "per_instance_state"):
                print("Initializing per_instance_state")
                db.per_instance_state = type('', (), {})()
                db.per_instance_state.base_command = np.zeros(3)
                db.per_instance_state.key_state = {}
                db.per_instance_state.prev_command = np.zeros(3)

                # Initialize key state tracking for all mapped keys
                for key in INPUT_KEYBOARD_MAPPING.keys():
                    db.per_instance_state.key_state[key] = False

            # Access state with safety checks
            base_command = getattr(db.per_instance_state, "base_command", np.zeros(3))
            key_state = getattr(db.per_instance_state, "key_state", {})
            if not key_state:
                key_state = {}
                for key in INPUT_KEYBOARD_MAPPING.keys():
                    key_state[key] = False
                db.per_instance_state.key_state = key_state

            # Handle keyboard input - same approach as in QuadrupedExample
            for key in key_state:
                # Check each key in the mapping
                current_state = getattr(db.inputs, key.lower(), False)

                # Check for key press (wasn't pressed, now is)
                if current_state and not key_state[key]:
                    # Key just pressed
                    base_command += np.array(INPUT_KEYBOARD_MAPPING[key])
                    key_state[key] = True
                    if hasattr(db.spot_policy, "_policy_counter") and db.spot_policy._policy_counter % 10 == 0:
                        print(f"Key pressed: {key}, Command: {base_command}")

                # Check for key release (was pressed, now isn't)
                elif not current_state and key_state[key]:
                    # Key just released
                    base_command -= np.array(INPUT_KEYBOARD_MAPPING[key])
                    key_state[key] = False
                    if hasattr(db.spot_policy, "_policy_counter") and db.spot_policy._policy_counter % 10 == 0:
                        print(f"Key released: {key}, Command: {base_command}")

            # Command smoothing for stability
            if hasattr(db.per_instance_state, "prev_command"):
                prev_command = db.per_instance_state.prev_command
                # Smoothing factor (0.0-1.0)
                smooth_factor = 0.7  # Higher value = smoother but slower response
                smoothed_command = smooth_factor * prev_command + (1.0 - smooth_factor) * base_command
                base_command = smoothed_command

            # Store previous command for next frame
            db.per_instance_state.prev_command = base_command.copy()

            # Update the state
            db.per_instance_state.base_command = base_command

            # Apply command to the policy controller - follow QuadrupedExample
            step_size = WORLD_SETTINGS["physics_dt"]
            if hasattr(db.spot_policy, "forward"):
                db.spot_policy.forward(step_size, base_command)
            else:
                print("WARNING: Policy has no forward method")

            # Set outputs for connections
            if hasattr(db, "outputs"):
                db.outputs.base_command = base_command

            # Log occasionally
            if hasattr(db.spot_policy, "_policy_counter") and db.spot_policy._policy_counter % 100 == 0:
                print(f"Frame: {db.spot_policy._policy_counter}, Command: {base_command}")

            # Check for stop request
            if hasattr(db, "stop_requested") and db.stop_requested:
                print("Stop requested - resetting physics ready state")
                db.physics_ready = False
                db.stop_requested = False
        except Exception as e:
            print(f"Error in compute inner loop: {str(e)}")
            traceback.print_exc()

    except Exception as e:
        print(f"Error in compute: {str(e)}")
        traceback.print_exc()

    return True


def cleanup(db):
    """
    Cleanup function to reset the state.
    Follows the QuadrupedExample cleanup pattern.
    """
    try:
        print("Performing cleanup...")

        # Reset robot
        if hasattr(db, "spot_policy") and db.spot_policy is not None:
            try:
                if hasattr(db.spot_policy, "post_reset"):
                    db.spot_policy.post_reset()
                db.spot_policy = None
                print("Robot controller cleaned up")
            except Exception as e:
                print(f"Error cleaning up robot: {e}")
                traceback.print_exc()

        # Clean up timeline callback
        if hasattr(db, "event_timer_callback"):
            db.event_timer_callback = None
            print("Timeline callback cleaned up")

        # Reset state
        if hasattr(db, "per_instance_state"):
            db.per_instance_state.base_command = np.zeros(3)
            db.per_instance_state.key_state = {}
            db.per_instance_state.prev_command = np.zeros(3)
            print("State data cleaned up")

        # Reset physics ready flag
        if hasattr(db, "physics_ready"):
            db.physics_ready = False

        # Reset stop request flag
        if hasattr(db, "stop_requested"):
            db.stop_requested = False

        # Reset global states
        global stabilization_counter, simulation_started
        stabilization_counter = 0
        simulation_started = False
        print("Global states reset")

    except Exception as e:
        print(f"Error in cleanup: {str(e)}")
        traceback.print_exc()