"""
Script Node for controlling the arm and body of the Spot robot in Isaac Sim.
This script allows for real-time control of the robot's arm and base joints
using keyboard input, similar to SpotWithArmFlatTerrainPolicy.

Author: Aditya Rauniyar (rauniyar@cmu.edu)
Modified: [Your Name]
"""

import numpy as np
import torch
import io
import omni.client
import yaml

# Robot
ROBOT_NAME = "spot"

# Define the joint order and default positions, as in the policy
JOINT_ORDER = ['fl_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn']
DEFAULT_POSITIONS = [0.1, -0.1, 0.1, -0.1, 0.9, 0.9, 1.1, 1.1, -1.5, -1.5, -1.5, -1.5]
ARM_JOINT_NAMES = ['arm0_sh0', 'arm0_sh1', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']
ARM_DEFAULT_POSITIONS = [0.0, -3.10843, 3.05258, 0.0, 0.0, 0.0, 0.0]

# Policy and environment paths
POLICY_FILE_PATH = "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_policy.pt"
POLICY_ENV_PATH = "/home/adi/repo/navi/simulation/assets/omniverse_spot/spot_env.yaml"

# Input keyboard mapping - expanded based on QuadrupedExample
INPUT_KEYBOARD_MAPPING = {
    "UP": [2.0, 0.0, 0.0],         # Forward
    "DOWN": [-2.0, 0.0, 0.0],      # Backward
    "RIGHT": [0.0, -2.0, 0.0],     # Strafe right
    "LEFT": [0.0, 2.0, 0.0],       # Strafe left
    "N": [0.0, 0.0, 2.0],          # Yaw positive
    "M": [0.0, 0.0, -2.0],         # Yaw negative
    "NUMPAD_8": [2.0, 0.0, 0.0],   # Forward alternative
    "NUMPAD_2": [-2.0, 0.0, 0.0],  # Backward alternative
    "NUMPAD_6": [0.0, -2.0, 0.0],  # Strafe right alternative
    "NUMPAD_4": [0.0, 2.0, 0.0],   # Strafe left alternative
    "NUMPAD_7": [0.0, 0.0, 2.0],   # Yaw positive alternative
    "NUMPAD_9": [0.0, 0.0, -2.0],  # Yaw negative alternative
}

def setup(db):

    # try to setup robot from path
    try:
        # Get the robot by prim path
        import omni.isaac.core.utils.stage as stage_utils
        from isaacsim.core.prims import SingleArticulation

        spot_prim_path = f"/{ROBOT_NAME}"

        db.robot = SingleArticulation(prim_path=spot_prim_path, name="spot_robot")
        db.robot.initialize()
    except Exception as e:
        print(f"Exception in setup: {e}")
        db.robot = None


    # Create persistent state
    if hasattr(db, "per_instance_state"):
        db.per_instance_state.base_command = np.zeros(3)
        db.per_instance_state.joint_positions = np.array(DEFAULT_POSITIONS + ARM_DEFAULT_POSITIONS, dtype=np.float64)
        db.per_instance_state.policy_data = {}
        db.per_instance_state.policy_counter = 0
        db.per_instance_state.action_scale = 0.2  # Important: Consider tuning this
        db.per_instance_state.previous_action = np.zeros(12)
        db.per_instance_state.joint_mapping = {}
        db.per_instance_state.arm_joint_mapping = {}  # Add arm joint mapping
        db.per_instance_state.policy_joint_names = None  # To store joint names from the policy env
        db.per_instance_state.key_state = {}  # Track key states for press/release detection

        # Initialize key state tracking for all mapped keys
        for key in INPUT_KEYBOARD_MAPPING.keys():
            db.per_instance_state.key_state[key] = False

        for i, name in enumerate(JOINT_ORDER):
            db.per_instance_state.joint_mapping[name] = i
        for i, name in enumerate(ARM_JOINT_NAMES):
            db.per_instance_state.arm_joint_mapping[name] = 12 + i

        # Load policy and environment data
        _load_policy(db, POLICY_FILE_PATH, POLICY_ENV_PATH)
        db.per_instance_state.full_default_pos = np.array(
            DEFAULT_POSITIONS + ARM_DEFAULT_POSITIONS, dtype=np.float64
        )

def cleanup(db):
    # Cleanup function to reset the state
    db.per_instance_state.base_command = np.zeros(3)
    db.per_instance_state.joint_positions = np.array(DEFAULT_POSITIONS + ARM_DEFAULT_POSITIONS, dtype=np.float64)
    db.per_instance_state.policy_data = {}
    db.per_instance_state.policy_counter = 0
    db.per_instance_state.action_scale = 0.2
    db.per_instance_state.previous_action = np.zeros(12)
    db.per_instance_state.joint_mapping = {}
    db.per_instance_state.arm_joint_mapping = {}
    db.per_instance_state.policy_joint_names = None
    db.per_instance_state.key_state = {}
    pass

def _load_policy(db, policy_file_path, policy_env_path):
    """
    Loads the policy model and environment parameters from Omniverse.
    """
    def parse_env_config(env_config_path):
        """
        Parses the environment configuration file.
        """
        class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
            def ignore_unknown(self, node):
                return None

        SafeLoaderIgnoreUnknown.add_constructor(
            "tag:yaml.org,2002:python/tuple", yaml.SafeLoader.construct_sequence
        )
        SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

        file_content = omni.client.read_file(env_config_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        data = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)
        return data

    try:
        # Load policy from Omniverse
        result, read_file_path, file_content = omni.client.read_file(policy_file_path)
        if result == omni.client.Result.OK:
            file = io.BytesIO(memoryview(file_content).tobytes())
            db.per_instance_state.policy_data['model'] = torch.jit.load(file)
            print(f"Policy model loaded successfully from {policy_file_path}")
        else:
            print(f"Error: Failed to load policy from {policy_file_path}. Result: {result}")
            db.per_instance_state.policy_data['model'] = None

        # Load environment parameters from Omniverse using the custom parser
        env_params = parse_env_config(policy_env_path)
        db.per_instance_state.env_params = env_params
        print(f"Environment parameters loaded from {policy_env_path}")

    except Exception as e:
        print(f"Exception in _load_policy: {e}")
        db.per_instance_state.policy_data['model'] = None
        db.per_instance_state.env_params = {}

# Helper functions
def quat_to_rot_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY], [xY + wZ, 1.0 - (xX + zZ), yZ - wX], [xZ - wY, yZ + wX, 1.0 - (xX + yY)]]
    )

def compute(db):
    # Make sure we have state initialized
    if not hasattr(db, "per_instance_state"):
        return True

    # Get persistent state
    base_command = db.per_instance_state.base_command
    joint_positions = db.per_instance_state.joint_positions
    env_params = db.per_instance_state.env_params
    policy_counter = db.per_instance_state.policy_counter
    action_scale = db.per_instance_state.action_scale
    previous_action = db.per_instance_state.previous_action
    joint_mapping = db.per_instance_state.joint_mapping
    arm_joint_mapping = db.per_instance_state.arm_joint_mapping
    full_default_pos = db.per_instance_state.full_default_pos
    policy_joint_names = db.per_instance_state.policy_joint_names
    key_state = db.per_instance_state.key_state

    # Handle keyboard input - detect key press/release like in QuadrupedExample
    for key in INPUT_KEYBOARD_MAPPING:
        # Get the current key state
        current_state = getattr(db.inputs, key.lower(), False)

        # Check for key press (wasn't pressed, now is)
        if current_state and not key_state[key]:
            # Key just pressed
            base_command += np.array(INPUT_KEYBOARD_MAPPING[key])
            key_state[key] = True

        # Check for key release (was pressed, now isn't)
        elif not current_state and key_state[key]:
            # Key just released
            base_command -= np.array(INPUT_KEYBOARD_MAPPING[key])
            key_state[key] = False

    # Update the persistent base command
    db.per_instance_state.base_command = base_command

    # Decimation rate
    decimation = 4
    if policy_counter % decimation == 0:
        # Compute observation and action
        observation = _compute_observation(db, base_command)
        raw_action = _compute_action(db, observation)

        # Debug: Print raw action and policy joint names
        print(f"Raw Action from Policy ({len(raw_action)}): {raw_action}")
        print(f"Policy Joint Names: {policy_joint_names}")

        # Reorder the action based on JOINT_ORDER
        reordered_action = np.zeros_like(raw_action)
        if policy_joint_names is not None and len(policy_joint_names) == len(raw_action):
            policy_name_to_action_index = {name: i for i, name in enumerate(policy_joint_names)}
            for i, target_joint_name in enumerate(JOINT_ORDER):
                if target_joint_name in policy_name_to_action_index:
                    original_index = policy_name_to_action_index[target_joint_name]
                    reordered_action[i] = raw_action[original_index]
                else:
                    print(f"Warning: Joint '{target_joint_name}' not found in policy output.")
                    reordered_action[i] = 0.0  # Or some default value
        else:
            print("Warning: Policy joint names not available or length mismatch. Using raw action.")
            reordered_action = raw_action

        db.per_instance_state.previous_action = reordered_action.copy()

        # Update joint positions
        full_joint_positions = full_default_pos.copy()  # Start with full default positions

        for i, target_joint_name in enumerate(JOINT_ORDER):
            if target_joint_name in joint_mapping:
                robot_joint_index = joint_mapping[target_joint_name]
                if robot_joint_index < 12:  # Legs
                    default_leg_pos = DEFAULT_POSITIONS[i]  # Use index i based on JOINT_ORDER
                    full_joint_positions[robot_joint_index] = default_leg_pos + (
                        reordered_action[i] * action_scale
                    )

        for i, joint_name in enumerate(ARM_JOINT_NAMES):
            if joint_name in arm_joint_mapping:
                robot_idx = arm_joint_mapping[joint_name]
                full_joint_positions[robot_idx] = ARM_DEFAULT_POSITIONS[i]

        db.per_instance_state.joint_positions = full_joint_positions

    # Apply base commands to outputs
    db.outputs.base_command = base_command

    # Set joint positions
    db.outputs.joints = db.per_instance_state.joint_positions
    print(f"Joint Positions: {db.outputs.joints}")

    db.per_instance_state.policy_counter += 1
    return True


def _compute_observation(db, command):
    """
    Compute the observation vector for the policy.
    """
    joint_positions = db.per_instance_state.joint_positions
    previous_action = db.per_instance_state.previous_action
    env_params = db.per_instance_state.env_params
    joint_mapping = db.per_instance_state.joint_mapping

    # Get robot state from the SingleArticulation object
    if hasattr(db, "robot") and db.robot is not None:
        # Get position and orientation in world frame
        robot_position, robot_orientation = db.robot.get_world_pose()

        # Get linear and angular velocities in world frame
        robot_linear_velocity = db.robot.get_linear_velocity()
        robot_angular_velocity = db.robot.get_angular_velocity()

        # Get joint velocities
        joint_velocities = db.robot.get_joint_velocities()
        print("Success: Robot state retrieved successfully.")
    else:
        # Fallback to placeholders if robot isn't available
        robot_position = np.array([0.0, 0.0, 0.0])
        robot_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        robot_linear_velocity = np.array([0.0, 0.0, 0.0])
        robot_angular_velocity = np.array([0.0, 0.0, 0.0])
        joint_velocities = np.zeros(19)

        print("Warning: Robot state not available. Using placeholders.")

    # Continue with the rest of your observation computation
    R_IB = quat_to_rot_matrix(robot_orientation)
    R_BI = R_IB.transpose()
    lin_vel_b = np.matmul(R_BI, robot_linear_velocity)
    ang_vel_b = np.matmul(R_BI, robot_angular_velocity)
    gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

    obs = np.zeros(48)
    obs[:3] = lin_vel_b
    obs[3:6] = ang_vel_b
    obs[6:9] = gravity_b
    obs[9:12] = command

    # Create policy observation with mapped joints
    leg_positions = np.zeros(12)
    leg_velocities = np.zeros(12)
    default_pos = np.array(DEFAULT_POSITIONS)
    for joint_name, robot_idx in db.per_instance_state.joint_mapping.items():
        if robot_idx < 12:  # Use robot_idx for comparison
            leg_positions[robot_idx] = joint_positions[robot_idx]
            leg_velocities[robot_idx] = joint_velocities[robot_idx]

    obs[12:24] = leg_positions - default_pos
    obs[24:36] = leg_velocities
    obs[36:48] = previous_action

    return obs

def _compute_action(db, observation):
    """
    Compute the action from the policy.

    Args:
        observation (np.ndarray): The observation.

    Returns:
        np.ndarray: The action.
    """
    if 'model' in db.per_instance_state.policy_data:
        model = db.per_instance_state.policy_data['model']
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).view(1, -1).float()
            action_tensor = model(obs_tensor).detach().view(-1).numpy()
        return action_tensor
    else:
        print("Warning: Policy model not loaded. Returning zero action.")
        return np.zeros(12)


def find_prim_path_by_keyword(keyword):
    """
    Find prim paths that contain the given keyword.
    Returns a list of matching prim paths.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    matching_prims = []

    # Traverse all prims in the stage
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()

        # Check if the keyword is in the path or name
        if keyword.lower() in prim_path.lower() or keyword.lower() in prim_name.lower():
            matching_prims.append(prim_path)

    return matching_prims