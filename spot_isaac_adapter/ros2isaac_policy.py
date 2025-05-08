# Minimal script with up/down controls for joint position

def setup(db):
    # Create persistent state with default positions
    if hasattr(db, "per_instance_state"):
        # Default joint positions
        db.per_instance_state.positions = [0.1, -0.1, 0.1, -0.1, 0.9, 0.9, 1.1, 1.1,
                                           -1.5, -1.5, -1.5, -1.5, 0.0, -3.1, 3.0, 0.0, 0.0, 0.0, 0.0]
        # Index of the joint to control (arm0_el0)
        db.per_instance_state.ARM_JOINT_INDEX = 14


def cleanup(db):
    pass


def compute(db):
    # Make sure we have state initialized
    if not hasattr(db, "per_instance_state") or not hasattr(db.per_instance_state, "positions"):
        return True

    # Get current positions and the joint index
    positions = db.per_instance_state.positions.copy()
    ARM_JOINT_INDEX = db.per_instance_state.ARM_JOINT_INDEX

    # Increment if up is pressed
    if hasattr(db.inputs, "up") and db.inputs.up:
        positions[ARM_JOINT_INDEX] += 0.1

    # Decrement if down is pressed
    if hasattr(db.inputs, "down") and db.inputs.down:
        positions[ARM_JOINT_INDEX] -= 0.1

    # Save updated positions
    db.per_instance_state.positions = positions

    # Convert to numpy array and set output
    import numpy as np
    db.outputs.joints = np.array(positions, dtype=np.float64)

    return True