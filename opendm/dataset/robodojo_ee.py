"""RoboDojo EE dataset registration (LeRobot v3.0 export).

Source data is RoboDojo's native LeRobot v3.0 export (``meta/`` + ``data/`` +
``videos/``), converted by ``script/robodojo_lerobot_to_opendm.py`` into the
OpenDM JSONL layout. See ``docs/zh/dm05_robodojo_ee.md``.

Action/state space: absolute end-effector pose, ``position(3) + axis-angle(3) +
gripper(1)`` per arm (14 dims total), matching OpenDM's EEF convention
(``ArrangeState`` / ``VLA_ARENA_EEF_STATE_DESC``). This is *not* the 14-dim
joint space used by the official ``robodojo_sim_cover_blocks`` entry.
"""

from opendm.constants.robot import RobotStateDesc, RobotType
from opendm.dataset.register import register_dataset

ROBODOJO_EE_STATE_DESC = (
    [RobotStateDesc.EEF] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.EEF] * 6
    + [RobotStateDesc.GRIPPER]
)

register_dataset(
    {
        "cover": {
            "jsonl_dir": "./data/robodojo_ee/jsonl",
            "image_dir": "./data/robodojo_ee/video",
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA,
            "state_desc": ROBODOJO_EE_STATE_DESC,
            "fps": 25,
            "speed": "0.5",
            "control_mode": "eef",
        },
    },
    prefix="robodojo_ee",
)
