"""Unitree R1 velocity environment configurations."""

from src.assets.robots import (
    R1_ACTION_SCALE,
    get_r1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_r1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree R1 flat terrain velocity configuration with DR overrides."""
    cfg = make_velocity_env_cfg()

    # --- Flat terrain overrides ---
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    if "height_scan" in cfg.observations["actor"].terms:
        del cfg.observations["actor"].terms["height_scan"]
    if "height_scan" in cfg.observations["critic"].terms:
        del cfg.observations["critic"].terms["height_scan"]

    # Disable terrain curriculum
    cfg.curriculum.pop("terrain_levels", None)

    # --- Keep other rough config settings ---
    cfg.scene.entities = {"robot": get_r1_robot_cfg()}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "pelvis"

    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = R1_ACTION_SCALE

    cfg.viewer.body_name = "torso_link"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.05

    # ========== OVERRIDE COMMAND RANGES TO FAVOR FORWARD ==========
    twist_cmd.ranges.lin_vel_x = (-0.5, 2.0)      # allow backward but encourage forward
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)      # narrow lateral to reduce side-walking
    # ang_vel_z remains default (-1.0, 1.0)

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    # ========== DR MODIFICATIONS ==========
    cfg.events["foot_friction"].params["ranges"] = (0.2, 1.5)
    cfg.events["base_com"].params["ranges"] = {0: (-0.07, 0.07), 1: (-0.07, 0.07), 2: (-0.07, 0.07)}

    cfg.events["body_mass"] = EventTermCfg(
        func=dr.body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ranges": (0.8, 1.2),
            "operation": "scale",
        },
    )

    cfg.observations["actor"].terms["base_ang_vel"].noise = {"type": "gaussian", "mean": 0.0, "std": 0.05}
    cfg.observations["actor"].terms["projected_gravity"].noise = {"type": "gaussian", "mean": 0.0, "std": 0.02}
    cfg.observations["actor"].terms["joint_pos"].noise = {"type": "gaussian", "mean": 0.0, "std": 0.01}
    cfg.observations["actor"].terms["joint_vel"].noise = {"type": "gaussian", "mean": 0.0, "std": 0.1}
    # ===========================================

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.15,
        r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.15,
        r".*waist_roll.*": 0.1,
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.25,
        r".*hip_yaw.*": 0.25,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.25,
        r".*waist_roll.*": 0.1,
        r".*shoulder_pitch.*": 0.25,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }

    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    # ========== INCREASE FORWARD REWARD WEIGHT ==========
    cfg.rewards["track_linear_velocity"].weight = 2.0   # default is 1.0

    # Play mode overrides
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg


