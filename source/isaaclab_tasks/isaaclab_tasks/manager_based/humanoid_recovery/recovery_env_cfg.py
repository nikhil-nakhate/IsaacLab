from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.humanoid_recovery.mdp as mdp


@configclass
class FlatSceneCfg(InteractiveSceneCfg):
    """Flat plane scene with a humanoid robot."""

    robot: ArticulationCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications."""

    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Events: randomize/reset to fallen poses to train recovery."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # randomize root state near ground with large roll/pitch so the robot starts fallen frequently
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.2), "roll": (-1.8, 1.8), "pitch": (-1.8, 1.8), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # jitter joints slightly around defaults for diversity
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg:
    """Rewards for recovery to standing."""

    # uprightness via projected gravity alignment (z-up)
    upright = RewTerm(func=mdp.uprightness_exp, params={"std": 0.25}, weight=2.0)
    # encourage base height (standing)
    base_height = RewTerm(func=mdp.base_height_tanh, params={"target": 1.0, "std": 0.25}, weight=3.0)

    # regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, params={"asset_cfg": SceneEntityCfg("robot")}, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Terminations for success/timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # success: standing upright above height threshold
    success = DoneTerm(func=mdp.standing_success, params={"height_threshold": 0.9, "upright_threshold": 0.85})


@configclass
class HumanoidRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based humanoid fall recovery environment on a flat plane."""

    scene: FlatSceneCfg = FlatSceneCfg(num_envs=2048, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # simulation and episode settings
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        # use the asset-integrated humanoid config (USD copied into assets)
        from isaaclab_assets.robots import UCR_HUMANOID_CFG

        self.scene.robot = UCR_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
