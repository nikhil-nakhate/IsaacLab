import gymnasium as gym

from . import recovery_env_cfg, agents

gym.register(
    id="Isaac-Humanoid-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:HumanoidRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidRecoveryPPORunnerCfg",
    },
)
