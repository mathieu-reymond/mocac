from envs.minecart.env import MinecartEnv, MinecartDeterministicEnv, MinecartSimpleDeterministicEnv
from gym.envs.registration import register


register(
    id='MinecartDeterministic-v0',
    entry_point='envs.minecart:MinecartDeterministicEnv',
    )
