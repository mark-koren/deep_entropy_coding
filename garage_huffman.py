from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from huffman_env import HuffmanEnv
from garage.tf.algos import TRPO
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy

env = TfEnv(HuffmanEnv(data_file ='/home/mkoren/deep_entropy_coding/DJIEncoded.p',
                                 parsed_file='/home/mkoren/deep_entropy_coding/DJIParsed.p',
                                 num_classes=16))

policy = CategoricalMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=([32]))

baseline = LinearFeatureBaseline(env_spec=env.spec)

# algo = TRPO(
#     env=env,
#     policy=policy,
#     baseline=baseline,
#     batch_size=4000,
#     max_path_length=100,
#     n_itr=40,
#     discount=0.99,
#     step_size=0.01,
#     plot=True)
algo = PPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2048,
    max_path_length=1000,
    n_itr=488,
    discount=0.99,
    step_size=0.01,
    optimizer_args=dict(batch_size=32, max_epochs=10),
    plot=False)
algo.train()
