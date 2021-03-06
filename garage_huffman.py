from garage.baselines import LinearFeatureBaseline
from garage.tf.baselines import DeterministicMLPBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.envs import normalize
from huffman_env import HuffmanEnv
from garage.tf.algos import TRPO
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.core.network import MLP
from garage.misc import logger
import argparse
import os.path as osp
import tensorflow as tf

# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='entropy')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='deep_entropy_coding/run02')
parser.add_argument('--path', type=str, default='/home/mkoren/')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, args.tabular_log_file)
text_log_file = osp.join(log_dir, args.text_log_file)
params_log_file = osp.join(log_dir, args.params_log_file)

logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

# env = TfEnv(HuffmanEnv(data_file ='/home/mkoren/Research/deep_entropy_coding/DJIEncoded.p',
#                                  parsed_file='/home/mkoren/Research/deep_entropy_coding/DJIParsed.p',
#                                  num_classes=8))


env = TfEnv(HuffmanEnv(data_file =args.path+'deep_entropy_coding/DJIEncoded.p',
                                 parsed_file=args.path+'deep_entropy_coding/DJIParsed4.p',
                                 freq_file=args.path+'deep_entropy_coding/DJIFreq4.p',
                                 num_classes=2,
                                 width=4))



policy = CategoricalMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=((512,256,128,64)))

# baseline = LinearFeatureBaseline(env_spec=env.spec)
baseline = GaussianMLPBaseline(env_spec=env.spec)
# policy = CategoricalMLPPolicy(
#     name="policy", env_spec=env.spec, hidden_sizes=(512,128,64,32))
# feature_network = MLP(output_dim=128,
#             hidden_sizes=(32,64),
#             hidden_nonlinearity=tf.nn.relu,
#             output_nonlinearity=tf.nn.relu,
#             name='MLP',
#             # hidden_w_init=ly.XavierUniformInitializer(),
#             # hidden_b_init=tf.zeros_initializer(),
#             # output_w_init=ly.XavierUniformInitializer(),
#             # output_b_init=tf.zeros_initializer(),
#             input_var=env.observation_space,
#             input_layer=None,
#             input_shape=(16,),
#             batch_normalization=False,
#             weight_normalization=False,
#     )
# policy = CategoricalLSTMPolicy(env_spec=env.spec,
#                  name="policy",
#                  hidden_dim=256,
#                  feature_network=feature_network,
#                  prob_network=None,
#                  state_include_action=False,
#                  hidden_nonlinearity=tf.tanh,
#                  forget_bias=1.0,
#                  use_peepholes=True)
# baseline = LinearFeatureBaseline(env_spec=env.spec)
# baseline = DeterministicMLPBaseline(env_spec=env.spec)
# baseline = GaussianMLPBaseline(env_spec=env.spec)
optimizer = ConjugateGradientOptimizer
optimizer_args = {'hvp_approach':FiniteDifferenceHvp(base_eps=1e-5)}
# algo = TRPO(
#     env=env,
#     policy=policy,
#     baseline=baseline,
# # <<<<<<< HEAD
#     # batch_size=1000,
#     # max_path_length=1000,
#     # n_itr=101,
#     # discount=1.0,
#     # step_size=10.0,
# # =======
#     batch_size=1000,
#     max_path_length=1000,
#     n_itr=101,
#     discount=1.0,
#     step_size=3.0,
#     clip_range=3.0,
#     optimizer=optimizer,
#     optimizer_args=optimizer_args,
# # >>>>>>> c553223c025426f088361a1fb90391b7aaee1969
#     plot=False)
algo = PPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=1000,
    max_path_length=1000,
    n_itr=101,
    discount=1.0,
    step_size=30.0,
    clip_range=0.001,
    optimizer_args=dict(batch_size=1, max_epochs=10, 
    	tf_optimizer_args=dict(learning_rate=1e-5)),
    plot=False)
algo.train()
