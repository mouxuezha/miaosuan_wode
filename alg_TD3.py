import copy
import numpy as np
# from dodo.algos import BaseActor, BaseLearner, RlAlgo
import threading
import random
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import csv
from typing import List, Optional
from numpy import ndarray
from torch.distributions import Categorical
from log import log
import pickle

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.max_action = 1

    def forward(self, state):
        a = self.ln1(self.l1(state))
        a = F.relu(a)
        a = self.ln2(self.l2(a))
        a = F.relu(a)
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 硬编码网络结构吗……算了也不是不行
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(256)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.ln1(self.l1(sa))
        q1 = F.relu(q1)
        q1 = self.ln2(self.l2(q1))
        q1 = F.relu(q1)
        q1 = self.l3(q1)

        q2 = self.ln3(self.l4(sa))
        q2 = F.relu(q2)
        q2 = self.ln4(self.l5(q2))
        q2 = F.relu(q2)
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.ln1(self.l1(sa))
        q1 = F.relu(q1)
        q1 = self.ln2(self.l2(q1))
        q1 = F.relu(q1)
        q1 = self.l3(q1)
        return q1

    def Q2(self, state, action):
        # 所以为啥原版没有这个函数呢？肯定是有他的道理的吧。
        sa = torch.cat([state, action], 1)
        q2 = self.ln3(self.l4(sa))
        q2 = F.relu(q2)
        q2 = self.ln4(self.l5(q2))
        q2 = F.relu(q2)
        q2 = self.l6(q2)
        return q2


class ReplayBuffer(object):
    def __init__(self, **config):
        """
        :param config:
        """
        # 获取buffer size
        self.buffer_size = config["buffer_size"]
        self.evaluate_episodes = config["evaluate_episodes"]
        self.n_agents = None
        self.state_shape = None
        self.obs_shape = None
        self.n_actions = None
        self.episode_limit = None
        # 内存管理
        self.current_idx = 0
        self.current_size = 0
        self.episode_idx = 0
        # 初始化replay buffer
        self.buffers = {}    # 这款实现不是用队列，所以用起来相对要难受一点。但是应该也还好。
        # 定义线程锁
        self.lock = threading.Lock()



    def prepare(self, **kwargs):
        self.n_agents = len(kwargs["possible_agents"])
        # todo: 后续需要扩展
        self.n_actions = kwargs["action_shape"]
        self.state_shape = kwargs["state_shape"]
        self.obs_shape = kwargs["obs_shape"]
        self.buffers = {
            "s": np.empty([self.buffer_size, *self.state_shape]),
            "r": np.empty([self.buffer_size, 1]),
            "a": np.empty([self.buffer_size, *self.n_actions]),
            "s_next": np.empty([self.buffer_size, *self.state_shape]),
            "terminated": np.empty([self.buffer_size, 1]),
            "info": {"episodes_rewards": np.zeros(self.evaluate_episodes)},
        }

        try:
            # self.load_buffer(kwargs["result_path"] + r'\buffer.pkl')
            self.load_buffer()
            print("exsiting buffer loaded")
        except:
            print("no exsiting buffer")

    def add(self, episode_batch):
        # batch_size = episode_batch["s"].shape[0]  # episode_number
        batch_size = 1
        # if len(episode_batch["s"].shape)>1:
        #     batch_size = episode_batch["s"].shape[0]  # epis
        # else:
        #     batch_size = 1

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            self.buffers["s"][idxs] = episode_batch["s"]
            self.buffers["r"][idxs] = episode_batch["r"]
            self.buffers["a"][idxs] = episode_batch["a"]
            self.buffers["s_next"][idxs] = episode_batch["s_next"]
            self.buffers["terminated"][idxs] = episode_batch["terminated"]
            self.buffers["info"]["episodes_rewards"][self.episode_idx] = episode_batch["info"]["episode_rewards"]
            self.episode_idx = (self.episode_idx + 1) % self.evaluate_episodes

        # # xxh 1014不保熟.不加个这个的话似乎self.current_size不更新。# 然而不是，更新的。
        # self.current_size = self.current_size + batch_size


    def sample(self, batch_size):
        mini_size = min(batch_size, self.current_size)
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, mini_size)
        for key in self.buffers.keys():
            if key != "info":
                temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc):
        inc = inc or 1
        if self.current_idx + inc <= self.buffer_size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.buffer_size:
            overflow = inc - (self.buffer_size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.buffer_size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.buffer_size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def get_current_index(self):
        return self.current_idx

    def get_current_size(self):
        # xxh试着加的，不保熟。
        return self.current_size

    def save_buffer(self,folder=r"E:\XXH\RLtraining" ):
        # 存和读取都是很关键的，不然后面直接无法调试了。
        file_name = folder + r'\buffer.pkl'
        file = open(file_name, "wb")
        pickle.dump(self.buffers, file)
        file.close()

    def load_buffer(self, folder=r"E:\XXH\RLtraining" ):
        file_name = folder + r'\buffer.pkl'
        file = open(file_name, "rb")
        self.buffers = pickle.load(file)
        file.close()

class TD3Actor:
    def __init__(self, config):
        # 所以这个和前面的actor是什么关系？这里面也是actor-critic。看起来是target网络的关系?好像也不是
        # 算法相关参数
        self.results_path = config["results_path"]
        self.alg_name = config["alg_name"]
        self.seed = config["seed"]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.is_use_gpu = config["is_use_gpu"]

        self.epsilon = config["epsilon"]
        self.min_epsilon = config["min_epsilon"]
        self.anneal_epsilon = config["anneal_epsilon"]
        self.epsilon_anneal_scale = config["epsilon_anneal_scale"]
        self.render_per_episode = config.get("render_per_episode", 100)
        self.enable_render = config.get("enable_render", False)
        self.load_model = config["load_model"]

        # 环境相关参数
        self.env_name = None
        self.task_name = None
        self.n_agents = None
        self.n_actions = None
        self.state_shape = None
        self.obs_shape = None
        self.task_path = None
        self.runs_path = None
        # 策略网络
        self.actor_net = None
        self.actor_target_net = None
        self.critic_net = None
        self.critic_target_net = None

        # 判断时间
        self.start_time = None
        self.end_time = None
        self.time_duration = None
        self.runs_path = config["results_path"]
        self.load_filename = config["load_filename"]
        self.learn_steps = 0

    def prepare(self, **env_params):
        print("learner.prepare env info is", env_params)
        # self.env_name = env_params['env_name']
        # self.task_name = env_params['task_name']

        # self.task_path = self.results_path + '{}_{}/{}/{}/'.format(
        #     self.env_name,
        #     self.alg_name,
        #     self.task_name,
        #     self.seed
        # )

        # self.runs_path = dodo_path.get_runs_dir()

        # todo: 针对Gym输入，后续需要扩展
        self.n_agents = len(env_params["possible_agents"])
        self.obs_shape = env_params["obs_shape"]
        self.n_actions = env_params["action_shape"]
        self.state_shape = env_params["state_shape"]

        # 定义actor ，critic网络
        self.actor_net = Actor(state_dim=self.state_shape[0], action_dim=self.n_actions[0])
        self.actor_target_net = copy.deepcopy(self.actor_net)

        self.critic_net = Critic(state_dim=self.state_shape[0], action_dim=self.n_actions[0])
        self.critic_target_net = copy.deepcopy(self.critic_net)

        if self.load_model == True:
            self.load_policy(self.runs_path, self.load_filename)
            print("load model OK!")

        if self.is_use_gpu:
            self.actor_net.cuda()
            self.actor_target_net.cuda()
            self.critic_net.cuda()
            self.critic_target_net.cuda()


    def load_policy(self, path, rec):
        # todo: 断点重连，后续需要补充优化器optimizater和步数衰减器stepscale
        # path_actor = os.path.join(path, 'TD3_net_actor_params_{}.pkl'.format(rec))
        # path_critic = os.path.join(path, 'TD3_net_critic_params_{}.pkl'.format(rec))
        path_actor = os.path.join(path, "{}_TD3_net_actor_params.pkl".format(rec))
        path_critic = os.path.join(path, "{}_TD3_net_critic_params.pkl".format(rec))
        # print(path_actor)
        if os.path.exists(path_actor) and os.path.exists(path_critic):
            # if os.path.exists(path_actor):
            map_location = "cuda:0" if self.is_use_gpu else "cpu"
            self.actor_net.load_state_dict(
                torch.load(path_actor, map_location=map_location)
            )
            self.critic_net.load_state_dict(
                torch.load(path_critic, map_location=map_location)
            )
        else:
            raise Exception("No actor model!")

    def episode_start(self, episode_id: int):
        """
        每次episode采样开始时候，判断是否载入新策略
        :param episode_id:
        :return: 1、载入新策略; 2、NN隐变量重置; 3、..
        """
        epsilon = 1.0 - episode_id * self.anneal_epsilon
        self.epsilon = epsilon if epsilon > self.min_epsilon else self.min_epsilon
        # 记录开始时间
        self.start_time = time.time()
        pass

    def episode_end(self, _episode_id: int):
        """
        暂无判断
        :param _episode_id:
        :return:
        """
        pass

    def should_render(self, episode_id: int):
        """
        处理是否render，暂无实现
        :param _episode_id:
        :return:
        """
        return episode_id % self.render_per_episode == 0

    def select_action(
        self,
        agent: int,
        observation: np.ndarray,
        avail_actions: np.ndarray,
        last_action: np.ndarray = None,
    ):
        """

        :param agent:
        :param observation:
        :param avail_actions:
        :param last_action:
        :return:
        """

        # 架构嘛搭的很帅，然而没有用起来貌似。
        if self.load_model == False:
            if self.learn_steps < 1000:
                self.learn_steps = self.learn_steps + 1
                return np.array(
                    [
                        random.uniform(-1, 1),
                        random.uniform(-1, 1),
                        random.uniform(-1, 1),
                        random.uniform(-1, 1),
                    ],
                    dtype=np.float32,
                )
            else:
                self.learn_steps = self.learn_steps + 1
                state = torch.FloatTensor(observation)
                if self.is_use_gpu:
                    state.cuda()
                return self.actor_net(state).cpu().data.numpy().flatten()
        else:
            self.learn_steps = self.learn_steps + 1
            state = torch.FloatTensor(observation)
            if self.is_use_gpu:
                state.cuda()
            return self.actor_net(state).cpu().data.numpy().flatten()

    def learner_feedback(self, obj: dict) -> None:
        self.actor_net.load_state_dict(obj["actor_net"].state_dict())
        self.critic_net.load_state_dict(obj["critic_net"].state_dict())


class TD3Learner:
    def __init__(self, **config):
        # 定义学习参数
        self.results_path = config["results_path"]

        self.alg_name = config["alg_name"]
        # 随机种子设置
        self.seed = config["seed"]
        self.is_use_gpu = config["is_use_gpu"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.lamda = config["lamda"]
        self.eps = config["eps"]
        self.clip = config["clip"]
        self.epochs = config["epochs"]
        self.entropy_coef = config["entropy_coef"]
        self.max_train_steps = config["max_train_steps"]
        self.batch_size = config["batch_size"]
        self.target_update_cycle = config["target_update_cycle"]
        self.update_policy_cycle = config["update_policy_cycle"]
        self.save_cycle = config["save_cycle"]
        self.evaluate_episodes = config["evaluate_episodes"]
        self.evaluate_cycle = config["evaluate_cycle"]
        self.learn_times = config["learn_times"]
        self.average_reward_name = config.get("average_reward_name", "脱靶量")
        self.use_lr_decay = config["use_lr_decay"]
        self.set_adam_eps = config["set_adam_eps"]
        self.policy_noise = config["policy_noise"]
        self.noise_clip = config["noise_clip"]
        self.discount = config["discount"]
        self.store_episode_rewards = config["store_episode_rewards"]
        self.store_buffer = config["store_buffer"]
        self.max_store_buffer_count = config["max_store_buffer_count"]
        self.runs_path = config["results_path"]
        self.load_model = config["load_model"]
        self.save_filename = config["save_filename"]
        self.load_filename = config["load_filename"]
        # 环境相关参数
        self.env_name = None
        self.task_name = None
        self.n_agents = None
        self.state_shape = None
        self.n_actions = None
        self.episode_limit = config["episode_limit"]
        self.obs_shape = None
        # 策略学习网络：actor ,critic
        self.actor_net = None
        self.actor_target_net = None
        self.critic_net = None
        self.critic_target_net = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actor_parameters = None
        self.critic_parameters = None

        # 定义buffer
        self.buffer = ReplayBuffer(**config)
        self.start_buffer_size = config["start_buffer_size"]

        # 学习管理
        self.learn_steps = 0
        self.sample_steps = 0
        self.task_path = None
        self.writer = None
        self.logprob = None
        self.values = None

        # 评估管理
        self.evaluate_steps = 0
        self.evaluate_rewards = 0

        self.max_action = 1
        if self.load_model == False:
            self.actor_loss = []
            self.critic_loss = []
        else:
            with open(self.results_path + r"loss_data.csv", newline="") as csvfile:
                reader = csv.reader(csvfile)
                data = [row for row in reader]
                data1 = data[0]
                self.actor_loss = list(map(float, data1))
                data2 = data[2]
                self.critic_loss = list(map(float, data2))

        # 不得把actor和critic弄到这里面？
        # self.what_net = TD3Actor(config)
        # 麻了，这里面似乎是整了actor和critic的。这么写应该是为了适应下面那个RlAlgo那个吧可能?
        # 以及这里申明放在了prepare里面。

        # xxh1014不保熟，把环境弄进来了。
        self.env = config["env"]

    def prepare(self, **env_params):

        # xxh 1014，不保熟。
        # self.what_net.prepare(**env_params)

        # self.env_name = env_params['env_name']
        # self.task_name = env_params['task_name']
        self.n_agents = len(env_params["possible_agents"])
        # self.n_agents = 1

        # self.task_path = self.results_path + '{}_{}/{}/{}/'.format(
        #     self.env_name,
        #     self.alg_name,
        #     self.task_name,
        #     self.seed
        # )
        # 配置训练过程结果存储路径
        # self.runs_path = dodo_path.get_runs_dir()

        # todo: 针对Gym输入，后续需要扩展
        self.obs_shape = env_params["obs_shape"]
        self.n_actions = env_params["action_shape"]
        self.state_shape = env_params["state_shape"]

        self.actor_net = Actor(state_dim=self.state_shape[0], action_dim=self.n_actions[0])
        self.actor_target_net = copy.deepcopy(self.actor_net)

        self.critic_net = Critic(state_dim=self.state_shape[0], action_dim=self.n_actions[0])
        self.critic_target_net = copy.deepcopy(self.critic_net)

        if self.load_model == True:
            self.load_policy(self.runs_path, self.load_filename)
            print("load model OK!")

        if self.is_use_gpu:
            self.actor_net.cuda()
            self.actor_target_net.cuda()
            self.critic_net.cuda()
            self.critic_target_net.cuda()

        self.actor_parameters = self.actor_net.parameters()
        self.critic_parameters = self.critic_net.parameters()
        # print(self.actor_parameters)

        # 定义优化器和优化参数
        if self.set_adam_eps:
            self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.actor_lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=self.critic_lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=self.critic_lr)

        # 初始化 replay buffer
        self.buffer.prepare(**env_params)

        # 学习管理
        self.learn_steps = 0
        self.evaluate_steps = 0

        # tensorboard 记录存储
        # self.writer = SummaryWriter(dodo_path.get_events_dir(), flush_secs=10)
        self.episode_start_time = None

    def store(self, episode_batch):
        """
        存储一局游戏数据
        :param episode_batch: 接收来自采集器输入
        :return:
        """
        one_episode_batch = self._data_normalization(episode_batch)
        self.buffer.add(one_episode_batch)
        self.sample_steps += 1

        if self.store_episode_rewards:
            info = self.buffer.buffers["info"]
            # 遍历episodes_rewards并将每个值存储到per_episode_reward中,并用dodo.log 保存
            with np.nditer(info["episodes_rewards"], flags=["c_index"]) as it:
                for per_episode_reward in it:
                    per_episode_reward = float(per_episode_reward)
                    # print(f'Epis`ode reward: {per_episode_reward}')
                    # dodo.log(topic="算法评估", metrics={
                    #     "每局奖励值": per_episode_reward
                    # })

        # if self.episode_start_time is None:
        #     self.episode_start_time = time.time()
        # else:
        #     time_duration = time.time() - self.episode_start_time
        #     dodo.log(topic="采样耗时",
        #              metrics={
        #                  "采样耗时(秒)": time_duration
        #              })
        #     self.episode_start_time = time.time()

        # if self.store_buffer:
        #     # Create directory if it doesn't exist
        #     os.makedirs('/task/buffer_episodes', exist_ok=True)
        #
        #     # Save the list to a JSON file
        #     episode_id = episode_batch.get('episode_id', 0)
        #     if self.max_store_buffer_count is None or episode_id < self.max_store_buffer_count:
        #         filename = f"/task/buffer_episodes/ppo_one_episode_batch_{episode_id}.json"
        #         with open(filename, 'w') as f:
        #             json.dump(episode_batch, f, cls=NumpyEncoder)

    def train(self):
        total_actor_loss, total_critic_loss = 0, 0
        state = self.env.reset()  # xxh 1014 不保熟。
        for i in range(self.learn_times):
            # evaluate_cycle步数，开始评估
            if self.learn_steps > 0 and self.learn_steps % self.evaluate_cycle == 0:
                self.evaluate()

            # 啥玩意？所以不用一边填充buffer一边从中采样？所以和环境互动的咋说。
            # 没毛病，论文里就是一边填充buffer一遍和环境互动
            is_done = self.interaction_with_env(self.env, state)
            buffer_size = self.buffer.get_current_size()
            if buffer_size<self.start_buffer_size:
                continue  # 不慌开始训练，先把buffer填了。


            # 从buffer中采样学习
            mini_batch = self.buffer.sample(self.batch_size)


            actor_loss, critic_loss = self._learn(mini_batch)
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss

            if self.learn_steps % 10 == 0:
                print("learn_steps = {}, actor_loss = {}, critic_loss = {},".format(
                        self.learn_steps, actor_loss, critic_loss))
                self.actor_loss.append(actor_loss)
                learn_steps = list(range(len(self.actor_loss)))
                plt.plot(learn_steps, self.actor_loss)
                plt.xlabel("learn_steps(*10)")
                plt.ylabel("actor_loss")
                plt.savefig(self.results_path + r"\actor_loss.png")
                plt.close()
                self.critic_loss.append(critic_loss)
                learn_steps = list(range(len(self.critic_loss)))
                plt.plot(learn_steps, self.critic_loss)
                plt.xlabel("learn_steps(*10)")
                plt.ylabel("critic_loss")
                plt.savefig(self.results_path + r"\critic_loss.png")
                plt.close()
                if self.learn_steps % 100 == 0:
                    with open(self.results_path + r"loss_data.csv", "w") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([self.actor_loss, self.critic_loss])
            # self.writer.add_scalar('train/loss', actor_loss, critic_loss, self.learn_steps)

            # 实时存储最新策略
            if self.learn_steps % self.update_policy_cycle == 0:
                self._save_policy(self.runs_path, "latest")
                # send_message['is_save_policy'] = True

            # # 存储历史策略
            # if self.learn_steps % self.save_cycle == 0:
            #     self._save_policy(self.runs_path, self.learn_steps)
            if is_done:
                break

        log(topic="训练",
            metrics={"actor_loss": float(total_actor_loss / self.learn_times),
                "critic_loss": float(total_critic_loss / self.learn_times),},)

        send_message = dict()
        send_message["actor_net"] = self.actor_net
        send_message["critic_net"] = self.critic_net

        return send_message

    def train_auto(self):
        # xxh 1014 不保熟。

        for episode_now in range(self.episode_limit):
            print("start number "+str(episode_now) + " episode. ")
            # self.reset()
            # 不对，每个episode里面应该是不能reset的。
            self.train()

    def reset(self):
        # 重置一下各种东西，包括路径什么的，让它能够真正意义上地重新算下一个
        # 定义buffer
        # self.buffer = ReplayBuffer(**config)
        # self.start_buffer_size = config["start_buffer_size"]
        # 讲道理buffer不用reset

        # 学习管理
        self.learn_steps = 0
        self.sample_steps = 0
        self.task_path = None
        self.writer = None
        self.logprob = None
        self.values = None

        # 评估管理
        self.evaluate_steps = 0
        self.evaluate_rewards = 0

        self.max_action = 1
        if self.load_model == False:
            self.actor_loss = []
            self.critic_loss = []
        else:
            with open(self.results_path + r"loss_data.csv", newline="") as csvfile:
                reader = csv.reader(csvfile)
                data = [row for row in reader]
                data1 = data[0]
                self.actor_loss = list(map(float, data1))
                data2 = data[2]
                self.critic_loss = list(map(float, data2))


        # 然后重新生成一个。
        pass

    def evaluate(self,):
        info = self.buffer.buffers["info"]
        self.evaluate_steps += 1

        # self.writer.add_scalar(
        #     'train/average_reward',
        #     np.average(info['episodes_rewards']),
        #     self.learn_steps
        # )

        # reward_metric_name = f"{self.average_reward_name}/{self.evaluate_cycle}局"
        # avg_reward = np.average(info['episodes_rewards'])

        # dodo.log(topic="结果统计", metrics={
        #     reward_metric_name: avg_reward,
        # })

        print(
            "learn steps = {}, average_reward = {}".format(
                self.learn_steps,
                np.average(info["episodes_rewards"]),
            )
        )
        log(
            topic="评估",
            metrics={
                "average_reward": np.average(info["episodes_rewards"]),
            },
        )

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    max_episode_len = max(transition_idx + 1, max_episode_len)
                    break
        if max_episode_len == 0:
            max_episode_len = self.episode_limit
        return max_episode_len

    def get_q_values(self, batch):
        """
        计算Q值
        """
        # 什么东西？空的？整不会了。不行还是得给它写了
        # xxh1014 不保熟。

        pass

    def _learn(self, learn_batch):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        # 移除info
        if "info" in learn_batch.keys():
            learn_batch.pop("info")
        else:
            pass

        # 转化为tensor
        for key in learn_batch.keys():
            if key == "a":
                learn_batch[key] = torch.tensor(np.array(learn_batch[key]), dtype=torch.float32)
            else:
                learn_batch[key] = torch.tensor(np.array(learn_batch[key]), dtype=torch.float32)

        states = learn_batch["s"]
        actions = learn_batch["a"]
        rewards = learn_batch["r"][0]
        next_states = learn_batch["s_next"]
        dones = learn_batch["terminated"]

        if self.is_use_gpu:
            states.cuda()
            actions.cuda()
            rewards.cuda()
            next_states.cuda()
            dones.cuda()
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target_net(next_states) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target_net(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q_lmda = rewards + dones * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_net(states, actions)
        # current_Q1_2, current_Q2_2 = self.critic(next_state, next_action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q_lmda) + F.mse_loss(current_Q2, target_Q_lmda)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor losse
        actor_loss = -self.critic_net.Q1(states, self.actor_net(states)).mean()
        # Optimize the actor

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learn_steps > 0 and self.learn_steps % self.target_update_cycle == 0:
            self.actor_net.load_state_dict(self.actor_net.state_dict())
            self.critic_net.load_state_dict(self.critic_net.state_dict())

        for param, target_param in zip(self.critic_net.parameters(), self.critic_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 学习步数+1
        self.learn_steps += 1

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self._lr_decay(self.learn_steps)

        return actor_loss.item(), critic_loss.item()

    def _compute_advantage(self, gamma, lamda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0

        for delta in td_delta[::-1]:
            advantage = gamma * lamda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float32)

    def _lr_decay(self, total_steps):
        actor_lr_now = self.actor_lr * (1 - total_steps / self.max_train_steps)
        critic_lr_now = self.critic_lr * (1 - total_steps / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p["lr"] = actor_lr_now
        for p in self.critic_optimizer.param_groups:
            p["lr"] = critic_lr_now

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _save_policy(self, path, rec):
        if not os.path.exists(path):
            os.makedirs(path)

        path_actor = os.path.join(path, "{}_TD3_net_actor_params.pkl".format(rec))
        path_critic = os.path.join(path, "{}_TD3_net_critic_params.pkl".format(rec))
        torch.save(self.actor_net.state_dict(), path_actor)
        torch.save(self.critic_net.state_dict(), path_critic)

        # # 根据时间存pkl
        #
        # timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        #
        # path_actor_time = os.path.join(path, f'TD3_net_actor_params_{timestamp}.pkl')
        # path_critic_time = os.path.join(path, f'TD3_net_critic_params_{timestamp}.pkl')
        # torch.save(self.actor_net.state_dict(), path_actor_time)
        # torch.save(self.critic_net.state_dict(), path_critic_time)

    def save_policy(self, path="/task/finals"):
        path_actor = os.path.join(path, "TD3_net_actor_params.pt")
        path_critic = os.path.join(path, "TD3_net_critic_params.pt")
        torch.save(self.actor_net, path_actor)
        torch.save(self.critic_net, path_critic)

    def load_policy(self, path, rec):
        # todo: 断点重连，后续需要补充优化器optimizater和步数衰减器stepscale
        # print(path)
        path_actor = os.path.join(path, "{}_TD3_net_actor_params.pkl".format(rec))
        path_critic = os.path.join(path, "{}_TD3_net_critic_params.pkl".format(rec))

        if os.path.exists(path_actor) and os.path.exists(path_critic):
            # if os.path.exists(path_actor):
            map_location = "cuda:0" if self.is_use_gpu else "cpu"

            self.actor_net.load_state_dict(
                torch.load(path_actor, map_location=map_location)
            )
            self.critic_net.load_state_dict(
                torch.load(path_critic, map_location=map_location)
            )
        else:
            raise Exception("No actor model!")

    def _data_normalization(self, episode_batch):
        """
        episode_batch: {
        "episode_id": 0,
        "transitions": [{
            "step": 1,
            "state": ndarray,
            "state_next": ndarray,
            "agents": ["agent1", "agent2", ...],
            "actions": [ndarray, ndarray, ...],
            "obs": [ndarray, ndarray,],
            "obs_next": [ndarray, ndarray,],
            "avail_actions": [ndarray, ndarray,],
            "avail_actions_next": [ndarray, ndarray,],
            "rewards": [float, float,...],
            "infos": [{}, {},...],
            "dones": [bool, bool, ...]
        }]
        }
        """
        max_episode_limit = 0
        for i in range(len(episode_batch["transitions"])):
            if episode_batch["transitions"][i]["rewards"][0] == 0:
                continue
            else:
                max_episode_limit = max_episode_limit + 1

        one_episode_batch = {
            "s": np.zeros([max_episode_limit, *self.state_shape]),
            "r": np.zeros([max_episode_limit, 1]),
            "a": np.zeros([max_episode_limit, *self.n_actions]),
            # 'a_logprob': np.zeros([max_episode_limit, 1]),
            "s_next": np.zeros([max_episode_limit, *self.state_shape]),
            # 'dw': np.zeros([max_episode_limit, 1]),
            "terminated": np.ones([max_episode_limit, 1]),
            # todo: 1局游戏评估指标
            "info": {
                "episode_rewards": 0,
            },
        }
        j = 0
        for i in range(len(episode_batch["transitions"])):
            if episode_batch["transitions"][i]["rewards"][0] == 0:
                continue
            else:
                one_episode_batch["s"][j] = np.array(episode_batch["transitions"][i]["state"])
                one_episode_batch["a"][j] = np.array(episode_batch["transitions"][i]["actions"][0])
                # one_episode_batch['a_logprob'][i][0] = episode_batch["transitions"][i]["actions_logprobs"][0]
                one_episode_batch["r"][j][0] = np.sum(episode_batch["transitions"][i]["rewards"])
                one_episode_batch["info"]["episode_rewards"] += one_episode_batch["r"][j][0]
                one_episode_batch["s_next"][j] = np.array(episode_batch["transitions"][i]["state_next"])
                # one_episode_batch['s_next'][i] = np.array(episode_batch["transitions"][i]["state_next"])
                dones = episode_batch["transitions"][i]["dones"]
                one_episode_batch["terminated"][j] = (np.array([1]) if dones[0] is True else np.array([0]))
                j = j + 1
        one_episode_batch["terminated"][j - 1] = np.array([1])
        return one_episode_batch

    # xxh 凭感觉加的，不保熟
    def interaction_with_env(self, env, state):
        # 就单纯的和环境互动呗。

        # select action with exploration noise
        # action_raw = self.what_net.select_action(114514, state, 1919810, 0)
        state_tensor = torch.from_numpy(state)
        # state_tensor.double()
        # state_tensor.to(torch.double)
        state_tensor = state_tensor.float()
        action_raw = self.actor_net(state_tensor)
        random_noise = (np.random.random((len(action_raw),)) - 0.5) * 2  # TODO 按说应该加点更专业的噪音，下次一定
        # 20240429，下次也不一定

        action = action_raw.detach().numpy() + random_noise

        # observe reward r and new state s'
        next_state, reward, is_done, info = env.step(action)

        # store transition tuple (s, a, r, s') in buffer.
        # 先用最简单的，每次存一个进去再说。
        episode = {}
        episode["s"] = state
        episode["r"] = reward
        episode["a"] = action
        episode["s_next"] = next_state
        episode["terminated"] = is_done
        episode["info"] = {"episode_rewards": 0}  # 尚未能看懂这个是干什么的。
        self.buffer.add(episode)

        if self.buffer.get_current_size()%114==5:
            self.buffer.save_buffer(self.results_path)
            self.buffer.save_buffer()
        # 然后再检测一下buffer里面的个数？如果个数够了再进行之后的训练，如果不够，那就先不慌。
        # buffer_size = self.buffer.get_current_size()
        # print("xxh modified, unfinished yet")
        return is_done

    def interaction_with_env2(self, env, state):
        # 这个是用来测试和实际用的时候搞的，就不加什么奇奇怪怪的东西了。
        state_tensor = torch.from_numpy(state)
        state_tensor = state_tensor.float()
        action_raw = self.actor_net(state_tensor)
        action = action_raw.detach().numpy()

        # observe reward r and new state s'
        next_state, reward, is_done, info = env.step(action)

        # store transition tuple (s, a, r, s') in buffer.
        # 先用最简单的，每次存一个进去再说。
        episode = {}
        episode["s"] = state
        episode["r"] = reward
        episode["a"] = action
        episode["s_next"] = next_state
        episode["terminated"] = is_done
        episode["info"] = {"episode_rewards": 0}  # 尚未能看懂这个是干什么的。
        self.buffer.add(episode)

        if self.buffer.get_current_size()%114 == 5:
            self.buffer.save_buffer(self.results_path)
            self.buffer.save_buffer()

        return episode

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
