from env_for_RL import EnvForRL
from alg_TD3 import TD3Learner
import numpy as np
import os


def get_defualt_value():

    config = {}

    # 定义学习参数
    config["results_path"] = "./RLtraining/models"
    config["alg_name"] = "TD3"
    # 随机种子设置
    config["seed"] = 0
    config["is_use_gpu"] = False
    config["gamma"] = 0.99
    config["tau"] = 0.005
    config["actor_lr"] = 0.0003
    config["critic_lr"] = 0.0003
    config["lamda"] = 0.95
    config["eps"] = 0.2
    config["clip"] = 1
    config["epochs"] = 1
    config["entropy_coef"] = 0.01
    config["max_train_steps"] = 100000
    config["batch_size"] = 160
    config["target_update_cycle"] = 50
    config["update_policy_cycle"] = 5
    config["save_cycle"] = 100
    config["evaluate_episodes"] = 16
    config["evaluate_cycle"] = 10
    config["learn_times"] = 600 # 3000/5=600
    config["use_lr_decay"] = True
    config["set_adam_eps"] = True
    config["policy_noise"] = 0.2
    config["noise_clip"] = 0.2
    config["discount"] = 0.99
    config["store_episode_rewards"] = False
    config["store_buffer"] = True
    config["max_store_buffer_count"] = 1
    config["results_path"] = "./RLtraining/models"
    config["load_model"] = False
    config["save_filename"] = 'latest'
    config["load_filename"] = 'latest'

    config["start_buffer_size"] = 3000
    config["buffer_size"] = config["start_buffer_size"] * 10
    config["episode_limit"] = 18 #  9999

    config = get_defualt_value_TD3Actor(config)

    return config

def get_defualt_value_TD3Actor(config):
    config["epsilon"] = 0.0001
    config["min_epsilon"] = 0.0001
    config["anneal_epsilon"] = 0.0001
    config["epsilon_anneal_scale"] = 1
    return config

def get_defualt_value_env(env):
    env_params = {}
    env_params["possible_agents"] = np.array([1])
    env_params["obs_shape"] = np.array([env.state_dim])
    env_params["action_shape"] = np.array([env.action_dim])
    env_params["state_shape"] = np.array([env.state_dim])
    return env_params

def auto_run_RL(location = r"./RLtraining"):
    config = get_defualt_value()

    env = EnvForRL()
    env_params = get_defualt_value_env(env)

    config["env"] = env  # 按照xxh之前的习惯，还是把环境直接弄进里面去比较合适。

    for i in range(10):

        # 搞一个location
        save_folder = generate_location(location, name = "models")
        config["results_path"] = save_folder

        # 然后开始走一遍。
        agent = TD3Learner(**config)

        agent.prepare(**env_params)
        try:
            agent.train_auto()
        except:
            print("sdk seems G, just save and contibue")

def generate_location(location = r"./RLtraining", name="models", **kargs):

    for i in range(114514):

        location_new = location + "/" + name + str(i)

        if not(os.path.exists(location_new)):
            # 这个路径还没被用。
            os.mkdir(location_new)
            return location_new

    return "fail to generate location"

def continue_train(location = r"./RLtraining/models0"):
    # 这个是尝试进行读取和续算，反正都是需要的
    config = get_defualt_value()

    env = EnvForRL()
    env_params = get_defualt_value_env(env)

    config["env"] = env  # 按照xxh之前的习惯，还是把环境直接弄进里面去比较合适。

    config["results_path"] = location
    config["load_model"] = True

    # 然后开始
    agent = TD3Learner(**config)
    agent.prepare(**env_params)

    # 读取已有的智能体
    agent.train_auto()

if __name__ == "__main__":
    
    # 思路打开，也不是说非要gym不可，心中有gym，到处都是gym

    flag = 0
    if flag == 0:
        while(True):
            config = get_defualt_value()

            env = EnvForRL()
            env_params = get_defualt_value_env(env)

            config["env"] = env  # 按照xxh之前的习惯，还是把环境直接弄进里面去比较合适。

            agent = TD3Learner(**config)

            agent.prepare(**env_params)

            agent.train_auto()
    elif flag == 1:
        # 这个是自动版的“一遍一遍跑然后存下来”了。
        location = r"./RLtraining"
        auto_run_RL(location)
    elif flag == 2:
        # 这个是手动版的“重新计算其中某一个点”
        location = r"./RLtraining/models0"
        continue_train(location)

