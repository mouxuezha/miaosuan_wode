from env_for_RL import EnvForRL
from alg_TD3 import TD3Learner
from alg_PPO import PPO
import numpy as np
import os

from auto_run import record_result, save_replay

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
    config["episode_limit"] = 514 #  9999

    config = get_defualt_value_TD3Actor(config)

    return config

def get_defualt_value_TD3Actor(config):
    config["epsilon"] = 0.0001
    config["min_epsilon"] = 0.0001
    config["anneal_epsilon"] = 0.0001
    config["epsilon_anneal_scale"] = 1
    return config

def get_defualt_value_PPO():
    
    # ####### initialize environment hyperparameters ######
    # env_name = "RoboschoolWalker2d-v1"
    
    # has_continuous_action_space = True  # continuous action space; else discrete

    # max_ep_len = 1000                   # max timesteps in one episode
    max_ep_len = 100  # 其实94就打完一把了在episode里面
    # max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    config["print_freq"] = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    config["log_freq"] = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    config["save_model_freq"] = int(1e5)          # save model frequency (in num timesteps)

    config["action_std"] = 0.6                    # starting std for action distribution (Multivariate Normal)
    config["action_std_decay_rate"] = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    config["min_action_std"] = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    config["action_std_decay_freq"] = int(2.5e5)  # action_std decay frequency (in num timesteps)
    # #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    
    config["update_timestep"] = max_ep_len * 4      # update policy every n timesteps

    config["K_epochs"]  =  80          # update policy for K epochs in one PPO update

    config["eps_clip"]= 0.2          # clip parameter for PPO

    config["gamma"] = 0.99            # discount factor

    config["lr_actor"] = 0.0003       # learning rate for actor network
    
    config["lr_critic"] = 0.001       # learning rate for critic network

    config["random_seed"] = 0         # set random seed if required (0 = no random seed)    
    config["results_path"] = "./RLtraining/models"
    return config

def get_defualt_value_env(env):
    env_params = {}
    env_params["possible_agents"] = np.array([1])
    env_params["obs_shape"] = np.array([env.state_dim])
    env_params["action_shape"] = np.array([env.action_dim])
    env_params["state_shape"] = np.array([env.state_dim])
    env_params["state_dim"] = env.state_dim
    env_params["action_dim"] = env.action_dim
    config["has_continuous_action_space"] = True  # this is for PPO, continuous action space; else discrete
    config["env_name"] = "cross_fire"
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

def auto_run_RL_PPO(location = r"./RLtraining"):
    env = EnvForRL()
    env_params = get_defualt_value_env(env)
    config = get_defualt_value_PPO(env_params)
    config["env"] = env  # 按照xxh之前的习惯，还是把环境直接弄进里面去比较合适。
    env_name = env_params["env_name"]
    random_seed = 0
    config["results_path"] = location
    


    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)    


    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(**config)
    
    # then start train.
    try:
        ppo_agent.train_auto()
    except:
        print("ppo_agent seems G.")

    
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

def auto_test(location = r"./RLtraining/models0", jieguo_location = r"./RLtraining/jieguo"):
    # 这个是自动的推演，给出一个agent，那就可以加载这个agent，然后进行推演。
    
    # 初始化一个用来后处理的class
    jieguo = record_result()
    jieguo.record_config("test RL crossfire, location = " + location)

    # 上来先把环境初始化出来，
    env = EnvForRL()
    env_params = get_defualt_value_env(env)
    
    # 然后把agent加载进来，上来先初始化一波各种设定
    config = get_defualt_value()
    config["env"] = env  # 按照xxh之前的习惯，还是把环境直接弄进里面去比较合适。
    config["results_path"] = location
    config["load_model"] = True   
    agent = TD3Learner(**config)
    agent.prepare(**env_params) # 读取这一步体现在这里面了

    # 然后开始推演了
    tuiyan_num = 114 # 推演的局数先设定一个
    for i in range(tuiyan_num):
        all_state = agent.tuiyan_single()

        zip_name = "tuiyan_" + str(i)
        zip_name = save_replay(zip_name,all_state)

        jieguo.get_result_single(all_state,zip_name)
        print("tuiyan_" + str(i) + " done, everyting looks ok. \n\n\n\n\n")
        
    jieguo.get_result_all(jieguo.all_games)

if __name__ == "__main__":
    
    # 思路打开，也不是说非要gym不可，心中有gym，到处都是gym

    flag = 1
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
    elif flag == 3:
        # 这个是自动版的“一遍一遍跑然后存下来”了。但是是PPO的。
        location = r"./RLtraining"
        auto_run_RL_PPO(location)

