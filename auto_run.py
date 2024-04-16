# this is the first shishi, En Taro XXH! 

import json
import os
import time
import zipfile
import sys 
import datetime
sys.path.append("/home/vboxuser/Desktop/miaosuan/starter-kit")
from train_env.cross_fire_env import CrossFireEnv
from train_env.scout_env import ScoutEnv
from train_env.defend_env import DefendEnv

class auto_run():
    def __init__(self,env_name="crossfire") -> None:

        # these flags identified the status from env.
        self.red_flag = 0
        self.blue_flag = 1
        self.white_flag = -1

        if env_name == "crossfire":
            self.__init_crossfire()
        elif env_name == "defend":
            self.__init_defend()
        elif env_name == "scout":
            self.__init_scout()
        else:
            raise Exception("auto_run: invalid env setting, G.")
        
        self.__init_analyse()
        pass

    def __init_defend(self):
        from ai.agent_guize import agent_guize
        self.env = DefendEnv(3,5,1)
        self.agent = agent_guize()
        self.begin = time.time()   

    def __init_crossfire(self):
        # from ai.agent import Agent
        # from agent_guize import agent_guize
        from ai.agent import Agent
        self.env = CrossFireEnv()
        self.agent = Agent()
        self.begin = time.time()
        # # varialbe to build replay
        # self.all_states = []

        # ai_user_name = 'myai'
        # ai_user_id = 1

        # state = self.env.setup({'user_name': ai_user_name, 'user_id': ai_user_id})
        # self.all_states.append(state[self.white_flag])
        # self.agent.setup(
        #     {
        #         "scenario": self.env.scenario_data,
        #         "basic_data": self.env.basic_data,
        #         "cost_data": self.env.cost_data,
        #         "see_data": self.env.see_data,
        #         "seat": 1,
        #         "faction": 0,
        #         "role": 0,
        #         "user_name": ai_user_name,
        #         "user_id": ai_user_id,
        #     }
        # )

    def __init_scout(self):
        from ai.agent import Agent
        self.env = ScoutEnv()
        self.agent = Agent()
        self.begin = time.time()

    def __init_analyse(self):
        self.reward_ave = 0 
        self.return_ave = 0 
        self.reward_list = [] 
        self.return_list = [] 
        self.config_str = ""
        self.all_games = []
    
    def run_single(self):
        # varialbe to build replay
        self.begin = time.time()

        self.all_states = []


        ai_user_name = 'myai'
        ai_user_id = 1
        try:
            state = self.env.setup({'user_name': ai_user_name, 'user_id': ai_user_id})
        except:
            state = self.env.setup({'user_name': ai_user_name, 'user_id': ai_user_id, 'seat':1})

        self.all_states.append(state[self.white_flag])
        self.agent.setup(
            {
                "scenario": self.env.scenario_data,
                "basic_data": self.env.basic_data,
                "cost_data": self.env.cost_data,
                "see_data": self.env.see_data,
                "seat": 1,
                "faction": 0,
                "role": 0,
                "user_name": ai_user_name,
                "user_id": ai_user_id,
            }
        )

        done = False
        while not done:
            actions = self.agent.step(state[self.red_flag])
            state, done = self.env.step(actions)
            self.all_states.append(state[self.white_flag])

        self.agent.reset()
        self.env.reset()


        # save replay
        save_replay(self.begin, self.all_states)
        return self.all_states
        # pass

class record_result():
    def __init__(self):
        self.reward_ave = 0 
        self.return_ave = 0 
        self.reward_list = [] 
        self.return_list = [] 
        self.config_str = ""
        self.all_games = []

    def record_config(self, config_str:str):
        self.config_str = config_str

    def get_result_single(self,all_states):
        if len(self.config_str)==0:
            raise Exception("auto_run: config_str is empty, G. do not be lazy.")
        self.all_games.append(all_states)
        
    def get_result_all(self,all_games):
        # get result from one round
        geshu = len(all_games)
        for i in range(geshu):
            state_single = all_games[i][-1]
            reward_single = state_single["reward"]
            return_single = state_single["return"]
            self.reward_list.append(reward_single)
            self.return_list.append(return_single)
            self.reward_ave = self.reward_ave + reward_single
            self.return_ave = self.return_ave + return_single
        self.reward_ave = self.reward_ave / geshu 
        self.return_ave = self.return_ave / geshu 
        
        # then save all these things.
        file_name = f"logs/replays/result_analyse.txt"
        if not os.path.exists("logs/replays/"):
            os.makedirs("logs/replays/")
        current_time = datetime.datetime.now()
        file_txt = open(file_name, "a+")
        file_txt.write(f"{self.config_str}\n")
        file_txt.write(f"time: {current_time}\n")
        file_txt.write(f"reward_ave: {self.reward_ave}\n")
        file_txt.write(f"return_ave: {self.return_ave}\n")
        file_txt.write(f"reward_list: {self.reward_list}\n")
        file_txt.write(f"return_list: {self.return_list}\n")
        file_txt.write("\n\n\n")
        file_txt.close()
        pass

# copy from demo code
def save_replay(replay_name, data):
    zip_name = f"logs/replays/{replay_name}.zip"
    if not os.path.exists("logs/replays/"):
        os.makedirs("logs/replays/")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(f"{replay_name}.json", json.dumps(data, ensure_ascii=False, indent=None, separators=(',', ':')).encode('gbk'))


if __name__ == "__main__":
    jieguo = record_result()
    jieguo.record_config("debug raolu, 增加了分类讨论的绕路逻辑，增加了被打的势场修正，原则上能好点。")
    for i in range(50):
        # shishi = auto_run(env_name="defend")
        shishi = auto_run(env_name="crossfire")
        # shishi = auto_run(env_name="scout")
        all_states_single = shishi.run_single()
        jieguo.get_result_single(all_states_single)
    jieguo.get_result_all(jieguo.all_games)

