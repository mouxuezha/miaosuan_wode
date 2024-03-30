# this is the first shishi, En Taro XXH! 

import json
import os
import time
import zipfile
import sys 
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
        

        
        pass

    def __init_defend(self):
        from agent_guize import agent_guize
        self.env = DefendEnv(3,5,1)
        self.agent = agent_guize()
        self.begin = time.time()   

    def __init_crossfire(self):
        # from ai.agent import Agent
        from agent_guize import agent_guize
        self.env = CrossFireEnv()
        self.agent = agent_guize()
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



    def run_single(self):
        # varialbe to build replay
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
        pass

# copy from demo code
def save_replay(replay_name, data):
    zip_name = f"logs/replays/{replay_name}.zip"
    if not os.path.exists("logs/replays/"):
        os.makedirs("logs/replays/")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(f"{replay_name}.json", json.dumps(data, ensure_ascii=False, indent=None, separators=(',', ':')).encode('gbk'))


if __name__ == "__main__":
    for i in range(5):
        # shishi = auto_run(env_name="defend")
        shishi = auto_run(env_name="crossfire")
        shishi.run_single()

