import json
import os
import time
import zipfile
import sys 
import datetime
sys.path.append("/home/vboxuser/Desktop/miaosuan/starter-kit")
import numpy,pickle
from copy import deepcopy
from collections import deque

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
        elif env_name == "renji":
            self.__init_renji()
        else:
            raise Exception("auto_run: invalid env setting, G.")
        
        self.__init_analyse()
        pass

    def __init_defend(self):
        from ai.agent import Agent
        from train_env.cross_fire_env import CrossFireEnv
        from train_env.scout_env import ScoutEnv
        from train_env.defend_env import DefendEnv        
        self.env = DefendEnv(3,5,1)
        self.agent = Agent()
        self.begin = time.time()   

    def __init_crossfire(self):
        # from ai.agent import Agent
        # from agent_guize import agent_guize
        from train_env.cross_fire_env import CrossFireEnv
        from train_env.scout_env import ScoutEnv
        from train_env.defend_env import DefendEnv
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
        from train_env.cross_fire_env import CrossFireEnv
        from train_env.scout_env import ScoutEnv
        from train_env.defend_env import DefendEnv        
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
    
    def __init_renji(self):
        from ai.agent import Agent
        from train_env import TrainEnv
        self.red1 = Agent()
        self.blue1 = Agent()
        self.env1 = TrainEnv()
        self.begin = time.time()
        self.starter_kit_location = "C:\\Users\\Administrator\\Desktop\\insiaosuan\\starter-kit"

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
        zip_name = save_replay(self.begin, self.all_states)
        return self.all_states, zip_name
        # pass

    def run_in_single_agent_mode(self):
        """
        run demo in single agent mode
        """
        print("running in single agent mode...")
        # instantiate agents and env
        red1 = self.red1
        blue1 = self.blue1
        env1 = TrainEnv()
        begin = time.time()

        # get data ready, data can from files, web, or any other sources
        with open(self.starter_kit_location + "/data/scenarios/1014-WRJD-5.json", encoding='utf8') as f:
            scenario_data = json.load(f)
        with open(self.starter_kit_location + "/data/maps/2022081021/basic.json", encoding='utf8') as f:
            basic_data = json.load(f)
        with open(self.starter_kit_location + '/data/maps/2022081021/cost.pickle', 'rb') as file:
            cost_data = pickle.load(file)
        see_data = numpy.load(self.starter_kit_location + "/data/maps/2022081021/see.npz")['data']

        # varialbe to build replay
        all_states = []

        # player setup info
        player_info = [{
            "seat": 1,
            "faction": 0,
            "role": 1,
            "user_name": "demo",
            "user_id": 0
        },
        {
            "seat": 11,
            "faction": 1,
            "role": 1,
            "user_name": "demo",
            "user_id": 10
        }]

        # env setup info
        env_step_info = {
            "scenario_data": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "player_info": player_info
        }

        # setup env
        state = env1.setup(env_step_info)
        all_states.append(state[self.white_flag])
        print("Environment is ready.")

        # setup AIs
        red1.setup(
            {
                "scenario": deepcopy(scenario_data),
                "basic_data":deepcopy(basic_data),
                "cost_data": deepcopy(cost_data),
                "see_data": deepcopy(see_data),
                "seat": 1,
                "faction": 0,
                "role": 1,
                "user_name": "demo",
                "user_id": 0,
                "state": state,
            }
        )
        blue1.setup(
            {
                "scenario": deepcopy(scenario_data),
                "basic_data":deepcopy(basic_data),
                "cost_data": deepcopy(cost_data),
                "see_data": deepcopy(see_data),
                "seat": 11,
                "faction": 1,
                "role": 1,
                "user_name": "demo",
                "user_id": 10,
                "state": state,
            }
        )
        print("agents are ready.")

        assign_task = {
            "actor": 1, #"int 动作发出者席位",
            "type": 209,
            "seat": 1, #"命令接收人id",
            "hex": 3456, #"任务目标位置",
            "radius": 20, #"侦察半径"
            "start_time": 2, #"起始时间",
            "end_time": 3000, #"结束时间",
            "unit_ids": [261,262,57,55,333], #"执行任务的单位ID列表",
            "route": [], #"执行此任务的途径点列表"
        }
        
        end_deploy1 = {
            "actor": 1, #"int，动作发出者",
            "type": 333
        }
        
        end_deploy2 = {
            "actor": 11, #"int，动作发出者",
            "type": 333
        }
        
        
        # loop until the end of game
        print("steping")
        done = False
        while not done:
            actions = []
            if state[self.red_flag]["time"]["cur_step"] == 0:
                actions.append(end_deploy1)
                actions.append(end_deploy2)

            if state[self.red_flag]["time"]["cur_step"] == 1:
                actions.append(assign_task)

            actions += red1.step(state[self.red_flag])
            actions += blue1.step(state[self.blue_flag])
            state, done = env1.step(actions)
            all_states.append(state[self.white_flag])

        env1.reset()
        red1.reset()
        blue1.reset()

        print(f"Total time: {time.time() - begin:.3f}s")

        # save replay
        zip_name = save_replay(self.begin, self.all_states)
        return self.all_states, zip_name

class record_result():
    def __init__(self):
        self.reward_ave = 0 
        self.return_ave = 0 
        self.reward_list = [] 
        self.return_list = [] 
        self.config_str = ""
        self.all_games = []
        self.time_list = [] 
        self.fupan_names = [] 
        self.begin = time.time()

    def record_config(self, config_str:str):
        self.config_str = config_str

    def get_result_single(self,all_states,zip_name):
        if len(self.config_str)==0:
            raise Exception("auto_run: config_str is empty, G. do not be lazy.")
        self.all_games.append(all_states)
        self.fupan_names.append(zip_name)
        this_moment = time.time()
        time_consumption = this_moment - self.begin
        print("time_comsumption:",time_consumption)
        self.time_list.append(time_consumption)
        self.begin = this_moment
        
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
        for i in range(geshu):
            file_txt.write("--------NO."+str(i)+"game--------\n")
            file_txt.write(f"reward: {self.reward_list[i]}\n")
            file_txt.write(f"return: {self.return_list[i]}\n")
            file_txt.write(f"time: {self.time_list[i]}\n")
            file_txt.write(f"fupan_name: {self.fupan_names[i]}\n")

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
    return zip_name 


if __name__ == "__main__":
    jieguo = record_result()
    jieguo.record_config("debug defend, merge")
    for i in range(3):
        # shishi = auto_run(env_name="defend")
        # shishi = auto_run(env_name="crossfire")
        # shishi = auto_run(env_name="scout")
        shishi = auto_run(env_name="renji")

        # all_states_single,zip_name = shishi.run_single()
        all_states_single,zip_name = shishi.run_in_single_agent_mode()
        jieguo.get_result_single(all_states_single,zip_name)
    jieguo.get_result_all(jieguo.all_games)