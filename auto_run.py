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
from memory_profiler import profile
from threading import Thread
import copy

import signal
import time

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
        from ai.agent import Agent
        self.env = DefendEnv(3,5,1)
        self.agent = Agent()
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
    
    @profile(stream=open("logs/replays/memory_profiler.log", "w+"))
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
    
    def handler(self, signum, frame):
        print("auto_run: 抓到超时了，但是生活还要继续，继续继续...")
        signal.pause()
        # raise TimeoutError()
    
    def run_single_with_time_limit(self,time_limit = 114.514):
        # 加了超时暂停的run_single,用于调试程序。
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(time_limit)
        self.all_states, zip_name = self.run_single()
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

    def get_result_multi(self,result_list,result_name_list):
        # get result from one round
        geshu = len(result_list)
        for i in range(geshu):
            all_states_single = result_list[i]
            zip_name_single = result_name_list[i]
            self.get_result_single(all_states_single,zip_name_single)

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
    
    def run_multi(self,num_thread,env_name="crossfire"):
        # init a void list for results
        self.result_list = [] 
        self.result_name_list = [] 
        for i in range(num_thread):
            self.result_list.append([])
            self.result_name_list.append([])

        # get some threads
        thread_list = []
        for i in range(num_thread):
            thread_single = Thread(target=self.run_single,args=(env_name, i))
            thread_list.append(thread_single)
        
        # then run, 
        for thread_single in thread_list:
            thread_single.start()
        
        # then wait for all threads to finish.
        for thread_single in thread_list:
            thread_single.join()
        
        # get result
        self.get_result_multi(self.result_list,self.result_name_list)
            
        return self.result_list,self.result_name_list

    def run_single(self,env_name, index):
        runner_in = auto_run(env_name=env_name)
        all_states_single,zip_name = runner_in.run_single()
        self.result_list[index] = copy.deepcopy(all_states_single)
        self.result_name_list[index] = copy.deepcopy(zip_name)
        return

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
    jieguo.record_config("debug crossfire, can shoot off when self.num>1300, and uing qianpai")
    for i in range(1):
        print("\n\n"+"round "+str(i)+"\n")

        # # single thread model
        # shishi = auto_run(env_name="defend")
        # shishi = auto_run(env_name="crossfire")
        # shishi = auto_run(env_name="scout")
        # all_states_single,zip_name = shishi.run_single()
        # jieguo.get_result_single(all_states_single,zip_name)
        
        # multi thread model
        jieguo.run_multi(2, "crossfire")

    jieguo.get_result_all(jieguo.all_games)

