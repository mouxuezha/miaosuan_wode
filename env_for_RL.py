# 整一个类似gym的东西,用于支撑强化学习训练.
# xxh 0330  # coding=UTF-8

import sys
import os
import datetime
import zipfile
import time
from transfer import transfer
from agent_guize import * 
from tools import *
import numpy as np

sys.path.append("/home/vboxuser/Desktop/miaosuan/starter-kit")
from ai.map import Map

class EnvForRL(object):
    def __init__(self):

        self.__init_folder()

        self.__init_crossfire()

        self.__init_dim()  # 设定state和actor维数和上下游各种东西

        self.state = np.zeros(self.state_dim,) 
        self.action = np.zeros(self.action_dim,)    
    
    def __init_folder(self):
        # 这些路径.使用的时候根据实际情况改改吧,先硬编码了
        # self.log_file = '/home/vboxuser/Desktop/miaosuan/miaosuan_wode/overall_result.txt'
        # self.log_folder = '/home/vboxuser/Desktop/miaosuan/miaosuan_wode/'       
        self.log_file = './RLtrainning/overall_result.txt'
        self.log_folder = './RLtrainning/'  

    def __init_dim(self):
        self.action_dim = 2
        self.red_obs_dim = 114514
        self.blue_obs_dim = 1919810
        self.state_dim = self.red_obs_dim + 1  # temp set  # 多出一个维度的时间戳。
        # self.state_dim = len(self.red_ID) * 2 + len(self.blue_ID) * 2  
              
    def __init_crossfire(self):
        # from ai.agent import Agent
        from train_env.cross_fire_env import CrossFireEnv
        self.agent_guize = agent_guize()
        # it should be point out that agent_guize is not the 'agent' in RL framework, instead, agent_guize is part of RL evironment.
        self.env = CrossFireEnv()
        self.begin = time.time()     
        # these flags identified the status from env.
        self.red_flag = 0
        self.blue_flag = 1
        self.white_flag = -1   

        self.red_obj_num = 16 # need debug.
        self.blue_obj_num = 4 
        
        self.get_target_cross_fire() 

        # varialbe to build replay
        self.all_states = [] 
        # this is to save replay

        ai_user_name = 'myai'
        ai_user_id = 1
        try:
            state_dict = self.env.setup({'user_name': ai_user_name, 'user_id': ai_user_id})
        except:
            state_dict = self.env.setup({'user_name': ai_user_name, 'user_id': ai_user_id, 'seat':1})

        self.state_dict = state_dict

        self.all_states.append(state_dict[self.white_flag])
        self.agent_guize.setup = {
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
        self.map = self.agent_guize.map
        self.map_data = self.agent_guize.map_data
        pass 

    def save_replay(self,replay_name, data):
        zip_name = f"logs/replays/{replay_name}.zip"
        if not os.path.exists("logs/replays/"):
            os.makedirs("logs/replays/")
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.writestr(f"{replay_name}.json", json.dumps(data, ensure_ascii=False, indent=None, separators=(',', ':')).encode('gbk'))
    
    def get_target_cross_fire(self):
        # call one time for one game.
        observation = self.status
        communications = observation["communication"]
        flag_done = False
        for command in communications:
            if command["type"] in [210] :
                self.my_direction = command
                self.target_pos = self.my_direction["hex"]
                self.end_time = self.my_direction["end_time"]
                flag_done = True
        if flag_done==False:
            raise Exception("get_target_cross_fire: G!")
        else:
            print("get_target_cross_fire: Done.")
        return  self.target_pos
    
    def reset(self):
        self.__init_crossfire()
        # self.env.reset() # use theirs.
        self.state = np.zeros(self.state_dim,) 
        self.action = np.zeros(self.action_dim,)  
    
    def get_altitude(self,target_pos):
        qian2wei = round(target_pos / 100)
        hou2wei = target_pos - qian2wei*100 
        map_data_single = self.map_data[qian2wei,hou2wei]
        altitude_single = map_data_single['elev']
        return altitude_single

    def get_state_unit(self, unit):
        # avoid copy and paste code.
        state_list = [] 
        sub_type = unit["sub_type"]
        hex_single = get_pos(unit)
        alt_single = unit["altitude"]
        xy_single = hex_to_xy(hex_single)
        keep_remain_time = unit["keep_remain_time"]
        speed = unit["speed"]
        state_list = [sub_type,xy_single[0],xy_single[1], alt_single, keep_remain_time, speed]

        hex_around = self.map.get_neighbors(hex_single)
        for i in range(len(hex_around)):
            hex_around_single = hex_around[i]
            xy_around_single = hex_to_xy(hex_around_single)
            if hex_around_single >0:
                alt_around_single = self.get_altitude(hex_around_single)
            else:
                alt_around_single = -1 
            state_list.append(xy_around_single[0],xy_around_single[1],alt_around_single)
        geshu = len(state_list)
        return state_list, geshu


    def get_state(self, state_dict):
        # get state_real from state_dict,

        index_now = 0  # shoudong laige zhizheng      
        self.state[index_now] = self.target_pos
        index_now = index_now + 1  
        
        # first, enemy obj state.
        # 0, me. 1 enemy. -1, all.
        state_dict_enemy = state_dict[1]
        enemy_obj_IDs = get_ID_list(state_dict_enemy)
        geshu = min(self.blue_obj_num,len(enemy_obj_IDs))
        for i in range(geshu):
            ID = enemy_obj_IDs[i]
            unit = get_bop(ID)
            state_list, geshu_single = self.get_state_unit(unit)

            self.state[index_now:(index_now+geshu_single)] = state_list[:]
            index_now = index_now + geshu 
        
        # then, some map info.
        







        # then transfer

    def step(self, action):
        
        # call agent_guize, 
        self.agent_guize.step(self.state_dict[self.red_flag])

        # then overwrite some abstract_state 
        self.command_tank(action_real)

        # then generate action dict 
        action_dict = self.act
        state, done = self.env.step(action_dict)
        self.all_states.append(state[self.white_flag])
        pass 

    def shadow_step(self):
        pass
    def command_tank(self, action_real):
        pass
    def render(self):
        print("unfinished yet")
        pass
    def Gostep(self):
        # 这个是把那些要维护状态的那些事情都搞进来。
        # 所有改act的东西都应该统一经过这里面。
        pass 

if __name__ == "__main__":
    flag = 0
    if flag == 0:
        print("EnvForRL: testing")
        shishi_env = EnvForRL()
        step_num_max = shishi_env.max_step

        shishi_env.reset()
        for i in range(step_num_max):
            # random_action = (np.random.random((4,)) - 0.5) * 2 * shishi_env.action_abs_max
            random_action = (np.random.random((2,)) - 0.5) * 2
            shishi_env.step(random_action)
        print("EnvForRL: 起码跑起来了")    