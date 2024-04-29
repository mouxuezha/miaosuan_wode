# 整一个类似gym的东西,用于支撑强化学习训练.
# xxh 0330  # coding=UTF-8

import sys
import os
import datetime
import zipfile
import time
import copy
from transfer import transfer
from ai.agent import *
from ai.tools import *
import numpy as np
import time 

sys.path.append("/home/vboxuser/Desktop/miaosuan/starter-kit")
from ai.map import Map

class EnvForRL(object):
    def __init__(self):

        self.__init_folder()

        self.__init_crossfire()

        self.__init_dim()  # 设定state和actor维数和上下游各种东西

        self.state = np.zeros(self.state_dim,) 
        self.action = np.zeros(self.action_dim,)  
        self.reward = 0 
        self.done = False  
        self.num = 0 # num in simulation 
        self.step_num = 0 # num for RL
        self.shadow_step_num = [30, 30, 30] # 50  # 平台实际推的帧数 = max_step * (shadow_step_num + 1)
        self.shadow_step_range = [0, 1000, 2000, 3000, 1145141919]
        # 整点花的，安排一个动态的。按说粗细粒度一起来的话应该要多多区分一点儿才是。
        self.step_num_real = 0
        self.max_step = 8208820

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
        self.agent = Agent()
        # it should be point out that agent is not the 'agent' in RL framework, instead, agent is part of RL evironment.
        self.env = CrossFireEnv()
        self.begin = time.time()     
        self.max_step = 2800 
        # these flags identified the status from env.
        self.red_flag = 0
        self.blue_flag = 1
        self.white_flag = -1   

        self.red_obj_num = 16 # need debug.
        self.blue_obj_num = 4 
        self.enemy_num_max = 3 
        
        # self.get_target_cross_fire() 

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
        self.old_state_dict = copy.deepcopy(self.state_dict)

        self.all_states.append(state_dict[self.white_flag])
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
        self.map = self.agent.map
        self.map_data = self.agent.map_data
        pass 
    
    def __init_scout(self):
        print("unfinished yet")
    def __init_defend(self):
        print("unfinished yet")

    def save_replay(self,replay_name, data):
        zip_name = f"logs/replays/{replay_name}.zip"
        if not os.path.exists("logs/replays/"):
            os.makedirs("logs/replays/")
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.writestr(f"{replay_name}.json", json.dumps(data, ensure_ascii=False, indent=None, separators=(',', ':')).encode('gbk'))
    
    def get_target_cross_fire(self):
        # call one time for one game.
        observation = self.state_dict[0]
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
        xy_single = hex_to_xy(target_pos)
        # qian2wei = round(target_pos / 100)
        # hou2wei = target_pos - qian2wei*100 
        qian2wei = xy_single[1]
        hou2wei = xy_single[0]
        map_data_single = self.map_data[qian2wei][hou2wei]
        altitude_single = map_data_single['elev']
        # print("get_altitude need debug")
        return altitude_single
    
    def get_terrain(self,target_pos):
        xy_single = hex_to_xy(target_pos)
        # qian2wei = round(target_pos / 100)
        # hou2wei = target_pos - qian2wei*100 
        qian2wei = xy_single[1]
        hou2wei = xy_single[0]
        map_data_single = self.map_data[qian2wei][hou2wei]
        terrain_single = map_data_single['cond']
        # print("get_terrain need debug")
        return terrain_single

    def get_state_unit(self, unit):
        # avoid copy and paste code.
        state_list = [] 
        sub_type = unit["sub_type"]
        hex_absolute = get_pos(unit,status=self.state_dict[self.red_flag]) 
        
        # use relative hex.
        xy_single = np.array(hex_to_xy(hex_absolute)) - np.array(hex_to_xy(self.target_pos)) 
        alt_single = unit["altitude"]
        keep_remain_time = unit["keep_remain_time"]
        speed = unit["speed"]
        weapon_CD = unit["weapon_cool_time"]
        state_list = [sub_type,xy_single[0],xy_single[1], alt_single, keep_remain_time, speed, weapon_CD]

        # hex_around = self.map.get_neighbors(hex_absolute)
        distance_start = 0 
        distance_end = 2 
        hex_around = self.map.get_grid_distance(hex_absolute,distance_start,distance_end)

        for hex_around_single_abs in hex_around:
            # hex_around_single = hex_around[i] - self.target_pos
            hex_around_single = hex_around_single_abs - self.target_pos
            # use relative hex.
            xy_around_single = hex_to_xy(hex_around_single)
            if hex_around_single_abs >0:
                alt_around_single = self.get_altitude(hex_around_single_abs)
            else:
                alt_around_single = -1 
                xy_around_single = [-100,-100] 
            try:
                threaten_field_single = self.agent.threaten_field[hex_around_single_abs] 
            except KeyError:
                threaten_field_single = 0 
            state_list = state_list + [xy_around_single[0],xy_around_single[1],alt_around_single,threaten_field_single]
        geshu = len(state_list)

        return state_list, geshu
    
    def get_state_global_terrain(self,**kargs):
        # 全局地形，只在一开始记录一次，后面就反正往里塞进去就是了。
        if self.num <2:
            # 说明是第一次。把地形数据的区域取出来
            pos_ave = self.agent.get_pos_average(self.agent.status["operators"])
            pos_center = self.agent.get_pos_average([pos_ave, self.target_pos], model="input_hexs")
            
            if "area" in kargs:
                area = kargs["area"]
            else:
                area = self.map.get_grid_distance(pos_center, 0, 30)
            state_terrain = [] 
            for i in range(len(area)):
                xy_single = hex_to_xy(area[i])
                dixing_single = self.get_terrain(area[i])
                state_terrain = state_terrain + [xy_single[0],xy_single[1],self.get_altitude(area[i])]
                state_terrain.append(dixing_single)
            state_terrain = np.array(state_terrain)
            self.state_terrain = state_terrain
            # 然后把地形数据取出来
            pass
        else:
            # 说明不是第一次，就直接把地形数据取出来
            state_terrain = self.state_terrain
            pass
        geshu = len(state_terrain)
        return state_terrain, geshu
        print("unfinished yet")
            
    def get_state(self, state_dict):
        # get state_real from state_dict,

        index_now = 0  # shoudong laige zhizheng      
        self.state[index_now] = self.target_pos
        index_now = index_now + 1  
        
        self.state = np.zeros(self.state_dim,)
        # reset the self.state to avoid dimension dismatch. 

        # first, enemy obj state.
        # it is not correct. we can not use undetected enemy_obj information.  
        # 0, me. 1 enemy. -1, all.
        # state_dict_enemy = state_dict[1]
        # enemy_obj_IDs = get_ID_list(state_dict_enemy)
        # geshu = min(self.blue_obj_num,len(enemy_obj_IDs))
        # for i in range(geshu):
        #     ID = enemy_obj_IDs[i]
        #     unit = get_bop(ID)
        #     state_list, geshu_single = self.get_state_unit(unit)

        #     self.state[index_now:(index_now+geshu_single)] = state_list[:]
        #     index_now = index_now + geshu 
        
        # first, enemy obj state. using self.detected_state
        units_enemy = self.agent.detected_state
        for i in range(self.enemy_num_max):
            try:
                unit = units_enemy[i]
                state_list, geshu_single = self.get_state_unit(unit)

                self.state[index_now:(index_now+geshu_single)] = state_list[:]
            except:
                # 没探到，就用默认值。
                self.state[index_now:(index_now+geshu_single)] = 0
        
        # then, my obj states, which include detected enemy units.
        state_dict_my = state_dict
        my_obj_IDs = get_ID_list(state_dict_my) 
        geshu = min((self.red_obj_num+self.blue_obj_num),len(my_obj_IDs))
        for i in range(geshu):
            ID = my_obj_IDs[i]
            unit = get_bop(ID,status=state_dict_my)
            state_list, geshu_single = self.get_state_unit(unit)

            self.state[index_now:(index_now+geshu_single)] = state_list[:]
            index_now = index_now + geshu_single 
        
        # then, gloable 地形，which is important 的东西.
        state_terrain, geshu_terrain = self.get_state_global_terrain()
        self.state[index_now:(index_now+geshu_terrain)] = state_terrain[:]

        # then transfer.  ? it seems not 需要. 
            
        return self.state

    def add_actions(self, action_dict):
        # 这个试图来一个比较阳间的“从已有的命令中过滤掉特定ID的命令，然后把需要执行的命令加到其中去”的函数。
        # 应该能够有助于scout赛道和defend赛道发命令。

        # 先过滤一下
        print("unfinished yet")
        pass

    def filter_action(self, ID, **kargs):
        # 过滤命令的独立出来一个好了.
        # TODO: filter by type or other keys
        act_new = []
        for act_single in self.act:
            if act_single["obj_id"] != ID:
                act_new.append(act_single)
        return act_new


    def calculate_reward_cross_fire(self):
        # calculate the rl reward according to self.act and status.
        rewrad_list = [] 

        # first, the distance from target_pos.
        # 0, me. 1 enemy. -1, all.
        units_all_now = self.state_dict[-1]["operators"]
        my_units_now = select_by_type(units_all_now,key="color",value=0)
        pos_ave = get_pos_average(my_units_now)
        jvli = self.map.get_distance(pos_ave, self.target_pos)
        reward_jvli = 5/(jvli+0.0001)
        rewrad_list.append(reward_jvli)

        # then, reward for enemy unit detected.
        enemy_units_detected_now =select_by_type(self.state_dict[0]["operators"], key="color",value=1)
        num_detected =len(enemy_units_detected_now)
        reward_detected = num_detected * 0.1
        rewrad_list.append(reward_detected)

        # then, reward for my unit keeping
        num_living = len(my_units_now)
        reward_living = num_living * 0.001 
        rewrad_list.append(reward_living)

        # then, reward for shoot.
        act_shoot = select_by_type(self.act,key="type",value=2) # shoot
        act_guided = select_by_type(self.act,key="type",value=2) # shoot
        reward_fire = (len(act_shoot) + len(act_guided)) * 0.5 
        rewrad_list.append(reward_fire)

        # then, reward for yazhi
        my_units_yazhied = select_by_type(my_units_now,key="keep", value=1)
        reward_yazhi = -0.5 * len(my_units_yazhied)
        rewrad_list.append(reward_yazhi)

        # finally add them.
        self.reward = 0 
        for reward_single in rewrad_list:
            self.reward = self.reward + reward_single

        return self.reward 

    def calculate_reward_scout(self):
        # calculate the rl reward according to self.act and status.
        # 尚霖琢磨一下这部分罢，reward咋定一定程度上是关系成败的
        rewrad_list = [] 
        print("unfinished yet")

    def calculate_reward_defend(self):
        # calculate the rl reward according to self.act and status.
        # 子航琢磨一下这部分罢，reward咋定一定程度上是关系成败的
        rewrad_list = [] 
        print("unfinished yet")

    def step(self, action):
        # 
        # get the target first.
        if self.num <2:
            target_pos = self.get_target_cross_fire()
        else:
            target_pos = self.target_pos        

        self.state =self.get_state(self.state_dict[self.red_flag])

        # call agent, 
        self.agent.step(self.state_dict[self.red_flag], model="RL")
        self.num = self.agent.num
        self.step_num = self.step_num + 1 
        
        # there must be some dim to indictate if a unit should be forced stop and change target.
        # then overwrite some abstract_state 
        action_real = action
        self.command_tank(action_real)
        # then generate self.act. 
        self.act = self.agent.Gostep_abstract_state()

        # then generate action dict 
        action_dict = self.act
        state, done = self.env.step(action_dict)
        self.all_states.append(state[self.white_flag])
        self.old_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict = state

        # then shadow step
        for i in range(len(self.shadow_step_num)):
            if (self.step_num_real >= self.shadow_step_range[i]) and (self.step_num_real < self.shadow_step_range[i+1]):
                # 在特定的步数范围内，就整上特定的shadow step 步数。
                shadow_step_num_local = self.shadow_step_num[i]
            else:
                shadow_step_num_local = 30
        for i in range(shadow_step_num_local):
            self.act = self.shadow_step()

        # then calculate the RL reward. do not use pre-defined env reward.
        self.reward = self.calculate_reward_cross_fire()
        # 然后处理一下返回值
        if self.num > self.max_step:
            # 说明步数已经够了,结束了.
            is_done = True
        else:
            is_done = False

        info = {}

        return self.state, self.reward, is_done, info

    def shadow_step(self):
        # 这个用来实现"不发RL指令单纯空跑几帧规则",不然一个episode好几千步,就寄了
        self.agent.step(self.state_dict[self.red_flag], model="guize")
        self.num = self.agent.num
        # then generate self.act. 
        self.act = self.agent.Gostep_abstract_state()
        action_dict = self.act
        state, done = self.env.step(action_dict)
        self.all_states.append(state[self.white_flag])
        self.old_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict = state        
        return self.act

    def command_tank(self, action_real):
        # assume that the first dim is "stop_flag"
        units = self.state_dict[self.red_flag]["operators"]
        if action_real[0] > 0:
            # which means it should be stopped and go another pos.
            IFV_units = self.agent.get_IFV_units()
            infantry_units = self.agent.get_infantry_units()
            UAV_units = self.agent.get_UAV_units()
            tank_units = [unit for unit in units if (unit not in IFV_units) and (unit not in infantry_units) and (unit not in UAV_units)]

            pos_ave_tank = get_pos_average(tank_units)
            
            # xy_new = np.array(action_real[1:3]) + hex_to_xy(self.target_pos)
            xy_new = np.array(action_real[1:3]) + hex_to_xy(pos_ave_tank)
            hex_new = xy_to_hex(xy_new)
            
            self.agent.group_A(tank_units,hex_new,model="force")
            pass
        else:
            # which means it should not change.
            print("AI thinks nothing should happen.")
            pass 

    def command_scout_demo(self, action_real):
        # 这面实现“怎么把智能体那边拿来的指令生成为self.act里面的动作”
        act_RL = self.agent._move_action(0,1145)
        """
            command
        """

        self.act = self.filter_action(0)
        self.act = self.add_actions(act_RL[-1])
        print("unfinished yet")    

    def command_defend_demo(self, action_real):
        # 这面实现“怎么把智能体那边拿来的指令生成为self.act里面的动作”
        act_RL = self.agent._move_action(0,1145)
        """
            command
        """

        self.act = self.filter_action(0)
        self.act = self.add_actions(act_RL[-1])
        print("unfinished yet")    

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
            random_action = (np.random.random((3,)) - 0.5) * 2
            # random_action[0]=np.random.randint(0,1)
            random_action[0] = 1
            shishi_env.step(random_action)
        print("EnvForRL: 起码跑起来了")    