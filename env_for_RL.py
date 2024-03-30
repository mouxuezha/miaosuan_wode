# 整一个类似gym的东西,用于支撑强化学习训练.
# xxh 0330  # coding=UTF-8

import sys
import os
import datetime
from transfer import transfer
from agent_guize import * 

class EnvForRL(object):
    def __init__(self):
        self.net_args = self.__init_agent()

        
        self.__init_folder()

        self.__init_dim()  # 设定state和actor维数和上下游各种东西     
    
    def __init_folder(self):
        # 这些路径.使用的时候根据实际情况改改吧,先硬编码了
        # self.log_file = '/home/vboxuser/Desktop/miaosuan/miaosuan_wode/overall_result.txt'
        # self.log_folder = '/home/vboxuser/Desktop/miaosuan/miaosuan_wode/'       
        self.log_file = './RLtrainning/overall_result.txt'
        self.log_folder = './RLtrainning/'  
    
    def __init_agent(self):
        # 设定state和actor维数和上下游各种东西
        print("unfinished yet")

    def __init_dim(self):
        self.action_dim = 2
        self.red_obs_dim = 114514
        self.blue_obs_dim = 1919810
        self.state_dim = self.red_obs_dim + 1  # temp set  # 多出一个维度的时间戳。
        # self.state_dim = len(self.red_ID) * 2 + len(self.blue_ID) * 2        
    def reset(self):
        pass 
    def step(self, action):
        pass 
    def shadow_step(self):
        pass
    def command_2tank(self, action_real):
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
            random_action = (np.random.random((4,)) - 0.5) * 2
            shishi_env.step(random_action)
        print("EnvForRL: 起码跑起来了")    