from .action import ActionGenerator
from .const import ActType, BopType, MoveType, TaskType, UnitSubType
from .executors.cross_fire import CrossFireExecutor
from .executors.defend import DefendExecutor
from .executors.scout import ScoutExecutor
from .map import Map


# this is wode agent for miaosuan, define some layers.
import os 
import sys 
import json
import copy,random
import numpy as np
from .tools import *
from .base_agent import BaseAgent
from typing import List , Dict, Mapping, Tuple
from time import time 
from functools import wraps

class Agent(BaseAgent):  # TODO: 换成直接继承BaseAgent，解耦然后改名字。
    def __init__(self):
        super(Agent,self).__init__()
        # abstract_state is useful
        #@szh0404 添加记录fort 状态的
        self.fort_assignments = {}
        #@szh0404 添加记录算子目的地 说法是用movepath 就不用 
        self.ops_destination = {}
        self.troop_stage = {}
        self.troop_defend_target = {}
        self.chariot_stage = {}
        self.chariot_defend_target = {}
        self.tank_stage = {}
        self.tank_defend_target = {}
        self.prepare_to_occupy  = {}
        #@szh0404 添加记录fort 状态的
        self.fort_assignments = {}
        #@szh0404 添加记录算子目的地 说法是用movepath 就不用 
        self.ops_destination = {}
        self.troop_stage = {}
        self.troop_defend_target = {}
        self.chariot_stage = {}
        self.chariot_defend_target = {}
        self.tank_stage = {}
        self.tank_defend_target = {}
        self.prepare_to_occupy  = {}
        self.enemy_info = {}  #用来记敌方位置信息  需要记录对应的时刻 key = obj_id
        self.filtered_enemyinfo = {}
        self.max_unseen_time = 150
    def setup(self, setup_info):
        self.scenario = setup_info["scenario"]
        # self.get_scenario_info(setup_info["scenario"])
        self.color = setup_info["faction"]
        self.faction = setup_info["faction"]
        self.seat = setup_info["seat"]
        self.role = setup_info["role"]
        self.user_name = setup_info["user_name"]
        self.user_id = setup_info["user_id"]
        self.priority = {
            ActionType.Occupy: self.gen_occupy,
            ActionType.Shoot: self.gen_shoot,
            ActionType.GuideShoot: self.gen_guide_shoot,
            ActionType.JMPlan: self.gen_jm_plan,
            ActionType.LayMine: self.gen_lay_mine,
            ActionType.ActivateRadar: self.gen_activate_radar,
            ActionType.ChangeAltitude: self.gen_change_altitude,
            ActionType.GetOn: self.gen_get_on,
            ActionType.GetOff: self.gen_get_off,
            ActionType.Fork: self.gen_fork,
            ActionType.Union: self.gen_union,
            ActionType.EnterFort: self.gen_enter_fort,
            ActionType.ExitFort: self.gen_exit_fort,
            ActionType.Move: self.gen_move,
            ActionType.RemoveKeep: self.gen_remove_keep,
            ActionType.ChangeState: self.gen_change_state,
            ActionType.StopMove: self.gen_stop_move,
            ActionType.WeaponLock: self.gen_WeaponLock,
            ActionType.WeaponUnFold: self.gen_WeaponUnFold,
            ActionType.CancelJMPlan: self.gen_cancel_JM_plan
        }  # choose action by priority
        self.observation = None
        self.map = Map(
            setup_info["basic_data"],
            setup_info["cost_data"],
            setup_info["see_data"]
        )  # use 'Map' class as a tool
        self.map_data = self.map.get_map_data()

        self.act_gen = ActionGenerator(self.seat)
        self.task_executors = {
            TaskType.CrossFire: CrossFireExecutor(),
            TaskType.Scout: ScoutExecutor(),
            TaskType.Defend: DefendExecutor(),
        }
    def reset(self):
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map = None
        self.scenario_info = None
        self.map_data = None

        self.num = 0 

    def time_decorator(func):
        @wraps(func)
        def core(self, *args, **kwargs):
            start = time()
            res = func(self, *args, **kwargs)
            print("{time_step}: function::{funcname} :: time costing: {time_costing}".format(\
                time_step = self.num, funcname = func.__name__, time_costing = time() - start ) )
            return res 
        return  core


    # guize_functions xxh
    def F2A(self,target_pos):
        units = self.status["operators"]
        for unit in units:
            self.set_move_and_attack(unit,target_pos,model="force")
            # A了A了，都这时候了还要个毛的脑子，直接头铁
        pass

    def group_A(self, units,target_pos, model='normal'):
        # print("group_A: unfinished yet")
        for unit in units:
            self.set_move_and_attack(unit,target_pos,model=model)
        return

        # 这里需要一个新的结阵逻辑。
        target_xy = self._hex_to_xy(target_pos)
        ave_pos = self.get_pos_average(units=units)
        ave_xy = self._hex_to_xy(ave_pos)
        
        # 还是先弄出一个类似向量的东西确定方位，
        vector_xy = target_xy - ave_xy
        # 然后确定阵形中能用的一系列位置、
        distance_start = 0
        distance_end = 2 
        pos_list = self.map.get_grid_distance(target_pos, distance_start, distance_end)
        # 然后再进行一波分配。
        # 这样，算每个点和vector_xy的夹角然后比较大小，然后索引，夹角小的就是在阵型的前面
        # 也有点问题，这个搞法应该是夹角小的就在阵型的矛头上。所以比较理想的应该是坦克装甲车辆在矛头上，步兵在两翼
        dot_list = [] 
        for pos_single in pos_list:
            xy_single = self._hex_to_xy(pos_single) 
            vector_single = xy_single - ave_xy
            dot_single = np.dot(vector_xy, vector_single) / np.linalg.norm(vector_xy) / np.linalg.norm(vector_single)
            dot_list.append(dot_single)
        
        # 组装一个矩阵用于排序：每一行应该是[pos, dot_single]，然后照着dot_single排序后往里填充东西。
        geshu = len(pos_list)
        array_sort = np.append(np.array(list(pos_list)).reshape(geshu,1), np.array(dot_list).reshape(geshu,1),axis=1)
        array_sorted = array_sort[array_sort[:,1].argsort()] # 按照第二列进行排序。
        
        # 然后就可以往里面分配了。专业一点，这里再搞一个分类
        IFV_units = self.get_IFV_units()
        infantry_units = self.get_infantry_units()
        UAV_units = self.get_UAV_units()
        others_units = [unit for unit in units if (unit not in IFV_units) and (unit not in infantry_units) and (unit not in UAV_units)]
        
        units_sorted=others_units+IFV_units+infantry_units
        # 坦克装甲车辆在矛头，步兵什么的在两翼和后面，每个格子来三个东西。
        for i in range(len(units_sorted)):
            unit_single = units_sorted[i]
            index_pos = round(i/3)  # 这个处理比较傻逼，但是不管了，也不是不能用。
            target_pos_single = array_sorted[index_pos,0]
            self.set_move_and_attack(unit,target_pos_single)    
    
    def group_A2(self,units,units_VIP):
        # 这个以低成本实现一个跟随的。units跟随units_VIP里面距离最近的一个，跟随的逻辑是直接瞄着其当前位置就去了。
        for unit in units:
            if len(units_VIP)==0:
                # 那就是被跟随的已经被杀完了，那就无所谓了
                # 来点随机性，防止全都堆在一起。
                target_pos_candidate = self.map.get_distance(self.target_pos,0,1)
                target_pos_selected = target_pos_candidate[random.randint(0,len(target_pos_candidate)-1)]
                self.set_move_and_attack(unit,target_pos_selected,model="force")
            else:
                # 找那一堆里面距离最近的来跟随。
                jvli_list = [] 
                for i in range(len(units_VIP)):
                    jvli_single = self.distance(unit,units_VIP[i])  
                    jvli_list.append(jvli_single)
                jvli_min = min(jvli_list)
                index_min = jvli_list.index(jvli_min)
                VIP_pos_single = units_VIP[index_min]["cur_hex"]
                target_pos_candidate = self.map.get_distance(VIP_pos_single,0,1)
                target_pos_selected = target_pos_candidate[random.randint(0,len(target_pos_candidate)-1)]
                self.set_move_and_attack(unit,VIP_pos_single,model="force")

    def get_pos_list_A(self, units, target_pos):
        # 上来先维护target_pos_list,包括判断威胁等级看是不是有必要绕路。
        pos_ave = self.get_pos_average(units)
        xy_ave = self._hex_to_xy(pos_ave)
        
        target_xy= self._hex_to_xy(target_pos)
        vector_xy = target_xy - xy_ave
        target_distance = self.map.get_distance(target_pos, pos_ave)

        # enemy_infantry_units = self.select_by_type(self.detected_state,key="sub_type",value=2)
        # tanks and other things included
        enemy_infantry_units = self.detected_state
        enemy_infantry_units_danger = [] 
        enemy_infantry_dot_danger = [] 
        enemy_infantry_jvli_danger = [] 
        # 如果敌方步兵在正前方了，那就别去了。同时满足距离和方向的才算。
        for enemy_infantry_unit in enemy_infantry_units:
            # 遍历看一下是不是需要跑。
            enemy_pos = enemy_infantry_unit["cur_hex"]
            enemy_xy = self._hex_to_xy(enemy_pos)
            vector_single = enemy_xy - xy_ave
            enemy_distance = self.map.get_distance(enemy_pos, pos_ave)

            dot_single = np.dot(vector_single, vector_xy) / np.linalg.norm(vector_xy+0.001) / np.linalg.norm(vector_single+0.001)
            
            # 还得加一个判断以防止敌人在目标点的延长线上导致的超级大回环。拟采用向量的模来判断。
            if enemy_distance > target_distance*1.1:
                # 哪怕是在完全正面都比目标点远了，要是在侧面就更远了，就不是很有所谓了。保险起见再来个安全裕度
                flag_far_enemy = True
            else:
                flag_far_enemy = False

            
            if enemy_distance<20 and dot_single>0.50 and flag_far_enemy==False:
                # 这两个阈值都是从案例里抠出来的。flag_far_enemy用来防止究极大回环
                enemy_infantry_units_danger.append(enemy_infantry_unit)
                enemy_infantry_dot_danger.append(dot_single)
                enemy_infantry_jvli_danger.append(enemy_distance)
                
        
        # 至此，就筛出了究极高威胁的敌方步兵的位置。然后是根据这些位置确定绕路的方向，以target_pos_list的形式放在list中。
        if len(enemy_infantry_units_danger)>0:
            # 然后向量计算求一下那个点应该往哪边找，点乘判断正负可也，
            
            # 先求两个垂直于路径的法向量。
            n1_xy = np.array([vector_xy[1], -1*vector_xy[0]]) / np.linalg.norm(vector_xy)
            n2_xy = -1*n1_xy
            jvli_min = min(enemy_infantry_jvli_danger)
            index_min = enemy_infantry_jvli_danger.index(jvli_min)
            enemy_pos= enemy_infantry_units_danger[index_min]["cur_hex"]
            enemy_xy = self._hex_to_xy(enemy_pos)
            

            # # method1: 先取个中间点出来
            def method1(enemy_infantry_units_danger):
                pos_ave_enemy = self.get_pos_average(enemy_infantry_units_danger)
                xy_ave_enemy = self._hex_to_xy(pos_ave_enemy)

                # vector_ave_enemy = xy_ave_enemy - xy_ave         
                vector_xy_enemy = enemy_xy - xy_ave

                # 然后检测哪个比较好。
                if np.dot(n1_xy,vector_xy_enemy)>0:
                    # 那说明是偏向这个方向，绕道的路就得往另一个方向去了。
                    n_xy_list = [n2_xy, n1_xy] 
                else:
                    n_xy_list = [n1_xy, n2_xy] 
                # 道理上不可能两个方向都在外面，因为起点终点在垂线的不同侧，且都在范围内。
                # 所以两边必有一边是能够绕路的。

                # # 然后然后开始计算距离点了。
                # pos_center = self.get_pos_average([pos_ave,target_pos], model="input_hexs")
                # # 然后算。反正两个方向，总得有一个对的。要是都不对也防一手。
                # xy_center = self._hex_to_xy(pos_center)

                # 这个改一下，向量运算的起点还是改成最近的敌方单位的位置恐怕好一些。
                xy_center = enemy_xy
                xy_center_distance = np.linalg.norm(vector_xy_enemy)
                # 绕路的距离应该和当前到目标点的距离有关系，太近了就别绕太远了。这里的距离应该是从xy来选
                if xy_center_distance<15:
                    # 那就是已经快到了，那就得限制一下绕路的距离
                    len_raolu = round(xy_center_distance)
                else:
                    len_raolu = 18 

                try:
                    xy_candidate = xy_center + len_raolu*n_xy_list[0]
                    pos_candidate = self._xy_to_hex(xy_candidate)
                except:
                    xy_candidate = xy_center + len_raolu*n_xy_list[1]
                    pos_candidate = self._xy_to_hex(xy_candidate) 

                return pos_candidate

            def method2(enemy_infantry_units_danger):   
                # method2: find zuiwaimain units and xiuzheng.
                # dot_min = min(enemy_infantry_dot_danger)
                # index_min = enemy_infantry_dot_danger.index(dot_min)
                
                vector_xy_enemy = enemy_xy - xy_ave
                if np.dot(n1_xy,vector_xy_enemy)>0 or True: #disabled for debug
                    # which means the direction is right.
                    # n_xy_list = [n1_xy, n2_xy] 

                    # 绕路的距离应该和当前到目标点的距离有关系，太近了就别绕太远了。这里的距离应该是从xy来选
                    xy_center_distance = np.linalg.norm(vector_xy_enemy)
                    if xy_center_distance<15:
                        # 那就是已经快到了，那就得限制一下绕路的距离
                        len_raolu = round(xy_center_distance)
                    else:
                        len_raolu = 18 

                    try:
                        xy_candidate = enemy_xy + len_raolu*n1_xy
                        pos_candidate = self._xy_to_hex(xy_candidate)
                    except:
                        # if it doesn't work, then use method1
                        pos_candidate =  method1(enemy_infantry_units_danger) 
                else:
                    pos_candidate =  method1(enemy_infantry_units_danger)
                return pos_candidate
            
            # 更合理的应该是再来一层，看探测到的东西是在同一个方向还是不同的方向，然后分别调。
            int_method = self.get_direction_list_A(units, target_pos,self.detected_state)
            if int_method == 1:
                pos_candidate = method1(enemy_infantry_units)
            elif int_method == 2:
                pos_candidate = method2(enemy_infantry_units)  
            else:
                raise Exception("invalid list_A method, G!")
            # pos_candidate = method2(enemy_infantry_units)            
        else:
            pos_candidate = target_pos
        return [pos_candidate, target_pos, target_pos] # 这里后面补一个target_pos是为了写循环的时候好写。

    def get_direction_list_A(self,units, target_pos,detected_state):
        # 这个用来判断到时候往哪边去绕。目前的说法是，如果发现了一个，就往中心线反方向去绕；如果发现了很多个且在同侧了，就还是往中心线反方向去绕，如果发现多个还在异侧了，那就智能method2，往外面去绕了。所以这个就返回一个int就好了
        int_method=0
        # 虽然重复计算了，但是适度的独立性是必要的
        pos_ave = self.get_pos_average(units)
        xy_ave = self._hex_to_xy(pos_ave)
        
        target_xy= self._hex_to_xy(target_pos)
        vector_xy = target_xy - xy_ave

        # 先求两个垂直于路径的法向量。
        n1_xy = np.array([vector_xy[1], -1*vector_xy[0]]) / np.linalg.norm(vector_xy)
        n2_xy = -1*n1_xy

        # 然后来一堆向量运算。
        flag_list_which_side = [] 
        for unit in detected_state:
            # 分别计算每一个探测到的东西的flag_list_which_side
            xy_single = self._hex_to_xy(unit["cur_hex"])
            vector_single = xy_single - xy_ave
            flag_list_which_side.append(np.dot(vector_single, n1_xy) > 0)

        # 然后开始判断了。
        if len(set(flag_list_which_side)) == 1:
            # 那就是只有一个，那就还好。
            int_method = 1
        elif len(set(flag_list_which_side)) >= 2:
            # 两个就要看是不是同侧了
            flag_one_side = True
            for i in range(len(flag_list_which_side)-1):
                if flag_list_which_side[0] * flag_list_which_side[i+1]>0 :
                    # 一个正的一个负的，那就说明不是一侧了，
                    flag_one_side = flag_one_side and False
                else:
                    # 一直都是正的，那就说明都是同一侧。
                    flag_one_side = flag_one_side and True
            if flag_one_side == True:
                # 那就是全都在同侧了，那也挺好的。
                int_method = 1 
            else:
                # 那就是有不同侧的，得想辙
                int_method = 2 
            pass
        return int_method

    def list_A(self, units, target_pos, **kargs):
        # “选取部队横越地图”，实现一个宏观层面的绕行机制。
        if len(units) ==0:
            # unit lost, nothing happen.
            return 
        if "target_pos_list" in kargs:
            # this is for debug, basicly
            target_pos_list = kargs["target_pos_list"]
        else:
            # target_pos_list作为类的一个属性在这里面自己维护了。
            try:
                target_pos_list = self.target_pos_list
                flag_exists = True
            except:
                flag_exists = False
                self.target_pos_list = [self.target_pos,self.target_pos,self.target_pos]

            # if (self.num<1500 and self.num%75==2) or not(flag_exists): # 原则上不用每一步都求解这个。只要位置变化了一次能够求一次就行了
                # target_pos_list = self.get_pos_list_A(units, target_pos)
                # self.target_pos_list = target_pos_list 
            # else:
                # target_pos_list = self.target_pos_list
            
            if self.flag_detect_update==True or (self.target_pos_list[0]==self.target_pos and self.num%75==2):
                # 说明是刚刚更新了detect，那就把list更新一下。
                target_pos_list = self.get_pos_list_A(units, target_pos)
                self.target_pos_list = target_pos_list 
            else:
                target_pos_list = self.target_pos_list
        


        # 强行判断是否到了，到了就改成目标点。越写越乱越写越丑了，但是先不管了，能用就行。
        pos_ave = self.get_pos_average(units)
        jvli = self.distance(pos_ave, target_pos_list[0])
        if jvli < 3:
            # 说明是到了
            target_pos_list[0] = self.target_pos
        # 还得再来个强行判断，以防止出现超级大回环。就是距离差不多了就直着过去了。
        jvli = self.distance(pos_ave, self.target_pos)

        # if there is no more time, then just chong.
        time_assume = round(jvli * 20 * 1.1)
        # time_assume = -114514
        if time_assume > (self.end_time - self.num):
            # then just chong, without using naozi
            for unit in units:
                self.set_move_and_attack(unit,self.target_pos,model="force")
        else:        
            for unit in units:
                # 如果到了某一个点，就去下一个点。搞成通用的，以防未来需要很多个路径点的时候不好搞。
                target_pos_list_temp = copy.deepcopy(target_pos_list)
                for i in range(len(target_pos_list_temp)-1):
                    target_pos_single = target_pos_list_temp[i]
                    pos_single = self.get_pos(unit)
                    if pos_single==target_pos_list_temp[-1]:
                        # arrived
                        break
                    if pos_single==target_pos_single:
                        # 说明到了这个点了，那就去下一个点。
                        target_pos = target_pos_list_temp[i+1]
                        target_pos_candidate = self.map.get_distance(target_pos,0,1)
                        target_pos_selected = target_pos_candidate[random.randint(0,len(target_pos_candidate)-1)]
                        self.set_move_and_attack(unit,target_pos_selected,model="force")
                        del target_pos_list_temp[i]
                        break 
                    else:
                        # 没到的话就无事发生。
                        # no, if not arrived, then go there.
                        self.set_move_and_attack(unit,target_pos_single,model="force")
                        # del target_pos_list_temp[i]
                        break 
        return self.target_pos_list
                
    def final_juhe(self, units):
        flag_arrived, units_arrived = self.is_arrive(units,self.target_pos,tolerance = 0 )

        # cao, baoli chu miracle, force products qiji.
        for king in units_arrived:
            king_ID = king["obj_id"]
            try:
                king_abstract_state = self.abstract_state[king_ID]["abstract_state"]
            except:
                king_abstract_state = "none"
                self.set_none(king_ID)

            if king_abstract_state == "juhe":
                # this one has juheing.
                pass
            else:
                # give the king a kinght for juhe
                for knight in units:
                    knight_ID = knight["obj_id"]
                    try:
                        knight_abstract_state = self.abstract_state[knight_ID]["abstract_state"]
                    except:
                        knight_abstract_state = "none"
                        self.set_none(knight_ID)
 
                    if (knight_abstract_state == "move_and_attack" and "next" in self.abstract_state[knight_ID]) or knight_abstract_state == "juhe" or (knight_ID==king_ID):
                        # this knight has his lord.
                        pass 
                    else:
                        # it's time to set juhe
                        self.set_juhe(king_ID, knight_ID)
                        break

        return 
    
    def final_xiache(self, units):
        # 这个是最后加个下车。
        # 改改逻辑，谁到了谁就下，不要等都到了才下。
        for unit in units:
            # 判断到没到
            flag_arrived, units_arrived = self.is_arrive([unit],self.target_pos,tolerance = 3 )
            if flag_arrived == False:
                # 没到就算了
                return
            
            # 判断停没停
            flag_is_stop = self.is_stop(unit)
            if flag_is_stop == False:
                # 没停就算了
                return

            # 判断能不能下车
            flag_can_xiache = (len(unit["passenger_ids"])>0)
            if flag_can_xiache == False:
                # 不能就算了
                return

            # 发下车指令
            self.set_off_board(unit)

    def UAV_patrol(self, target_pos):
        # 这个会覆盖给无人机的其他命令，优先执行“飞过去打一炮”，然后再把别的命令弄出来。

        # 不要重复下命令，不然就把时间都刷没了

        # 先把UAV取出来
        UAV_units = self.select_by_type(self.status["operators"],key="sub_type",value=5)
        # 然后把目标取出来
        if len(self.detected_state)>0 and False:
            target_unit = self.detected_state[0]
            target_pos = target_unit["cur_hex"]
            # 然后设定状态就开始过去了。
            for UAV_unit in UAV_units:
                if self.abstract_state[UAV_unit["obj_id"]]["abstract_state"]!="UAV_move_on":
                    self.set_UAV_move_on(UAV_unit["obj_id"],target_pos=target_pos)
        else:
            # if nothing detected, then nothing happen.
            # no, if nothing detected, then random patrol target
            pos_ave =self.get_pos_average(self.status["operators"]) 
            pos_center = self.get_pos_average([pos_ave,target_pos], model="input_hexs")
            
            pos_around_list = list(self.map.get_grid_distance(pos_center,3,5))
            target_pos_random = pos_around_list[random.randint(0,len(pos_around_list)-1)]
            if target_pos_random == -1:
                target_pos_random = pos_center

            # 然后设定状态就开始过去了。
            for UAV_unit in UAV_units:
                # if self.abstract_state[UAV_unit["obj_id"]]["abstract_state"]!="UAV_move_on":
                #     # self.set_UAV_move_on(UAV_unit["obj_id"],target_pos=target_pos_random)
                #     self.set_UAV_move_on(UAV_unit["obj_id"],target_pos=target_pos_random)    
                self.set_move_and_attack(UAV_unit["obj_id"],target_pos=target_pos_random,model="force")        
            pass

    def UAV_patrol2(self,unscouted_input):
        # 这个会覆盖给无人机的其他命令，优先执行“探索未知区域”，这个依赖于从尚霖那里抄的通视那一系列东西。
        # 不要重复下命令，不然就把时间都刷没了
        # 保持一定的灵活性，别啥都用成员变量

        # 先把UAV取出来
        UAV_units = self.select_by_type(self.status["operators"],key="sub_type",value=5)
        UAV_unit = UAV_units[0] # 反正只有一个无人机，就别骗自己了。

        # 然后找一个离得最近的没有探索的区域。
        area_unscouted = list(self.unscouted)
        geshu = len(area_unscouted)
        jvli_list = [] 
        if geshu > 0:
            # 说明有没探索的区域。
            for i in range(geshu):
                # 然后把所有的距离都算一遍
                pos_single = area_unscouted[i]
                juli = self.distance(pos_single, UAV_unit["cur_hex"])
                jvli_list.append(juli)
            # 然后找到离得最近的那个
            min_jvli = min(jvli_list)
            min_jvli_index = jvli_list.index(min_jvli)
            selected_pos = area_unscouted[min_jvli_index]
        else:
            # 要是失效了就还是飞去目标点好了。
            # 原则上应该整一个“要是失效了就飞个之字形过去”之类的说法。
            selected_pos= self.target_pos
        # 然后设定状态就开始过去了。
        self.set_move_and_attack(UAV_unit,selected_pos,model="force")

    def IFV_transport(self,model="on"):
        # 这个会覆盖给步战车和步兵的其他命令。优先执行“开过去接人”。
        # on 就是上车，off就是下车。
        # print("unfinished yet")
        # 先把步兵和步兵车选出来。
        # IFV_units = self.select_by_type(self.status["operators"],key="sub_type",value=1)
        # infantry_units = self.select_by_type(self.status["operators"],key="sub_type",value=2)
        # 有一个遗留问题，如果不是同时被打烂，那就会不匹配，所以就要好好搞搞。
        if self.num < 114514:
            IFV_units = self.get_IFV_units()
            infantry_units = self.get_infantry_units()
            self.IFV_units = copy.deepcopy(IFV_units)
            self.infantry_units = copy.deepcopy(infantry_units)
        else:
            IFV_units = self.IFV_units
            infantry_units = self.infantry_units

        
        if model == "on":
            geshu=min(len(IFV_units), len(infantry_units))
        elif model == "off":
            geshu=max(len(IFV_units), len(infantry_units))
        # 然后循环发命令。
        # 这个命令不重复发，发过了就不发了.
        for i in range(geshu):
            IFV_unit = IFV_units[i]
            IFV_abstract_state = self.abstract_state[IFV_unit["obj_id"]]

            # check if the on board or off board abstract state setted. attension, when the unit is moving.
            flag_ordered = False
            if "abstract_state" in IFV_abstract_state:
                if (IFV_abstract_state["abstract_state"]=="on_board") or (IFV_abstract_state["abstract_state"]=="off_board") :
                    flag_ordered = True
                elif IFV_abstract_state["abstract_state"]=="move_and_attack":
                    if "next" in IFV_abstract_state:
                        flag_ordered = (IFV_abstract_state["next"]["abstract_state"]=="on_board") or (IFV_abstract_state["next"]["abstract_state"]=="off_board")
            else:
                flag_ordered=True
                # finished. 


            if (not flag_ordered) and (model == "on"):
                self.set_on_board(IFV_unit,infantry_units[i])
            elif (not flag_ordered) and (model == "off"):
                # this is not right if jieju done.
                # self.set_off_board(IFV_unit,infantry_units[i])

                infantry_ID_list = IFV_unit["get_off_partner_id"]+IFV_unit["get_on_partner_id"] + IFV_unit["passenger_ids"]
                if len(infantry_ID_list)>0:
                    # self.set_off_board(IFV_unit, infantry_ID_list[0])
                    self.set_off_board(IFV_unit)
                else:
                    print("nothing to off board")
                    pass

                # get infantry_unit_id from

    def IFV_transport_check(self):
        # 检测步兵是不是全部在车上或者不在车上。
        flag_on = True
        flag_off = True 
        # if there is no bubing, regarded as on board. # TODO: this it not safe
        infantry_units = self.select_by_type(self.status["operators"],key="sub_type",value=2)
        if len(infantry_units)==0:
            flag_on = True
            flag_off = False 

        for infantry_unit in infantry_units:
            if infantry_unit["on_board"] == True:
                flag_on = flag_on and True
                flag_off = flag_off and False
            else:
                flag_on = flag_on and False
                flag_off = flag_off and True
        
        return flag_on, flag_off 
    
    def jieju_check(self,model="all",**kargs):
        # check if the jieju finished.
        if model == "IFV":
            units = self.get_IFV_units() + self.get_infantry_units()
        elif model == "all":
            units = self.status["operators"]
        elif model == "part":
            units = kargs["units"]
            pass
        
        flag_finished = True 
        for unit in units:
            try:
                abstract_state_single = self.abstract_state[unit["obj_id"]]
            except:
                self.set_none(unit["obj_id"])
                abstract_state_single = self.abstract_state[unit["obj_id"]]

            if "abstract_state" in abstract_state_single:
                if abstract_state_single["abstract_state"] == "jieju":
                    flag_finished and False
                if abstract_state_single["abstract_state"] == "move_and_attack":
                    if "flag_jieju" in abstract_state_single:
                        flag_finished and False
                pass
            else:
                # no abstract for it.
                pass

            if len(self._check_actions(unit["obj_id"], model="jieju"))==0 and unit["forking"]==0 and unit["blood"]==1 :
                # not forking, can not fork anymore, so the forking process finished.
                flag_finished = flag_finished and True
            else:
                flag_finished = flag_finished and False
        
        return flag_finished

    def distinguish_saidao(self):
        # 区分当前是哪个赛道，然后把相应需要的全局变量和初始化做了。
        observation = self.status
        communications = observation["communication"] 
        flag_cross_fire = False      
        flag_defend = False
        flag_scout = False
        if self.num <2:
            for command in communications:
                if command["type"] in [210] :
                    # 说明是cross fire 赛道
                    flag_cross_fire = True
                if command["type"] in [209] :
                    # 说明是Scout 赛道
                    flag_scout = True
                if command["type"] in [208] :
                    # 说明是Defend 赛道
                    flag_defend = True
            
            # 然后搞一下相应的初始化。
            if flag_cross_fire:
                self.env_name = "cross_fire" 
                if self.num <2:
                    target_pos = self.get_target_cross_fire()
                else:
                    target_pos = self.target_pos
            elif flag_scout:
                self.env_name = "scout" 
                self.get_target_scout()
            elif flag_defend:
                self.env_name = "defend" 
                self.get_target_defend()
            else:
                raise Exception("invalid saidao, G")
    
    def distinguish_saidao2(self,task):
        flag_cross_fire = False
        flag_scout = False
        flag_defend = False
        flag_attack = False
        self.end_time = task["end_time"]
        if self.num <2:
            if task["type"] in [210] :
                # 说明是cross fire 赛道
                flag_cross_fire = True
            if task["type"] in [209] :
                # 说明是Scout 赛道
                flag_scout = True
            if task["type"] in [208] :
                # 说明是Defend 赛道
                flag_defend = True 
            if task["type"] in [207]:
                # 说明是attack 任务
                flag_attack = True

            # 然后搞一下相应的初始化。
            if flag_cross_fire:
                self.env_name = "cross_fire" 
                target_pos = self.get_target_cross_fire(task)
            elif flag_scout:
                self.env_name = "scout" 
                self.get_target_scout()
            elif flag_defend:
                self.env_name = "defend" 
                self.get_target_defend()
            elif flag_attack:
                self.env_name = "attack"
                target_pos = self.get_target_attack(task)
            else:
                # raise Exception("invalid saidao, G")
                print("invalid saidao, G")
                # 不要再报异常了。不然非法任务直接就把AI干死了
            
    def get_target_cross_fire(self,task):
        # call one time for one game.
        observation = self.status
        communications = observation["communication"]
        flag_done = False
        command = task
        if command["type"] in [210] :
            self.my_direction = command
            self.target_pos = self.my_direction["hex"]
            self.end_time = self.my_direction["end_time"]
            flag_done = True
        # for command in communications:
        #     if command["type"] in [210] :
        #         self.my_direction = command
        #         self.target_pos = self.my_direction["hex"]
        #         self.end_time = self.my_direction["end_time"]
        #         flag_done = True
        if flag_done==False:
            raise Exception("get_target_cross_fire: G!")
            # print("WTF, it should be cross_fire, GAN")
            # self.my_direction = []
            # self.target_pos = self.my_direction["hex"]
            # self.end_time = self.my_direction["end_time"]
        else:
            print("get_target_cross_fire: Done.")
        return  self.target_pos
    def get_target_attack(self,task):
        # call one time for one game.
        observation = self.status
        communications = observation["communication"]
        flag_done = False
        command = task
        if command["type"] in [207] :
            self.my_direction = command
            self.target_pos = self.my_direction["hex"]
            self.end_time = self.my_direction["end_time"]
            flag_done = True
        if flag_done==False:
            raise Exception("get_target_attack: G!")
        else:
            print("get_target_attack: Done.")
        return  self.target_pos
    
    def get_target_defend(self):
        observation = self.status
        communications = observation["communication"]
        flag_done = False

        defend_pos = []
        for communication in communications:
            defend_pos_single = communication["hex"]
            defend_pos.append(defend_pos_single)    
            
        #@szh0404 这个地方看一下是不是在这里初始化BopSubType
        self.troop_stage = {op["obj_id"]: ""  for op in self.observation["operators"] if op["sub_type"]==BopSubType.Infantry}
        self.chariot_stage = {op['obj_id']: "" for op in self.observation["operators"] if op["sub_type"] == BopSubType.Chariot}
        self.tank_stage =  {op['obj_id']: "" for op in self.observation["operators"] if op["sub_type"] == BopSubType.Tank}
        ops = self.get_defend_armorcar_units() + self.get_defend_infantry_units() + self.get_defend_tank_units()
        self.ops_destination = {op['obj_id']: [] for op in ops if op["color"] == self.color}
        self.prepare_to_occupy = {op['coord']: [] for op in self.observation["cities"]}
        

        
        return defend_pos

    def get_target_scout(self):
        pass

    # then step functions 
    def step(self, observation: dict, model="guize"):
        # if model = guize, then generate self.act in step, else if model = RL, then generate self.act in env rather than here.
        self.act = []
        self.observation = observation
        self.ob = self.observation
        self.update_time()
        self.update_tasks()
        self.update_all_units()
        self.update_valid_actions()

        # self.num = self.num + 1 
        self.num_real = self.num # 这个用来以防万一，因为后面的self.num要改成相对的。
        if self.num == 1:
            print("Debug, moving")
        else:
            if self.num%100==99:
                print("Debug, self.num = "+str(self.num))
        self.observation = observation
        self.status = observation # so laji but fangbian.

        self.team_info = observation["role_and_grouping_info"]
        self.controposble_ops = observation["role_and_grouping_info"][self.seat]["operators" ]

        # 重新整一个，用来处理人机混合。
        for task in self.tasks:  # 遍历每个分配给本席位任务
            time_start = task["start_time"] # 这个用来修改self.num,实现相对的时长。
            self.num = self.num - time_start # 这里改成相对的时长，后面再改回去
            # get the target first.
            self.distinguish_saidao2(task)
            if task["type"] == 210:
                # 集结==cross fire
                self.env_name="cross_fire"
                self.Gostep_abstract_state()
                self.step_cross_fire_test()
            elif task["type"] == 209:
                # 侦察
                self.env_name="scout"
                self.step_scout(task)
            elif task["type"] == 208:
                # 防御
                self.env_name="defend"
                self.step_defend()
            elif task["type"] == 207:
                # 进攻，之前是没有的。
                self.env_name="attack"
                self.Gostep_abstract_state()
                self.step_attack()
            self.num = self.num + time_start # 完了一个循环之后再改回去。原则上这里加了之后self.num应该等于self.num_real
        
        # 再来一个，人机混合的时候如果没有任务，那就A过去。
        if len(self.tasks) == 0:
            # 先A过去，然后一转防御
            self.Gostep_abstract_state()
            self.step_default()

        return self.act

    def step0(self):
        # this is the first one for learning the guize of miaosuan, 1226 xxh.
        unit0 = self.get_bop(0)
        pos_0 = unit0["cur_hex"]
        target_pos = pos_0 + 3
        if self.num == 1:
            self.set_move_and_attack(unit0["obj_id"], target_pos)
        elif self.num > 114.514:
            self._move_action(unit0["obj_id"],target_pos)
            self._check_actions(unit0["obj_id"])
            self._fire_action(unit0["obj_id"])
            self._check_actions(unit0["obj_id"], model="test")
            self._check_actions(unit0["obj_id"], model="fire")
        pass
        
        # self.Gostep_abstract_state()
    
    def step_cross_fire(self):
        # this is to tackle cross_fire.
        target_pos = self.target_pos
        units=self.status["operators"]           
        IFV_units = self.get_IFV_units()
        infantry_units = self.get_infantry_units()
        UAV_units = self.get_UAV_units()
        tank_units = self.get_tank_units()
        # 这个是获取别的units用来准备一开始就解聚
        # others_units = list(set(units) - set(IFV_units) - set(infantry_units) - set(UAV_units))
        others_units = [unit for unit in units if (unit not in IFV_units) and (unit not in infantry_units) and (unit not in UAV_units)]

        flag_on,flag_off = self.IFV_transport_check()
        jieju_flag2 = self.jieju_check(model="part", units=IFV_units)

        if flag_on==False:
            # 如果刚开始且没上车，那就先上车
            self.IFV_transport(model="on")
        elif flag_on==True:
            # self.IFV_transport(model="off") # this is for test
            if jieju_flag2==False:
                for unit in IFV_units:
                    self.set_jieju(unit)

        jieju_flag = self.jieju_check(model="part", units=others_units)
        # if self.num<500 and jieju_flag==False:
        if jieju_flag==False:
            # 那就是没解聚完，那就继续解聚。
            for unit in others_units:
                self.set_jieju(unit)
        
        # F2A.
        # if jieju_flag == True and self.num<800:
        if jieju_flag == True:
            if self.num < 200:
                model="normal"
            else:
                model="force"
            self.group_A(others_units,target_pos,model=model)
            # self.group_A2(others_units,IFV_units)
        elif self.num>300:
            self.group_A(others_units,target_pos,model="force")

        if jieju_flag2 == True:
            # self.group_A(IFV_units,target_pos,model="force")
            self.list_A(IFV_units,target_pos)
        elif self.num>300:
            self.list_A(IFV_units,target_pos)

        # if arrived, then juhe.
        if self.num>800:
            self.final_juhe(tank_units)
            self.final_juhe(IFV_units)

        # if self.num>1000:
        #     # 最后一波了，直接F2A了
        #     self.F2A(target_pos)
        #     pass # disabled for tiaoshi
        
        if (self.num % 100==0) and (self.num>-200) and (self.num<2201):
            # 保险起见，等什么上车啊解聚啊什么的都完事儿了，再说别的。
            # deal with UAV.
            # self.UAV_patrol(target_pos)
            # kaibai is fine.
            self.group_A(UAV_units,target_pos)
        return 

    def step_cross_fire2(self):
        # this is to test group_A2.
        target_pos = self.target_pos
        units=self.status["operators"]           
        IFV_units = self.get_IFV_units()
        infantry_units = self.get_infantry_units()
        UAV_units = self.get_UAV_units()
        tank_units = self.get_tank_units()
        # 这个是获取别的units用来准备一开始就解聚
        # others_units = list(set(units) - set(IFV_units) - set(infantry_units) - set(UAV_units))
        others_units = [unit for unit in units if (unit not in IFV_units) and (unit not in infantry_units) and (unit not in UAV_units)]

        flag_on,flag_off = self.IFV_transport_check()
        jieju_flag2 = self.jieju_check(model="part", units=IFV_units)

        if flag_on==False:
            # 如果刚开始且没上车，那就先上车
            self.IFV_transport(model="on")
        elif flag_on==True and self.num<800:
            # self.IFV_transport(model="off") # this is for test
            if jieju_flag2==False:
                for unit in IFV_units:
                    self.set_jieju(unit)

        jieju_flag = self.jieju_check(model="part", units=others_units)
        # if self.num<500 and jieju_flag==False:
        if jieju_flag==False and self.num<800:
            # 那就是没解聚完，那就继续解聚。
            for unit in others_units:
                self.set_jieju(unit)
        
        # F2A.
        # if jieju_flag == True and self.num<800:
        if jieju_flag == True and jieju_flag2==True:
            if self.num < 200:
                model="normal"
            else:
                model="force"
            # self.group_A(others_units,target_pos,model=model)
            self.group_A2(others_units,IFV_units)
        elif self.num>300:
            # self.group_A(others_units,target_pos,model="force")
            self.group_A2(others_units,IFV_units)

        if jieju_flag2 == True:
            # self.group_A(IFV_units,target_pos,model="force")
            # self.list_A(IFV_units,target_pos,target_pos_list = [2024,2024,self.target_pos] )
            self.list_A(IFV_units,target_pos)
        if self.num>300:
            self.list_A(IFV_units,target_pos)

        # if arrived, then juhe.
        if self.num>800:
            self.final_juhe(tank_units)
            self.final_juhe(IFV_units)

        # if self.num>1000:
        #     # 最后一波了，直接F2A了
        #     self.F2A(target_pos)
        #     pass # disabled for tiaoshi
        
        if (self.num % 100==0) and (self.num>-200) and (self.num<2201):
            # 保险起见，等什么上车啊解聚啊什么的都完事儿了，再说别的。
            # deal with UAV.
            # self.UAV_patrol(target_pos)
            # kaibai is fine.
            self.group_A(UAV_units,target_pos)
        return 

    def step_cross_fire_test(self):
        # this is to test group_A2.
        print("step_cross_fire_test: successfully get in, self.num="+str(self.num))
        target_pos = self.target_pos
        units=self.status["operators"]           
        IFV_units = self.get_IFV_units()
        infantry_units = self.get_infantry_units()
        UAV_units = self.get_UAV_units()
        tank_units = self.get_tank_units()
        # 这个是获取别的units用来准备一开始就解聚
        # others_units = list(set(units) - set(IFV_units) - set(infantry_units) - set(UAV_units))
        others_units = [unit for unit in units if (unit not in IFV_units) and (unit not in infantry_units) and (unit not in UAV_units)]

        flag_on,flag_off = self.IFV_transport_check()
        jieju_flag2 = self.jieju_check(model="part", units=IFV_units)

        if flag_on==False:
            # 如果刚开始且没上车，那就先上车
            self.IFV_transport(model="on")
        elif flag_on==True and self.num<800:
            # self.IFV_transport(model="off") # this is for test
            if jieju_flag2==False:
                for unit in IFV_units:
                    self.set_jieju(unit)

        jieju_flag = self.jieju_check(model="part", units=others_units)
        # if self.num<500 and jieju_flag==False:
        if jieju_flag==False and self.num<800:
            # 那就是没解聚完，那就继续解聚。
            for unit in others_units:
                self.set_jieju(unit)
        
        # F2A.
        # if jieju_flag == True and self.num<800:
        qianpai_units = self.get_qianpai_units()
        others2_units = [unit for unit in units if (unit not in qianpai_units)]
        if jieju_flag == True and jieju_flag2==True:
            # self.group_A(others_units,target_pos,model=model)
            self.group_A2(others2_units,qianpai_units)
        elif self.num>300:
            # self.group_A(others_units,target_pos,model="force")
            self.group_A2(others2_units,qianpai_units)

        if jieju_flag2 == True:
            # self.group_A(IFV_units,target_pos,model="force")
            # self.list_A(IFV_units,target_pos,target_pos_list = [2024,2024,self.target_pos] )
            self.list_A(qianpai_units,target_pos)
        elif self.num>350:
            self.list_A(qianpai_units,target_pos)

        # # if arrived, then juhe.
        # if self.num>800:
        #     self.final_juhe(tank_units)
        #     self.final_juhe(IFV_units)

        if self.num>1500:
            # 最后一波了，直接F2A了
            self.F2A(target_pos)
            pass # disabled for tiaoshi
        
        if (self.num % 100==0) and (self.num>-200) and (self.num<1000):
            # 保险起见，等什么上车啊解聚啊什么的都完事儿了，再说别的。
            # deal with UAV.这里面是带骑脸目标、停车、引导打击等逻辑的，但是好像不是太适合现在这个场景。
            self.UAV_patrol(target_pos)
            
            # kaibai is fine.逃避可耻但有用
            # self.group_A(UAV_units,target_pos)

            # 抢救一下，无人机给一些新的说法
            # self.UAV_patrol2(self.unscouted)
        else:
            self.group_A(UAV_units,target_pos)


        self.final_xiache(units)
        
        return 

    def step_scout(self,task):
        # unfinished yet.

        # self.update_time()
        # self.update_tasks()
        print("step_scout: successfully get in, self.num="+str(self.num))

        if not self.tasks:
            return []  # 如果没有任务则待命
        
        self.prefer_shoot()  # 优先选择射击动作

        self.task_executors[task["type"]].execute(task, self)  
    
    def step_attack(self):
        # 先解决有无问题。F2A总会吧。
        print("step_jingong: successfully get in, self.num="+str(self.num))
        units = self.status["operators"] 
        self.group_A2(units,[])        # 直接框框A过去。
        self.final_xiache(units) 
        return
    
    def step_default(self):
        # 这个是处理没有收到信号的时候的情况，先A过去然后在那里防御。
        target_pos = 2652
        units = self.status["operators"]
        start_time = 1000
        if self.num < start_time:
            self.group_A(units,target_pos)
        else:
            self.num = self.num - start_time 
            # 然后假装防御一会儿.这就需要改成相对的路径了
            self.step_defend()

            self.num = self.num + start_time 



    ###################### defend  ############################    
    @time_decorator
    def step_defend(self):
        # # unfinished yet.
        
        # # 先把场景目标点在哪读出来
        # defend_pos = [0,0,0] # three in hex form
        # # get the target first.
        # if self.num <2:
        #     defend_pos = self.get_target_defend()
        #     self.defend_pos = defend_pos
        # else:
        #     defend_pos = self.defend_pos    

        # # 经典分兵编队
        # units=self.status["operators"]           
        # IFV_units = self.get_IFV_units()
        # infantry_units = self.get_infantry_units()
        # UAV_units = self.get_UAV_units()
        # others_units = [unit for unit in units if (unit not in IFV_units) and (unit not in infantry_units) and (unit not in UAV_units)]

        # # 怎么判断A到了呢？姑且可以是全停下就算是A到了。或者是直接步数
        # jieju_flag = self.jieju_check(model="part", units=others_units)
        # if self.num<100 and jieju_flag==False:
        #     # 那就是没解聚完，那就继续解聚。
        #     for unit in (others_units+infantry_units+IFV_units):
        #         self.set_jieju(unit)
        # else:
        #     index_chong = round(((self.num+101) % 600) / 200 ) - 1  # 这个就应该是0,1,2
            
        #     self.group_A((others_units+UAV_units), defend_pos[index_chong])
        #     for unit in IFV_units+infantry_units:
        #         self.set_open_fire(unit)

        # print("step_defend: unfinished yet.")
        print("step_defend: successfully get in, self.num="+str(self.num))

        #@szh 0404 添加fort状态
        self.fort_assignments = {op["obj_id"]: op["entering_fort_partner"]+op["fort_passengers"] for op in self.observation["operators"] if op["type"]==BopType.Fort}
        #@szh 0404 更新trooop stage 和 chariot stage
        chariots = [op for op in self.observation["operators"] if op["type"]==BopType.Vehicle and op["color"] == self.color]
        troops =   [op for op in self.observation["operators"] if op["sub_type"]==BopSubType.Infantry and op["color"] == self.color]
        tanks =    [op for op in self.observation["operators"] if op["sub_type"]==BopSubType.Tank and op["color"] == self.color]
        ops = self.get_defend_infantry_units() + self.get_defend_armorcar_units() + self.get_defend_tank_units()
        ops_dests = [op for op in ops if op["color"] == self.color]
        for op in chariots:
            if op["obj_id"] not in self.chariot_stage.keys():
                self.chariot_stage[ op["obj_id"] ] =""    # 对应可能是新解聚的情况
                # self._defend_jieju_and_move( op["obj_id"] )
        for op in troops:
            if op["obj_id"] not in self.troop_stage.keys():
                self.troop_stage[ op["obj_id"] ]  =""
                self._defend_jieju_and_move( op["obj_id"] )
                
        for op in tanks:
            if op["obj_id"] not in self.tank_stage.keys():
                self.tank_stage[ op["obj_id"] ]  =""
                # self._defend_jieju_and_move( op["obj_id"] )
        for op in ops_dests:
            if op["obj_id"] not in self.ops_destination.keys():
                # self.ops_destination[ op["obj_id"]]  = ""
                self.ops_destination[ op["obj_id"]]  = []
        self.reset_occupy_state()                         # 重新看有没有空点
        self.update_prepare_to_occupy()
        self.update_enemyinfo()
        self.update_filter_enemyinfo()
        if self.num <= 900:
            for troop in self.get_defend_infantry_units():
                if self.num <=2:
                    closest_city = min(
                        self.observation["cities"],
                        key=lambda city: self.distance(troop["cur_hex"], city["coord"]),
                    )
                    self.ops_destination[ troop["cur_hex"] ]  =  [closest_city["coord"]]
                self.defend_BT_Troop(troop["obj_id"])
            if self.num >= 90 and len(self.filtered_enemyinfo) > 0: 
                if self.defend_shrink_by_power():
                    if self.defend_let_our_power_shrink_to_city():
                        return 
                if self.defend_attack_by_power():
                    if self.defend_let_our_power_attack_to_city():
                        return 
            for chariot in self.get_defend_armorcar_units():
                self.defend_BT_Chariot(chariot["obj_id"])
            for tank in self.get_defend_tank_units():
                self.defend_BT_Tank(tank["obj_id"])
        else:
            self.defend_goto_cities()

    #@szh 0404 所有算子执行最终夺控
    def defend_goto_cities(self):
        our_units = self.get_defend_infantry_units() + self.get_defend_armorcar_units() + self.get_defend_tank_units()
        for u in our_units:
            closest_city = min(
                self.observation["cities"],
                key=lambda city: self.distance(u["cur_hex"], city["coord"])
            )
            if u["cur_hex"] in [c["coord"] for c in self.observation["cities"]]:
                self.gen_occupy(u["obj_id"])
            if self.distance(u["cur_hex"], closest_city["coord"]) >= 2:
                self._move_action(u["obj_id"], closest_city["coord"])
            else:
                self.__tank_handle_open_fire(u["obj_id"])
        
        return 

    #@szh 0404 添加转化为二级冲锋的
    def gen_change_state(self, obj_id, chongfeng):
        """Generate change state action with some probability."""
        bop = self.get_bop(obj_id)
        if not bop:
            return
        # 步兵正常机动切换成二级冲锋
        if bop["sub_type"] == 2  and chongfeng == 2 :
            target_state = 3
            act_change_state = {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.ChangeState,
                "target_state": target_state,
            }
            self._action_check_and_append(act_change_state)
            return
        elif bop["sub_type"] == 2  and chongfeng == 1 :
            target_state = 2
            act_change_state =  {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.ChangeState,
                "target_state": target_state,
            }
            self._action_check_and_append(act_change_state)
            return 
        elif bop["sub_type"] == 2  and chongfeng == 0 :
            act_stop =  {
                    "actor": self.seat,
                    "obj_id": obj_id,
                    "type": ActionType.StopMove,
                }
            self._action_check_and_append(act_stop)
            return 
        
    #@szh 0404 抄写解聚的
    def gen_fork(self, obj_id):
        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Fork}

    #@szh 0404 抄写的进夺控点的
    def gen_enter_fort(self, obj_id, target_fort):
        # 敌方算子位置
        # enemy_hexes = [o["cur_hex"] for o in self.observation["operators"] if o["color"] != self.color]
        # city_hexes =  [city["coord"] for city in self.observation["cities"] if city["flag"] != self.color]
        # city_hexes_no_op = []
        # # 没有敌方算子的夺控点
        # if city_hexes:
        #     for city_hex in city_hexes:
        #         neighbours = self.map.get_neighbors(city_hex)
        #         if not set(neighbours) & set(enemy_hexes):
        #             city_hexes_no_op.append(city_hex)
        # if not city_hexes_no_op:
            #target_fort = random.choice(candidate)["target_obj_id"]
        self.fort_assignments[target_fort].append(obj_id)
        act_enter_fort =  {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.EnterFort,
            "target_obj_id": target_fort,
        }
        self._action_check_and_append(act_enter_fort)
        return act_enter_fort

    #@szh 0328 直接调用 select_by_type 
    def get_defend_infantry_units(self)->List[Dict]:
        #  返回unit 对象 list 形式
        infops = self.select_by_type(self.status["operators"],key="sub_type",value=2)
        return [op for op in infops if op["color"] ==  self.color]
    def get_defend_tank_units(self):
         #  返回unit 对象 list 形式
        tankops = self.select_by_type(self.status["operators"],key="sub_type",value=0)
        return  [op for op in tankops if op["color"] ==  self.color]
    def get_defend_armorcar_units(self):
         #  返回unit 对象 list 形式
        armorcar_ops = self.select_by_type(self.status["operators"],key="sub_type",value=1)
        return   [op for op in armorcar_ops if op["color"] ==  self.color]

    #@szh 0328 规划战斗工事 只有人员的工事
    def get_fightingbase_units(self)->List[Dict]:
         #  返回unit 对象 list 形式
        return self.select_by_type(self.status["operators"],key="sub_type",value=11)
    def get_hiddingbase_units(self)->List[Dict]:
        return self.select_by_type(self.status["operators"],key="sub_type",value=20)
    
    #@szh @0417 写一个更新 enemy info 的函数
    def update_enemyinfo(self):
        maxsize = 3
        ene_ops = self.defend_enemy_info()
        for eop in ene_ops:
            if eop["obj_id"] not in self.enemy_info.keys():
                self.enemy_info[ eop["obj_id"] ] = RecurrentQueue(maxsize)
            if "time_step" not in eop.keys():
                eop["time_step"] = self.num
            self.enemy_info[ eop["obj_id"] ]._push_back(eop)
        return
    #@szh @0417 写一个过滤 enemy info 的函数  太长时间没看到的 
    def update_filter_enemyinfo(self):
        max_unseen_time  = self.max_unseen_time
        self.filtered_enemyinfo = self.enemy_info
        member_to_del  = []
        for obid , v in self.enemy_info.items():
            info2_index = (v.rear - 1  +  v.maxsize ) % v.maxsize
            top_record = v._get_item_by_index(info2_index)
            if self.num - top_record["time_step"] > max_unseen_time:
                member_to_del.append(obid)
        for del_obid in member_to_del:
            self.filtered_enemyinfo[del_obid]
        return 
    
            
        
        
        
                
                
    #@szh 0329 补充获得敌方detect 函数
    def defend_enemy_hex(self)->List[int]:
        return [op["cur_hex"] for op in self.status["operators"] if op["color"] != self.color]
    def defend_enemy_info(self)->List[Dict]:
        return [op for op in self.status["operators"] if op["color"] != self.color]


    #@szh 0329 通视情况 这里用see enemy bop ids 可以查看所有对方
    def defend_see_enemy_bop_ids(self, unit_id : str)-> List[str]:
        detected_enemy_unit = []
        unit = self.get_bop(unit)
        if "see_enemy_bop_ids" in unit.keys():
            detected_enemy_unit = unit["see_enemy_bop_ids"]
        return detected_enemy_unit

    #@szh 0402 步兵班找最近的工事 注意这个函数不要随意调用，在调用时
    #需要进一步想一下怎么去安排步兵班
    #严格来说步兵班进入工事应该根据场景去自动分配 不要在这里写
    def defend_troop_find_nearest_fort(self,
        troop_id:str,
    )->str:
        troop_unit = self.get_bop(troop_id)
        if troop_unit is None or troop_unit["sub_type"] != 2 or troop_unit["in_fort"] is True:
            return None
        our_hidding_fort_list = self.get_hiddingbase_units()
        our_fighting_fort_list = self.get_fightingbase_units()
        # 这里注意一共最多两个步兵班所以只选最近的两个
        fort_list = our_fighting_fort_list + our_hidding_fort_list
        fort_list = list(set(fort_list))
        min_dis = 1000
        nearest_fort  = None
        for fort_info  in fort_list:
            fort_cur_hex = fort_info["cur_hex"]
            dis_ = self.distance(fort_cur_hex, troop_unit["cur_hex"])
            if dis_ < min_dis: 
                min_dis = dis_
                nearest_fort = fort_info
        return nearest_fort
    
            
            
             
        

    

    #@szh 0329 detect 信息 再做个filter 根据地点选择一定范围内的敌方算子 
    def defend_detect_enemy_by_scope(self, 
        center_pos: int,
        filter_scope: int,
        mode: str =  "normal"
        )->List[Dict]:    
        """
         后续扩展哪些类型的敌方算子要加入进来  例如 炮弹等有明确目标的就不必考虑 
         在defend 场景下只需要考虑 坦克 步兵  战车
        """
        detected_units_state = self.get_detected_state(self.observation) # 返回的是 list of units
        # detected_units_state : 一个包括对方所有算子的list  基于此 选择筛选一定范围的敌方算子
        detect_unit_in_scope = []
        for detect_unit in detected_units_state:
            detect_unit_id  = detect_unit["obj_id"] 
            detect_unit_pos = self.get_pos(detect_unit_id)
            if self.distance(center_pos , detect_unit_pos) < filter_scope:
                if mode == "normal":
                    detect_unit_in_scope.append(detect_unit_id)
        return detect_unit_in_scope

    #@szh0328 防御点坐标    
    def get_defend_cities_pos(self):
        observation = self.status
        if "cities" not in observation.keys():
            return []
        city_coord =  [city["coord"] for city in self.observation["cities"]]
        return city_coord
    
    # 防御点信息
    #@szh0328 包括 分值哈的
    def get_defend_cities_info(self):
        observation = self.status
        if "cities" not in observation.keys():
            return []
        city_info = [city for city in self.observation["cities"]]
        return city_info
            
    #@szh0328 步兵班进入工事
    #@szh0403 开火调用——open attack fire
    def defend_gen_shoot(self): 
        pass
    
    #@szh0402 找夺控点距离为1范围内的格子编号 作为夺控要点
    def defend_get_key_point_around_fort(self,center_pos:int, mode: str)->List[int]:
        distance_min, distance_max = 1,2
        if mode == "key": 
            distance_min, distance_max = 1,2    
        elif mode == "youji":
            distance_min, distance_max = 2,3
        return list(self.map.get_grid_distance(center_pos, distance_min,distance_max))
    
    #@szh0404 根据敌方位置   把夺控点“后方”的我方游击点位置筛选出来
    def defend_filter_key_point_by_enemy_pos(
        self, 
        center_pos: int,                  #  参考点  夺控点
        key_point_candidate: List[int],   # list of cur_hex 
        filter_mode:str
    )->List[int]:
        # filter_mode ==  "enemy"  用敌方位置去筛选  找距离
        # filter_mode ==  "index"  用地图编号去筛选
        # filter_mode ==  "mix"    结合index 和  enemy pos
        # filter_mode ==  "none" 
        if filter_mode == "enemy": 
            destination = key_point_candidate 
            enemypos_list = self.defend_enemy_hex()  #
            if len(enemypos_list) > 0:
                min_dis_ = 10000
                min_hex_ = 0
                for enemypos in enemypos_list:
                    if self.distance(enemypos , center_pos) < min_dis_ :
                        min_dis_ = self.distance(enemypos , center_pos)
                        min_hex_ = enemypos
                destination =  [p for p in key_point_candidate if self.distance(p, min_hex_) > min_dis_ ]
                if destination is None or len(destination) == 0:
                    destination =   [center_pos]
            elif len(enemypos_list) ==0:
                destination = key_point_candidate+[center_pos]
            return destination
        if filter_mode == "index" and len(self.defend_enemy_hex()) > 0:
            # 这里可以再 思考一下具体怎么用
            # for p in key_point_candidate:
            #     if (p% 100) > center_pos%100 or \
            #         (p/100) == center_pos/100 and 
            destination =  [p for p in key_point_candidate if (p% 100) > center_pos%100]
            if destination is None or len(destination) == 0:
                destination =   [center_pos]
            return destination
        if filter_mode == "none" and len(self.defend_enemy_hex()) > 0:
            return key_point_candidate


    def __handle_open_fire(self, attacker_ID):
        # 有啥能打的就都打一遍呗。这个还是共用CD的，不用考虑遍历。共用CD挺傻逼的讲道理。

        # 如果在机动，就停下来。
        flag_is_stop = self.is_stop(attacker_ID)
        if not(flag_is_stop):
            # 没有is stop就是在机动呗，那就停下来。
            self._stop_action(attacker_ID)

        # 这个写法相当于每一步都检测一次，能打就打
        # 在机动或者正在停的时候反正也检测不到有效的开火命令，所以这条空过几次感觉问题也不大
        self._fire_action(attacker_ID)
    
    #@szh0404  将点按照特定要求排序  如高程 地形等 
    def defend_sort_point_by_type(self,
        key_point_candidate: List[int],   # list of cur_hex 
        sort_type:str # eg elev                            
    )-> List[int]:
        if key_point_candidate is None:
            return []
        if len(key_point_candidate) == 0:
            return []
        if sort_type == "elev":            # 根据高程排序
            return sorted(key_point_candidate, \
            key = lambda x:self.map_data[x//100][x%100]["elev"] , reverse= True)

    #@szh 0404 将点进行筛选 只有地形为0，1，2 的可以
    def defend_filter_key_point_by_scope(
        self,
        key_point_candidate: List[int],   # list of cur_hex 
    )->List[int]:
        return [c for c in key_point_candidate if self.map.basic[c // 100][c % 100]["cond"] in [0,1,2] ]

    #@szh0402 步兵班的行为树
    @time_decorator
    def defend_BT_Troop(self, obj_id):
        #设几个状态转换的量 这里得是每个id 都要
        bop_troop = self.get_bop(obj_id)
        # self.troop_stage[obj_id]= "start"
        if self.troop_stage[obj_id] != "fire":
            # 新场景下直接先解聚
            # 得判断一下是不是在forking 是的话直接退出
            closest_city = min(
                self.observation["cities"],
                key=lambda city: self.distance(bop_troop["cur_hex"], city["coord"]),
             )
            if bop_troop["cur_hex"] == closest_city["coord"]:
                    self.gen_occupy(obj_id)
            self.ops_destination["obj_id"] =  [closest_city["coord"]]
            if self.num <=120:
                if self.color != closest_city["flag"] :
                    self.ops_destination[obj_id] = [closest_city["coord"]]
                    self.gen_change_state(obj_id, 2)
                    self._move_action( obj_id,  self.ops_destination[obj_id][0] )
                    return
            #     # if self.color == closest_city["flag"] and bop_troop["cur_hex"] == closest_city["coord"] and self.troop_stage[obj_id] == "":
            #     #     self.gen_change_state(obj_id, 0)
                          
            if bop_troop["forking"]:
                return 
            act, _jieju_flag_done =  self._jieju_action(obj_id)
            if self.num <= 120 and bop_troop["move_to_stop_remain_time"]>0 :
                _jieju_flag_done = True

            if _jieju_flag_done == False: # 未执行解聚直接进入到
                self.defend_troop_start_stage_zhandian(obj_id)
            
        elif self.troop_stage[obj_id] == "fire":
            self.defend_troop_fire_stage_zhandian(obj_id)
            
    @time_decorator    
    def defend_BT_Chariot(self,obj_id):
        bop_chariot = self.get_bop(obj_id)
        if self.chariot_stage[obj_id] == "":
            # 战车也是先解聚
            if bop_chariot["forking"]:
                return 
            self.defend_chariot_start_stage_zhandian(obj_id)
                
        elif self.chariot_stage[obj_id] == "fire":
            self.defend_chariot_fire_stage_zhandian(obj_id)
            
    @time_decorator
    def defend_BT_Tank(self, obj_id):
        bop_tank =self.get_bop(obj_id)
        if self.tank_stage[obj_id] == "":
            if bop_tank["forking"]:
                return 
            self.defend_tank_start_stage_zhandian(obj_id)
                
        elif self.tank_stage[obj_id] == "fire":
            self.defend_tank_fire_stage_zhandian(obj_id)
            pass    

    def defend_key_point_filter_and_sort(self, 
          key_point_candidates: List,
          closest_city:Dict,
          filter_mode:str,
          sort_mode:str
        )-> List[int]:
        after_filter_key_point = False
        if len(self.defend_filter_key_point_by_scope(key_point_candidates)):
            key_point_candidates = self.defend_filter_key_point_by_scope(key_point_candidates)
            key_point_candidates =  self.defend_filter_key_point_by_enemy_pos(\
                closest_city["coord"], key_point_candidates,\
                filter_mode= filter_mode
                )
            key_point_candidates = self.defend_sort_point_by_type(\
                key_point_candidates, 
                sort_mode
                )   
            if len(key_point_candidates) > 0 :
                after_filter_key_point = True  
        if after_filter_key_point == False:
            return []
        return key_point_candidates
        
    #@szh 0404 针对重型战车找最优的保护夺控点的
    def defend_chariot_find_best_cover_points(self, center_pos: int, dis_min:int, dis_max:int)->List[int]:
        """
             围绕center——pos 找距离为dis 且能够通视最多的点
             对于重型战车 选择距离夺控点 4-6 这个位置的
        """
        best_cover_points = []
        neighbor_in_dis = self.map.get_grid_distance(center_pos, dis_min, dis_max)
        # 筛地形
        filtered_by_scope = self.defend_filter_key_point_by_scope(neighbor_in_dis)
        # 筛位置  夺控点后方的
        enemy_detected = [op for op in self.observation["operators"] if op["color"]!=self.color]
        if len(enemy_detected) == 0:
            # 如果没有发现敌方算子 则无法判断敌方位置 那就寄了  则直接返回一个夺控点
            return [center_pos]
        #closest_enemy, min_dis = self.get_bop_closest(bop, enemy_detected)
        filter_by_pos = self.defend_filter_key_point_by_enemy_pos(
            center_pos,
            filtered_by_scope,
            filter_mode = "enemy" 
        )
        flag_has_forest_nearby = False
        # 1.先找周围一定范围有没有树林啥的 有的话去一个集合  再根据高程排序
        # 在 4-7 的一个范围内
        forest_hex = [ p for p in filter_by_pos if self.map.basic[p // 100][p % 100]["cond"] in [1] ]
        if len(forest_hex) != 0:
            flag_has_forest_nearby = True
        if flag_has_forest_nearby:
            forest_hex.sort(key = lambda p : self.map.basic[p // 100][p % 100]["elev"] , reverse= True)
            return forest_hex
        # 2.没有树林则按高程排序 
        filter_by_pos.sort(key = lambda p : self.map.basic[p // 100][p % 100]["elev"] , reverse= True)
        return filter_by_pos  
        # -----------------下面的暂时不用-----------------------
        # 找通视觉点最多的 cover point   选择8格这个通视量
        dis_max_cover = dis_max      # 8 格左右 
        neighbor_to_cover_can_see = self.map.get_grid_distance(center_pos, 0, dis_max_cover)
        cover_points_around_city = self.defend_filter_key_point_by_enemy_pos(
            center_pos,
            neighbor_to_cover_can_see,
            filter_mode = "enemy"
        )
        cover_points_filter_by_pos_by_enemy = self.defend_filter_key_point_by_enemy_pos(
            center_pos,
            cover_points_around_city,
            filter_mode = "enemy" 
        )
        # 选择的都是夺控点前面的点
        cover_points_test_can_see = list( set(cover_points_around_city) -  set(cover_points_filter_by_pos_by_enemy))
        # 逐个点检测同时情况
        # 统计tongshi点个数
        
    #@szh 0404 执行夺控指令
    def gen_occupy(self, obj_id):
        action_gen = {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.Occupy
            }
            # self.act.append(action_gen)
        self._action_check_and_append(action_gen)
        
    #@szh 0417 写一个计算当前敌我双方力量配比的
    def defend_calculate_power(self, color:int, candidates:List[Dict])->int :
        power_ = 0
        if color == self.color:
            #ops = self.get_defend_armorcar_units() + self.get_defend_infantry_units() + self.get_defend_tank_units()
            ops = candidates
            for op in ops:
                if op["sub_type"] ==  BopSubType.Infantry:
                    power_ += FriendValue.Infantry*op["blood"]
                elif op["sub_type"] == BopSubType.Tank:
                    power_ += FriendValue.Tank*op["blood"]
                elif op["sub_type"] == BopSubType.Chariot:
                    power_ += FriendValue.Chariot*op["blood"]   
        else:
            #ops = [op for op in self.observation["operators"] if op["color"]!=self.color]
            ops = candidates
            for op in ops:
                if op["sub_type"] ==  BopSubType.Infantry:
                    power_ += EnemyValue.Infantry*op["blood"]
                elif op["sub_type"] == BopSubType.Tank:
                    power_ += EnemyValue.Tank*op["blood"]
                elif op["sub_type"] == BopSubType.Chariot:
                    power_ += EnemyValue.Chariot*op["blood"]
        return power_

    #@szh 0417 写一个收缩战略  观察到敌我力量悬殊 收缩到一个点
    @time_decorator
    def defend_shrink_by_power(self):
        our_ops = self.get_defend_tank_units() + self.get_defend_infantry_units()+ self.get_defend_armorcar_units()
        ene_ops_in_obs = [op["obj_id"] for op in self.observation["operators"] if op["color"]!=self.color]
        ene_ops = [op for op in self.observation["operators"] if op["color"]!=self.color]
        for obid , v in self.filtered_enemyinfo.items():
            info2_index = (v.rear - 1  +  v.maxsize ) % v.maxsize
            top_record = v._get_item_by_index(info2_index)
            if obid not in ene_ops_in_obs:
                ene_ops.append(top_record)
        our_power = self.defend_calculate_power(self.color, our_ops)
        enemy_power = self.defend_calculate_power(~self.color,ene_ops)
        if enemy_power >= int(2 * our_power):
            return True
        return False
    
    #@szh 0417 写一个进攻战略 如果我方力量较敌方更强  直接进点占点
    @time_decorator
    def defend_attack_by_power(self):
        our_ops = self.get_defend_tank_units() + self.get_defend_infantry_units()+ self.get_defend_armorcar_units()
        ene_ops_in_obs = [op["obj_id"] for op in self.observation["operators"] if op["color"]!=self.color]
        ene_ops = [op for op in self.observation["operators"] if op["color"]!=self.color]
        for obid , v in self.filtered_enemyinfo.items():
            info2_index = (v.rear - 1  +  v.maxsize ) % v.maxsize
            top_record = v._get_item_by_index(info2_index)
            if obid not in ene_ops_in_obs:
                ene_ops.append(top_record)   
        our_power = self.defend_calculate_power(self.color, our_ops)
        enemy_power = self.defend_calculate_power(~self.color, ene_ops)
        if enemy_power < int(0.5* our_power):
            return True
        return False
    
    #@szh 0417 写一个收缩战略  观察到敌我力量悬殊 全线收缩
    def defend_let_our_power_shrink_to_city(self)->bool:  # True  就收缩  False 就继续原来的策略
        """
            有步兵班就先去步兵班那
            没有就去夺控点
            后面需要确定一下是去夺控点好还是直接去步兵班的位置好
        """
        our_tanks = self.get_defend_tank_units()
        our_infan  = self.get_defend_infantry_units()
        our_chariot = self.get_defend_armorcar_units()
        flag_has_infantry_to_occ = False
        city_pos = [c["coord"] for c in self.observation["cities"]]
        cities = [c for c in self.observation["cities"]]
        ene_city_pos = [c["coord"] for c in cities if c["flag"] != self.color]  # List[int]
        if len(our_infan) > 0:
            # 直接跟步兵班那里
            # 选距离最近的步兵班
            for tank in our_tanks:
                self.__tank_handle_open_fire(tank["obj_id"])
                if self.ops_destination[tank["obj_id"]] != [] and tank["cur_hex"] == self.ops_destination[tank["obj_id"]][0]:
                    self.ops_destination[ tank["obj_id"] ].pop(0)
                cloest_infan = min(
                    our_infan,
                    key = lambda infan : self.distance( infan["cur_hex"], tank["cur_hex"])
                )
                if tank["cur_hex"] in city_pos:
                    self.gen_occupy(tank["obj_id"])
                if len(ene_city_pos):
                    closest_ene_city = min(
                        ene_city_pos, key = lambda enec : self.distance(enec , tank["cur_hex"])
                    )
                    closest_ene_city_nbs = self.map.get_neighbors(closest_ene_city) + [ closest_ene_city ]
                    ene = [op for op in self.observation["operators"] if op["color"] != self.color and op["cur_hex"] in closest_ene_city_nbs]
                    if len(ene) == 0:
                        self.ops_destination[ tank["obj_id"] ].insert(0, closest_ene_city)
                        
                     
                self.ops_destination[ tank["obj_id"] ].append(cloest_infan["cur_hex"])
                self._move_action(tank["obj_id"] , self.ops_destination[ tank["obj_id"] ][0] )
                if tank["cur_hex"] == cloest_infan["cur_hex"]:
                    self.gen_change_state(tank["obj_id"], 0)
                    
            for chariot in our_chariot:
                cloest_infan = min(
                    our_infan,
                    key = lambda infan : self.distance( infan["cur_hex"], chariot["cur_hex"])
                )
                if chariot["cur_hex"] in city_pos:
                    self.gen_occupy(chariot["obj_id"])  # 能占点把点站上
                if len(ene_city_pos):
                    closest_ene_city = min(
                        ene_city_pos, key = lambda enec : self.distance(enec , chariot["cur_hex"])
                    )
                    closest_ene_city_nbs = self.map.get_neighbors(closest_ene_city) + [ closest_ene_city ]
                    ene = [op for op in self.observation["operators"] if op["color"] != self.color and op["cur_hex"] in closest_ene_city_nbs]
                    if len(ene) == 0:
                        self.ops_destination[ chariot["obj_id"] ].insert(0, closest_ene_city)
                self.ops_destination[ chariot["obj_id"] ].append( cloest_infan["cur_hex"] )
                self._move_action(chariot["obj_id"], self.ops_destination[ chariot["obj_id"]][0])
                if chariot["cur_hex"] == cloest_infan["cur_hex"]:
                    self.gen_change_state(chariot["obj_id"], 0)
                    self.__handle_open_fire(chariot["obj_id"])
            flag_has_infantry_to_occ = True
                
        return flag_has_infantry_to_occ
                
    
    #@szh 0417 写一个进攻战略  板载!!!
    def defend_let_our_power_attack_to_city(self)->bool:
        """
        有不是我方夺控的  坦克: 夺控点六格内没人-- 直接冲夺控点  冲完夺控点去支援  六格内有人在夺控点周围六个内晃悠
        战车: 
        
        """
        our_tanks = self.get_defend_tank_units()
        our_infan  = self.get_defend_infantry_units()
        our_chariot = self.get_defend_armorcar_units()
        scities = [c for c in self.observation["cities"]]
        ene_cities = [c for c in scities if c["flag"] != self.color]
        ene_cities_pos = [c["coord"] for c in ene_cities]
        if len(ene_cities) == 0:
            return False  
        # 估计敌人方位
        # enepos = self.filtered_enemyinfo
        # enemy_closest_city = None  # 找敌人离的最近的城市 六格随机游走
        ene_cities_map = {c["coord"] : 1000 for c in ene_cities}
        for ec in ene_cities:
            total_dis = 0
            for enid , eninfo_que in self.filtered_enemyinfo.items():
                last_index = (eninfo_que.rear - 1  +  eninfo_que.maxsize ) % eninfo_que.maxsize
                recent_record = eninfo_que._get_item_by_index(last_index)
                total_dis += self.distance(ec["coord"], recent_record["cur_hex"])
            ene_cities_map[ ec["coord"] ] = total_dis
        min_dis = 10000
        min_ene_city_pos = None
        for k, v in ene_cities_map.items():
            if v < min_dis:
                min_ene_city_pos = k 
                min_dis = v
        if min_ene_city_pos is not None :
            for tank in our_tanks:
                self.__tank_handle_open_fire(tank["obj_id"])
                ene_city_candidates = self.map.get_neighbors(min_ene_city_pos) + [min_ene_city_pos]
                tar_pos = random.choice(ene_city_candidates)
                # 找敌人离的最近的城市 六格随机游走
                if tank["speed"] == 0 or self.ops_destination[ tank["obj_id"] ] == []:
                    self.ops_destination[ tank["obj_id"] ] = [tar_pos]
                    self._move_action(tank["obj_id"] , self.ops_destination[ tank["obj_id"] ][0] )
                if tank["cur_hex"] in ene_cities_pos:
                    self.gen_occupy(tank["obj_id"])
            for chariot in our_chariot:
                # 找能尽可能覆盖这个点的位置
                nbs = self.map.get_neighbors(chariot["cur_hex"]) + [chariot["cur_hex"]]
                tars = self.map.get_neighbors(min_ene_city_pos) + [min_ene_city_pos]
                max_can_see = 0
                best_point_can_see = None
                if chariot["cur_hex"] in ene_cities_pos:
                    self.gen_occupy(chariot["obj_id"])    # 能占点就占点
                for nb in nbs:
                    total_can_see = 0
                    for tp in tars:
                        if self.map.can_see(nb, tp, 0):
                            total_can_see += 1
                    if total_can_see > max_can_see:
                        max_can_see = total_can_see
                        best_point_can_see = nb
                if best_point_can_see is not None:
                    if chariot["speed"] == 0 or self.ops_destination[ chariot["obj_id"] ] == []:
                        self.ops_destination[ chariot["obj_id"] ] = [best_point_can_see]
                        self._move_action(chariot["obj_id"] , self.ops_destination[ chariot["obj_id"] ][0] )
                    if chariot["cur_hex"] == best_point_can_see:
                        self.__handle_open_fire(chariot["obj_id"])
                    
            return True

                    
                # cloest_infan = min(
                #     our_infan,
                #     key = lambda infan : self.distance( infan["cur_hex"], chariot["cur_hex"])
                # )
                # self.ops_destination[ chariot["obj_id"] ] = cloest_infan["cur_hex"]
                # self._move_action(chariot["cur_hex"], self.ops_destination[ chariot["obj_id"]])
                # if chariot["cur_hex"] == cloest_infan["cur_hex"]:
                #     self.__handle_open_fire()
            
            
                # scities = [c for c in self.observation["cities"]]
                # scities.sort(key = lambda c: self.distance(c["coord"], bop["cur_hex"]))
                # second_near_city, third_near_city = scities[1], scities[2]
                # second_near_ene =  [op for op in self.observation["operators"] if op["color"]!=self.color \
                #             and self.map.get_distance(op["cur_hex"], second_near_city["coord"]) <= 1]
                # third_near_ene = [op for op in self.observation["operators"] if op["color"]!=self.color \
                #             and self.map.get_distance(op["cur_hex"], third_near_city["coord"]) <= 2]
        return False       
        
        
            
        
        
    
    #@szh 0404 重型战车的部署策略
    @time_decorator
    def defend_chariot_start_stage_zhandian(self, obj_id):
        #先解聚
        destination = None
        bop = self.get_bop(obj_id)  
        if bop["speed"] != 0:  # 有未完成的机动
            return
        if len(self.ops_destination[obj_id]) != 0 and self.ops_destination[obj_id][0] == bop["cur_hex"]:
            self.ops_destination[obj_id].pop(0)
        # self.__handle_open_fire(obj_id)           # 先开火打一发
        if self.ops_destination[obj_id] != [] and bop["cur_hex"] ==  self.ops_destination[obj_id][0]:
            self.chariot_stage[obj_id] = "fire"
            return  
        if self.ops_destination[obj_id] != [] and bop["cur_hex"] != self.ops_destination[obj_id][0]:
            self._move_action(obj_id, self.ops_destination[obj_id][0])
            return
        # 原则上来说一定有closest
        if bop["cur_hex"]  in  [c["coord"] for c in self.observation["cities"]]:
            self.gen_occupy(obj_id)
        closest_city = min(
            self.observation["cities"],
            key=lambda city: self.distance(bop["cur_hex"], city["coord"])
        )
        if closest_city["flag"] != self.color: # 先占点
            self.ops_destination[obj_id] = [closest_city["coord"]]
            self._move_action(obj_id, self.ops_destination[obj_id][0])
            if bop["cur_hex"] == closest_city["coord"]:
                self.gen_occupy(obj_id)
        # 判断是否为机动算子
        # 判断夺控点周围有没有我方算子
        if self.color == closest_city["flag"]:
            tar = self.defend_check_nearby_enemy(obj_id)
            if len(tar) > 0:
                self._move_action(obj_id, tar[0])
                return
        our_troop_in_neighbor = [
            op for op in self.get_defend_infantry_units() if self.distance(op["cur_hex"],closest_city["coord"]) \
            <=1 
        ]          
        our_units = self.get_defend_infantry_units() + self.get_defend_armorcar_units() + self.get_defend_tank_units()
        destination = None
        # 这部分加上 去看一下为啥会停下来
        # tar = self.defend_check_nearby_enemy(obj_id)
        # if len(tar) > 0:
        #     self._move_action(obj_id, tar[0])
        #     return
        if bop["speed"] == 0 or self.ops_destination[obj_id] == []:  #
            # 判断和敌方单位距离
            pts_candidates = self.map.get_grid_distance(\
                bop["cur_hex"], 2, 4
            )
            pts_candidates = self.defend_filter_key_point_by_scope(pts_candidates)
            pts_candidates = self.defend_filter_key_point_by_enemy_pos(
                bop["cur_hex"],
                pts_candidates, filter_mode="enemy"
            )
            if pts_candidates is None:
                pts_candidates = list(self.map.get_grid_distance(\
                bop["cur_hex"], 1, 4
                ))
            target_pos = random.choice(pts_candidates)
            self._move_action(obj_id, target_pos)
            self.ops_destination[obj_id] = [target_pos]
            self.chariot_stage[obj_id] =  "fire"
        
        if len(our_troop_in_neighbor) > 0:       #  附近有步兵班转为机动算子
            # 先找一个没有守点夺控点
            flag_move_to_another_city = False
            for city in self.observation["cities"]:
                if  city["coord"] == closest_city["coord"] or len(self.prepare_to_occupy[city["coord"]])>=1:
                    continue
                defend_force = [ op for op in our_units if self.distance(op["cur_hex"] , city["coord"]) <=2 ]
                if len(defend_force) == 0:
                    # 没有守备力量的夺控点  
                    if len(self.defend_count_current_pos_enemy(city["coord"], 3)) >=2:
                        destination = self.defend_chariot_find_best_cover_points(
                            city["coord"], 4 , 7
                        )
                        if len(destination) == 0:
                            return #  4-7   找不到就别玩
                        flag_move_to_another_city = True 
                    else: # 进点
                        destination = [city["coord"]]
                        flag_move_to_another_city = True 
                    self.ops_destination[obj_id] = [destination[0]]
                    self._move_action(obj_id, destination[0])
                    break
            if flag_move_to_another_city:
                self.chariot_stage[obj_id] = "fire"
            else:
                destination = self.defend_chariot_find_best_cover_points(city["coord"], 3, 5)
                if len(destination) == 0:
                    return 
                self.ops_destination[obj_id] = [destination[0]]
                self._move_action(obj_id, self.ops_destination[obj_id][0])
                return 
        # 暂时还在守这个点
        else:
            # 先占点
            if self.color == closest_city["flag"]:      # 隐蔽

                if len(self.defend_count_current_pos_enemy(closest_city["coord"], 2) )>= 2:   
                    destination = self.defend_chariot_find_best_cover_points(city["coord"], 3, 5)
                else:
                    destination =  self.defend_chariot_find_best_cover_points(city["coord"], 0, 1)
                if len(destination) == 0:
                    return 
            
            else:
                # 判断当前 夺控点周围敌方算子数量
                if len(self.defend_count_current_pos_enemy(closest_city["coord"], 2) ) >=2 :
                    destination = self.defend_chariot_find_best_cover_points(city["coord"], 3, 5)
                else:
                    destination =  [ closest_city["coord"] ]
                    self.prepare_to_occupy[ closest_city["coord"]  ].append(obj_id)
            self.ops_destination[obj_id] = [destination[0]]
            self._move_action(obj_id, self.ops_destination[obj_id][0])
            self.chariot_stage[obj_id] = "fire"
            return 
        return 

    
    #@szh0405 重写一个tank 优先占点的
    @time_decorator
    def defend_tank_start_stage_zhandian(self, obj_id):
        destination = None
        # tank 初始时刻判断敌方算子到我方距离 距离太近能打到就先别解聚  尤其算子在我方工事”前面“的时候
        bop = self.get_bop(obj_id)
        self.__tank_handle_open_fire(obj_id)
        #if len(bop["move_path"]) > 0:  # 有未完成的机动
        if bop["speed"] != 0:
            return 
        closest_city = min(
            self.observation["cities"],
            key=lambda city: self.distance(bop["cur_hex"], city["coord"]),
        )
        self.__tank_handle_open_fire(obj_id)
        flag_close_city_nobody_oc = False
        if bop["cur_hex"] == closest_city["coord"] and self.color != closest_city["flag"]:
            self.gen_occupy(obj_id)
        if self.ops_destination[obj_id] != [] and bop["cur_hex"] ==  self.ops_destination[obj_id][0]:
            self.tank_stage[obj_id] = "fire"
            return 
    
        # tar = self.defend_check_nearby_enemy(obj_id)
        # if len(tar) > 0:
        #     self._move_action(obj_id, tar[0])
        #     return
        # if self.ops_destination[obj_id] == bop["cur_hex"]:
        #     self.ops_destination[obj_id] = ""
            
        ourunits = self.get_defend_infantry_units() + self.get_defend_armorcar_units()
        neighbors = self.map.get_grid_distance(closest_city["coord"], 0, 1)
        
        for ou in ourunits:
            if ou["cur_hex"] in neighbors and ou["obj_id"] != obj_id:
                flag_close_city_nobody_oc = True
        if flag_close_city_nobody_oc == False:     #没有其他我方算子
            # 准备夺控
            city_hex = closest_city["coord"]
            self.prepare_to_occupy[city_hex].append(obj_id)
            self.ops_destination[obj_id] = [ closest_city["coord"] ]
            self._move_action(obj_id, closest_city["coord"])
            if bop["cur_hex"] == closest_city["coord"]:
                self.gen_occupy(obj_id)            #执行夺控
            #这里是 如果占完点后 敌人过来了  要往后撤一撤  而且状态转换要写清楚
            if self.color ==  closest_city["flag"]:  # 占完夺控点了
                # 转为进攻 
                self.tank_stage[obj_id] = "fire"
                return 
                   
        else:
            # 如果有空的夺控点且没人参与夺控 去夺控 
            for city in self.observation["cities"]:
                flag_move_to_another_city = False
                if city["coord"] == closest_city["coord"] or len(self.prepare_to_occupy[city["coord"]])>=1:
                    continue
                flag_another_city_has_defend_force = False
                another_city_neighbors = self.map.get_grid_distance(city["coord"], 0, 2)
                for u in ourunits:
                    if  u["cur_hex"] in another_city_neighbors:
                        flag_another_city_has_defend_force = True
                if flag_another_city_has_defend_force == False:  # 可去该夺控点去占点
                    if len(self.defend_count_current_pos_enemy(city["coord"], 3)) >= 2:
                        destination = self.defend_chariot_find_best_cover_points(city["coord"], 3, 5)
                        if len(destination) == 0:
                            return 
                        flag_move_to_another_city = True
                        break
                    else: 
                        destination = [city["coord"]]
                        flag_move_to_another_city = True 
                    self.ops_destination[obj_id] = [destination[0]]
                    self._move_action(obj_id, destination[0])
                    break
            if flag_move_to_another_city:
                self.tank_stage[obj_id] = "fire"
                return
            else:
                destination = self.defend_chariot_find_best_cover_points(city["coord"], 3, 5)
                if len(destination) == 0:
                    return 
                self.ops_destination[obj_id] = [destination[0]]
                self._move_action(obj_id, self.ops_destination[obj_id][0])
                self.tank_stage[obj_id] = "fire"
                return   

    #@szh0404 步兵班开始交火   直接站桩
    def defend_troop_fire_stage_zhangzhuang(self, obj_id):
        #步兵班在开火阶段就站着a
        destination = None
        bop = self.get_bop(obj_id)
        closest_city = min(
            self.observation["cities"],
            key=lambda city: self.distance(bop["cur_hex"], city["coord"]),
        )
        if len(bop["move_path"]) > 0 :
            destination = bop["move_path"][-1]
            return 
        self.__handle_open_fire(obj_id)
        # 检查 是否有步兵

    #@szh0417 检查自己是不是离敌方算子最近的
    def defend_check_nearest_to_enemy(self,obj_id):
        bop = self.get_bop(obj_id)
        enemy_hex = self.defend_enemy_hex()
        if len(enemy_hex) == 0:
            return False
        current_total_ = sum([self.distance(bop["cur_hex"], en_hex) for en_hex in enemy_hex])
        ourunits = self.get_defend_armorcar_units() + self.get_defend_tank_units() + self.get_defend_infantry_units()
        flag_nearest_to_enemy = True
        for ou in ourunits:
            total_dis = sum([self.distance(ou["cur_hex"],en_hex) for en_hex in enemy_hex])
            if total_dis < current_total_:
                flag_nearest_to_enemy = False
        return flag_nearest_to_enemy
                
            
    #@szh0417 检查到附近有敌人立刻发指令后撤 利用 对方不占点问题  返回
    def defend_check_nearby_enemy(self, obj_id)->List[int]:
        bop = self.get_bop(obj_id)
        enemy_around = self.defend_count_current_pos_enemy(
            bop["cur_hex"], 3
        )
        # 同时检查自己是不是离敌方算子最近的
        if len(enemy_around) >= 2 or self.defend_check_nearest_to_enemy(obj_id):
            closest_enemy , min_dis = self.get_bop_closest(bop, self.defend_enemy_info())
            target_candidate = self.map.get_grid_distance(bop["cur_hex"],2,4)
            target_candidate = self.defend_filter_key_point_by_scope(target_candidate)
            target_candidate = [p for p in target_candidate if self.distance(p, closest_enemy["cur_hex"]) > min_dis + 1]
            target_candidate.sort(key = lambda p : self.map.basic[p // 100][p % 100]["elev"] , reverse= True)
            destination = target_candidate
            if destination is None:
                destination = [p for p in self.map.get_neighbors(bop["cur_hex"]) if self.distance(p, closest_enemy["cur_hex"]) > min_dis + 1]
            self._move_action(obj_id, destination[0])
            self.ops_destination[obj_id] = [destination[0]]
            return destination
        else:
            return [] 
    


    #@szh0404 步兵班解聚后占点  
    @time_decorator
    def defend_troop_start_stage_zhandian(self, obj_id):
        # 找最近的夺控点
        destination = None
        bop = self.get_bop(obj_id)  
        # if len(bop["move_path"]) > 0:  
        if bop["speed"] != 0:
            return 
        if bop["weapon_cool_time"] == 0:
            self.__handle_open_fire(obj_id)           # 先开火打一发
        if bop["forking"]:
            return 
        if len(self.ops_destination[obj_id]) != 0 and self.ops_destination[obj_id][0] == bop["cur_hex"]:
            self.ops_destination[obj_id].pop(0)
        if bop["cur_hex"] in [c["coord"] for c in self.observation["cities"]]:
            self.gen_occupy(obj_id) 
        # 放在fire stage
        if self.ops_destination[obj_id] is not None and self.ops_destination[obj_id] != []:
            if  bop["cur_hex"] != self.ops_destination[obj_id][0]:
                self.gen_change_state(obj_id, 2)
                self._move_action(obj_id, self.ops_destination[obj_id][0])
                return 

        # 这个条件判断需要再考虑考虑
        if self.ops_destination[obj_id] != [] and bop["cur_hex"] == self.ops_destination[obj_id][0]:
            if bop["cur_hex"] in [ nearby_hidding_fort["cur_hex"] ] and self.ops_destination[obj_id][0] ==  nearby_hidding_fort["cur_hex"]: 
                hforts = [op for op in self.observation["operators"] if op["sub_type"]== 20]
                destination = [o["obj_id"] for o in hforts if o["cur_hex"] == bop["cur_hex"] ]
                self.gen_enter_fort(obj_id, destination[0])
                self.troop_stage[obj_id] = "fire"
                return
        
        # 原则上来说一定有closest
        closest_city = min(
            self.observation["cities"],
            key=lambda city: self.distance(bop["cur_hex"], city["coord"]),
        )
        self.ops_destination["obj_id"] =  [closest_city["coord"]]
        if self.color != closest_city["flag"]:
            self.ops_destination[obj_id] = [closest_city["coord"]]
            self.gen_change_state(obj_id, 2)
            self._move_action(obj_id,  self.ops_destination[obj_id][0] )
            return
        # 找最近的隐蔽工事 
        #our_hidding_fort_list = self.get_hiddingbase_units()
        # 工事里能不能解聚
        # 依据隐蔽工事和夺控距离 判断该隐蔽工事和夺控点是不是在一起的 且可进入
        nearby_hidding_fort = None
        hidding_forts = [op for op in self.observation["operators"] if op["sub_type"]==  20 and not self.fort_assignments[op["obj_id"]]]
        
        if hidding_forts is not None and len(hidding_forts) > 0:
            nearby_hidding_fort = min(
                hidding_forts,
                key=lambda fort: self.distance(bop["cur_hex"], fort["cur_hex"]),
            )
            if self.distance(bop["cur_hex"], nearby_hidding_fort["cur_hex"]) >= 2:
                nearby_hidding_fort = None
        flag_troop_prepare_enter_fort = False
        if nearby_hidding_fort is not None:
            if self.troop_stage[obj_id] == "prepare_to_enter_fort"  and len(self.fort_assignments[nearby_hidding_fort["obj_id"]]) == 0:      
                self.gen_enter_fort(obj_id, nearby_hidding_fort["obj_id"]) 
                self.ops_destination[obj_id] = [ nearby_hidding_fort["cur_hex"] ]
                if bop["cur_hex"] == nearby_hidding_fort["cur_hex"]: 
                    self.troop_stage[obj_id] = "fire"
                flag_troop_prepare_enter_fort = True

            elif nearby_hidding_fort and not bop["entering_fort"] and not bop["in_fort"] \
                and len(self.fort_assignments[nearby_hidding_fort["obj_id"]]) < 1 and flag_troop_prepare_enter_fort == False:
                destination = nearby_hidding_fort["cur_hex"]
                
                if self.distance(bop["cur_hex"], destination) < 2:
                    self.gen_change_state(obj_id, 2)
                elif self.distance(bop["cur_hex"], destination) <= 2:
                    self.gen_change_state(obj_id, 1)
                self._move_action(obj_id,destination)  # 反正先每帧发个冲锋指令 
                self.troop_stage[obj_id] = "prepare_to_enter_fort"  
                self.ops_destination[obj_id] = [destination]
                return
        # 这里增加支援其他夺控点
        if bop["in_fort"]:
            return 
        flag_city_in_control = False 
        if closest_city["flag"] == self.color:
            flag_city_in_control = True
            
        if flag_city_in_control ==  False:
            self.ops_destination[obj_id] = [closest_city["coord"]]
            self._move_action(obj_id, closest_city["coord"])
            if closest_city["coord"] == bop["cur_hex"]:
                self.gen_occupy(obj_id)

        flag_has_another_defend_unit = 0
        flag_can_support_another_city = False
        ourtroop = self.get_defend_infantry_units()
        for t in ourtroop:
            nearby_hidding_fort_hex = nearby_hidding_fort["cur_hex"] if nearby_hidding_fort is not None else closest_city["coord"]
            if len(self.ops_destination[t["obj_id"]]) == 0:
                continue
            if self.ops_destination[t["obj_id"]][0] in [closest_city["coord"], nearby_hidding_fort_hex]:
                flag_has_another_defend_unit += 1
        if flag_has_another_defend_unit > 1 :
            flag_can_support_another_city = True
            for c in self.observation["cities"]:
                flag_c_need_support =  True
                if c["name"] == closest_city["name"]:
                    continue
                cn = list(self.map.get_grid_distance(c["coord"], 0, 2)) 
                for t in ourtroop:
                    if len(self.ops_destination[ t["obj_id"] ]) == 0:
                        continue
                    if self.ops_destination[ t["obj_id"] ][0] in cn:
                         flag_c_need_support = False
                if flag_c_need_support and self.distance(bop["cur_hex"], c["coord"]) <=4 :   #可去支援
                    if len(self.defend_count_current_pos_enemy(c["coord"], 2)) <= 2:    # 敌方太多就别去了
                        self.ops_destination[obj_id] = [c["coord"]]
                        return 
                    
       
        if destination is not None and len(destination) > 0:
            self._move_action(obj_id, destination[0])
        # 直接进点
        self._move_action(obj_id, closest_city["coord"])
        self.ops_destination[obj_id] = [ closest_city["coord"] ]
        self.troop_stage[obj_id] = "fire"
        return 
    

    @time_decorator
    def defend_troop_fire_stage_zhandian(self, obj_id):
        destination = None
        bop = self.get_bop(obj_id)
        
        closest_city = min(
            self.observation["cities"],
            key=lambda city: self.distance(bop["cur_hex"], city["coord"]),
        )
        if len(self.ops_destination[obj_id]) != 0 and  self.ops_destination[obj_id][0] == bop["cur_hex"]:
            self.ops_destination[obj_id].pop(0)
        if self.ops_destination[obj_id] != [] and bop["cur_hex"] != self.ops_destination[obj_id][0]:
            self.gen_change_state(obj_id, 2)
            self._move_action(obj_id, self.ops_destination[obj_id][0])
            return
        
        if bop["weapon_cool_time"] == 0:
            self.__handle_open_fire(obj_id)
        if bop["cur_hex"] == closest_city["coord"]:
            self.gen_occupy(obj_id)
        # 0423 步兵班火力支援方案 在周围没有对方算子的条件下步兵班 找夺控点周围七个格子中通视效果最好的 
        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 1)) == 0:  
            if len(self.defend_count_current_pos_enemy(bop["cur_hex"], 1)) == 0:
                # 可以选择支援
                has_ene_cities = [c for c in self.observation["cities"] if len(self.defend_count_current_pos_enemy(
                    c["coord"], 1  #  考虑调整为1  或  2 
                )) > 0]
                if has_ene_cities is not None and len(has_ene_cities) > 0:
                    closest_has_ene_city = min(
                        has_ene_cities,
                        key = lambda enec: self.distance(closest_city["coord"], enec["coord"])
                    )
                    if closest_has_ene_city is not None:
                        #准备支援
                        closest_ene_city_nbs  = self.map.get_neighbors( closest_has_ene_city["coord"]) + [ closest_has_ene_city["coord"] ]
                        closest_city_nbs = self.map.get_neighbors(closest_city["coord"]) + [ closest_city["coord"] ]
                        max_can_see = 0
                        best_point_can_see = None  #  int 
                        for nb in closest_city_nbs:
                            total_can_see = 0
                            for tp in closest_ene_city_nbs:
                                if self.map.can_see(nb, tp, 0):
                                    total_can_see += 1
                            if total_can_see > max_can_see:
                                max_can_see = total_can_see
                                best_point_can_see = nb
                        if best_point_can_see is not None:
                            self.ops_destination[obj_id].insert(0, best_point_can_see)
                            
                
             

        hforts = [op for op in self.observation["operators"] if op["sub_type"]== 20]
        hforts_hex = [op["cur_hex"] for op in self.observation["operators"] if op["sub_type"]== 20]
        if bop["cur_hex"] in hforts_hex and len( self.ops_destination[obj_id] ) and self.ops_destination[obj_id][0] in hforts_hex:
            destination = [o["obj_id"] for o in hforts if o["cur_hex"] == bop["cur_hex"] ]
            self.gen_enter_fort(obj_id, destination[0])
    #@szh 0404  更新prepare _to  _occupy 的内容
    def update_prepare_to_occupy(self):
        # ops = self.get_defend_tank_units() + self.get_defend_armorcar_units()
        # cities_map = {c["name"]  : c["coord"]  for c in self.observation["cities"]  }
        # 检查 每个算子des 是不是当前的这个  步兵班除外
        for k, v in self.prepare_to_occupy.items():   # K:coord  v list of obj_id
            v = list(set(v))
            for i in range(len(v)):  # v_i obj_id
                if self.ops_destination[ v[i] ][0] != k:
                    v.pop(i)          

    def get_bop_closest(self, bop, refer_bops: list):
        '''
        返回用作参照的算子中最近的一个及距离
        '''
        min_dis = 100000
        bop_closest = None
        for refer_bop in refer_bops:
            bop_dis = self.distance(bop["cur_hex"], refer_bop["cur_hex"])
            # 如果refer_bop是自身则跳过
            if bop['obj_id'] != refer_bop['obj_id'] and bop_dis < min_dis:
                min_dis = bop_dis
                bop_closest = refer_bop
        return bop_closest, min_dis

    #@szh 0404 检查当前格有没有敌方算子  s
    def defend_count_current_pos_enemy(self, cur_pos:int, scope:int)->List[int] :
        """
            scope 范围为 scope 的格子有多少个敌方算子
        """
        enemys = [op for op in self.observation["operators"] if op["color"]!=self.color ]
        if scope == 0:
            return [ op for op in enemys if op["cur_hex"] == cur_pos]
        return  [op for op in enemys if self.distance(op["cur_hex"], cur_pos) <= scope]
            
      #@szh 0404 战车开火策略
    @time_decorator
    def defend_chariot_fire_stage_zhandian(self, obj_id):
        destination = None
        bop = self.get_bop(obj_id)
        if bop["speed"] != 0:         # 如果当前还在行进中
            #destination = bop["move_path"][-1]
            return 
        closest_city = min(
            self.observation["cities"],
            key=lambda city: self.distance(bop["cur_hex"], city["coord"]),
        )
        # if bop["weapon_cool_time"] == 0:        # 如果到达冷却时间
        #     self.__handle_open_fire(obj_id)
        if len(self.ops_destination[obj_id]) != 0 and  self.ops_destination[obj_id][0] == bop["cur_hex"]:
            self.ops_destination[obj_id].pop(0)

        if self.ops_destination[obj_id] != [] and  bop["cur_hex"] != self.ops_destination[obj_id][0]:
            self._move_action(obj_id, self.ops_destination[obj_id][0])
            return

        if bop["cur_hex"] in [c["coord"] for c in self.observation["cities"]]:
            self.gen_occupy(obj_id)
            if len(self.defend_count_current_pos_enemy(closest_city["coord"], 2)) >= 3: #附近有敌方算子 而且很多  溜了溜了
                destination = self.defend_chariot_find_best_cover_points(bop["cur_hex"], 4, 6)
                self.ops_destination[obj_id].insert(0,destination[0])
                self._move_action( obj_id, self.ops_destination[obj_id][0] )
                return 
        tar = self.defend_check_nearby_enemy(obj_id)
        if len(tar) > 0:
            self._move_action(obj_id, tar[0])
            return
        if bop["speed"] == 0  or self.ops_destination[obj_id] == []:  #
            # 判断和敌方单位距离
            pts_candidates = self.map.get_grid_distance(\
                bop["cur_hex"], 2, 4
            )
            pts_candidates = self.defend_filter_key_point_by_scope(pts_candidates)
            pts_candidates = self.defend_filter_key_point_by_enemy_pos(
                bop["cur_hex"],
                pts_candidates, filter_mode="enemy"
            )
            if pts_candidates is None:
                pts_candidates = list(self.map.get_grid_distance(\
                bop["cur_hex"], 1, 3
                ))
            target_pos = random.choice(pts_candidates)
            self._move_action(obj_id, target_pos)
            self.ops_destination[obj_id] = [target_pos]

        # 可以覆盖上边的
        city_empty = self.defend_check_city_no_hex()
        if self.distance(bop["cur_hex"], closest_city["coord"]) >= 5 and len(city_empty) > 0:
            city_empty.sort(key = lambda c: self.distance(c["coord"], bop["cur_hex"]))
            self.ops_destination[obj_id] = [ city_empty[0]["coord"] ]
            self._move_action(obj_id, self.ops_destination[obj_id][0])   
            return  
        else:
            scities = [c for c in self.observation["cities"]]
            scities.sort(key = lambda c: self.distance(c["coord"], bop["cur_hex"]))
            second_near_city, third_near_city = scities[1], scities[2]
            second_near_ene =  [op for op in self.observation["operators"] if op["color"]!=self.color \
                        and self.map.get_distance(op["cur_hex"], second_near_city["coord"]) <= 1]
            third_near_ene = [op for op in self.observation["operators"] if op["color"]!=self.color \
                        and self.map.get_distance(op["cur_hex"], third_near_city["coord"]) <= 2]
            if len(second_near_ene) == 0 and len(third_near_ene) != 0:
                self.ops_destination[obj_id] = [ second_near_city["coord"] ]
                self._move_action(obj_id, self.ops_destination[obj_id][0])
                return  

        closest_enemy , min_dis = self.get_bop_closest(bop, self.defend_enemy_info())
        if min_dis <= 2: # 优先避免同格交战
            #找自身围为2的游击点  但得和对方算子拉开距离
                youji_point_candidates = self.defend_get_key_point_around_fort(\
                    bop["cur_hex"],
                    mode = "youji"
               )
                youji_point_candidates = self.defend_filter_key_point_by_scope(youji_point_candidates)
                youji_point_candidates = [p for p in youji_point_candidates if self.distance(p, closest_enemy["cur_hex"]) >= min_dis]
                if len(youji_point_candidates)  > 0 :
                    destination = [random.choice(youji_point_candidates)]
                if type(destination) == int:
                    destination = [destination]
                self.ops_destination[obj_id]  = [destination[0]]
                self._move_action(obj_id, self.ops_destination[obj_id][0])  
                return  
           
        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 1)) == 0 and self.color != closest_city["flag"]:
            destination = [closest_city["coord"]]

        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 1)) == 0 and self.color == closest_city["flag"]:
           # 周边6个格随机走
            if len(self.defend_count_current_pos_enemy(closest_city["coord"], 2)) >2 :
                destination = self.defend_chariot_find_best_cover_points(closest_city["coord"], 2, 4)       
            else:
                destination = self.defend_chariot_find_best_cover_points(closest_city["coord"], 1, 3)  
                
        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 1)) != 0 and self.color != closest_city["flag"]:
            ourunits = self.get_defend_infantry_units() + self.get_defend_armorcar_units() + self.get_defend_tank_units()
            for c in self.observation["cities"]:
                flag_c_need_support =  True
                if c["name"] == closest_city["name"]:
                    continue
                cn = list(self.map.get_grid_distance(c["coord"], 0, 2))  # 0-2 范围没有我方
                for t in ourunits:
                    if len(self.ops_destination[ t["obj_id"] ]) == 0:
                        continue
                    if self.ops_destination[ t["obj_id"] ][0] in cn:
                         flag_c_need_support = False
                if flag_c_need_support and self.color != c["flag"] :   #可去支援
                    if len(self.defend_count_current_pos_enemy(c["coord"], 1)) == 0 and \
                        len(self.defend_count_current_pos_enemy(c["coord"], 3)) <= 2:
                        destination = [c["coord"]]
                        self.prepare_to_occupy[c["name"]].append(obj_id) 
                        break
        if destination is None:
            destination = [closest_city["coord"]]            
        self.ops_destination[obj_id] = [destination[0]]
        self._move_action(obj_id, self.ops_destination[obj_id][0]) 

        # if bop["weapon_cool_time"] == 0 and closest_enemy is not None and self.distance(closest_enemy["cur_hex"], bop["cur_hex"]) >=2: 
        #     self.__handle_open_fire(obj_id)   
    
    #@szh0404  reset 占领点状态
    def reset_occupy_state(self):
        cities = [ci for ci in self.observation["cities"] ]
        ourunits = self.get_defend_armorcar_units() + self.get_defend_infantry_units() + self.get_defend_tank_units()
        for c in cities:
            flag_city_has_our_units_nearby = False
            neighbors_hex = list(self.map.get_grid_distance(c["coord"], 0, 1)) +  [c["coord"]]  
            for ou in ourunits:
                if ou["cur_hex"] in neighbors_hex:
                    flag_city_has_our_units_nearby = True
                    break
             
            if flag_city_has_our_units_nearby == False:
                self.prepare_to_occupy[c["name"]] = [] 

    #@szh0417 重新写一个tank openfire
    def __tank_handle_open_fire(self, attacker_ID):
        self._fire_action(attacker_ID)

    #@szh0417 写一个检查夺控点
    def defend_check_city_no_hex(self)->List[Dict]:
        cities = [c for c in self.observation["cities"]]
        city_ene = []
        city_has_no_ene = []
        for c in cities:
            if c["flag"] == self.color:
                continue
            city_ene = [op for op in self.observation["operators"] if op["color"]!=self.color \
                        and self.map.get_distance(op["cur_hex"], c["coord"]) <= 1]
            if city_ene is None or len(city_ene) == 0:
                city_has_no_ene.append(c)
        return city_has_no_ene
    
    # #@szh0417 写一个如果距离太远 向夺控点靠拢
    # def defend_check_distance_too_far(self, obj_id)->bool:
        
        
    
    
            
        
        
    #@szh0404  tank开火
    @time_decorator
    def defend_tank_fire_stage_zhandian(self, obj_id):
        destination = None
        bop = self.get_bop(obj_id)
        closest_city = min(
            self.observation["cities"],
            key=lambda city: self.distance(bop["cur_hex"], city["coord"]),
        )
        # 判断是否去其他点支援
        # if bop["weapon_cool_time"] == 0:        # 如果到达冷却时间
        self.__tank_handle_open_fire(obj_id)
        if len(self.ops_destination[obj_id]) and self.ops_destination[obj_id][0] == bop["cur_hex"]:
            self.ops_destination[obj_id].pop(0)
        """
            # 策略  找没有啥敌人的夺控点一占
            # 1. 先判断当前是否在夺控点上  
            #    在的话    判断周围是否敌方算子   有的话    撤离  
                                               没有的话   接着不动
              2. 判断当前有无空闲夺控点
                                    有的话     有敌方算子  不去了  当前
                                            没有敌方算子   去夺控
            # 避免同格交战
            
        """
        # 找最近的且未夺控的点 一种是没人夺控  就是中立颜色              
        if self.ops_destination[obj_id] != [] and bop["cur_hex"] != self.ops_destination[obj_id][0]:
            self._move_action(obj_id, self.ops_destination[obj_id][0])
            return
        tar = self.defend_check_nearby_enemy(obj_id)
        if len(tar) > 0:
            self._move_action(obj_id, tar[0])
            return
        if bop["cur_hex"] in [c["coord"] for c in self.observation["cities"]]:
            self.gen_occupy(obj_id)
        
        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 2)) >= 3 or self.defend_check_nearest_to_enemy(obj_id): 
            #附近有敌方算子 而且很多  溜了溜了
            destination = self.defend_chariot_find_best_cover_points(bop["cur_hex"], 3, 5)
            self.ops_destination[obj_id]  = [ destination[0] ]
            self._move_action(obj_id, self.ops_destination[obj_id][0])
            return   
        if bop["speed"] == 0  or self.ops_destination[obj_id] == []:  
            # 判断和敌方单位距离
            pts_candidates = self.map.get_grid_distance(\
                bop["cur_hex"], 2, 4
            )
            pts_candidates = self.defend_filter_key_point_by_scope(pts_candidates)
            pts_candidates = self.defend_filter_key_point_by_enemy_pos(
                bop["cur_hex"],
                pts_candidates, filter_mode="enemy"
            )
            if pts_candidates is None:
                pts_candidates = list(self.map.get_grid_distance(\
                bop["cur_hex"], 1, 4
                ))
            target_pos = random.choice(pts_candidates)
            self._move_action(obj_id, target_pos)
            self.ops_destination[obj_id] = [ target_pos ]

        
        city_empty = self.defend_check_city_no_hex()
        if self.distance(bop["cur_hex"], closest_city["coord"]) >= 4 and len(city_empty) > 0:
            city_empty.sort(key = lambda c: self.distance(c["coord"], bop["cur_hex"]))
            self.ops_destination[obj_id] = [ city_empty[0]["coord"] ]
            self._move_action(obj_id, self.ops_destination[obj_id][0])   
            return         
                
        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 1)) == 0 and self.color != closest_city["flag"]:
            destination = [ closest_city["coord"] ]
            # self.ops_destination[obj_id] = 
            # self._move_action(obj_id, self.ops_destination[obj_id])
            # return 
        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 1)) == 0 and self.color == closest_city["flag"]:
           # 周边6个格随机走
            if len(self.defend_count_current_pos_enemy(closest_city["coord"], 3)) >2 :
                destination = self.defend_chariot_find_best_cover_points(closest_city["coord"], 2, 4)    
                # self.ops_destination[obj_id] = destination[0]
                # self._move_action(obj_id, self.ops_destination[obj_id])    
            else:
                neighbors1 = self.map.get_grid_distance(closest_city["coord"], 0, 1)
                destination =  [random.choice(list(neighbors1))]
                # self.ops_destination[obj_id] = destination
                # self._move_action(obj_id, self.ops_destination[obj_id]) 
            #  得考虑一下 self .color  ==  怎么办
        if len(self.defend_count_current_pos_enemy(closest_city["coord"], 1)) != 0 and self.color != closest_city["flag"]:
            # 找附近没有敌人的我方夺控点 
            flag_can_support_another_city = False
            ourunits = self.get_defend_infantry_units() + self.get_defend_armorcar_units() + self.get_defend_tank_units()
            for c in self.observation["cities"]:
                flag_c_need_support =  True
                if c["name"] == closest_city["name"]:
                    continue
                cn = list(self.map.get_grid_distance(c["coord"], 0, 1))  # 0-2 范围没有我方
                for t in ourunits:
                    if len(self.ops_destination[ t["obj_id"] ]) == 0:
                        continue
                    if self.ops_destination[ t["obj_id"] ][0] in cn:
                         flag_c_need_support = False
                if flag_c_need_support and self.color != c["flag"] :   #可去支援
                    if len(self.defend_count_current_pos_enemy(c["coord"], 1)) == 0 and \
                        len(self.defend_count_current_pos_enemy(c["coord"], 3)) <= 2:
                        destination = [ c["coord"] ]
                        self.prepare_to_occupy[c["name"]].append(obj_id) 
                        break
            # 这里看一下  得加上如果不能支援应该怎么
        if destination is None:
            destination = list(self.defend_chariot_find_best_cover_points(bop["cur_hex"], 5, 7)) + [closest_city["coord"]]
        
        # 避免同格交战
        closest_enemy, min_dis = self.get_bop_closest(
            bop,  
            [op for op in self.observation["operators"] if op["color"]!=self.color]
        )
        if min_dis <= 2:
            youji_point_candidates = self.defend_get_key_point_around_fort(\
                bop["cur_hex"],
                mode = "youji"
            )
            youji_point_candidates = self.defend_filter_key_point_by_scope(youji_point_candidates)
            youji_point_candidates = [p for p in youji_point_candidates if self.distance(p, closest_enemy["cur_hex"]) > min_dis]
            if len(youji_point_candidates)  > 0 :
                destination = [random.choice(youji_point_candidates)]
        if type(destination) == int:
            destination = [destination]
        self.ops_destination[obj_id]  = [ destination[0] ]
        self._move_action(obj_id, self.ops_destination[obj_id][0])  
        return      
           
    #@szh0404  打个补丁 解聚之后周围随机数移动  
    def _defend_jieju_and_move(self, obj_id):
        bop = self.get_bop(obj_id)
        neighbors = list( self.map.get_grid_distance(bop["cur_hex"], 0, 1) )
        destination = random.choice(neighbors)
        self._move_action(obj_id, destination)
        if bop["sub_type"] == BopSubType.Infantry:
            self.gen_change_state(obj_id, 2)

class EnemyValue:
    Infantry, Tank, Chariot =  10, 25, 10
class FriendValue:
    Infantry, Tank, Chariot =  30, 15, 5
class RecurrentQueue(object):
    def __init__(self, maxsize):
        self.queue = [None] * maxsize
        self.maxsize = maxsize
        self.front = 0
        self.rear = 0
    def _length(self):
        return (self.rear - self.front + self.maxsize) % self.maxsize

    def _push_back(self, data):
        if (self.rear + 1)%self.maxsize == self.front:
            self._pop_front()
        self.queue[self.rear] = data
        self.rear = (self.rear + 1)%self.maxsize

    def _pop_front(self):
        if self.rear == self.front:  # 一般来说不会出现这种情况
            return  None
        else:
            data = self.queue[self.front]
            self.queue[self.front] = None
            self.front = (self.front + 1)%self.maxsize
            return data
    def _get_item_by_index(self, index):
        # if index >= self._length() or index < 0:
        #     return
        # print(index)
        return self.queue[ (index) % self.maxsize ]

#####################################################