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


class Agent(BaseAgent):  # TODO: 换成直接继承BaseAgent，解耦然后改名字。
    def __init__(self):
        super(Agent,self).__init__()
        # abstract_state is useful

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

    def find_pos_vector(self,pos_here, pos_list,vector_xy):
        # 从pos_list中找到“从pos_here到其中的点”的向量方向最符合vector_xy的，然后返回相应的pos四位数，用于后续的move。
        # 这个函数得好好写写，因为会高频调用。
        # 最符合方向，那就是归一化之后的内积最大嘛，先这么搞
        xy_here = self._hex_to_xy(pos_here)
        index = 0
        index_selected = 0 
        dot_max = -1.1
        for pos_single in pos_list:
            xy_single = self._hex_to_xy(pos_single)
            vector_single = xy_single-xy_here
            dot_single = np.dot(vector_single,vector_xy) / np.linalg.norm(vector_xy) / np.linalg.norm(vector_single)
            # dot_single in [-1, 1], 所以肯定会有某个点被选出来的
            if dot_single>dot_max:
                # 那就说明这个符合的更好
                dot_max = dot_single
                index_selected = index
            index = index + 1 
        return  pos_list[index_selected] # 返回值应该是一个四位的int，能拿去用的那种。
    
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
                self.set_move_and_attack(unit,self.target_pos,model="force")
            else:
                # 找那一堆里面距离最近的来跟随。
                jvli_list = [] 
                for i in range(len(units_VIP)):
                    jvli_single = self.distance(unit,units_VIP[i])  
                    jvli_list.append(jvli_single)
                jvli_min = min(jvli_list)
                index_min = jvli_list.index(jvli_min)
                VIP_pos_single = units_VIP[index_min]["cur_hex"]
                self.set_move_and_attack(unit,VIP_pos_single,model="force")

    def get_pos_list_A(self, units, target_pos):
        # 上来先维护target_pos_list,包括判断威胁等级看是不是有必要绕路。
        pos_ave = self.get_pos_average(units)
        xy_ave = self._hex_to_xy(pos_ave)
        
        target_xy= self._hex_to_xy(target_pos)
        vector_xy = target_xy - xy_ave

        # enemy_infantry_units = self.select_by_type(self.detected_state,key="sub_type",value=2)
        # tanks and other things included
        enemy_infantry_units = self.detected_state
        enemy_infantry_units_danger = [] 
        enemy_infantry_dot_danger = [] 
        # 如果敌方步兵在正前方了，那就别去了。同时满足距离和方向的才算。
        for enemy_infantry_unit in enemy_infantry_units:
            # 遍历看一下是不是需要跑。
            enemy_pos = enemy_infantry_unit["cur_hex"]
            enemy_xy = self._hex_to_xy(enemy_pos)
            vector_single = enemy_xy - xy_ave
            enemy_distance = self.map.get_distance(enemy_pos, pos_ave)

            dot_single = np.dot(vector_single, vector_xy) / np.linalg.norm(vector_xy+0.001) / np.linalg.norm(vector_single+0.001)
            
            
            if enemy_distance<17 and dot_single>0.80:
                # 这两个阈值都是从案例里抠出来的。
                enemy_infantry_units_danger.append(enemy_infantry_unit)
                enemy_infantry_dot_danger.append(dot_single)
        
        # 至此，就筛出了究极高威胁的敌方步兵的位置。然后是根据这些位置确定绕路的方向，以target_pos_list的形式放在list中。
        if len(enemy_infantry_units_danger)>0:
            # 然后向量计算求一下那个点应该往哪边找，点乘判断正负可也，
            
            # 先求两个垂直于路径的法向量。
            n1_xy = np.array([vector_xy[1], -1*vector_xy[0]]) / np.linalg.norm(vector_xy)
            n2_xy = -1*n1_xy

            # # method1: 先取个中间点出来
            def method1(enemy_infantry_units_danger):
                pos_ave_enemy = self.get_pos_average(enemy_infantry_units_danger)
                xy_ave_enemy = self._hex_to_xy(pos_ave_enemy)
                vector_ave_enemy = xy_ave_enemy - xy_ave            
                # 然后检测哪个比较好。
                if np.dot(n1_xy,vector_xy)>0:
                    # 那说明是偏向这个方向，绕道的路就得往另一个方向去了。
                    n_xy_list = [n2_xy, n1_xy] 
                else:
                    n_xy_list = [n1_xy, n2_xy] 
                # 道理上不可能两个方向都在外面，因为起点终点在垂线的不同侧，且都在范围内。
                # 所以两边必有一边是能够绕路的。

                # 然后然后开始计算距离点了。
                
                pos_center = self.get_pos_average([pos_ave,target_pos], model="input_hexs")
                # 然后算。反正两个方向，总得有一个对的。要是都不对也防一手。
                xy_center = self._hex_to_xy(pos_center)
                try:
                    xy_candidate = xy_center + 15*n_xy_list[0]
                    pos_candidate = self._xy_to_hex(xy_candidate)
                except:
                    xy_candidate = xy_center + 15*n_xy_list[1]
                    pos_candidate = self._xy_to_hex(xy_candidate) 

                return pos_candidate
               
            # method2: find zuiwaimain units and xiuzheng.
            dot_min = min(enemy_infantry_dot_danger)
            index_min = enemy_infantry_dot_danger.index(dot_min)
            enemy_pos= enemy_infantry_units_danger[index_min]["cur_hex"]
            enemy_xy = self._hex_to_xy(enemy_pos)
            vector_xy_enemy = self._hex_to_xy(enemy_pos)
            if np.dot(n1_xy,vector_xy_enemy)>0:
                # which means the direction is right.
                # n_xy_list = [n1_xy, n2_xy] 
                try:
                    xy_candidate = enemy_xy + 15*n1_xy
                    pos_candidate = self._xy_to_hex(xy_candidate)
                except:
                    # if it doesn't work, then use method1
                    pos_candidate =  method1(enemy_infantry_dot_danger) 
            else:
                pos_candidate =  method1(enemy_infantry_units_danger)            
        else:
            pos_candidate = target_pos
        return [pos_candidate, target_pos, target_pos] # 这里后面补一个target_pos是为了写循环的时候好写。
                
    def list_A(self, units, target_pos):
        # “选取部队横越地图”，实现一个宏观层面的绕行机制。
        if len(units) ==0:
            # unit lost, nothing happen.
            return 
        # target_pos_list作为类的一个属性在这里面自己维护了。
        if (self.num<400 and self.num%20==2) or not("self.target_pos_list" in locals()): # 原则上不用每一步都求解这个。只要位置变化了一次能够求一次就行了
            target_pos_list = self.get_pos_list_A(units, target_pos)
            self.target_pos_list = target_pos_list 
        else:
            target_pos_list = self.target_pos_list
        
        for unit in units:
            # 如果到了某一个点，就去下一个点。搞成通用的，以防未来需要很多个路径点的时候不好搞。
            for i in range(len(target_pos_list)-1):
                target_pos_single = target_pos_list[i]
                pos_single = self.get_pos(unit)
                if pos_single==target_pos_list[-1]:
                    # arrived
                    break
                if pos_single==target_pos_single:
                    # 说明到了这个点了，那就去下一个点。
                    target_pos = target_pos_list[i+1]
                    self.set_move_and_attack(unit,target_pos,model="force")
                else:
                    # 没到的话就无事发生。
                    # no, if not arrived, then go there.
                    self.set_move_and_attack(unit,target_pos_single,model="normal")
                    pass
                
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

    def UAV_patrol(self, target_pos):
        # 这个会覆盖给无人机的其他命令，优先执行“飞过去打一炮”，然后再把别的命令弄出来。

        # 不要重复下命令，不然就把时间都刷没了

        # 先把UAV取出来
        UAV_units = self.select_by_type(self.status["operators"],key="sub_type",value=5)
        # 然后把目标取出来
        if len(self.detected_state)>0:
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
                if self.abstract_state[UAV_unit["obj_id"]]["abstract_state"]!="UAV_move_on":
                    self.set_UAV_move_on(UAV_unit["obj_id"],target_pos=target_pos_random)            
            pass

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
                    self.set_off_board(IFV_unit, infantry_ID_list[0])
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
                if abstract_state_single["abstract_state"] == "jieju":
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
            # print("WTF, it should be cross_fire, GAN")
            # self.my_direction = []
            # self.target_pos = self.my_direction["hex"]
            # self.end_time = self.my_direction["end_time"]
        else:
            print("get_target_cross_fire: Done.")
        return  self.target_pos

    def get_target_defend(self):
        observation = self.status
        communications = observation["communication"]
        flag_done = False

        defend_pos = []
        for communication in communications:
            defend_pos_single = communication["hex"]
            defend_pos.append(defend_pos_single)
        
        return defend_pos

    def get_target_scout(self):
        pass

    # then step functions 
    def step(self, observation: dict, model="guize"):
        # if model = guize, then generate self.act in step, else if model = RL, then generate self.act in env rather than here.
        self.num = self.num + 1 
        if self.num == 1:
            print("Debug, moving")
        else:
            if self.num%100==99:
                print("Debug, self.num = "+str(self.num))
        self.observation = observation
        self.status = observation # so laji but fangbian.

        self.team_info = observation["role_and_grouping_info"]
        self.controposble_ops = observation["role_and_grouping_info"][self.seat][
            "operators"
        ]

        # get the target first.
        self.distinguish_saidao()

        # the real tactics in step*() function.
        # self.step0()
        if self.env_name=="cross_fire":
            # update the actions
            if model == "guize":
                self.Gostep_abstract_state()
            elif model =="RL":
                pass
            # self.step_cross_fire()
            self.step_cross_fire_test()
        elif self.env_name=="scout":
            self.act = self.step_scout()
        elif self.env_name=="defend":
            self.Gostep_abstract_state()
            self.step_defend()
        else:
            raise Exception("G!")

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

    def step_cross_fire_test(self):
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


    def step_scout(self):
        # unfinished yet.
        self.ob = self.observation
        self.update_time()
        self.update_tasks()
        if not self.tasks:
            return []  # 如果没有任务则待命
        self.update_all_units()
        self.update_valid_actions()

        self.actions = []  # 将要返回的动作容器
        self.prefer_shoot()  # 优先选择射击动作

        for task in self.tasks:  # 遍历每个分配给本席位任务
            self.task_executors[task["type"]].execute(task, self)
        return self.actions   
    
    def step_defend(self):
        # unfinished yet.
        
        # 先把场景目标点在哪读出来
        defend_pos = [0,0,0] # three in hex form
        # get the target first.
        if self.num <2:
            defend_pos = self.get_target_defend()
            self.defend_pos = defend_pos
        else:
            defend_pos = self.defend_pos    

        # 经典分兵编队
        units=self.status["operators"]           
        IFV_units = self.get_IFV_units()
        infantry_units = self.get_infantry_units()
        UAV_units = self.get_UAV_units()
        others_units = [unit for unit in units if (unit not in IFV_units) and (unit not in infantry_units) and (unit not in UAV_units)]

        # 怎么判断A到了呢？姑且可以是全停下就算是A到了。或者是直接步数
        jieju_flag = self.jieju_check(model="part", units=others_units)
        if self.num<100 and jieju_flag==False:
            # 那就是没解聚完，那就继续解聚。
            for unit in (others_units+infantry_units+IFV_units):
                self.set_jieju(unit)
        else:
            index_chong = round(((self.num+101) % 600) / 200 ) - 1  # 这个就应该是0,1,2
            
            self.group_A((others_units+UAV_units), defend_pos[index_chong])
            for unit in IFV_units+infantry_units:
                self.set_open_fire(unit)

        print("step_defend: unfinished yet.")
