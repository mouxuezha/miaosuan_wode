# this is wode agent for miaosuan, define some layers.
import os 
import sys 
import json
sys.path.append("/home/vboxuser/Desktop/miaosuan/starter-kit")

from ai.agent import Agent
from ai.map import Map
import copy,random
import numpy as np
from tools import *

class BopType:
    Infantry, Vehicle, Aircraft = range(1, 4)


class ActionType:
    (
        Move,
        Shoot,
        GetOn,
        GetOff,
        Occupy,
        ChangeState,
        RemoveKeep,
        JMPlan,
        GuideShoot,
        StopMove,
        WeaponLock,
        WeaponUnFold,
        CancelJMPlan,
        Fork,
        Union,
        ChangeAltitude,
        ActivateRadar,
        EnterFort,
        ExitFort,
        LayMine,
    ) = range(1, 21)


class MoveType:
    Maneuver, March, Walk, Fly = range(4)


class agent_guize(Agent):  # TODO: 换成直接继承BaseAgent，解耦然后改名字。
    def __init__(self):
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map = None
        self.scenario_info = None
        self.map_data = None
        self.seat = None
        self.faction = None
        self.role = None
        self.controposble_ops = None
        self.team_info = None
        self.my_direction = None
        self.my_mission = None
        self.user_name = None
        self.user_id = None
        self.history = None
        # abstract_state is useful
        self.abstract_state = {} 
        # ["move_and_attack", only for tank, 
        # "move_and_attack2", for other units
        # "follow_and_guard",  
        # "hide_and_alert",  for most dongxi
        # "charge_and_xiache", for infantry and che
        # ]

        self.act = [] # list to save all commands generated.

        self.status = {}
        self.status_old = {} 
        self.detected_state = {} 
        self.detected_state2 = {} 

        self.num = 0 

        self.abstract_state = {}  # key 是装备ID，value是抽象状态

        # self.threaten_source_set = set() 
        self.threaten_source_list = [] 
        # dict is unhashable, so use list
        # 这个是用来避障的。元素是dic，包括威胁源的位置、种类和时间延迟，后面不够再来补充可也。{pos: int , type: int, delay: int} # 不需要标记时间炮火持续时间那种，持续完了直接删了就行了。但是飞行中是要标记的。
        # type: 0 for enemy units, 1 for artillery fire, 2 for unit lost . -1 for my unit, negative threaten
        self.threaten_field = {} # 垃圾一点儿了，直接整一个字典类型来存势场了。{pos(int) : field_value(double)}
        self.env_name = "cross_fire" # defualt. 

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

    def reset(self):
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map = None
        self.scenario_info = None
        self.map_data = None

        self.num = 0 

    def get_scenario_info(self, scenario: int):
        SCENARIO_INFO_PATH = os.path.join(
            os.path.dirname(__file__), f"scenario_{scenario}.json"
        )
        with open(SCENARIO_INFO_PATH, encoding="utf8") as f:
            self.scenario_info = json.load(f)

    def get_bop(self, obj_id):
        """Get bop in my observation based on its id."""
        for bop in self.observation["operators"]:
            if obj_id == bop["obj_id"]:
                return bop

    def gen_occupy(self, obj_id, candidate):
        """Generate occupy action."""
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.Occupy,
        }

    def gen_shoot(self, obj_id, candidate):
        """Generate shoot action with the highest attack level."""
        best = max(candidate, key=lambda x: x["attack_level"])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.Shoot,
            "target_obj_id": best["target_obj_id"],
            "weapon_id": best["weapon_id"],
        }

    def gen_guide_shoot(self, obj_id, candidate):
        """Generate guide shoot action with the highest attack level."""
        best = max(candidate, key=lambda x: x["attack_level"])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.GuideShoot,
            "target_obj_id": best["target_obj_id"],
            "weapon_id": best["weapon_id"],
            "guided_obj_id": best["guided_obj_id"],
        }

    def gen_jm_plan(self, obj_id, candidate):
        """Generate jm plan action aimed at a random city."""
        weapon_id = random.choice(candidate)["weapon_id"]
        jm_pos = random.choice([city["coord"] for city in self.observation["cities"]])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.JMPlan,
            "jm_pos": jm_pos,
            "weapon_id": weapon_id,
        }

    def gen_get_on(self, obj_id, candidate):
        """Generate get on action with some probability."""
        get_on_prob = 0.5
        if random.random() < get_on_prob:
            target_obj_id = random.choice(candidate)["target_obj_id"]
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.GetOn,
                "target_obj_id": target_obj_id,
            }

    def gen_get_off(self, obj_id, candidate):
        """Generate get off action only if the bop is within some distance of a random city."""
        bop = self.get_bop(obj_id)
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if bop and self.map.get_distance(bop["cur_hex"], destination) <= 10:
            target_obj_id = random.choice(candidate)["target_obj_id"]
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.GetOff,
                "target_obj_id": target_obj_id,
            }

    def gen_change_state(self, obj_id, candidate):
        """Generate change state action with some probability."""
        change_state_prob = 0.001
        if random.random() < change_state_prob:
            target_state = random.choice(candidate)["target_state"]
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.ChangeState,
                "target_state": target_state,
            }

    def gen_remove_keep(self, obj_id, candidate):
        """Generate remove keep action with some probability."""
        remove_keep_prob = 0.2
        if random.random() < remove_keep_prob:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.RemoveKeep,
            }

    def gen_move(self, obj_id, candidate):
        """Generate move action to a random city."""
        bop = self.get_bop(obj_id)
        if bop["sub_type"] == 3:
            return
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if self.my_direction:
            destination = self.my_direction["info"]["target_pos"]
        if bop and bop["cur_hex"] != destination:
            move_type = self.get_move_type(bop)
            route = self.map.gen_move_route(bop["cur_hex"], destination, move_type)
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.Move,
                "move_path": route,
            }

    def get_move_type(self, bop):
        """Get appropriate move type for a bop."""
        bop_type = bop["type"]
        if bop_type == BopType.Vehicle:
            if bop["move_state"] == MoveType.March:
                move_type = MoveType.March
            else:
                move_type = MoveType.Maneuver
        elif bop_type == BopType.Infantry:
            move_type = MoveType.Walk
        else:
            move_type = MoveType.Fly
        return move_type

    def gen_stop_move(self, obj_id, candidate):
        """Generate stop move action only if the bop is within some distance of a random city.

        High probability for the bop with passengers and low for others.
        """
        bop = self.get_bop(obj_id)
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if self.map.get_distance(bop["cur_hex"], destination) <= 10:
            stop_move_prob = 0.9 if bop["passenger_ids"] else 0.01
            if random.random() < stop_move_prob:
                return {
                    "actor": self.seat,
                    "obj_id": obj_id,
                    "type": ActionType.StopMove,
                }

    def gen_WeaponLock(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        prob_weaponlock = 0.001
        if (
            max(self.map_data[bop["cur_hex"] // 100][bop["cur_hex"] % 100]["roads"]) > 0
            or random.random() < prob_weaponlock
        ):
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.WeaponLock}

    def gen_WeaponUnFold(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if self.map.get_distance(bop["cur_hex"], destination) <= 10:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.WeaponUnFold,
            }

    def gen_cancel_JM_plan(self, obj_id, candidate):
        cancel_prob = 0.0001
        if random.random() < cancel_prob:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.CancelJMPlan,
            }

    def gen_grouping_info(self, observation):
        def partition(lst, n):
            return [lst[i::n] for i in range(n)]

        operator_ids = []
        for operator in observation["operators"] + observation["passengers"]:
            if operator["color"] == self.color:
                operator_ids.append(operator["obj_id"])
        lists_of_ops = partition(operator_ids, len(self.team_info.keys()))
        grouping_info = {"actor": self.seat, "type": 100}
        info = {}
        for teammate_id in self.team_info.keys():
            info[teammate_id] = {"operators": lists_of_ops.pop()}
        grouping_info["info"] = info
        return [grouping_info]

    def gen_battle_direction_info(self, observation):
        direction_info = []
        for teammate_id in self.team_info.keys():
            direction = {
                "actor": self.seat,
                "type": 201,
                "info": {
                    "company_id": teammate_id,
                    "target_pos": random.choice(observation["cities"])["coord"],
                    "start_time": 0,
                    "end_time": 1800,
                },
            }
            direction_info.append(direction)
        return direction_info

    def gen_battle_mission_info(self, observation):
        mission_info = []
        for teammate_id in self.team_info.keys():
            mission = {
                "actor": self.seat,
                "type": 200,
                "info": {
                    "company_id": teammate_id,
                    "mission_type": random.randint(0, 2),
                    "target_pos": random.choice(observation["cities"])["coord"],
                    "route": [
                        random.randint(0, 9000),
                        random.randint(0, 9000),
                        random.randint(0, 9000),
                    ],
                    "start_time": 0,
                    "end_time": 1800,
                },
            }
            mission_info.append(mission)
        return mission_info

    def gen_fork(self, obj_id, candidate):
        prob = 0.01
        if random.random() < prob:
            return None
        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Fork}

    def gen_union(self, obj_id, candidate):
        prob = 0.1
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": random.choice(candidate)["target_obj_id"],
            "type": ActionType.Union,
        }

    def gen_change_altitude(self, obj_id, candidate):
        prob = 0.05
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": random.choice(candidate)["target_altitude"],
            "type": ActionType.ChangeAltitude,
        }

    def gen_activate_radar(self, obj_id, candidate):
        prob = 1
        if random.random() < prob:
            return None
        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.ActivateRadar}

    def gen_enter_fort(self, obj_id, candidate):
        prob = 0.5
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.EnterFort,
            "target_obj_id": random.choice(candidate)["target_obj_id"],
        }

    def gen_exit_fort(self, obj_id, candidate):
        prob = 0.1
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.ExitFort,
            "target_obj_id": random.choice(candidate)["target_obj_id"],
        }

    def gen_lay_mine(self, obj_id, candidate):
        prob = 1
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": 20,
            "target_pos": random.randint(0, 9177),
        }

    # assistant functions 
    def get_detected_state(self,state):
        # it is assumed that only my state passed here.
        # xxh 1226 legacy issues: how to get the state of emeny without the whole state?
        # 0106,it is said that detected enemy also included in my state.
        self.detected_state = []
        units = state["operators"]
        
        # for unit in units:
        #     detected_IDs = unit["see_enemy_bop_ids"]
        #     for detected_ID in detected_IDs:
        #         detected_state_single = self.select_by_type(units,key="obj_id", value=detected_ID)
        #         self.detected_state = self.detected_state + detected_state_single

        self.status_old=copy.deepcopy(self.status)

        color_enemy = 1 - self.color
        detected_state_single = self.select_by_type(units,key="color", value=color_enemy)

        # 这里其实有点问题，逻辑应该是探测到的敌方单位就再也不删除了，有状态更新就更新，没有就保持不变。
        detected_state_new = copy.deepcopy(self.detected_state)
        
        # 去重和更新。
        # for unit_old in self.detected_state:
        for unit in detected_state_single:
            flag_updated = False
            # for unit in detected_state_single:
            for unit_old in self.detected_state:
                if unit_old["obj_id"] == unit["obj_id"]:
                    # 说明这个是已经探索过的了，那就用新的
                    detected_state_new.append(unit)
                    flag_updated == True
                    break
            if flag_updated == False:
                # 说明是new detected.
                # 这个会高估威胁，打爆了的敌人会继续存在于态势中。
                # 但是crossfire里面真能打爆敌人的场景也不是很多，所以也就罢了。
                detected_state_new.append(unit)
            

                                
        self.detected_state = detected_state_new
        # 至此可以认为，过往所有探测到的敌人都保持在这里面了。

        return self.detected_state
    
    def get_move_type(self, bop):
        """Get appropriate move type for a bop."""
        bop_type = bop["type"]
        if bop_type == BopType.Vehicle:
            if bop["move_state"] == MoveType.March:
                move_type = MoveType.March
            else:
                move_type = MoveType.Maneuver
        elif bop_type == BopType.Infantry:
            move_type = MoveType.Walk
        else:
            move_type = MoveType.Fly
        return move_type
    def get_scenario_info(self, scenario: int):
        SCENARIO_INFO_PATH = os.path.join(
            os.path.dirname(__file__), f"scenario_{scenario}.json"
        )
        with open(SCENARIO_INFO_PATH, encoding="utf8") as f:
            self.scenario_info = json.load(f)
    def get_bop(self, obj_id, **kargs):
        """Get bop in my observation based on its id."""
        # 这个实现好像有点那啥，循环里带循环的，后面看看有没有机会整个好点的。xxh0307
        if "status" in kargs:
            observation = kargs["status"]
        else:
            observation = self.observation
        for bop in observation["operators"]:
            if obj_id == bop["obj_id"]:
                return bop
    def select_by_type(self,units,key="obj_id",value=0):
        # a common tool, for selecting units according to different keys.
        units_selected = [] 
        for unit in units:
            value_candidate = unit[key]
            if type(value_candidate) == str:
                if value in value_candidate:
                    # so this one is selected
                    units_selected.append(unit)
            else:
                if value==value_candidate:
                    # so this one is selected
                    units_selected.append(unit)
        
        return units_selected
    

    def get_prior_list(self,unit):
        # different weapons have common CD, so one prior list for one unit seems enough.
        # unit is a operator, a dict.
        unit_type = unit["sub_type"]
        prior_list = [] 
        # -1 for everything else, use it carefully 
        if unit_type == 0: # tank 
            prior_list = [0, 1, 4]
        elif unit_type == 1: # fighting vechile
            prior_list = [0,1,4,-1]
        elif unit_type == 2:  # infantry
            prior_list = [-1] # infantry attack everything
        elif unit_type == 3: # pao 
            prior_list = [-1] # pao did not use prior list 
        elif unit_type == 4: # unmanned che 
            prior_list = [5,7,6,8,2,0,1,-1] # the only airdefender 
        elif unit_type == 5: # UAV 
            prior_list = [0,1,4,-1] # it can do guided attack 
        elif unit_type == 6: # helicopter
            prior_list = [] # unused yet
        elif unit_type == 7: # xunfei missile
            prior_list = [0,1,4,-1] # only one shot.
        elif unit_type == 8: # transport helicopter 
            prior_list = [] # unused yet 
        else:
            raise Exception("get_prior_list: invalid unit_type in get_prior_list, G. ")
            prior_list = [0, 1, 4, -1]
            pass 

        return prior_list
    def get_pos(self,attacker_ID, **kargs):
        # just found pos according to attacker_ID
        # print("get_pos: unfinished yet")
        if type(attacker_ID) == int:
            unit0 = self.get_bop(attacker_ID,**kargs)
        else:
            unit0 = attacker_ID
        
        try:
            pos_0 = unit0["cur_hex"]
        except:
            pos_0 = -1 
        return pos_0

    def is_stop(self,attacker_ID, model="now"):
        # 这个就是单纯判断一下这东西是不是停着
        if type(attacker_ID) == list:
            # which means units inputted.
            flag_is_stop = True
            for unit in attacker_ID:
                this_abstract_state = self.abstract_state[unit["obj_id"]]
                if "abstract_state" in this_abstract_state:
                    if this_abstract_state["abstract_state"]=="move_and_attack":
                        flag_is_stop = False
                flag_is_stop = flag_is_stop and unit["stop"]
        else:
            # normal
            unit = self.get_bop(attacker_ID)
            flag_is_stop = unit["stop"]
        
        return flag_is_stop 

    def is_exist(self,attacker_ID,**kargs):
        # check if this obj still exist.
        attacker_ID = self._set_compatible(attacker_ID)
        if "units" in kargs:
            units = kargs["units"]
        else:
            units = self.status["operators"]

        flag_exist = False 
        for bop in units:
            if attacker_ID == bop["obj_id"]:
                flag_exist = flag_exist or True
        
        return flag_exist
    
    def is_arrive(self,units,target_pos,tolerance=0):
        # check if the units
        units_arrived = [] 
        flag_arrive = True
        for unit in units:
            if abs(unit["cur_hex"] - target_pos) <= tolerance :
                units_arrived.append(unit)
            else:
                flag_arrive = False
        return flag_arrive, units_arrived

    def get_ID_list(self,status):
        # get iterable ID list from status or something like status.
        operators_dict = status["operators"]
        ID_list = [] 
        for operator in operators_dict:
            # filter, only my operators pass
            if operator["color"] == self.color:
                # my operators
                ID_list.append(operator["obj_id"])
        return ID_list

    def get_pos_average(self,units,model="input_units"):
        geshu = len(units)
        x_sum = 0 
        y_sum = 0 
        x_ave = 0
        y_ave = 0 
        for i in range(geshu):
            if model == "input_units":
                # which means everything is ok.
                pos_this = self.get_pos(units[i]["obj_id"])
            elif model == "input_hexs":
                pos_this = units[i]
            xy_this = self._hex_to_xy(pos_this)

            x_sum = x_sum + xy_this[0]
            y_sum = y_sum + xy_this[1]
        
        x_ave = round(x_sum/geshu)
        y_ave = round(y_sum/geshu)
        pos_ave = self._xy_to_hex([x_ave,y_ave])

        return pos_ave

    def distance(self, target_pos, attacker_pos):
        if type(target_pos) == int:
            target_pos = target_pos
        else:
            target_pos = target_pos["cur_hex"]
        if type(attacker_pos) == int:
            attacker_pos = attacker_pos
        else:
            attacker_pos = attacker_pos["cur_hex"]       

        jvli = self.map.get_distance(target_pos, attacker_pos)
        # print("distance: unfinished yet")
        return jvli
    
    def get_IFV_units(self,**kargs):
        if "units" in kargs:
            units_input = kargs["units"]
        else:
            units_input = self.status["operators"]
        # double select by type.
        IFV_units = self.select_by_type(units_input,key="sub_type",value=1)
        IFV_units = self.select_by_type(IFV_units,key="color",value=0)
        return IFV_units
    
    def get_infantry_units(self,**kargs):
        if "units" in kargs:
            units_input = kargs["units"]
        else:
            units_input = self.status["operators"]        
        # same
        infantry_units = self.select_by_type(units_input,key="sub_type",value=2)
        infantry_units = self.select_by_type(infantry_units,key="color",value=0)
        return infantry_units       

    def get_UAV_units(self,**kargs):
        if "units" in kargs:
            units_input = kargs["units"]
        else:
            units_input = self.status["operators"]         
        UAV_units = self.select_by_type(units_input,key="sub_type",value=5)
        UAV_units = self.select_by_type(UAV_units,key="color",value=0)
        return UAV_units

    def get_tank_units(self,**kargs):
        if "units" in kargs:
            units_input = kargs["units"]
        else:
            units_input = self.status["operators"]         
        UAV_units = self.select_by_type(units_input,key="sub_type",value=0)
        UAV_units = self.select_by_type(UAV_units,key="color",value=0)
        return UAV_units

    # basic AI interface.
    def _move_action(self,attacker_ID, target_pos):
        bop = self.get_bop(attacker_ID)
        move_type = self.get_move_type(bop)
        route = self.map.gen_move_route(bop["cur_hex"], target_pos, move_type)
        
        action_move =  {
                "actor": self.seat,
                "obj_id": attacker_ID,
                "type": ActionType.Move,
                "move_path": route,
            }
        # self.act.append(action_move)
        self._action_check_and_append(action_move)
        return self.act

    def _stop_action(self,attacker_ID):
        # 简简单单stop，没什么好说的
        action_stop = {
                "actor": self.seat,
                "obj_id": attacker_ID,
                "type": ActionType.StopMove
                }
        # self.act.append(action_stop)
        
        # if self.num > 600:
        #     print("debug: disable all stop action")
        # else:
        #     self._action_check_and_append(action_stop)
        self._action_check_and_append(action_stop)
        return self.act

    def _fire_action(self,attacker_ID, target_ID="None", weapon_type="None"):
        # check if th fire action is valid.
        fire_actions = self._check_actions(attacker_ID, model="fire")
        # fire_actions = self._check_actions(attacker_ID, model="test")
        # if not, nothing happen.
        if len(fire_actions)>0:
            # get attack type
            if 2 in fire_actions:
                attack_type = 2 # direct
            elif 9 in fire_actions:
                attack_type = 9 # guided
            else:
                # raise Exception("_fire_action: invalid attack_type")
                # it seems warning here is better, not exception.
                print("_fire_action: invalid attack_type, did not fire")
                return self.act
            
            target_list = fire_actions[attack_type]
            target_ID_list = [] 
            weapon_type_list = [] 
            for i in range(len(target_list)):
                target_ID_i = target_list[i]["target_obj_id"]
                weapon_type_i = target_list[i]["weapon_id"]
                target_ID_list.append(target_ID_i)  
                weapon_type_list.append(weapon_type_i) 

            # get target_ID 
            if target_ID!="None":
                # target_selected
                # decide target ID
                target_ID_selected, index_target_ID = self._selecte_compare_list(target_ID, target_list)
            else:
                # no target selected.
                # best = max(candidate, key=lambda x: x["attack_level"])
                # target_ID_selected = best["target_obj_id"]
                
                # index_target_ID = ? 

                target_ID_selected = target_ID_list[0]
                index_target_ID = 0 

            # decide weapon_ID
            weappon_type_selected = weapon_type_list[index_target_ID]
            # 0219 need debug here, about how to select weapon type. 

            # then generate action
            action_gen = {
            "actor": self.seat,
            "obj_id": attacker_ID,
            "type": ActionType.Shoot,
            "target_obj_id": target_ID_selected,
            "weapon_id": weappon_type_selected,
            }
            print("_fire_action: done")
            # self.act.append(action_gen)
            self._action_check_and_append(action_gen)
            return self.act
        else:
            # no valid fire_action here, nothing happen 
            # print("_fire_action: no valid fire_action here, nothing happen")
            return self.act
    
    def _guide_shoot_action(self, attacker_ID):
        # 这个直接抄了，原则上UAV骑脸之后都会打到设想中的目标
        """Generate guide shoot action with the highest attack level."""
        candidate = self.status["judge_info"]
        if len(candidate) == 0:
            # which means that target has been destroyed.
            # return to avoid key value error
            flag_done = True
            return self.act, flag_done
        
        best = max(candidate, key=lambda x: x["attack_level"])
        if best["attack_level"] >0:
            # 说明确实是有东西可以给打
            action_guide_shoot = {
                "actor": self.seat,
                "obj_id": attacker_ID,
                "type": ActionType.GuideShoot,
                "target_obj_id": best["target_obj_id"],
                "weapon_id": best["weapon_id"],
                "guided_obj_id": best["guided_obj_id"],
            }
            self._action_check_and_append(action_guide_shoot)
            flag_done = True
            return self.act,flag_done
        else:
            # 说明没东西可以打了
            flag_done = False
            return self.act, flag_done

    def _selecte_compare_list(self, target_ID, target_ID_list):
        # give an obj(target_ID) and a list(target_ID_list), if the obj is in the list, then reture it and its index. 
        # if selected obj can not be reached, then randomly get one or find by prior list.

        if len(target_ID_list)==0:
            raise Exception("_selecte_compare_list: invalid list inputted")

        if target_ID in target_ID_list:
            # if so, attack
            target_ID_selected = target_ID
            index_target_ID = target_ID_list.index(target_ID_selected)  # find the first one and attack. need debug 0218
            print("_selecte_compare_list: find the first one and attack. need debug 0218")
        else:
            # if selected target ID can not be reached, then randomly get one or find by prior list.
            target_ID_selected = target_ID_list[0]
            index_target_ID = 0 

        return target_ID_selected, index_target_ID

    def _check_actions(self, attacker_ID, model="void"):
        # found all valid action of attacker_ID.
        # model: "void" for return valid actions
        # "fire" fire for return all valid fire action
        # "board" for all about on board and off board. 
        obj_id=attacker_ID
        total_actions = {} 
        observation = self.status 


        if obj_id not in self.controposble_ops:
            return total_actions
        
        try:
            total_actions = observation["valid_actions"][attacker_ID]
        except:
            # print("no valid_actions here") # 这个正常运行别开，不然命令行全是这个。
            total_actions = {}

        if model == "void":
            # if model is "void", then skip selection and return the total actions.
            return total_actions
        else:
            # select the actions by set the model
            if model == "fire":
                selected_action_list = [2,9]
            elif model == "board":
                selected_action_list = [3,4] 
            elif model == "jieju":
                selected_action_list = [14] 
            elif model == "juhe":
                selected_action_list = [15]
            elif model == "test":
                selected_action_list = [] 
                for i in range(10):
                    selected_action_list.append(i)
                
            selected_actions = {}
            for action_type in selected_action_list:
                if action_type in total_actions:
                    selected_actions[action_type] = total_actions[action_type]
            return selected_actions

        return total_actions
    
        # loop and find
        for valid_actions in observation["valid_actions"].items():
            for (
                action_type
            ) in self.priority:  
                if action_type not in valid_actions:
                    continue
                
                # find which is valid, but don't gen here.

                # find the action generation method based on type
                gen_action = self.priority[action_type]
                action = gen_action(obj_id, valid_actions[action_type])
                if action:
                    total_actions.append(action)
                    break  # one action per bop at a time
        
        return total_actions

    def _hidden_actiion(self,attacker_ID):
        action_hidden =  {
                "actor": self.seat,
                "obj_id": attacker_ID,
                "type": 6,
                "target_state": 4,
            }
        # {
        # "actor": "int 动作发出者席位",
        # "obj_id": "算子ID int",
        # "type": 6,
        # "target_state": "目标状态 0-正常机动 1-行军 2-一级冲锋 3-二级冲锋, 4-掩蔽 5-半速"
        # }
        # self.act.append(action_hidden)
        self._action_check_and_append(action_hidden)
        return self.act

    def _jieju_action(self,attacker_ID):
        candidate_actions = self._check_actions(attacker_ID,model="jieju")
        action_jieju = {} 
        flag_done = False
        if len(candidate_actions)>0:
            action_jieju = {
                "actor": self.seat,
                "obj_id": attacker_ID,
                "type": 14
            }
            flag_done = True
        else:
            # raise Exception("_jieju_action: invalid action here.")
            flag_done = False
        # miaosuan platform did not accept void action {},
        # so there must be some check.
        # self.act.append(action_jieju)
        self._action_check_and_append(action_jieju)
        return self.act,flag_done
    
    def _juhe_action(self,attacker_ID, target_ID):
        candidate_actions = self._check_actions(attacker_ID,model="juhe")
        action_juhe = {} 
        flag_done = False
        if len(candidate_actions)>0:
            action_juhe = {
                "actor": self.seat,
                "obj_id": attacker_ID,
                "target_obj_id": target_ID,
                "type": 15
            }
            flag_done = True  
        else:
            # no invalid juhe action.
            flag_done = False
        self._action_check_and_append(action_juhe)
        return self.act,flag_done    

    def _on_board_action(self,attacker_ID,infantry_ID):
        # print("_on_board_action unfinished yet")
        action_on_board = {
                        "actor": self.seat,
                        "obj_id": infantry_ID,
                        "type": 3,
                        "target_obj_id": attacker_ID
                    }
        # 这个只能决定有没有发出命令，发了之后会怎么样就不在这里闭环了。
        self._action_check_and_append(action_on_board)
        return self.act
    
    def _off_board_action(self, attacker_ID,infantry_ID):
        action_off_board = {
                            "actor": self.seat,
                            "obj_id": attacker_ID,
                            "type": 4,
                            "target_obj_id": infantry_ID
                        }
        # 这个只能决定有没有发出命令，发了之后会怎么样就不在这里闭环了。
        self._action_check_and_append(action_off_board)
        return self.act
    
    def _action_check_and_append(self,action):
        # miaosuan platform did not accept void action {},
        # so there must be some check.
        if action=={}:
            pass 
        else:
            self.act.append(action)
        return self.act
    
    def _hex_to_xy(self,hex):
        # 要搞向量运算来出阵形，所以还是有必要搞一些转换的东西的。
        xy = hex_to_xy(hex)
        return xy
    
    def _xy_to_hex(self,xy):
        hex = xy_to_hex(xy)
        return hex

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
        
    # abstract_state and related functinos
    def Gostep_abstract_state(self,**kargs):
        # 先更新一遍观测的东西，后面用到再说
        self.detected_state = self.get_detected_state(self.status)
        # self.update_detectinfo(self.detected_state)  # 记录一些用于搞提前量的缓存

        self.update_field() # 这个是更新一下那个用于避障的标量场。

        # 清理一下abstract_state,被摧毁了的东西就不要在放在里面了.
        abstract_state_new = {}
        filtered_status = self.__status_filter(self.status)
        ID_list = self.get_ID_list(filtered_status)
        for attacker_ID in ID_list:
            if attacker_ID in self.abstract_state:
                try:
                    abstract_state_new[attacker_ID] = self.abstract_state[attacker_ID]
                except:
                    # 这个是用来处理新增加的单位的，主要是用于步兵上下车。
                    abstract_state_new[attacker_ID] = {"abstract_state": "none"}
                    # self.set_none(attacker_ID)
            else:
                # 下车之后的步兵在filtered_status有在abstract_state没有，得更新进去
                abstract_state_new[attacker_ID] = {}

        self.abstract_state = abstract_state_new

        self.act = []
        # 遍历一下abstract_state，把里面每个单位的命令都走一遍。
        for my_ID in self.abstract_state:
            my_abstract_state = self.abstract_state[my_ID]
            if my_abstract_state == {}:
                # 默认状态的处理, it still needs discuss, about which to use.
                # self.set_hidden_and_alert(my_ID)
                self.set_none(my_ID) 
                # self.set_jieju(my_ID) 
            else:
                # 实际的处理
                my_abstract_state_type = my_abstract_state["abstract_state"]
                if my_abstract_state_type == "move_and_attack":
                    # self.__handle_move_and_attack2(my_ID, my_abstract_state["target_pos"])
                    self.__handle_move_and_attack3(my_ID, my_abstract_state["target_pos"])
                elif my_abstract_state_type == "hidden_and_alert":
                    self.__handle_hidden_and_alert(my_ID)  # 兼容版本的，放弃取地形了。
                elif my_abstract_state_type == "open_fire":
                    self.__handle_open_fire(my_ID)
                elif my_abstract_state_type == "none":
                    self.__handle_none(my_ID)  # 这个就是纯纯的停止。
                elif my_abstract_state_type == "jieju":
                    self.__handle_jieju(my_ID) # 解聚，解完了就会自动变成none的。
                elif my_abstract_state_type == "juhe":
                    self.__handle_juhe(my_ID,my_abstract_state["target_ID"],my_abstract_state["role"])
                elif my_abstract_state_type == "UAV_move_on":
                    self.__handle_UAV_move_on(my_ID,target_pos=my_abstract_state["target_pos"])
                elif my_abstract_state_type == "on_board":
                    self.__handle_on_board(my_ID,my_abstract_state["infantry_ID"],my_abstract_state["flag_state"])
                    # 这个参数选择其实不是很讲究，要不要在这里显式传my_abstract_state["flag_state"]，其实还是可以论的。
                elif my_abstract_state_type == "off_board":
                    self.__handle_off_board(my_ID,my_abstract_state["infantry_ID"],my_abstract_state["flag_state"])
        return self.act
        pass

    def __status_filter(self,status):
        # print("__status_filter: unfinished yet.")
        return status
    
    def update_threaten_source(self):
        # 0229 单位损失得储存，别的倒是不太有所谓。

        # 要跨步骤存的先拿出来好了。
        # threaten_source_set_type2 = set() 
        threaten_source_list_type2 = [] 
        for source_single in self.threaten_source_list:
            if source_single["type"] == 2:
                source_single["delay"] = source_single["delay"] - 1 
                if source_single["delay"]>=0:
                    threaten_source_list_type2.append(source_single)
        
        # 然后清了，重新再开
        self.threaten_source_list = []   

        # 敌人首先得整进来。
        # units = self.status["operators"]
        # units_enemy = self.select_by_type(units,str="obj_id",value=1)
        units_enemy = self.detected_state

        for unit in units_enemy:
            # {pos: int , type: int, delay: int}
            threaten_source_single = {"pos":unit["cur_hex"], "type":0, "delay":0}
            self.threaten_source_list.append(threaten_source_single)

        # 然后是炮火打过来的点
        artillery_point_list = self.status["jm_points"]
        for artillery_point_single in artillery_point_list:
            if artillery_point_single["status"] == 1: # 在爆炸
                threaten_source_single = {"pos":artillery_point_single["pos"], "type":1, "delay":0}
            elif artillery_point_single["status"] == 0: # 在飞行
                threaten_source_single = {"pos":artillery_point_single["pos"], "type":1, "delay":75-artillery_point_single["fly_time"]}# 0319，实测中发现，由于移动时间会延迟，容易出现走上去的时候没事儿但是走开了就有事儿，所以时间得放开一些
            else:
                pass # 失效了
            self.threaten_source_list.append(threaten_source_single)
                
        # 然后是有单位损失的位置,通过比较status_old和status来给出，所以这个函数要放在更新abstract_state之前。
        ID_list_now = self.get_ID_list(self.status)
        ID_list_old = self.get_ID_list(self.status_old)
        for ID in ID_list_old:
            flag = not(ID in ID_list_now ) 
            if flag:
                # unit lost
                threaten_source_single = {"pos":self.get_pos(ID,status=self.status_old), "type":2, "delay":70}
                self.threaten_source_list.append(threaten_source_single)

        # “旁边有己方单位的地方更加安全”，所以己方单位作为一个负的威胁源，或者说威胁汇，恐怕是成立的。
        for ID in ID_list_now:
            threaten_source_single = {"pos":self.get_pos(ID), "type":-1} 


        # 最后把上一步需要持续考虑的算进来，岂不美哉。
        self.threaten_source_list =  self.threaten_source_list  + threaten_source_list_type2

        return self.threaten_source_list

    def update_field(self):
        # 标量场的话，得选定需要计算的范围，搞精细一点，所有单位的周围几个格子，然后还是得有个机制检验要不要变轨
        # 矢量场的话，似乎直接给每个单位算个斥力就行了？还简单点。
        # 不对，这可恶的六角格，算不了矢量场啊好像。
        # 总之这个放在GoStep里。

        # 选定所有单位周围的两个格子，然后去重
        distance_start = 0 
        distance_end = 2 
        ID_list = self.get_ID_list(self.status)
        pos_set = set() 
        for attacker_ID in ID_list:
            pos_attacker = self.get_pos(attacker_ID)
            pos_set_single = self.map.get_grid_distance(pos_attacker, distance_start, distance_end)
            pos_set = pos_set | pos_set_single
        
        # 选定所有单位的格子好了

        # 然后更新影响的来源，标量场嘛无所谓了。
        self.threaten_source_list = self.update_threaten_source()

        # 然后更新那一堆点里面的标量场。
        self.threaten_field = {}
        for pos_single in pos_set:
            field_value = self.update_field_single(pos_single, self.threaten_source_list)
            threaden_field_single = {pos_single:field_value}
            self.threaten_field.update(threaden_field_single)
            
        
        # # need debug 0229
            
        # print("update_field: finished, but was not used to modify the path yet, 0229")
        
        return self.threaten_field
    
    def update_field_single(self, pos_single, threaten_source_list):
        # 虽然可以直接self.调用，但是还是搞成显式的输入输出比较好。
        field_value = 0
        # TODO：还可以精细化一些，对不同类型的单位可以分别定义势场。
        for threaten_source in threaten_source_list:
            # 求出两个格子的距离
            jvli = self.map.get_distance(pos_single,threaten_source["pos"])

            # type: 0 for enemy units, 1 for artillery fire, 2 for unit lost 
            a1 = 10 # 在敌人那格，0距离，type=0，thus field=a1
            a2 = 1 
            if threaten_source["type"] == 0:
                # 有敌方单位，就别去送了。
                field_value = field_value + a1*3 / (a2 + jvli) # never touch the enemy. 
            elif threaten_source["type"] == 1:
                # 有炮火覆盖的地方，如果快开始爆炸了就别过去送了，绕一下。
                if threaten_source["delay"] == 0:
                    field_value = field_value + a1 / (a2 + 1 + jvli)
            elif threaten_source["type"] == 2: 
                # 之前有东西损失过的地方，如果人家CD快转好了就别过去送了，绕一下。
                if threaten_source["delay"] < 30:
                    field_value =field_value + a1 / (a2 + 1 + jvli)
            elif threaten_source["type"] == -1:
                # 有己方单位存活，认为那附近安全一点。负的威胁度
                field_value =field_value + -1*a1*0.2 / (a2 + 1 + jvli)
            
        return field_value

    def update_detectinfo(self, detectinfo):
        print("update_detectinfo: not finished yet, and it seems not necessarry")
        return 
        # 处理一下缓存的探测。
        # 好吧,这个并不需要。探测池子里面给到的“上一步”似乎是对的。
        for target_ID in detectinfo:
            for filter_ID in self.weapon_list:
                if filter_ID in target_ID:
                    continue  # 如果探测到的是弹药，那就不要了。

            target_state = {}

            target_state_single = {}
            lon = detectinfo[target_ID]["targetLon"]
            lat = detectinfo[target_ID]["targetLat"]
            alt = detectinfo[target_ID]["targetAlt"]
            pos = [lon, lat, alt]

            target_state_single["pos"] = pos
            target_state_single["num"] = self.num
            if target_ID in self.detected_state2:
                # 那就是有的
                if "this" in self.detected_state2[target_ID]:
                    # 反正暴力出奇迹，只存两步，线性插值，怎么简单怎么来。
                    target_state["last"] = copy.deepcopy(self.detected_state2[target_ID]["this"])
            target_state["this"] = copy.deepcopy(target_state_single)
            self.detected_state2[target_ID] = target_state

        # 整个过滤机制，时间太长的探测信息就直接不保存了
        list_deleted = []
        for target_ID in self.detected_state2:
            if (self.num - self.detected_state2[target_ID]["this"]["num"]) > 500:
                # 姑且是500帧之前的东西就认为是没用了。
                list_deleted.append(target_ID)
        for target_ID in list_deleted:
            del self.detected_state2[target_ID]
        return
    
    def _set_compatible(self,attacker_ID):
        # 这个用于保证兼容性，让那些set的东西既可以接收attacker_ID，也可以接收unit作为参数
        if (type(attacker_ID) == dict) and ("obj_id" in attacker_ID):
            #说明输入进来的是unit
            real_attacker_ID = attacker_ID["obj_id"]
        elif type(attacker_ID) == int:
            # 说明进来的是attacker_ID
            real_attacker_ID = attacker_ID
        else:
            raise Exception("invalid attacker_ID")
        return real_attacker_ID
    
    def _abstract_state_timeout_check(self,attacker_ID):
        # 统一整一个，check一下保持这个状态是不是超过时间了。如果超过了就结束，别卡在里面。
        # 不要滥用这个。
        if not("remain_time") in self.abstract_state[attacker_ID]:
            raise Exception("_abstract_state_timeout_check: invalid use, this abstract_state did not have remain_time")
        self.abstract_state[attacker_ID]["remain_time"]=self.abstract_state[attacker_ID]["remain_time"]+1
        if self.abstract_state[attacker_ID]["remain_time"]>300:
            # 总的停止时长如果超过一个阈值，就不玩了，直接结束这个状态。
            # self.__finish_abstract_state(attacker_ID)
            # return
            return True
        else:
            return False
        
    def set_move_and_attack(self, attacker_ID, target_pos,model="normal"):
        # 还得是直接用字典，不要整列表。整列表虽然可以整出类似红警的点路径点的效果，但是要覆盖就得额外整东西。不妥
        # 直接做成模式可选择的就好了，要覆盖就覆盖，不要的话可以不覆盖。
        attacker_ID = self._set_compatible(attacker_ID)
        if model=="normal":
            # 默认模式不覆盖，检测如果已经是move_and_attack状态了，就不做操作，
            # 如果是别的，就还是覆盖了。
            try:
                this_abstract_state= self.abstract_state[attacker_ID]["abstract_state"]
                if (this_abstract_state == "move_and_attack") or (this_abstract_state=="jieju") or (this_abstract_state=="on_board") or (this_abstract_state=="off_board"):
                    pass
                else:
                    self.abstract_state[attacker_ID] = {"abstract_state": "move_and_attack", "target_pos": target_pos,"flag_moving": False, "jvli": 114514, "flag_evading":False}
            except:
                # 上面要是没try到，就说明抽象状态里面还没有这个ID，那就先设定一下也没啥不好的。
                self.abstract_state[attacker_ID] = {"abstract_state": "move_and_attack", "target_pos": target_pos,"flag_moving": False, "jvli": 114514, "flag_evading":False}
        elif model == "force":
            # 这个就是不管一切的强势覆盖，如果连着发就会覆盖
            self.abstract_state[attacker_ID] = {"abstract_state": "move_and_attack", "target_pos": target_pos,"flag_moving": False, "jvli": 114514, "flag_evading":False}


    def set_hidden_and_alert(self, attacker_ID):
        # 这个就是原地坐下，调成隐蔽状态。
        # TODO: 不是原地坐下了，要找周围好的地形。
        if (type(attacker_ID) == dict) or (type(attacker_ID) == list):
            # 说明是直接把status输入进来了。那就得循环。
            for attacker_ID_single in attacker_ID:
                attacker_ID_single = self._set_compatible(attacker_ID_single)
                self.abstract_state[attacker_ID_single] = {"abstract_state": "hidden_and_alert", "flag_shelter": False}
        else:
            attacker_ID = self._set_compatible(attacker_ID)
            self.abstract_state[attacker_ID] = {"abstract_state": "hidden_and_alert", "flag_shelter": False}
        pass

    def set_none(self,attacker_ID,**kargs):
        # yangjian xiefa, all set_none operations use this function, rather than modifing abstract_state directly.
        attacker_ID = self._set_compatible(attacker_ID)
        self.abstract_state[attacker_ID] = {"abstract_state": "none"}
        if "next" in kargs:
            self.abstract_state[attacker_ID]["next"] = kargs["next"]
        # pass 
    
    def set_jieju(self, attacker_ID):
        # just jieju if it is possible. 
        attacker_ID = self._set_compatible(attacker_ID)
        self.abstract_state[attacker_ID] = {"abstract_state": "jieju"}
    
    def set_juhe(self, attacker_ID, target_ID):
        attacker_ID = self._set_compatible(attacker_ID)
        self.abstract_state[attacker_ID] = {"abstract_state": "juhe", "target_ID":target_ID, "role":"king"}
        self.abstract_state[target_ID] = {"abstract_state": "juhe", "target_ID":attacker_ID,"role":"knight","waiting_num":77}

    def set_open_fire(self,attacker_ID,**kargs):
        # 对只能站定打的东西来说，这个状态就有意义了。
        attacker_ID = self._set_compatible(attacker_ID)
        self.abstract_state[attacker_ID]={"abstract_state":"open_fire"}
        if "next" in kargs:
            self.abstract_state[attacker_ID]["next"] = kargs["next"]
        # 现在这版还没有集火的说法，就是单纯的能打谁就打谁。
        # 因为假设了这个庙算是一个“开火稀疏”的场景，所以能打就打应该是没问题的。

    def set_UAV_move_on(self,attacker_ID, target_pos):
        # 这个用来处理UAV的行动，基本逻辑是如果有敌人就飞到敌人旁边去定下来等着打引导打击。
        # 所以还得改个标志位，来体现是不是有东西能够提供引导打击。

        # 记录一下现有的 # 不对，逻辑上这个应该在set_UAV_move_on之前，别给他整乱了
        # 不对，就应该放这儿，UAV的逻辑应该是那种飞过去打一波走了的逻辑，应该自带状态转换。看起来就算是空状态也不影响这个状态转换的逻辑
        attacker_ID = self._set_compatible(attacker_ID)
        if not("abstract_state" in self.abstract_state[attacker_ID]):
            return
        if self.abstract_state[attacker_ID]["abstract_state"]!="UAV_move_on":
            # 防止无限嵌套，得检测一下是不是本来就已经在UAV_move_on了
            # 不然好像要写python写出内存泄漏了，乐.jpg
            abstract_state_previous = copy.deepcopy(self.abstract_state[attacker_ID])
            self.abstract_state[attacker_ID]={"abstract_state":"UAV_move_on", "target_pos":target_pos,"flag_moving": False, "jvli": 1919810, "flag_attacked":False, "stopped_time":0 }
            self.abstract_state[attacker_ID]["next"] = abstract_state_previous
        else:
            # 如果已经是UAV_move_on了，那就不用改了
            pass

    def set_on_board(self,attacker_ID, infantry_ID,**kargs):
        # 这个就是开过去接到车上就算是完事了。
        attacker_ID = self._set_compatible(attacker_ID)
        infantry_ID = self._set_compatible(infantry_ID)

        self.abstract_state[attacker_ID] = {"abstract_state": "on_board",
                                                "infantry_ID": infantry_ID,
                                                "flag_state": 1,
                                                "num_wait": 0,"remain_time":0}
        # 值得思考一下，这个是否应该支持next机制。要是都支持可能会搞得比较乱，需要更多调试，可能会比较帅
        if "next" in kargs:
            self.abstract_state[attacker_ID]["next"] = kargs["next"]

    def set_off_board(self,attacker_ID, infantry_ID,**kargs):
        attacker_ID = self._set_compatible(attacker_ID)
        infantry_ID = self._set_compatible(infantry_ID)        
        self.abstract_state[attacker_ID] = {"abstract_state": "off_board",
                                                "infantry_ID": infantry_ID,
                                                "flag_state": 1,
                                                "num_wait": 0, "remain_time":0}
        # 值得思考一下，这个是否应该支持next机制。要是都支持可能会搞得比较乱，需要更多调试，可能会比较帅
        if "next" in kargs:
            self.abstract_state[attacker_ID]["next"] = kargs["next"]

    def set_capture(self, attacker_ID, target_pos):
        # 这个是类似上下车那种复合型的，就是冲到那里，夺控，然后结束。冲的过程就open fire了。
        pass
    
    def __handle_set_capture(self, attacker_ID, target_pos):
        pass 

    def __handle_move_and_attack(self, attacker_ID, target_pos):
        # 这个是改进开火的。不带避障
        flag_attack = True  # 调试，开始打炮了。

        if flag_attack:
            self._fire_action(attacker_ID)
        else:
            print("XXHtest: attack disabled in __handle_move_and_attack")

        # 然后该打的打完了，就继续move呗
        attacker_pos = self.get_pos(attacker_ID)
        jvli = self.distance(target_pos,attacker_pos)  
        if jvli > 0:
            # 那就是还没到，那就继续移动
            if self.abstract_state[attacker_ID]["flag_moving"] == False:
                # 那就是没动起来，那就得让它动起来。
                self._move_action(attacker_ID, target_pos)
                self.abstract_state[attacker_ID]["flag_moving"] = True
            if (self.abstract_state[attacker_ID]["jvli"] == jvli) and (self.num>100):
                self.__finish_abstract_state(attacker_ID)
            else:
                self.abstract_state[attacker_ID]["jvli"] = jvli
        else:
            # 那就是到了，那就要改抽象状态里面了。
            self.__finish_abstract_state(attacker_ID)
    
    def __handle_move_and_attack2(self,attacker_ID,target_pos):
        # 这个是新版的，带基于势场的避障。
        flag_attack = True  # 调试，开始打炮了。

        if flag_attack:
            self._fire_action(attacker_ID)
        else:
            print("XXHtest: attack disabled in __handle_move_and_attack")

        # 然后该打的打完了，就继续move呗
        attacker_pos = self.get_pos(attacker_ID)

        # 这版的避障逻辑：检测周围一圈的格子，如果有势场大于某个阈值的，就触发避障，找势场最小的方向去先走一格。
        # 为了防止卡住，得整个随机的，找势场最小的两格随机一个。# TODO: need debug 0307
        # 于是先检测周围一圈的格子：
        neighbor_pos_list = self.map.get_neighbors(attacker_pos)
        neighbor_field_list = [] 
        for i in range(len(neighbor_pos_list)):
            neighbor_pos_single = neighbor_pos_list[i]
            if neighbor_pos_single ==-1:
                neighbor_field_single = 0
            else:
                neighbor_field_single = self.threaten_field[neighbor_pos_single]
            neighbor_field_list.append(neighbor_field_single)

        # 于是再检测阈值，看有没有超过的，如果有就躲一下，没有就继续往目标去。
        unit= self.get_bop(attacker_ID)
        if max(neighbor_field_list)>5:
            # 说明附近威胁有点大，触发规避动作
            if unit["stop"]==0:
                # which means this unit is moving
                self._stop_action(attacker_ID)
            elif unit["stop"]==1:
                #which means this unit is stop.
                # then 触发规避，就找威胁最小的那个格子然后过去。
                neighbor_field_min = min(neighbor_field_list)
                neighbor_field_min_index = neighbor_field_list.index(neighbor_field_min)
                neighbor_field_min_pos = neighbor_pos_list[neighbor_field_min_index]
                # 这下定位出威胁最小的那个格子了，那过去吧。
                self._move_action(attacker_ID, neighbor_field_min_pos)
            
            # 这部分是用于debug的：
            self.abstract_state[attacker_ID]["flag_evading"] = True # 现在这个设定，就是如果检测到一次就直接过去了，然后应该就不走了停在那儿了。
        else:
        # elif self.abstract_state[attacker_ID]["flag_evading"] == False:
            # 说明附近威胁尚可，那就无事发生，还是采用之前那个逻辑，往地方去就完事了。
            jvli = self.distance(target_pos,attacker_pos)  
            if jvli > 0:
                # 那就是还没到，那就继续移动
                if unit["stop"]==0:
                    pass
                elif unit["stop"]==1:
                    self._move_action(attacker_ID, target_pos)
                self.abstract_state[attacker_ID]["flag_moving"] = not(unit["stop"])

                self.abstract_state[attacker_ID]["jvli"] = jvli
            else:
                # 那就是到了，那就要改抽象状态里面了。
                self.__finish_abstract_state(attacker_ID)            
            pass 

    def __handle_move_and_attack3(self, attacker_ID, target_pos):
        # 这个是进一步升级来的，不要路径点序列了，一次只走一格，每一格都判断威胁程度 
        # 0319,这个确实能用，也确实能够爽爽避障，但是是走一格停一轮的，用在穿越火线里面是不科学的。
               
        unit = self.get_bop(attacker_ID)
        attacker_pos =self.get_pos(attacker_ID)
        attacker_xy = self._hex_to_xy(attacker_pos)
        target_xy = self._hex_to_xy(target_pos)
        vector_xy = target_xy - attacker_xy

        # 先打了再说。
        self._fire_action(attacker_ID)

        # 然后该打的打完了，就继续move呗
        attacker_pos = self.get_pos(attacker_ID)
        # if arrived, then stay.
        if np.linalg.norm(vector_xy) <0.000001:
            self.__finish_abstract_state(attacker_ID)
            return 
        # 来个来个向量运算，计算出周围一圈点中，符合威胁度要求的点中，最符合向量的一个。
        # 有威胁就开这个，没威胁就找出最符合向量的
        try:
            flag_arrive = (unit["move_path"][-1]==attacker_pos)
            # 路径里的点到了，就认为到了
        except:
            # 要是没有路径点，那就看是不是stop
            flag_arrive=False
            # if unit["stop"]==0:
            #     flag_arrive=False
            # elif unit["stop"]==1:
            #     flag_arrive=True
            if unit["speed"]==0:
                flag_arrive=True
            else:
                flag_arrive=False

        if flag_arrive==False:
            # 走着呢，由于都是走一格，所以这个就没有什么所谓了吧。
            pass 
        elif flag_arrive==True:
            #which means this unit is stop.
            # 找威胁符合的点中方向最好的

            # 于是先检测周围一圈的格子：
            neighbor_pos_list = self.map.get_neighbors(attacker_pos)
            neighbor_field_list = [] 
            neighbor_pos_list_selected = []
            neighbor_field_list_selected = []
            for i in range(len(neighbor_pos_list)):
                neighbor_pos_single = neighbor_pos_list[i]
                if neighbor_pos_single ==-1:
                    neighbor_field_single = 0
                else:
                    neighbor_field_single = self.threaten_field[neighbor_pos_single]
                neighbor_field_list.append(neighbor_field_single)

                if neighbor_field_single<5:
                    # 选出一些比较安全的点。如果没有比较安全的点就只能用全部点了。
                    neighbor_pos_list_selected.append(neighbor_pos_single)
                    neighbor_field_list_selected.append(neighbor_field_single)
            
            # 然后根据威胁情况看后面往哪里去。
            if len(neighbor_pos_list_selected)>3:
                # 说明周围存在相对安全一些的区域
                pos_next = self.find_pos_vector(attacker_pos, neighbor_pos_list_selected,vector_xy)
                pass
            else:
                # 说明周围全是高威胁的区域了，那还不如拼一枪。
                pos_next = self.find_pos_vector(attacker_pos, neighbor_pos_list, vector_xy)
                pass
            
            # 选出来之后就过去呗。
            self._move_action(attacker_ID, pos_next)
        
        jvli = self.distance(target_pos,attacker_pos)  
        if jvli > 0:
            # 那就是还没到，那就继续移动
            self.abstract_state[attacker_ID]["flag_moving"] = not(unit["stop"])
            self.abstract_state[attacker_ID]["jvli"] = jvli
        else:
            # 那就是到了，那就要改抽象状态里面了。
            self.__finish_abstract_state(attacker_ID)      

    def __handle_hidden_and_alert(self, attacker_ID):
        # 先来个基础版的，原地蹲下，不要取地形了。
        # 0219: 这个可能有风险，庙算里面开始状态转换之后似乎就走不了了
        self._hidden_actiion(attacker_ID)
        pass

    def __handle_none(self, attacker_ID):
        # nothing happen, literaly
        pass 

    def __handle_jieju(self, attacker_ID):
        # if it can be jieju, then jieju, or change the abstract_state.
        self.act,flag_done = self._jieju_action(attacker_ID)
        unit = self.get_bop(attacker_ID)
        if len(self._check_actions(attacker_ID, model="jieju"))==0 and unit["forking"]==0:
            # 进了这里才是没有正在解聚且不能再解聚了。_jieju_action的flag_done主要是有没有发出过命令。
            pos_attacker = self.get_pos(attacker_ID)
            distance_start = 1
            distance_end = 2 
            candidate_pos_list = self.map.get_grid_distance(pos_attacker, distance_start, distance_end)
            candidate_pos_list = list(candidate_pos_list)
            geshu = len(candidate_pos_list)
            index_random = random.randint(0,geshu-1)
            target_pos_random = candidate_pos_list[index_random]
            
            # # 然后移动过去。不要move and attack了，直接移过去。
            # self._move_action(attacker_ID,target_pos_random)
            self.set_move_and_attack(attacker_ID,target_pos_random,model="force")
        else:
            # 那就是还能接着解聚。
            pass
        # if flag_done==False:
        #     # can not jieju anymore, change to none.
        #     self.set_none(attacker_ID)
        # else:
        #     # finish jieju for one time, try to jieju further.
        #     # move to random location.
        #     pass
        return 

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

    def __handle_juhe(self, attacker_ID,target_ID,role):
        # if attacker is king, then wait, else if attacker is knight, then go and find the king.
        flag_finish = self.is_exist(attacker_ID) and self.is_exist(attacker_ID)
        if flag_finish == False:
            # one of them is not exist anymore, which means juhe finished.
            self.__finish_abstract_state(attacker_ID)
        attacker_pos = self.get_pos(attacker_ID)
        target_pos = self.get_pos(target_ID)
        jvli = self.distance(target_pos,attacker_pos)          

        if role == "king":
            # then wait and send juhe zhiling
            # if jvli is not 0, then wait.
            if jvli ==0:
                # then send juhe zhiling.
                flag_done = self._juhe_action(attacker_ID,target_ID)
                pass 
            else:
                # then wait. holy wait.
                # abstract_state_next = self.abstract_state[attacker_ID]
                # # self.set_open_fire(attacker_ID)
                # self.set_none(attacker_ID)
                # self.abstract_state[attacker_ID]["next"] = abstract_state_next 
                pass
        elif role == "knight":
            if jvli ==0:
                # then wait, don't do anything include set_none.
                print("knight is waiting")
                self.abstract_state[attacker_ID]["waiting_num"] = self.abstract_state[attacker_ID]["waiting_num"] -1 
                if self.abstract_state[attacker_ID]["waiting_num"] == 0:
                    # its king is not exists anymore.
                    self.__finish_abstract_state(attacker_ID)
                pass 
            else:
                # then go and find its king.
                abstract_state_next = self.abstract_state[attacker_ID]
                self.set_move_and_attack(attacker_ID,target_pos,model="normal")
                self.abstract_state[attacker_ID]["next"] = abstract_state_next 
                pass 
        else:
            raise Exception("invalid role when jieju.")


    def __handle_UAV_move_on(self, attacker_ID, target_pos):
        # 飞到目标点附近，站下来，如果完成了一次引导打击，就算是结束这个状态
        # 开口就是老入侵者战机/金乌轰炸机了

        # 如果已经打了一个引导打击了，那就退出去。不然就继续无人机出击。
        if self.abstract_state[attacker_ID]["flag_attacked"]==True:
            self.__finish_abstract_state(attacker_ID)
            return 

        # check一下停止的时长。
        if (self.is_stop(attacker_ID)):
            self.abstract_state[attacker_ID]["stopped_time"]=self.abstract_state[attacker_ID]["stopped_time"]+1
        if self.abstract_state[attacker_ID]["stopped_time"]>100:
            # 总的停止时长如果超过一个阈值，就不玩了，直接结束这个状态。
            self.__finish_abstract_state(attacker_ID)
            return
        
        # 前面那些check都过了，再来说函数实现的事情。
        # 看距离，飞过去。
        attacker_pos = self.get_pos(attacker_ID)
        jvli = self.distance(target_pos,attacker_pos)  
        if jvli > 2: # 飞到附近就行了，不用非要骑到脸上。
            # 那就是还没到，那就继续移动
            if self.abstract_state[attacker_ID]["flag_moving"] == False:
                # 那就是没动起来，那就得让它动起来。
                self._move_action(attacker_ID, target_pos)
                self.abstract_state[attacker_ID]["flag_moving"] = True
            self.abstract_state[attacker_ID]["jvli"] = jvli
        else:
            # 那就是到了，那就停下来准备打它
            flag_is_stop = self.is_stop(attacker_ID)
            if(flag_is_stop==False):
                self._stop_action(attacker_ID) 
                # 这个stop其实不是很必要，会自己停下来的。
                
                # 然后引导打击
                # 开始耦合了，按理来说应该多给几个车发出停下指令，准备好打，还要考虑车面临的威胁高不高，还要考虑车里有没有兵。
                # 但是这里先写个最垃圾的，如果无人机就位了，就把所有车都停了。而且是开环控制。
                IFV_units = self.get_IFV_units()
                for IFV_unit in IFV_units:
                    # 按理来说直接这么写就完事了，虽然可能下一步才更新，但是反正得好几帧才能停下，不差这点了。
                    next_IFV_abstract_state = copy.deepcopy(self.abstract_state[IFV_unit["obj_id"]])
                    # 如果里面有步兵就不执行这个任务，没有步兵才执行。
                    infantry_ID_list = IFV_unit["get_off_partner_id"]+IFV_unit["get_on_partner_id"] + IFV_unit["passenger_ids"]
                    if len(infantry_ID_list) >0 :
                        # 说明这个里面有兵，那原则上就不让它停下来等着打了。
                        pass 
                    else:
                        # 说明这个里面没有兵
                        self.set_open_fire(IFV_unit, next=next_IFV_abstract_state)
            else:
                # 到这里原则上已经停好了，UAV和IFV都停好了
                # 那就想想办法干它一炮
                self.act, flag_done = self._guide_shoot_action(attacker_ID)
                if flag_done==True:
                    # 说明引导打击命令合法地发出去了，就认为是打出去了
                    self.abstract_state[attacker_ID]["flag_attacked"] = True
                    # 然后那几个IFV也不用挂着了，该干啥干啥去好了
                    # 也是有隐患的，如果中间IFV的状态被改了而且next被清了，可能就要寄。
                    IFV_units = self.get_IFV_units()
                    for IFV_unit in IFV_units:
                        self.__finish_abstract_state(IFV_unit)

    def __handle_on_board(self,attacker_ID, infantry_ID, flag_state):
        # 这个得细心点弄一下。
        attacker_pos = self.get_pos(attacker_ID)
        try:
            infantry_pos = self.get_pos(infantry_ID)
        except:
            # 这个就是步兵不在态势里了。
            infantry_pos = -1
        
        # 接管步兵的控制权。在上车完成之前，步兵的抽象状态不再生效。
        flag_infantry_exist = self.is_exist(infantry_ID)
        if flag_infantry_exist:
            self.set_none(infantry_ID,next=self.abstract_state[infantry_ID])
        else:
            # finished or infantry unit lost.
            flag_state = 3
            self.abstract_state[attacker_ID]["flag_state"] = flag_state
        
        flag_time_out = self._abstract_state_timeout_check(attacker_ID)
        if flag_time_out:
            self.__finish_abstract_state(attacker_ID)
            self.__finish_abstract_state(infantry_ID)
            return
        
        jvli = self.distance(attacker_pos,infantry_pos)  

        if flag_state == 1:
            # 没上车且距离远，那就得过去。
            if jvli < 1:
                # 那就是到了，转变为可以上车的状态。

                # 上车命令。
                self._on_board_action(attacker_ID,infantry_ID)
                flag_state = 2
                self.abstract_state[attacker_ID]["flag_state"] = flag_state
            elif jvli < 3000:
                # 距离不够，那就过去接。简化逻辑，只写一个过去接，不假设过程中会动或者什么的。
                abstract_state_next = copy.deepcopy(self.abstract_state[attacker_ID])
                self.set_move_and_attack(attacker_ID, infantry_pos,model="force")
                self._stop_action(infantry_ID) # 步兵的动作给它停了，乖乖站好。TODO: 这里“乖乖站好”的实现，原则上也应该用abstract_state实现才比较优雅
                self.abstract_state[attacker_ID]["next"] = abstract_state_next  # 然后把它放回去，准备跑完了之后再复原。
            else:
                # 那就是步兵已经寄了，那就直接退化成move and attack就完事儿了。
                # self.set_move_and_attack(attacker_ID, target_LLA)
                self.__finish_abstract_state(attacker_ID) # 一样的，退出状态，原则上next里面就会是move_and_attack
        if flag_state == 2:
            # 没上车且正在上,或者说条件姑且具备了。
            if self.abstract_state[attacker_ID]["num_wait"] > 0:
                # 那就是等着呢，那就等会儿好了。
                self.abstract_state[attacker_ID]["num_wait"] = self.abstract_state[attacker_ID]["num_wait"] - 1
                pass
            else:
                if jvli < 1:
                    # 那就是到了，那就上车。
                    self._stop_action(attacker_ID)
                    self._on_board_action(attacker_ID,infantry_ID)
                    # self._stop_action(infantry_ID)
                    self.abstract_state[attacker_ID]["num_wait"] = 75
                elif jvli <= 3000:
                    # 那就是没到且可以去。
                    flag_state = 1
                    self.abstract_state[attacker_ID]["flag_state"] = flag_state
                
            pass  
        if flag_state == 3:
            # 那这意思就是上车完事了，就结束退出。开冲放在别的地方开冲了。
            # 开冲。 如果到了就放下来分散隐蔽，兵力分散火力集中。
            # 不要再闭环到1了，这样防止这东西死循环。
            self.__finish_abstract_state(attacker_ID)
            # self.__finish_abstract_state(infantry_ID)
            # there is no infantry, so.

    def __handle_off_board(self,attacker_ID, infantry_ID, flag_state):
        # 这个之前是没有的。思路也是一样的，分状态分距离。# 而且这个需要好好整一下
        unit_infantry = self.get_bop(infantry_ID)
        unit_attacker = self.get_bop(attacker_ID)
        # 一样的，接管步兵的控制权。在上车完成之前，步兵的抽象状态不再生效。
        flag_infantry_exist = self.is_exist(infantry_ID)
        if flag_infantry_exist:
            self.set_none(infantry_ID,next=self.abstract_state[infantry_ID])
            flag_state = 3 # finished xiache,
        else:
            pass 

        # 然后启动超时check
        flag_time_out = self._abstract_state_timeout_check(attacker_ID)
        if flag_time_out:
            self.__finish_abstract_state(attacker_ID)
            # self.__finish_abstract_state(infantry_ID)
            return

        if flag_state == 1:
            # 具备条件了，但是还没有发下车命令。那就发个下车指令然后开始等着。
            # 没停车就停车，停车了就发指令。
            if self.is_stop(attacker_ID) == False:
                self._stop_action(attacker_ID)
                # 停车，一直check到标志位变了，到停稳
            else:
                # 那就是停下了，那就发命令下车。
                self._off_board_action(attacker_ID,infantry_ID)
                self.abstract_state[attacker_ID]["num_wait"] = 75
                # 发出命令之后等着。
                flag_state = 2
                self.abstract_state[attacker_ID]["flag_state"] = flag_state            
            pass
        elif flag_state == 2:
            # 发了下车命令了，正在等下车CD
            self.abstract_state[attacker_ID]["num_wait"] = self.abstract_state[attacker_ID]["num_wait"] - 1
            if self.abstract_state[attacker_ID]["num_wait"]==1:
                # 说明下车下好了，转换状态
                flag_state = 3
                self.abstract_state[attacker_ID]["flag_state"] = flag_state       
            pass
        elif flag_state == 3:
            # 下车下完了，就可以结束任务了。
            self.__finish_abstract_state(attacker_ID)
            # self.__finish_abstract_state(infantry_ID) # 结束对步兵的控制

    def __finish_abstract_state(self, attacker_ID):
        # print("__finish_abstract_state: unfinished yet")
        attacker_ID = self._set_compatible(attacker_ID) # 来个这个之后就可以直接进unit了
        # 统一写一个完了之后清空的，因为也不完全是清空，还得操作一些办法。
        # 暴力堆栈了其实是，笨是笨点但是有用。
        if attacker_ID in self.abstract_state:
            pass
        else:
            # 这个是用来处理步兵上下车逻辑的。上车之后删了，下车之后得出来
            # self.abstract_state[attacker_ID] = {}  # 统一取成空的，后面再统一变成能用的。
            self.set_none(attacker_ID)

        if "next" in self.abstract_state[attacker_ID]:
            next_abstract_state = self.abstract_state[attacker_ID]['next']
        else:
            next_abstract_state = {}
        self.abstract_state[attacker_ID] = next_abstract_state
        pass
        

    # guize_functions
    def F2A(self,target_pos):
        units = self.status["operators"]
        for unit in units:
            self.set_move_and_attack(unit,target_pos,model="force")
            # A了A了，都这时候了还要个毛的脑子，直接头铁
        pass

    def group_A(self, units,target_pos, model='normal'):
        print("group_A: unfinished yet")
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

        enemy_infantry_units = self.select_by_type(self.detected_state,key="sub_type",value=2)
        enemy_infantry_units_danger = [] 
        # 如果敌方步兵在正前方了，那就别去了。同时满足距离和方向的才算。
        for enemy_infantry_unit in enemy_infantry_units:
            # 遍历看一下是不是需要跑。
            enemy_pos = enemy_infantry_unit["cur_hex"]
            enemy_xy = self._hex_to_xy(enemy_pos)
            vector_single = enemy_xy - xy_ave
            enemy_distance = self.map.get_distance(enemy_pos, pos_ave)

            dot_single = np.dot(vector_single, vector_xy) / np.linalg.norm(vector_xy) / np.linalg.norm(vector_single)
            
            if enemy_distance<15 and dot_single>0.95:
                # 这两个阈值都是从案例里抠出来的。
                enemy_infantry_units_danger.append(enemy_infantry_unit)
        
        # 至此，就筛出了究极高威胁的敌方步兵的位置。然后是根据这些位置确定绕路的方向，以target_pos_list的形式放在list中。
        if len(enemy_infantry_units_danger)>0:
            pos_ave_enemy = self.get_pos_average(enemy_infantry_units_danger)
            xy_ave_enemy = self._hex_to_xy(pos_ave_enemy)
            vector_ave_enemy = xy_ave_enemy - xy_ave
            
            # 然后向量计算求一下那个点应该往哪边找，点乘判断正负可也，
            # 先求两个垂直于路径的法向量。
            n1_xy = np.array([vector_xy[1], -1*vector_xy[0]]) / np.linalg.norm(vector_xy)
            n2_xy = -1*n1_xy
            # 然后检测哪个比较好。
            if np.dot(n1_xy,vector_xy)>0:
                # 那说明是偏向这个方向，绕道的路就得往另一个方向去了。
                n_xy_list = [n2_xy, n1_xy] 
            else:
                n_xy_list = [n1_xy, n2_xy] 
            # 道理上不可能两个方向都在外面，因为起点终点在垂线的不同侧，且都在范围内。
            # 所以两边必有一边是能够绕路的。

            # 然后然后开始计算距离点了。
            # 先取个中间点出来
            pos_center = self.get_pos_average([pos_ave,target_pos], model="input_hexs")
            # 然后算。反正两个方向，总得有一个对的。要是都不对也防一手。
            xy_center = self._hex_to_xy(pos_center)
            try:
                xy_candidate = xy_center + 15*n_xy_list[0]
                pos_candidate = self._xy_to_hex(xy_candidate)
            except:
                xy_candidate = xy_center + 15*n_xy_list[1]
                pos_candidate = self._xy_to_hex(xy_candidate)    
        else:
            pos_candidate = target_pos
        return [pos_candidate, target_pos, target_pos] # 这里后面补一个target_pos是为了写循环的时候好写。
                
    def list_A(self, units, target_pos, model="normal"):
        # “选取部队横越地图”，实现一个宏观层面的绕行机制。
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
                    self.set_move_and_attack(unit,target_pos,model=model)
                else:
                    # 没到的话就无事发生。
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
            if len(self._check_actions(unit["obj_id"], model="jieju"))==0 and unit["forking"]==0 and unit["blood"]==1:
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

    # then step
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
        # communications = observation["communication"]

        # # get the detected state
        # units = observation["operators"]
        # detected_state = self.get_detected_state(observation)
        # get the target first.
        self.distinguish_saidao()
        # update the actions
        if model == "guize":
            self.Gostep_abstract_state()
        elif model =="RL":
            pass
        # the real tactics in step*() function.
        # self.step0()

        self.step_cross_fire()

        # self.step_defend()

        # # update the actions
        # self.Gostep_abstract_state()

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
            self.group_A(others_units,target_pos,model="force")
            # self.group_A2(others_units,IFV_units)
        if jieju_flag2 == True:
            self.group_A(IFV_units,target_pos,model="force")
            # self.list_A(IFV_units,target_pos)

        # if arrived, then juhe.
        if self.num>800:
            self.final_juhe(tank_units)
            self.final_juhe(IFV_units)

        if self.num>1000:
            # 最后一波了，直接F2A了
            self.F2A(target_pos)
            pass # disabled for tiaoshi
        
        if (self.num % 100==0) and (self.num>-200) and (self.num<2201):
            # 保险起见，等什么上车啊解聚啊什么的都完事儿了，再说别的。
            # deal with UAV.
            # self.UAV_patrol(target_pos)
            # kaibai is fine.
            self.group_A(UAV_units,target_pos)
        return 


    def step_scout(self):
        # unfinished yet.
        
        print("step_scout: unfinished yet.")

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