# this is wode agent for miaosuan, define some layers.
import sys 
sys.path.append("/home/vboxuser/Desktop/miaosuan_code/sdk")
from ai.agent import Agent
from ai.base_agent import BaseAgent
from map import Map
class agent_guize(BaseAgent):
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
        self.controllable_ops = None
        self.team_info = None
        self.my_direction = None
        self.my_mission = None
        self.user_name = None
        self.user_id = None
        self.history = None
        # abstract_state is useful
        self.absract_state = {} 

        self.act = [] # list to save all commands generated.

        self.state = {}
        self.detected_state = {} 

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

    def get_detected_state(self,state):
        # it is assumed that only my state passed here.
        self.detected_state = {} 
        units = state["operators"]
        for unit in units:
            detected_IDs = unit["see_enemy_bop_ids"]
            for detected_ID in detected_IDs:
                detected_state_single = {} 
                # xxh 1226 legacy issues: how to get the state of emeny without the whole state?
                self.detected_state[detected_ID] = detected_state_single
        return self.detected_state

    def _move_action(self,attacker_ID, target_pos):
        
        pass

    def _fire_action(self,attacker_ID, target_ID, weapon_type):
        pass


    # abstract_state and related functinos
    def Gostep_absract_state(self,**kargs):
        pass

    # guize_functions
    def F2A(self):
        pass

    def group_A(self):
        pass

    # then step
    def step(self, observation: dict):
        self.observation = observation
        self.team_info = observation["role_and_grouping_info"]
        self.controllable_ops = observation["role_and_grouping_info"][self.seat][
            "operators"
        ]
        communications = observation["communication"]

        # get the detected state
        units = observation["operators"]
        detected_state = self.get_detected_state(observation)
        
        # the real tactics in step*() function.
        self.step0()

        # update the actions
        self.Gostep_absract_state()

        pass

    def step0(self):
        # this is the first one for learning the guize of miaosuan, 1226 xxh.
        self.gen_move(self.units[0])
        pass