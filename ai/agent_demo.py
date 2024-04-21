from dataclasses import dataclass

from .action import ActionGenerator
from .const import ActType, BopType, MoveType, TaskType, UnitSubType
from .executors.cross_fire import CrossFireExecutor
from .executors.defend import DefendExecutor
from .executors.scout import ScoutExecutor
from .map import Map


class Agent:
    def __init__(self):
        self.color = None
        self.seat = None
        self.role = None
        self.user_id = None
        self.user_name = None
        self.map = None
        self.priority = None
        self.ob = None
        self.actions = None

    def setup(self, setup_info):
        self.color = setup_info["faction"]
        self.seat = setup_info["seat"]
        self.role = setup_info["role"]
        self.user_id = setup_info["user_id"]
        self.user_name = setup_info["user_name"]
        self.priority = {}
        self.map = Map(
            setup_info["basic_data"],
            setup_info["cost_data"],
            setup_info["see_data"],
        )
        self.act_gen = ActionGenerator(self.seat)
        self.task_executors = {
            TaskType.CrossFire: CrossFireExecutor(),
            TaskType.Scout: ScoutExecutor(),
            TaskType.Defend: DefendExecutor(),
        }

    def step(self, observation: dict):
        self.ob = observation
        self.update_time()
        if self.time.is_deployment_stage:
            return self.make_deploy()  # 在部署阶段让步兵上车
        self.update_tasks()
        if not self.tasks:
            return []  # 如果没有任务则待命
        self.update_all_units()
        self.update_valid_actions()
        self.update_cities()

        self.actions = []  # 将要返回的动作容器
        self.prefer_shoot()  # 优先选择射击动作
        self.prefer_occupy()  # 优先选择夺控动作

        for task in self.tasks:  # 遍历每个分配给本席位任务
            self.task_executors[task["type"]].execute(task, self)
        if self.actions:
            print(f"actions at step {self.time.cur_step}: {self.actions}")
        return self.actions

    def update_time(self):
        cur_step = self.ob["time"]["cur_step"]
        stage = self.ob["time"]["stage"]
        self.time = Time(cur_step, stage)

    def make_deploy(self):
        actions = []
        for obj_id, unit in self.owned.items():
            if unit["sub_type"] == UnitSubType.Infantry:
                actions.append(self.act_gen.deploy_get_on(obj_id, unit["launcher"]))
        actions.append(self.act_gen.end_deploy())
        return actions

    def update_tasks(self):
        self.tasks = []
        for task in self.ob["communication"]:
            if (
                task["seat"] == self.seat
                and task["start_time"] <= self.time.cur_step < task["end_time"]
                and task["type"] in self.task_executors
            ):  # 分配给本席位的、有效时间内的、能执行的任务
                self.tasks.append(task)

    def update_all_units(self):
        self.friendly = {}  # 我方非乘员算子，包括属于我方其他席位的算子
        self.enemy = {}  # 可观察到的敌方算子
        self.on_board = {}  # 我方乘员算子，因为在车上，所以无法直接操纵
        self.owned = {}  # 属于当前席位的算子
        self.valid_units = {}  # 当前能够动作的算子
        owned_set = set(self.ob["role_and_grouping_info"][self.seat]["operators"])
        valid_set = set(self.ob["valid_actions"])
        for unit in self.ob["operators"]:
            if unit["color"] == self.color:
                self.friendly[unit["obj_id"]] = unit
                if unit["obj_id"] in owned_set:
                    self.owned[unit["obj_id"]] = unit
                    if unit["obj_id"] in valid_set:
                        self.valid_units[unit["obj_id"]] = unit
            else:
                self.enemy[unit["obj_id"]] = unit
        for unit in self.ob["passengers"]:
            self.on_board[unit["obj_id"]] = unit
            if unit["obj_id"] in owned_set:
                self.owned[unit["obj_id"]] = unit

    def update_valid_actions(self):
        self.valid_actions = {}
        self.flag_act = {}  # 记录已经生成了动作的算子，避免一个算子同一步执行多个动作
        for obj_id in self.valid_units:
            self.valid_actions[obj_id] = self.ob["valid_actions"][obj_id]
            self.flag_act[obj_id] = False

    def update_cities(self):
        self.cities = {}
        for city in self.ob["cities"]:
            self.cities[city["coord"]] = city

    def prefer_shoot(self):
        for obj_id, val_act in self.valid_actions.items():
            if self.flag_act[obj_id]:
                continue
            available_targets = val_act.get(ActType.Shoot, None)
            if available_targets:
                best = max(available_targets, key=lambda x: x["attack_level"])
                act = self.act_gen.shoot(
                    obj_id, best["target_obj_id"], best["weapon_id"]
                )
                self.actions.append(act)
                self.flag_act[obj_id] = True

    def prefer_occupy(self):
        for obj_id, val_act in self.valid_actions.items():
            if self.flag_act[obj_id]:
                continue
            if ActType.Occupy in val_act:
                act = self.act_gen.occupy(obj_id)
                self.actions.append(act)
                self.flag_act[obj_id] = True

    def gen_move_route(self, unit, destination):
        """计算指定算子按照适当方式机动到目的地的最短路径"""
        unit_type = unit["type"]
        if unit_type == BopType.Vehicle:
            if unit["move_state"] == MoveType.March:
                move_type = MoveType.March
            else:
                move_type = MoveType.Maneuver
        elif unit_type == BopType.Infantry:
            move_type = MoveType.Walk
        else:
            move_type = MoveType.Fly
        return self.map.gen_move_route(unit["cur_hex"], destination, move_type)


@dataclass
class Time:
    """维护当前推演时间"""

    cur_step: int
    stage: int

    @property
    def is_deployment_stage(self):
        return self.stage == 1
