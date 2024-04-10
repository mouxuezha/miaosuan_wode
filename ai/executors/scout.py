import random
import numpy as np

from ..const import ActType

# 六边形网格的方向，从右开始逆时针，根据奇偶行数分两种情况
directions = [
    [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)],
    [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)],
]


def hex2xy(hex_coord):
    """将六边形坐标转换为直角坐标"""
    col = hex_coord % 100
    row = hex_coord // 100
    return col, row


def xy2hex(x, y):
    """将直角坐标转换为六边形坐标"""
    return x * 100 + y


def find_corners(center, area):
    """area生成的顺序是从左到右，从上到下"""
    xy_points = np.array([divmod(point, 100) for point in area])
    first_row = area[0] // 100
    col_max_first = max(xy_points[xy_points[:, 0] == first_row][:, 1])

    center_row = center // 100
    center_row_points = xy_points[xy_points[:, 0] == center_row]
    col_min_center = min(center_row_points[:, 1])
    col_max_center = max(center_row_points[:, 1])

    last_row = area[-1] // 100
    col_min_last = min(xy_points[xy_points[:, 0] == last_row][:, 1])

    corners = [
        xy2hex(center_row, col_max_center),
        xy2hex(first_row, col_max_first),
        area[0],
        xy2hex(center_row, col_min_center),
        xy2hex(last_row, col_min_last),
        area[-1],
    ]
    if col_min_center == col_min_last:
        corners.pop(3)
    if col_max_center == col_max_first:
        corners.pop(0)
    return corners


def get_end_point(start, direc, dist):
    """给定起点、方向、距离，确定终点"""
    row, col = divmod(start, 100)
    flag = row & 1
    delta_row, delta_col = (dist + 1) // 2 * directions[flag][
        direc
    ] + dist // 2 * directions[1 - flag][direc]
    end_row, end_col = row + delta_row, col + delta_col


def get_direc(end, start):
    """
    用立方体坐标粗暴地算一个方向
    0-东，1-东北，2-西北，3-西，4-西南，5-东南，6-北，7-南
    """
    row1, col1 = divmod(start, 100)  # TODO: 单独写xy2qrs函数
    q1 = col1 - (row1 - (row1 & 1)) // 2
    r1 = row1
    s1 = -q1 - r1
    row2, col2 = divmod(end, 100)
    q2 = col2 - (row2 - (row2 & 1)) // 2
    r2 = row2
    s2 = -q2 - r2
    if r2 == r1:
        return 0 if q2 > q1 else 3
    elif s2 == s1:
        return 1 if r2 < r1 else 4
    elif q2 == q1:
        return 2 if s2 > s1 else 5
    elif abs(r2 - r1) > abs(q2 - q1) and abs(r2 - r1) > abs(s2 - s1):
        return 6 if r2 < r1 else 7
    else:
        return -1


class ScoutExecutor:
    """执行侦察任务的策略"""

    def __init__(self) -> None:
        self.area = None
        self.xy_points = None
        self.unscouted = None
        self.air_num = 0
        self.air_traj = {}
        self.enemy_pos = {}  # {hex: [obj_id, cond]}
        self.units = {}  # {obj_id: [last, cur]}
        self.suspected = set()

    def setup(self, task, agent):
        air_start = []
        print("ScoutExecutor init, agent info:")
        for obj_id, unit in agent.owned.items():
            print(f"obj_id: {obj_id}, unit: {unit['type']}")
            if unit["type"] == 3:
                self.air_num += 1
                self.air_traj[obj_id] = []
                air_start.append(unit["cur_hex"])
        rough_start = sum(air_start) // len(air_start)

        self.area = list(agent.map.get_grid_distance(task["hex"], 0, task["radius"]))
        self.area.sort()
        self.unscouted = set(self.area.copy())
        self.area2xy()
        self.allocate_traj(rough_start, task["hex"])

    def area2xy(self):
        """将侦察区域转换为直角坐标"""
        points = np.array([divmod(point, 100) for point in self.area])
        first_row = points[0][0]
        last_row = points[-1][0]
        self.xy_points = []
        for i in range(first_row, last_row + 1):
            self.xy_points.append(points[points[:, 0] == i])
        return self.xy_points

    def allocate_traj(self, start, center):
        """将侦察区域分配给各个无人机"""
        start_row = start // 100
        center_row = center // 100
        first_row = self.area[0] // 100
        last_row = self.area[-1] // 100
        total_traj = []
        if start_row >= center_row:
            layers = list(range(first_row + 1, last_row + 1, 3))
            if last_row - layers[-1] > 1:
                layers.append(last_row)
        else:
            layers = list(range(last_row - 1, first_row - 1, -3))
            if layers[-1] - first_row > 1:
                layers.append(first_row)
        flag = 0
        for i in layers:
            j = i - first_row
            if flag == 0:
                total_traj.append(
                    xy2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
                )
                total_traj.append(
                    xy2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
                )
            else:
                total_traj.append(
                    xy2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
                )
                total_traj.append(
                    xy2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
                )
            flag = 1 - flag
        print(f"total traj: {total_traj}")
        split = len(total_traj) // self.air_num + 1
        obj_ids = list(self.air_traj.keys())
        for i in range(self.air_num):
            self.air_traj[obj_ids[i]] = total_traj[split * i : split * (i + 1)]

    def update_unit(self, obj_id, cur_hex):
        if obj_id not in self.units.keys():
            self.units[obj_id] = [-1, cur_hex]
        else:
            self.units[obj_id][0] = self.units[obj_id][1]
            self.units[obj_id][1] = cur_hex

    def update_unscouted(self, agent, cur_hex):
        neighbors = set(agent.map.get_neighbors(cur_hex))
        self.unscouted -= neighbors
        self.suspected -= neighbors
        for obj_id, air_traj in self.air_traj.items():
            for j in range(len(air_traj) - 1, -1, -1):
                if air_traj[j] in neighbors:
                    air_traj.pop(j)

    def can_you_shoot_me(self, agent, cur_hex):
        cond = agent.map.basic[cur_hex // 100][cur_hex % 100]["cond"]
        radius = 12 if cond in [1, 2] else 20
        area = list(agent.map.get_grid_distance(cur_hex, 0, radius))
        area.sort()
        shoot_area = []
        for h in area:
            if agent.map.can_see(cur_hex, h, 0):
                shoot_area.append(h)
        return set(shoot_area)

    def get_nearest(self, agent, cur_hex, to_detect):
        """获取最近的未探测区域"""
        min_dist = 1000
        nearest_hex = -1
        for h in to_detect:
            dist = agent.map.get_distance(cur_hex, h)
            if dist < min_dist:
                min_dist = dist
                nearest_hex = h
        return nearest_hex

    def guess_enemy(self, cur_units, agent):
        assert len(self.units) - len(cur_units) == 1
        old_units = set(self.units.keys())
        diff = list(old_units - cur_units)
        missed_unit = diff[0]

        area_last = self.can_you_shoot_me(agent, self.units[missed_unit][0])
        area_cur = self.can_you_shoot_me(agent, self.units[missed_unit][1])
        tmp_suspect = area_cur - area_last & self.unscouted - set(self.enemy_pos.keys())
        print(f"tmp suspect num: {len(tmp_suspect)}")
        if len(self.suspected) == 0:
            self.suspected = tmp_suspect
        else:
            inter = self.suspected & tmp_suspect
            if len(inter) > 0:
                self.suspected = inter
            else:
                self.suspected = self.suspected | tmp_suspect
        self.units.pop(missed_unit)
        print(f"missed unit: {missed_unit}, suspected range: {len(self.suspected)}")

    def re_allocate_air(self, agent):
        tmp = list(self.suspected)
        center = sum(tmp) // len(tmp)
        dist = []
        for obj_id, unit in agent.owned.items():
            if unit["type"] == 3:
                dist.append([obj_id, agent.map.get_distance(unit["cur_hex"], center)])
        dist.sort(key=lambda x: x[1])
        new_traj_point = self.get_nearest(
            agent, agent.owned[dist[0][0]]["cur_hex"], tmp
        )
        old_traj = self.air_traj[dist[0][0]]
        point_before_insert = self.get_nearest(agent, new_traj_point, old_traj)
        idx = old_traj.index(point_before_insert)
        self.air_traj[dist[0][0]].insert(idx + 1, new_traj_point)
        print(f"***reallocate obj: {dist[0][0]}, new point: {new_traj_point}***")

    def execute(self, task, agent):
        """"""
        if agent.time.cur_step < 3:
            self.setup(task, agent)
        # self.area = list(agent.map.get_grid_distance(task["hex"], 0, task["radius"]))
        if not self.area:
            print("ScoutExecutor: area is empty")
            return  # 侦察区域不能为空
        if len(agent.enemy) > 0:
            for obj_id, unit in agent.enemy.items():
                enemy_hex = unit["cur_hex"]
                if enemy_hex not in self.enemy_pos.keys():
                    x, y = hex2xy(enemy_hex)
                    self.enemy_pos[enemy_hex] = [obj_id, agent.map.basic[x][y]["cond"]]

        if len(agent.owned) < len(self.units):
            cur_units = set(agent.owned.keys())
            self.guess_enemy(cur_units, agent)
            self.re_allocate_air(agent)

        available_units = set(task["unit_ids"])
        if not available_units:  # 没有指定算子则使用全部算子
            available_units = set(agent.valid_units)
        for obj_id, unit in agent.valid_units.items():
            if unit["cur_pos"] == 0:
                cur_hex = unit["cur_hex"]
                self.update_unit(obj_id, cur_hex)
                self.update_unscouted(agent, cur_hex)
                # print(f"remain: {len(self.unscouted)}")

            if obj_id not in available_units or agent.flag_act[obj_id]:
                continue  # 算子不参与此任务或已经生成了动作

            if unit["type"] == 2 and ActType.Fork in agent.valid_actions[obj_id]:
                # 车辆优先解聚
                agent.actions.append(agent.act_gen.fork(obj_id))
                agent.flag_act[obj_id] = True
                continue

            if ActType.Move not in agent.valid_actions[obj_id]:
                continue  # 算子正在机动，不再生成机动动作

            to_detected = (
                list(self.suspected) if self.suspected else list(self.unscouted)
            )
            destination = random.choice(to_detected)
            if unit["type"] == 3:
                if len(self.air_traj[obj_id]):
                    destination = self.air_traj[obj_id].pop(0)
                else:
                    destination = self.get_nearest(agent, unit["cur_hex"], to_detected)
            route = agent.gen_move_route(unit, int(destination))
            if route:
                agent.actions.append(agent.act_gen.move(obj_id, route))
                agent.flag_act[obj_id] = True
