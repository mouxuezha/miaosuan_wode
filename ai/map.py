import os
import heapq
import random
import numpy as np

from .const import BopType, CondType
from .tools import time_decorator

ob_range = [
    [10, 25, 1, 10, 0],
    [10, 25, 1, 25, 0],
    [2, 2, -1, 2, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]  # 依次为步兵、车辆、无人机的无地形遮蔽可观察距离
max_shoot_range = [10, 20, 0, 0, 0]  # 依次为步兵、车辆、无人机的最大射程


class Map:
    def __init__(self, basic_data, cost_data, see_data):
        """
        Load basic map data, move cost data and see data.

        You could do a lot more funny stuff using these three kind of data, e.g.
        get the whole see matrix of a given position via
        `self.see[mode][row, col]` or dynamically modify `self.cost` according
        to the observation of obstacles to generate customized move path.

        :param scenario: int
        """
        self.basic = basic_data["map_data"]

        self.max_row = len(self.basic)
        self.max_col = len(self.basic[0])

        self.cost = cost_data
        self.see = see_data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ob_file_path = os.path.join(current_dir, "ob_mat.npy")
        # self.ob = np.load(ob_file_path)
        self.ob_area3 = {}

    def is_valid(self, pos):
        """
        Check if `pos` is inside the map.

        :param pos: int
        :return: bool
        """
        row, col = divmod(pos, 100)
        return 0 <= row < self.max_row and 0 <= col < self.max_col

    def get_map_data(self):
        """Not very useful. Kept for backward compatibility."""
        return self.basic

    def get_neighbors(self, pos):
        """
        Get neighbors of `pos` in six directions.

        :param pos: int
        :return: List[int]
        """
        if not self.is_valid(pos):
            return []
        row, col = divmod(pos, 100)
        return self.basic[row][col]["neighbors"]

    def can_see(self, pos1, pos2, mode):
        """
        Check if `pos1` can see `pos2` with given `mode`.

        `self.see[mode]` is a `numpy.ndarray` that supports multidimensional
        indexing. So you could get the whole see matrix of a given position via
        `self.see[mode][row, col]`.

        By bit-masking see matrices of different positions, you could easily get
        all desired positions. For example,
        `np.argwhere((self.see[0][8, 24] & self.see[0][24, 8]) == True)` returns
        all positions that can be seen from both `0824` and `2408`. Tweak the
        condition and you could create a lot more interesting stuff.

        :param pos1: int
        :param pos2: int
        :param mode: int
        :return: bool
        """
        if (
            not self.is_valid(pos1)
            or not self.is_valid(pos2)
            or not 0 <= mode < len(self.see)
        ):
            return False
        row1, col1 = divmod(pos1, 100)
        row2, col2 = divmod(pos2, 100)
        return self.see[mode][row1, col1, row2, col2]

    def get_distance(self, pos1, pos2):
        """
        Get distance between `pos1` and `pos2`.

        :param pos1: int
        :param pos2: int
        :return: int
        """
        if not self.is_valid(pos1) or not self.is_valid(pos2):
            return -1
        # convert position to cube coordinate
        row1, col1 = divmod(pos1, 100)
        q1 = col1 - (row1 - (row1 & 1)) // 2
        r1 = row1
        s1 = -q1 - r1
        row2, col2 = divmod(pos2, 100)
        q2 = col2 - (row2 - (row2 & 1)) // 2
        r2 = row2
        s2 = -q2 - r2
        # calculate Manhattan distance
        return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2

    def gen_move_route(self, begin, end, mode):
        """
        Generate one of the fastest move path from `begin` to `end` with given
        `mode` using A* algorithm.

        Here I use python standard `heapq` package just for convenience. The
        idea of A* search algorithm is quite simple. It combines BFS with
        priority queue and adds some heuristic to accelerate the search.

        When an obstacle appears, you could modify the corresponding
        `self.cost[mode][row][col]` to some big number. Then this function is
        still able to give you one of the fastest path available.

        As you might have thought, it's ok to inject even more insights to the
        so-called `cost`. For example, when feeling a "threat", you could
        increase cost or add heuristic at some positions to avoid passing
        through these "dangerous" regions.

        For those who want to find all positions that can be reached within a
        given amount of time, Dijkstra is all you need. It's only BFS with
        priority queue. To find all positions that can be reached within a given
        amount of time, just modify the exit condition inside `search()` and
        save all valid positions during search. It's left for you brilliant
        developers to implement.

        :param begin: int
        :param end: int
        :param mode: int
        :return: List[int]
        """
        if (
            not self.is_valid(begin)
            or not self.is_valid(end)
            or not 0 <= mode < len(self.cost)
            or begin == end
        ):
            return []
        frontier = [(0, random.random(), begin)]
        cost_so_far = {begin: 0}
        came_from = {begin: None}

        def a_star_search():
            while frontier:
                _, _, cur = heapq.heappop(frontier)
                if cur == end:
                    break
                row, col = divmod(cur, 100)
                for neigh, edge_cost in self.cost[mode][row][col].items():
                    neigh_cost = cost_so_far[cur] + edge_cost
                    if neigh not in cost_so_far or neigh_cost < cost_so_far[neigh]:
                        cost_so_far[neigh] = neigh_cost
                        came_from[neigh] = cur
                        heuristic = self.get_distance(neigh, end)
                        heapq.heappush(
                            frontier, (neigh_cost + heuristic, random.random(), neigh)
                        )

        def reconstruct_path():
            path = []
            if end in came_from:
                cur = end
                while cur != begin:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
            return path

        a_star_search()
        return reconstruct_path()

    def get_grid_distance(self, center: int, distance_start: int, distance_end: int):
        """
        计算在地图上以center为圆心的距离为大于等于distance_start,小于等于distance_end的所有点
        :param source_grid: 基准点
        :param distance_start, distant_end: 距离
        :return: set
        """
        gridset = set()
        if not self.is_valid(center) or not 0 <= distance_start <= distance_end:
            return gridset
        if distance_end == 0:
            gridset.add(center)
            return gridset

        row = center // 100
        col = center % 100
        # 确定正方形的四个顶点
        row1 = min(row + distance_end + 1, self.max_row)
        row0 = max(row - distance_end, 0)
        col1 = min(col + distance_end + 1, self.max_col)
        col0 = max(col - distance_end, 0)

        for row_x in range(row0, row1):
            for col_x in range(col0, col1):
                pos = row_x * 100 + col_x
                if self.is_valid(pos):
                    dis = self.get_distance(center, pos)
                    if (dis >= distance_start) and (dis <= distance_end):
                        gridset.add(pos)
        return gridset

    def get_see_mode(self, unit1_type, unit2_type):
        """
        Get the see mode of `unit1` observing `unit2`.
        :return: int
        """
        if unit1_type == BopType.Aircraft:
            if unit2_type == BopType.Aircraft:
                return 1  # 低空对低空
            else:
                return 2  # 低空对地
        elif unit2_type == BopType.Aircraft:
            return -1  # 地对低空不可见
        else:
            return 0  # 地对地

    def can_observe(self, pos1, pos2, unit1_type, unit2_type):
        """
        Check if `pos1` can observe `pos2`.
        :return: bool
        """
        cond2 = self.basic[pos2 // 100][pos2 % 100]["cond"]
        unit1_ob_range = ob_range[unit1_type - 1][unit2_type - 1]
        if cond2 in [CondType.Jungle, CondType.City]:
            unit1_ob_range /= 2
        mode = self.get_see_mode(unit1_type, unit2_type)

        if self.get_distance(pos1, pos2) > unit1_ob_range:
            return False
        elif self.can_see(pos1, pos2, mode):
            return True
        else:
            return False

    def can_shoot(self, pos1, pos2, unit1_type, unit2_type):
        """
        Check if `pos1` can shoot `pos2` with given `mode`.
        :return: bool
        """
        if self.get_distance(pos1, pos2) > max_shoot_range[unit1_type - 1]:
            return False
        elif self.can_observe(pos1, pos2, unit1_type, unit2_type):
            return True
        else:
            return False

    def get_ob_mode(self, unit_type, target_type):
        """0-地面看步兵，1-地面看车辆，2-无人机看地面，-1-地面看无人机看不到"""
        if unit_type == BopType.Aircraft:
            return 2
        else:
            if target_type == BopType.Infantry:
                return 0
            elif target_type == BopType.Vehicle:
                return 1
            else:
                return -1

    def get_ob_area(self, center: int, unit_type: int, exclude_area=None):
        """
        Get the observation area of a unit.
        :return: set
        """
        # TODO: 待观察算子暂时只考虑车
        ob_area = []
        radius = ob_range[unit_type - 1][BopType.Vehicle - 1]
        max_area = self.get_grid_distance(center, 0, radius)
        if exclude_area:
            max_area -= exclude_area
        for h in list(max_area):
            if self.can_observe(center, h, unit_type, BopType.Vehicle):
                ob_area.append(h)
        return set(ob_area)

    def get_ob_area2(self, center: int, unit_type: int, target_type: int,
        passive=False, constrain_area=None):
        """
        调用算好的ob矩阵
        :return: set
        """              
        def idx2hex(idx):
            return idx // self.max_col * 100 + idx % self.max_col
              
        idx = center // 100 * self.max_col + center % 100
        if passive:
            mode = self.get_ob_mode(target_type, unit_type)
            if mode == -1:
                return set()
            ob_area_idx = np.where(self.ob[mode][:, idx])[0]
        else:
            mode = self.get_ob_mode(unit_type, target_type)
            if mode == -1:
                return set()
            ob_area_idx = np.where(self.ob[mode][idx, :])[0]
        ob_area_hex = [idx2hex(x) for x in ob_area_idx]
        ob_area = set(ob_area_hex)
        if constrain_area:
            ob_area &= constrain_area
        return ob_area

    def get_ob_area3(self, center: int, unit_type: int, target_type: int,
        passive=False, constrain_area=None):
        """
        人混阶段算好的ob矩阵可能不能调用，做个备保
        :return: set
        """
        def mode2radius(mode):
            # switch case to determine radius based on mode
            radius = {
                0: 10,
                1: 25,
                2: 2
            }.get(mode, 0)
            return radius
        if constrain_area is None:
            constrain_area = set()
        if (center, unit_type, target_type, passive, tuple(constrain_area)) in self.ob_area3.keys():
            ob_area = self.ob_area3[(center, unit_type, target_type, passive, tuple(constrain_area))]
        else:        
            ob_area_list = []
            if passive:
                mode = self.get_ob_mode(target_type, unit_type)
                radius = mode2radius(mode)
                cond = self.basic[center // 100][center % 100]["cond"]
                if cond in [CondType.Jungle, CondType.City]:
                    radius //= 2
                candidate = self.get_grid_distance(center, 0, radius)
                see_mode = self.get_see_mode(target_type, unit_type)
                for c in candidate:
                    if self.can_see(c, center, see_mode):
                        ob_area_list.append(c)
            else:
                mode = self.get_ob_mode(unit_type, target_type)
                radius = mode2radius(mode)
                candidate = self.get_grid_distance(center, 0, radius)
                for c in candidate:
                    if self.can_observe(center, c, unit_type, target_type):
                        ob_area_list.append(c)
            ob_area = set(ob_area_list)
            if constrain_area:
                ob_area &= constrain_area
            self.ob_area3[(center, unit_type, target_type, passive, tuple(constrain_area))] = ob_area
        return ob_area

    def get_shoot_area(self, center: int, unit_type: int, exclude_area=None):
        """
        Get the shoot area of a unit.
        :return: set
        """
        st_area = []
        radius = max_shoot_range[unit_type - 1]
        if radius == 0:
            return set()
        max_area = self.get_grid_distance(center, 0, radius)
        # 其实还没想好要这个干嘛
        if exclude_area:
            max_area -= exclude_area
        for h in list(max_area):
            # 前面已经用射程限制过范围了，能观察到肯定能打
            if self.can_observe(center, h, unit_type, BopType.Vehicle):
                st_area.append(h)
        return set(st_area)
