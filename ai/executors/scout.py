import random
import numpy as np
import heapq
from sklearn.cluster import KMeans

from ..const import ActType, BopType, CondType, MoveType

# 六边形网格的方向，从右开始逆时针，根据奇偶行数分两种情况
directions = [
    [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)],
    [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)],
]


def hex2rc(hex_coord):
    """将六边形坐标转换为直角坐标"""
    col = hex_coord % 100
    row = hex_coord // 100
    return col, row


def rc2hex(row, col):
    """将直角坐标转换为六边形坐标"""
    return row * 100 + col


def rc2qrs(row, col):
    """将直角坐标转换为立方体坐标"""
    q = col - (row - (row & 1)) // 2
    r = row
    s = -q - r
    return q, r, s


def qrs2rc(q, r, s):
    """将立方体坐标转换为直角坐标"""
    row = r
    col = q + (r - (r & 1)) // 2
    return row, col


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
        rc2hex(center_row, col_max_center),
        rc2hex(first_row, col_max_first),
        area[0],
        rc2hex(center_row, col_min_center),
        rc2hex(last_row, col_min_last),
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


def get_direction(start, end):
    """
    用立方体坐标粗暴地算一个方向
    0-东，1-东北，2-西北，3-西，4-西南，5-东南，6-北，7-南
    """
    row1, col1 = divmod(start, 100)
    q1, r1, s1 = rc2qrs(row1, col1)

    row2, col2 = divmod(end, 100)
    q2, r2, s2 = rc2qrs(row2, col2)

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


def decide_move_type(unit):
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
    return move_type


class ScoutExecutor:
    """执行侦察任务的策略"""

    def __init__(self) -> None:
        self.area = []
        self.xy_points = [] # [np.array,]
        self.unscouted = set()
        self.air_num = 0
        self.air_traj = {}
        self.enemy_pos = {}  # {obj_id: hex}
        self.units = {}  # {obj_id: [last, cur]}
        self.suspected = set()
        self.repeat_map = {} # 用于附加寻路代价，更倾向于探索未知区域
        self.threat_map = {} # 用于寻路代价值，更倾向于避开敌人射程
        self.air_ob = {}
        self.car_ob = {}
        self.car_to_detect = set()
        self.car_target = None
        self.car_cluster = [] # [hex, chosen_flag]

    def setup(self, task, agent):
        air_start = []
        print("ScoutExecutor init, agent info:")
        for obj_id, unit in agent.owned.items():
            print(f"obj_id: {obj_id}, unit: {unit['type']}")
            if unit["type"] == BopType.Aircraft:
                self.air_num += 1
                self.air_traj[obj_id] = []
                air_start.append(unit["cur_hex"])
        rough_start = sum(air_start) // len(air_start)

        self.area = list(agent.map.get_grid_distance(task["hex"], 0, task["radius"]))
        self.area.sort()
        self.unscouted = set(self.area.copy())
        for point in self.area:
            air_ob_area = agent.map.get_ob_area2(point, BopType.Aircraft, BopType.Vehicle)
            self.air_ob[point] = len(air_ob_area)
            car_ob_area = agent.map.get_ob_area2(point, BopType.Vehicle, BopType.Vehicle)
            self.car_ob[point] = len(car_ob_area)
        self.max_air_ob_num = max(self.air_ob.values())
        self.max_car_ob_num = max(self.car_ob.values())
        self.area2xy()
        self.allocate_traj(rough_start, task["hex"])
        self.repeat_map = {key: 0 for key in self.area}
        self.threat_map = {key: 0 for key in self.area}
        # self.qrs_points = np.array([rc2qrs(hex2rc(point)) for point in self.area])       

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
                    rc2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
                )
                total_traj.append(
                    rc2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
                )
            else:
                total_traj.append(
                    rc2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
                )
                total_traj.append(
                    rc2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
                )
            flag = 1 - flag
        print(f"total traj: {total_traj}")
        split = len(total_traj) // self.air_num + 1
        obj_ids = list(self.air_traj.keys())
        for i in range(len(obj_ids)):
            self.air_traj[obj_ids[i]] = total_traj[split * i : split * (i + 1)]
            print(f"obj_id: {obj_ids[i]}, traj: {self.air_traj[obj_ids[i]]}")

    def allocate_traj_hex(self, start, center):
        # 方向向量，模长均为2
        q_vec = np.array([3**0.5, 1])
        r_vec = np.array([0, -2])
        s_vec = np.array([-3**0.5, 1])
        vecs = [q_vec, r_vec, s_vec]
        
        r_start, c_start = divmod(start, 100)
        r_center, c_center = divmod(center, 100)
        m_vec = np.array([r_center - r_start, c_start - c_center])
        
        # 计算与m_vec点积最大的方向
        dot_prods = [abs(np.dot(m_vec, vec)) for vec in vecs]
        direc = dot_prods.index(max(dot_prods))
        sort_idx = np.argsort(self.qrs_points[:, direc])
        if np.dot(m_vec, vecs[direc]) > 0:
            sort_idx = sort_idx[::-1]
        sorted_qrs_points = self.qrs_points[sort_idx]
        
        first = sorted_qrs_points[0][direc]
        last = sorted_qrs_points[-1][direc]
        
    def update_unit(self, obj_id, cur_hex):
        """更新算子的当前和上一格位置信息"""
        if obj_id not in self.units.keys():
            self.units[obj_id] = [-1, cur_hex]
        else:
            self.units[obj_id][0] = self.units[obj_id][1]
            self.units[obj_id][1] = cur_hex
            
    def update_cluster(self, agent):
        n = len(self.units) - self.air_num
        if n == 0:
            return []
        self.car_target = []
        for point in self.car_to_detect:
            r, c = divmod(point, 100)
            c += 0.5 * (r & 1)
            obed_area = agent.map.get_ob_area2(point, BopType.Vehicle, BopType.Vehicle, passive=True)
            self.car_target.append([r, c, self.car_ob[point] / len(obed_area)])
        self.car_target = np.array(self.car_target)  
        kmeans = KMeans(n_clusters=n).fit(self.car_target[:, :2])
        clusters = kmeans.labels_
        
        self.car_cluster = [] # [hex, chosen_flag]
        for i in range(n):
            clusters_points = self.car_target[clusters == i]
            idx = np.argmax(clusters_points[:,2])
            r, c, _ = clusters_points[idx]
            point = int(r) * 100 + int(c)
            self.car_cluster.append([point, 0])
        print(self.car_cluster)    

    def update_unscouted(self, agent, cur_hex, unit_type):
        scouted = set(self.area) - self.unscouted
        new_ob = agent.map.get_ob_area(cur_hex, unit_type, scouted) & set(self.area)
        last_unscout_num = len(self.unscouted)
        self.unscouted -= new_ob
        cur_unscout_num = len(self.unscouted)
        
        # 顺道把奖励地图更新了
        for h in new_ob:
            self.repeat_map[h] += 0.2
        if cur_hex in self.area:
            self.repeat_map[cur_hex] += 0.2
        
        # 顺道把可疑区域一块更新了
        last_suspect_num = len(self.suspected)
        self.suspected -= new_ob
        cur_suspect_num = len(self.suspected)
        
        # 丑陋的air_traj更新1
        if last_suspect_num > cur_suspect_num:
            for obj_id, traj in self.air_traj.items():
                if traj and traj[0] in new_ob:
                    traj.pop(0)
                    if cur_suspect_num > 0:
                        self.re_allocate_air(agent)
        return last_suspect_num > cur_suspect_num or last_unscout_num > cur_unscout_num
                        
    def can_you_shoot_me(self, agent, cur_hex):
        cond = agent.map.basic[cur_hex // 100][cur_hex % 100]["cond"]
        radius = 12 if cond in [CondType.Jungle, CondType.City] else 20
        area = list(agent.map.get_grid_distance(cur_hex, 0, radius))
        area.sort()
        shoot_area = []
        for h in area:
            if agent.map.can_see(cur_hex, h, 0):
                shoot_area.append(h)
        return set(shoot_area)

    def get_nearest(self, agent, cur_hex, to_detect):
        """获取待探测区域最近的点"""
        min_dist = 1000
        nearest_hex = -1
        for h in to_detect:
            dist = agent.map.get_distance(cur_hex, h)
            if dist < min_dist:
                min_dist = dist
                nearest_hex = h
        return nearest_hex
    
    def get_farthest(self, agent, cur_hex, to_detect):
        """获取待探测区域最远的点"""
        max_dist = 0
        farthest_hex = -1
        for h in to_detect:
            dist = agent.map.get_distance(cur_hex, h)
            if dist > max_dist:
                max_dist = dist
                farthest_hex = h
        return farthest_hex    

    def guess_enemy(self, cur_units, agent):
        old_units = set(self.units.keys())
        diff = list(old_units - cur_units)
        for missed_unit in diff:
            area_last = self.can_you_shoot_me(agent, self.units[missed_unit][0])
            area_cur = self.can_you_shoot_me(agent, self.units[missed_unit][1])
            tmp_suspect = area_cur - area_last & self.unscouted - set(self.enemy_pos.values())

            if len(self.suspected) == 0:
                self.suspected = tmp_suspect
            else:
                inter = self.suspected & tmp_suspect
                if len(inter) > 0:
                    # 可能是两簇角点导致误判，暂时只通过精准化unscouted减小suspected范围
                    self.suspected = inter
                else:
                    self.suspected = self.suspected | tmp_suspect
            self.units.pop(missed_unit)
            print(f"missed unit: {missed_unit}, tmp suspect num: {len(tmp_suspect)}, final suspect num: {len(self.suspected)}")

    def re_allocate_air(self, agent):
        """
        根据可疑区域重新分配无人机的侦察路径
        有车被打掉了或者suspected范围减小后调用
        """
        # TODO:这里假设只有一簇可疑区域，派遣一架无人机的情况
        tmp = list(self.suspected)
        center = sum(tmp) // len(tmp)
        dist = []
        for obj_id, unit in agent.owned.items():
            if unit["type"] == BopType.Aircraft:
                dist.append([obj_id, agent.map.get_distance(unit["cur_hex"], center)])
        dist.sort(key=lambda x: x[1])
        new_traj_point = self.get_nearest(agent, agent.owned[dist[0][0]]["cur_hex"], tmp)
        
        # 找到原路径中离待插入点最近的，在其后插入？
        # old_traj = self.air_traj[dist[0][0]]
        # point_before_insert = self.get_nearest(agent, new_traj_point, old_traj)
        # idx = old_traj.index(point_before_insert)
        # self.air_traj[dist[0][0]].insert(idx + 1, new_traj_point)
        
        # 有可疑区域优先探索，一次只插入一个
        if new_traj_point not in self.air_traj[dist[0][0]] or \
            new_traj_point not in agent.owned[dist[0][0]]["move_path"]:
            self.air_traj[dist[0][0]].insert(0, new_traj_point)
            # self.suspected.remove(new_traj_point)
            print(f"***reallocate obj: {dist[0][0]}, new point: {new_traj_point}***")
   
    def my_a_star(self, agent, unit, end):
        move_type = decide_move_type(unit)
        begin = unit["cur_hex"]
        
        frontier = [(0, random.random(), begin)]
        cost_so_far = {begin: 0}
        came_from = {begin: None}
        
        def a_star_search():
            while frontier:
                _, _, cur = heapq.heappop(frontier)
                if cur == end:
                    break
                row, col = divmod(cur, 100)
                for neigh, edge_cost in agent.map.cost[move_type][row][col].items():
                    neigh_cost = cost_so_far[cur] + edge_cost
                    if neigh in self.area:
                        if unit["type"] == BopType.Vehicle:
                            neigh_cost += self.threat_map[neigh]
                            neigh_cost -= self.car_ob[neigh] / self.max_car_ob_num / 5
                        # TODO: 这个else可能不需要
                        else:
                            neigh_cost -= self.air_ob[neigh] / self.max_air_ob_num / 5
                        neigh_cost += self.repeat_map[neigh]
                    else:
                        neigh_cost += 0.5
                    if neigh not in cost_so_far or neigh_cost < cost_so_far[neigh]:
                        cost_so_far[neigh] = neigh_cost
                        came_from[neigh] = cur
                        heuristic = agent.map.get_distance(neigh, end)
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

    def check_enemy(self, agent):
        """
        检查agent.enemy变化情况，更新敌人位置和威胁地图
        """
        
        def update_enemy_threat_area(agent, enemy_hex, coeff=1):
            """
            根据敌人所处地形更新威胁地图，更新值是拍脑袋定的
            """
            st_area = agent.map.get_shoot_area(enemy_hex, BopType.Vehicle) & set(self.area)
            for h in st_area:
                if not agent.map.can_observe(h, enemy_hex, BopType.Vehicle, BopType.Vehicle):
                    self.threat_map[h] += 0.4 * coeff
                else:
                    self.threat_map[h] += 0.2 * coeff
                    
        cur_enemy = set(agent.enemy.keys())
        old_enemy = set(self.enemy_pos.keys())
        if len(cur_enemy) > len(old_enemy):
            new_enemy = cur_enemy - old_enemy
            for obj_id in new_enemy:
                enemy_hex = agent.enemy[obj_id]["cur_hex"]
                self.enemy_pos[obj_id] = enemy_hex
                update_enemy_threat_area(agent, enemy_hex, 1)
        elif len(cur_enemy) < len(old_enemy):
            lost_enemy = old_enemy - cur_enemy
            for obj_id in lost_enemy:
                enemy_hex = self.enemy_pos.pop(obj_id)
                update_enemy_threat_area(agent, enemy_hex, -1)
  
    def execute(self, task, agent):
        """
        侦察执行逻辑
        """
        if agent.time.cur_step < 3:
            self.setup(task, agent)

        if not self.area:
            print("ScoutExecutor: area is empty")
            return  # 侦察区域不能为空
        
        self.check_enemy(agent)
        
        if len(agent.owned) < len(self.units):
            cur_units = set(agent.owned.keys())
            self.guess_enemy(cur_units, agent)
            if self.suspected:
                self.re_allocate_air(agent)

        available_units = set(task["unit_ids"])
        if not available_units:  # 没有指定算子则使用全部算子
            available_units = set(agent.valid_units)
        
        change_flag = False
        move_flag = False
        for obj_id, unit in agent.valid_units.items():
            if unit["cur_pos"] == 0: # 完成一格移动
                cur_hex = unit["cur_hex"]
                self.update_unit(obj_id, cur_hex)
                change_flag |= self.update_unscouted(agent, cur_hex, unit["type"])
                # 丑陋的air_traj更新2
                if unit["type"] == BopType.Aircraft:
                    if self.air_traj[obj_id] and cur_hex == self.air_traj[obj_id][0]:
                        self.air_traj[obj_id].pop(0)
                # print(f"remain: {len(self.unscouted)}")
            if unit["type"] == BopType.Vehicle and ActType.Move in agent.valid_actions[obj_id]:
                move_flag = True
        
        self.car_to_detect = set()
        for point in self.suspected:
            self.car_to_detect |= agent.map.get_ob_area2(point, BopType.Vehicle, BopType.Vehicle, passive=True)
        self.car_to_detect |= self.unscouted
        self.car_to_detect &= set(self.area)
        if change_flag and move_flag and agent.time.cur_step > 151:
            self.update_cluster(agent)
                
        for obj_id, unit in agent.valid_units.items():
            if obj_id not in available_units or agent.flag_act[obj_id]:
                continue  # 算子不参与此任务或已经生成了动作

            if unit["type"] == BopType.Vehicle and ActType.Fork in agent.valid_actions[obj_id]:
                # 车辆优先解聚
                agent.actions.append(agent.act_gen.fork(obj_id))
                agent.flag_act[obj_id] = True
                continue

            if ActType.Move not in agent.valid_actions[obj_id]:
                continue  # 算子正在机动，不再生成机动动作
            
            # 无人机的侦察逻辑，开始按照分配的路径点依次移动，有可疑区域优先探索
            # 可疑区域探索完毕后，逐个探索未探测区域离其当时位置最远的点
            def air_scout(obj_id, cur_hex):
                if len(self.air_traj[obj_id]):
                    destination = self.air_traj[obj_id][0]
                else:
                    destination = self.get_farthest(agent, cur_hex, self.unscouted)
                return destination
            
            # 车辆的侦察逻辑，优先可疑区域其次未探索，暂时随机选点，后续避开已知敌人射程？
            def vehicle_scout():
                if self.car_cluster:
                    tmp = [x[0] for x in self.car_cluster if x[1] == 0]
                    destination = self.get_nearest(agent, cur_hex, tmp)
                    print(f"destination: {destination}")
                    for i in range(len(self.car_cluster)):
                        if self.car_cluster[i][0] == destination:
                            self.car_cluster[i][1] = 1
                            break
                else:
                    destination = random.choice(list(self.car_to_detect))
                return destination
            
            if unit["type"] == BopType.Aircraft:
                destination = air_scout(obj_id, unit["cur_hex"])
            else:
                destination = vehicle_scout()
            
            # route = agent.gen_move_route(unit, int(destination))
            route = self.my_a_star(agent, unit, int(destination))
            if route:
                # agent.actions.append(agent.act_gen.move(obj_id, route))
                agent.actions.append(agent.act_gen.move(obj_id, route[:2]))
                agent.flag_act[obj_id] = True
