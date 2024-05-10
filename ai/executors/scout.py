import random
import numpy as np
import heapq

from ..const import ActType, BopType, CondType, MoveType

# 六边形网格的方向，从右开始逆时针，根据奇偶行数分两种情况
directions = [
    [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)],
    [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)],
]

render_type = {
  "attention": "put a ! mark",
  "sub_type": "put a 2d or 3d model",
  "target": "put a flag",
  "see_range": "render hex in green",
  "fire_range": "render hex in red",
  "custom": "can specify string to write and color to use"
}

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
        self.enemy_pos = {}  # {obj_id: [hex,last_fire_step]}
        self.units = {}  # {obj_id: [last, cur]}
        self.suspected = set()
        self.repeat_map = {} # 用于附加寻路代价，更倾向于探索未知区域
        self.threat_map = {} # 用于寻路代价值，更倾向于避开敌人射程
        self.owned_units = set()
        self.new_units = set()
        self.suspect_hist = [] # [[step, set()],]
        self.future_ob = {}  # {(start, end)): set()}

    def setup(self, task, agent):
        air_start = []
        print("ScoutExecutor init, agent info:")
        available_units = set(task["unit_ids"])
        if not available_units:  # 没有指定算子则使用全部算子
            available_units = set(agent.valid_units)
        for obj_id, unit in agent.valid_units.items():
            if obj_id not in available_units or agent.flag_act[obj_id]:
                continue  # 算子不参与此任务或已经生成了动作
            print(f"obj_id: {obj_id}, unit: {unit['type']}")
            if unit["type"] == BopType.Aircraft:
                self.air_num += 1
                self.air_traj[obj_id] = []
                air_start.append(unit["cur_hex"])
        rough_start = sum(air_start) // len(air_start)

        self.area = list(agent.map.get_grid_distance(task["hex"], 0, task["radius"]))
        self.area.sort()
        self.unscouted = set(self.area.copy())
        self.area2xy()
        self.allocate_traj(agent, rough_start, task["hex"])
        self.repeat_map = {key: 0 for key in self.area}
        self.threat_map = {key: 0 for key in self.area}
        self.owned_units = set(agent.owned.keys())

    def area2xy(self):
        """将侦察区域转换为直角坐标"""
        points = np.array([divmod(point, 100) for point in self.area])
        first_row = points[0][0]
        last_row = points[-1][0]
        self.xy_points = []
        for i in range(first_row, last_row + 1):
            self.xy_points.append(points[points[:, 0] == i])
        return self.xy_points

    def allocate_traj(self, agent, start, center):
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
                # total_traj.append(
                #     rc2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
                # )
                # total_traj.append(
                #     rc2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
                # )
                tmp_start = rc2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
                tmp_end = rc2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
            else:
                # total_traj.append(
                #     rc2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
                # )
                # total_traj.append(
                #     rc2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
                # )
                tmp_start = rc2hex(self.xy_points[j][-2][0], self.xy_points[j][-2][1])
                tmp_end = rc2hex(self.xy_points[j][1][0], self.xy_points[j][1][1])
            if total_traj:
                total_traj += agent.map.gen_move_route(total_traj[-1], tmp_start, 3)
            total_traj += agent.map.gen_move_route(tmp_start, tmp_end, 3)
            flag = 1 - flag

        # split = len(total_traj) // self.air_num + 1
        split = len(total_traj) // self.air_num
        obj_ids = list(self.air_traj.keys())
        # for i in range(len(total_traj) - 1):
        #     self.future_ob[(total_traj[i], total_traj[i + 1])] = self.future_ob_area(
        #         agent, agent.owned[obj_ids[0]], total_traj[i + 1], total_traj[i]
        #     )
        for i in range(len(obj_ids)):
            # self.air_traj[obj_ids[i]] = total_traj[split * i : split * (i + 1)]
            self.air_traj[obj_ids[i]] = total_traj[split * i : split * (i + 1) + 1]
            print(f"obj_id: {obj_ids[i]}, traj: {self.air_traj[obj_ids[i]]}")

    def future_ob_area(self, agent, unit, end, begin=None):
        ob_area = set()
        path = self.my_a_star(agent, unit, end, begin)
        for point in path:
            ob_area |= agent.map.get_ob_area3(
                point, BopType.Aircraft, BopType.Vehicle
            )
        return ob_area

    def update_air_traj(self, agent):
        for obj_id, traj in self.air_traj.items():
            # while traj and traj[0] not in self.unscouted:
            #     point = traj.pop(0)
                # print(f"------point {point} has been observed------")
            if obj_id not in agent.owned.keys():
                continue
            for i in range(len(traj)-1, -1, -1):
                if i == 0:
                    last_hex = agent.owned[obj_id]["cur_hex"]
                else:
                    last_hex = traj[i-1]
                tmp_future_ob = self.future_ob.get((last_hex, traj[i]))
                if not tmp_future_ob:
                    tmp_future_ob = self.future_ob_area(agent, agent.owned[obj_id], traj[i], last_hex)
                    self.future_ob[(last_hex, traj[i])] = tmp_future_ob
                if not (self.unscouted & tmp_future_ob):
                    # print(f"remove {traj[i]} from obj{obj_id} traj")           
                    traj.pop(i)

    def update_unit(self, obj_id, cur_hex):
        """更新算子的当前和上一格位置信息"""
        if obj_id not in self.units.keys():
            self.units[obj_id] = [-1, cur_hex]
        else:
            self.units[obj_id][0] = self.units[obj_id][1]
            self.units[obj_id][1] = cur_hex

    def update_unscouted(self, agent, cur_hex, unit_type):
        # scouted = set(self.area) - self.unscouted
        # new_ob = agent.map.get_ob_area(cur_hex, unit_type, scouted) & set(self.area)
        new_ob = self.unscouted & agent.map.get_ob_area3(
            cur_hex, unit_type, BopType.Vehicle)
        last_unscout_num = len(self.unscouted)
        self.unscouted -= new_ob
        cur_unscout_num = len(self.unscouted)
        
        # 顺道把奖励地图更新了
        for h in new_ob:
            self.repeat_map[h] += 0.2
        if cur_hex in self.area:
            self.repeat_map[cur_hex] += 0.4
        
        # 顺道把可疑区域一块更新了
        last_suspect_num = len(self.suspected)
        # self.suspected -= new_ob
        excluded_suspect = self.suspected & new_ob
        self.suspected -= excluded_suspect
        cur_suspect_num = len(self.suspected)
        
        # if last_suspect_num > cur_suspect_num and cur_suspect_num > 0:
        #     self.re_allocate_air(agent)

        # TODO：若无人机后续路径点已被侦察到，则重新规划路径？
        # 好像还是得写一个普适的任意区域生成扫描路径的函数，但感觉比较困难
        
        # 丑陋的air_traj更新1
        traj_ch_flag = False
        if last_suspect_num > cur_suspect_num:
            for obj_id, traj in self.air_traj.items():                
                while traj and traj[0] in excluded_suspect:
                    point = traj.pop(0)
                    print(f"------point {point} has been observed------")
                    traj_ch_flag = True
        return traj_ch_flag, last_unscout_num > cur_unscout_num            
                    
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
        # # assert len(self.units) - len(cur_units) == 1
        # old_units = set(self.units.keys())
        # diff = list(old_units - cur_units)
        # for missed_unit in diff:
        #     area_last = self.can_you_shoot_me(agent, self.units[missed_unit][0])
        #     area_cur = self.can_you_shoot_me(agent, self.units[missed_unit][1])
        #     tmp_suspect = area_cur - area_last & self.unscouted - set(self.enemy_pos.values())
        # # tmp_suspect = area_cur - area_last & self.unscouted
        #     if len(self.suspected) == 0:
        #         self.suspected = tmp_suspect
        #     else:
        #         inter = self.suspected & tmp_suspect
        #         if len(inter) > 0:
        #             # 可能是两簇角点导致误判，暂时只通过精准化unscouted减小suspected范围
        #             self.suspected = inter
        #         else:
        #             self.suspected = self.suspected | tmp_suspect
        #     self.units.pop(missed_unit)
        #     print(f"missed unit: {missed_unit}, tmp suspect num: {len(tmp_suspect)}, final suspect num: {len(self.suspected)}")
        old_units = set(self.units.keys())
        diff = list(old_units - cur_units)
        for missed_unit in diff:
            area_last = self.can_you_shoot_me(agent, self.units[missed_unit][0])
            area_cur = self.can_you_shoot_me(agent, self.units[missed_unit][1])
            tmp_suspect = area_cur - area_last
            for k, v in self.enemy_pos.items():
                if v[0] in tmp_suspect and (self.num - v[1]) >= 75:
                    tmp_suspect = set()
                    break
            tmp_suspect &= self.unscouted
            self.suspect_hist.append([self.num, tmp_suspect])
            # suspect更新1
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
            if missed_unit in self.air_traj.keys():
                self.air_traj.pop(missed_unit)
            # self.car_to_detect.pop(missed_unit) # TODO: check for pop necessity
            print(
                f"step {self.num} missed unit: {missed_unit}, tmp suspect num: {len(tmp_suspect)}, final suspect num: {len(self.suspected)}"
            )
        # for p in self.suspected:
        #     self.ob_suspect |= agent.map.get_ob_area3(
        #         p, BopType.Vehicle, BopType.Vehicle,
        #         True, set(self.area)
        #     )

    def re_allocate_air(self, agent, available_units):
        """
        根据可疑区域重新分配无人机的侦察路径
        有车被打掉了或者suspected范围减小后调用
        """
        # TODO:这里假设只有一簇可疑区域，派遣一架无人机的情况
        tmp = list(self.suspected)
        center = sum(tmp) // len(tmp)
        dist = []
        for obj_id, unit in agent.owned.items():
            if obj_id not in available_units:
                continue
            if unit["type"] == BopType.Aircraft:
                dist.append([obj_id, agent.map.get_distance(unit["cur_hex"], center)])
        dist.sort(key=lambda x: x[1])
        # new_traj_point = self.get_nearest(agent, agent.owned[dist[0][0]]["cur_hex"], tmp)
        obj_id_allo = dist[0][0]
        new_traj_point, _ = self.get_nearest(
            agent, agent.owned[obj_id_allo]["cur_hex"], tmp
        )
        
        # 找到原路径中离待插入点最近的，在其后插入？
        # old_traj = self.air_traj[dist[0][0]]
        # point_before_insert = self.get_nearest(agent, new_traj_point, old_traj)
        # idx = old_traj.index(point_before_insert)
        # self.air_traj[dist[0][0]].insert(idx + 1, new_traj_point)
        
        # 有可疑区域优先探索，一次只插入一个
        # if new_traj_point not in self.air_traj[dist[0][0]] or \
        #     new_traj_point not in agent.owned[dist[0][0]]["move_path"]:
        if not self.air_traj[obj_id_allo] or new_traj_point not in self.air_traj[obj_id_allo]:
            self.air_traj[dist[0][0]].insert(0, new_traj_point)
            # self.suspected.remove(new_traj_point)
            # print(f"***reallocate obj: {dist[0][0]}, new point: {new_traj_point}***")
            print(
                f"***reallocate obj: {obj_id_allo}, new traj: {self.air_traj[obj_id_allo][:4]}***"
            )
            
    def my_a_star(self, agent, unit, end, begin=None):
        move_type = decide_move_type(unit)
        if not begin:
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
        cur_enemy = set(agent.enemy.keys())
        old_enemy = set(self.enemy_pos.keys())
        for obj_id in cur_enemy:
            enemy_hex = agent.enemy[obj_id]["cur_hex"]
            enemy_type = agent.enemy[obj_id]["type"]
            if obj_id in old_enemy and enemy_hex == self.enemy_pos[obj_id][0]:
                pass
            else: # 新敌人或者敌人位置变化
                if obj_id not in old_enemy:
                    print(f"!!!!!! step {self.num} find new enemy {obj_id} at {enemy_hex}!!!!!!")
                else: # 敌人位置变化，去掉旧的
                    self.update_enemy_threat_area(agent, self.enemy_pos[obj_id][0], enemy_type, -1)
                    msg_ids_to_del = []
                    for x in agent.status["communication"]:
                        if x["type"] == 205 and str(obj_id) in x["msg_body"]["description"]:
                            msg_ids_to_del.append(x["msg_id"])
                    agent.batch_del_xuanran(msg_ids_to_del)
                # 新渲染                    
                word = str(obj_id) + '\n' + agent.enemy[obj_id]["name"]#MyUnitSubType[agent.enemy[obj_id]["sub_type"]]
                render = {
                    "actor": agent.seat,
                    "type": 205,
                    "msg_body": {
                    "hexs": [enemy_hex], 
                    "graphic_type": "custom",
                    "word": word,
                    "color": "#ffffff",
                    "description": str(enemy_hex) + " " + word
                    }  
                }
                agent.act.append(render)
                if enemy_type in [BopType.Infantry, BopType.Vehicle]:
                    enemy_ob_area = list(agent.map.get_ob_area3(enemy_hex, enemy_type, BopType.Vehicle))
                    shoot_area = list(agent.map.get_shoot_area(enemy_hex, enemy_type))
                    ob_render = {
                        "actor": agent.seat,
                        "type": 205,
                        "msg_body": {
                            "hexs": enemy_ob_area, 
                            "graphic_type": "see_range",
                            "description": str(obj_id) + " " + str(enemy_hex) + " ob range"
                        }  
                    }
                    shoot_render = {
                        "actor": agent.seat,
                        "type": 205,
                        "msg_body": {
                            "hexs": shoot_area, 
                            "graphic_type": "fire_range",
                            "description": str(obj_id) + " " + str(enemy_hex) + " fire range"
                        }  
                    }
                    agent.act.append(ob_render)
                    agent.act.append(shoot_render)
                # suspect更新2
            
            last_fire_step = 0
            if enemy_type in [BopType.Infantry, BopType.Vehicle]:
                ids_to_pop = []
                for i in range(len(self.suspect_hist)):
                    if enemy_hex in self.suspect_hist[i][1]:
                        self.suspected -= self.suspect_hist[i][1]
                        for p in self.suspect_hist[i][1]:
                            self.ob_suspect &= agent.map.get_ob_area3(
                                p, BopType.Vehicle, BopType.Vehicle,
                                True, set(self.area)
                            )
                        print(f"remove {len(self.suspect_hist[i][1])} suspected points, remaining: {len(self.suspected)}")
                        last_fire_step = self.suspect_hist[i][0]
                        ids_to_pop.append(i)
                for i in ids_to_pop[::-1]:
                    self.suspect_hist.pop(i)        
            self.enemy_pos[obj_id] = [enemy_hex, last_fire_step]
            self.update_enemy_threat_area(agent, enemy_hex, enemy_type, 1)
        # elif len(cur_enemy) < len(old_enemy):
        #     lost_enemy = old_enemy - cur_enemy
        #     for obj_id in lost_enemy:
        #         enemy_hex = self.enemy_pos.pop(obj_id)
        #         self.update_enemy_threat_area(agent, enemy_hex, -1)
        #     return -1

    
    def update_enemy_threat_area(self, agent, enemy_hex, enemy_type, coeff=1):
        """
        根据敌人所处地形更新威胁地图，更新值是拍脑袋定的
        """
        st_area = agent.map.get_shoot_area(enemy_hex, enemy_type) & set(self.area)
        for h in st_area:
            if not agent.map.can_observe(h, enemy_hex, BopType.Vehicle, enemy_type):
                self.threat_map[h] += 0.4 * coeff
            else:
                self.threat_map[h] += 0.2 * coeff
    
    def execute(self, task, agent):
        """
        侦察执行逻辑
        """
        # 无人机的侦察逻辑，开始按照分配的路径点依次移动，有可疑区域优先探索
        # 可疑区域探索完毕后，逐个探索未探测区域离其当时位置最远的点
        def air_scout(obj_id, cur_hex):
            if len(self.air_traj[obj_id]):
                destination = self.air_traj[obj_id].pop(0)
            elif self.unscouted:
                destination = self.get_farthest(agent, cur_hex, self.unscouted)
            else:
                destination = random.choice(self.area)
            return destination
        
        # 车辆的侦察逻辑，优先可疑区域其次未探索，暂时随机选点，后续避开已知敌人射程？
        def vehicle_scout():
            if self.suspected:
                to_detect = list(self.suspected)
            elif self.unscouted:
                to_detect = list(self.unscouted)
            else:
                to_detect = self.area
            destination = random.choice(to_detect)
            return destination
        
        self.num = agent.num
        if self.num == 0:
            self.setup(task, agent)
            
        if not self.area:
            print("ScoutExecutor: area is empty")
            return  # 侦察区域不能为空
        
        self.check_enemy(agent)
        
        if len(agent.owned) > len(self.owned_units):
            self.new_units |= set(agent.owned.keys()) - self.owned_units
        available_units = set(task["unit_ids"])
        if not available_units:  # 没有指定算子则使用全部算子
            available_units = set(agent.valid_units)
        available_units |= self.new_units
          
        if len(available_units) < len(self.units):
            self.guess_enemy(available_units, agent)
            if self.suspected:
                self.re_allocate_air(agent, available_units)
        
        # if self.num in [74,75,150,151]:
        #     # 解聚完了看看
        #     print(f"available: {len(available_units)}, owned: {len(agent.owned)}, valid: {len(agent.valid_units)}")
    
        # 遍历更新探测状态
        air_traj_change = False
        for obj_id, unit in agent.valid_units.items():
            if obj_id not in available_units or agent.flag_act[obj_id]:
                continue  # 算子不参与此任务或已经生成了动作
            
            if unit["cur_pos"] == 0: # 完成一格移动
                cur_hex = unit["cur_hex"]
                self.update_unit(obj_id, cur_hex)
                traj_ch, _ = self.update_unscouted(agent, cur_hex, unit["type"])
                air_traj_change |= traj_ch
                # print(f"remain: {len(self.unscouted)}")
                # 丑陋的1air_traj更新2
                if unit["type"] == BopType.Aircraft:
                    if self.air_traj[obj_id] and cur_hex == self.air_traj[obj_id][0]:
                        self.air_traj[obj_id].pop(0)
                        
        # air_traj更新3
        self.update_air_traj(agent)
        # for obj_id, traj in self.air_traj.items():
        #     print(f"obj_id: {obj_id}, traj: {len(traj)}")
        if air_traj_change and self.suspected:
            # print("reallocate air")
            self.reallocate_air(agent)

        # 添加动作
        for obj_id, unit in agent.valid_units.items():
            if obj_id not in available_units or agent.flag_act[obj_id]:
                continue  # 算子不参与此任务或已经生成了动作
            
            if unit["type"] == BopType.Vehicle and ActType.Fork in agent.valid_actions[obj_id]:
                # 车辆优先解聚
                agent.act.append(agent.act_gen.fork(obj_id))
                agent.flag_act[obj_id] = True
                continue

            if ActType.Move not in agent.valid_actions[obj_id]:
                continue  # 算子正在机动，不再生成机动动作
                        
            if unit["type"] == BopType.Aircraft:
                destination = air_scout(obj_id, unit["cur_hex"])
            else:
                destination = vehicle_scout()
            
            # route = agent.gen_move_route(unit, int(destination))
            route = self.my_a_star(agent, unit, int(destination))
            if route:
                agent.act.append(agent.act_gen.move(obj_id, route))
                agent.flag_act[obj_id] = True
