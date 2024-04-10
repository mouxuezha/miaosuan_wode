from ..const import ActType


class CrossFireExecutor:
    """执行集结任务的策略"""

    def __init__(self) -> None:
        pass

    def execute(self, task, agent):
        """所有参与任务的算子沿最短路径机动的简单策略"""
        available_units = set(task["unit_ids"])
        if not available_units:  # 没有指定算子则使用全部算子
            available_units = set(agent.valid_units)
        for obj_id, unit in agent.valid_units.items():
            if obj_id not in available_units or agent.flag_act[obj_id]:
                continue  # 算子不参与此任务或已经生成了动作
            if ActType.Move not in agent.valid_actions[obj_id]:
                continue  # 算子正在机动，不再生成机动动作
            route = agent.gen_move_route(unit, task["hex"])
            if route:
                agent.actions.append(agent.act_gen.move(obj_id, route))
                agent.flag_act[obj_id] = True
