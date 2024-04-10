from .const import ActType


class ActionGenerator:
    """动作生成器，根据数据结构生成各类动作，不含智能体策略"""

    def __init__(self, seat) -> None:
        self.seat = seat

    def move(self, obj_id, move_path):
        return {
            "type": ActType.Move,
            "actor": self.seat,
            "obj_id": obj_id,
            "move_path": move_path,
        }

    def shoot(self, obj_id, target_id, weapon_id):
        """生成射击动作"""
        return {
            "type": ActType.Shoot,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
            "weapon_id": weapon_id,
        }

    def get_on(self, obj_id, target_id):
        """生成上车动作"""
        return {
            "type": ActType.GetOn,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
        }

    def get_off(self, obj_id, target_id):
        """生成下车动作"""
        return {
            "type": ActType.GetOff,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
        }

    def occupy(self, obj_id):
        """生成夺控动作"""
        return {"type": ActType.Occupy, "actor": self.seat, "obj_id": obj_id}

    def change_state(self, obj_id, target_state):
        """生成切换状态动作"""
        return {
            "type": ActType.ChangeState,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_state": target_state,
        }

    def remove_keep(self, obj_id):
        """生成移除压制动作"""
        return {"type": ActType.RemoveKeep, "actor": self.seat, "obj_id": obj_id}

    def jm_plan(self, obj_id, jm_pos, weapon_id):
        """生成间瞄射击动作"""
        return {
            "type": ActType.JMPlan,
            "actor": self.seat,
            "obj_id": obj_id,
            "jm_pos": jm_pos,
            "weapon_id": weapon_id,
        }

    def guide_shoot(self, obj_id, target_id, weapon_id, guided_id):
        """生成引导射击动作"""
        return {
            "type": ActType.GuideShoot,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
            "weapon_id": weapon_id,
            "guided_obj_id": guided_id,
        }

    def stop_move(self, obj_id):
        """生成停止机动动作"""
        return {"type": ActType.StopMove, "actor": self.seat, "obj_id": obj_id}

    def weapon_lock(self, obj_id):
        """生成武器锁定动作"""
        return {"type": ActType.WeaponLock, "actor": self.seat, "obj_id": obj_id}

    def weapon_unlock(self, obj_id):
        """生成武器展开动作"""
        return {"type": ActType.WeaponUnlock, "actor": self.seat, "obj_id": obj_id}

    def cancel_JM_plan(self, obj_id):
        """生成取消间瞄计划动作"""
        return {"type": ActType.CancelJMPlan, "actor": self.seat, "obj_id": obj_id}

    def fork(self, obj_id):
        """生成解聚动作"""
        return {"type": ActType.Fork, "actor": self.seat, "obj_id": obj_id}

    def union(self, obj_id, target_id):
        """生成聚合动作"""
        return {
            "type": ActType.Union,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
        }

    def change_altitude(self, obj_id, target_altitude):
        """生成改变高程动作"""
        return {
            "type": ActType.ChangeAltitude,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_altitude": target_altitude,
        }

    def activate_radar(self, obj_id):
        """生成开启校射雷达动作"""
        return {"type": ActType.ActivateRadar, "actor": self.seat, "obj_id": obj_id}

    def enter_fort(self, obj_id, target_id):
        """生成进入工事动作"""
        return {
            "type": ActType.EnterFort,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
        }

    def exit_fort(self, obj_id, target_id):
        """生成退出工事动作"""
        return {
            "type": ActType.ExitFort,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
        }

    def lay_mine(self, obj_id, target_pos):
        """生成布雷动作"""
        return {
            "type": ActType.LayMine,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_pos": target_pos,
        }

    def deploy_get_on(self, obj_id, target_id):
        """生成部署上车动作"""
        return {
            "type": ActType.DeployGetOn,
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": target_id,
        }

    def end_deploy(self):
        """生成结束部署动作"""
        return {"type": ActType.EndDeploy, "actor": self.seat}
