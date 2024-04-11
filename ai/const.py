class ActType:
    Move = 1
    Shoot = 2
    GetOn = 3
    GetOff = 4
    Occupy = 5
    ChangeState = 6
    RemoveKeep = 7
    JMPlan = 8
    GuideShoot = 9
    StopMove = 10
    WeaponLock = 11
    WeaponUnlock = 12
    CancelJMPlan = 13
    Fork = 14
    Union = 15
    ChangeAltitude = 16
    ActivateRadar = 17
    EnterFort = 18
    ExitFort = 19
    LayMine = 20
    DeployGetOn = 303
    EndDeploy = 333


class BopType:
    Infantry = 1
    Vehicle = 2
    Aircraft = 3


class MoveType:
    Maneuver = 0
    March = 1
    Walk = 2
    Fly = 3


class TaskType:
    Defend = 208
    Scout = 209
    CrossFire = 210


class UnitSubType:
    Tank = 0
    IFV = 1
    Infantry = 2
    Artillery = 3
    UGV = 4
    UAV = 5
    AttackHelicopter = 6
    # 巡飞弹7
    # 运输直升机8
    # 侦察型战车9
    # 炮兵校射雷达车10
    # 人员战斗工事11
    # 车辆工事12
    # 布雷车13
    # 扫雷车14
    # 防空高炮15
    # 便携防空导弹排16
    # 车载防空导弹车17
    # 皮卡车18
    # 天基侦察算子19
    # 人员隐蔽工事20


class CondType:
    Plain = 0       # 开阔地
    Jungle = 1      # 丛林
    City = 2        # 居民地
    Soft = 3        # 松软地
    River = 4       # 河流
    Obstacle = 5    # 路障
