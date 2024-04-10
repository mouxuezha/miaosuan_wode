import json
import os
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZipFile

from train_env import CrossFireEnv, DefendEnv, ScoutEnv

from ai.agent import Agent

RED, BLUE, WHITE = 0, 1, -1


def make_env_by_track(track_id):
    if track_id == 1:
        env = CrossFireEnv(can_fire=True)
    elif track_id == 2:
        env = ScoutEnv(can_fire=True)
    elif track_id == 3:
        env = DefendEnv(my_unit_num=3, enemy_unit_num=3, enemy_style=1)
    return env


def rollout(track_id):
    env = make_env_by_track(track_id)
    agent = Agent()
    trajectories = []

    agent_info = {
        "user_id": 1,
        "user_name": "my_agent",
        "seat": "p1",  # 智能体的席位，比赛时由外部指定
    }
    state = env.setup(agent_info)
    trajectories.append(state[WHITE])
    setup_info = {
        "basic_data": env.basic_data,
        "cost_data": env.cost_data,
        "see_data": env.see_data,
        "scenario": None,  # 智能体选拔阶段无法获取想定信息
        "state": None,  # 智能体选拔阶段无法获取初始态势信息
        "faction": 0,
        "role": 0,
    }
    setup_info |= agent_info
    agent.setup(setup_info)

    done = False
    while not done:
        actions = agent.step(state[RED])
        state, done = env.step(actions)
        trajectories.append(state[WHITE])
    env.reset()
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    replay_name = f"track_{track_id}_{cur_time}"
    save_replay(replay_name, trajectories)


def save_replay(replay_name, trajectories):
    os.makedirs("replays/", exist_ok=True)
    zip_name = f"replays/{replay_name}.zip"
    with ZipFile(zip_name, "w", ZIP_DEFLATED, compresslevel=9) as z:
        for i, ob in enumerate(trajectories):
            data = json.dumps(ob, ensure_ascii=False, separators=(",", ":"))
            z.writestr(f"{replay_name}/{i}", data)


def main():
    # rollout(track_id=1)
    rollout(track_id=2)
    # rollout(track_id=3)


if __name__ == "__main__":
    main()
