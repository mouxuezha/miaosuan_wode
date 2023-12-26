# this is wode agent for miaosuan, define some layers.

from sdk.ai.agent import Agent

class agent_guize(Agent):
    def __init__(self):
        super().__init__()

        # abstract_state is useful
        self.absract_state = {} 

        self.act = [] # list to save all commands generated.

        self.observation = {}

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
        
        pass