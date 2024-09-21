from pddlgym import core
import numpy as np
import random 
import pddlgym
import torch

class Translator:
    
    """
        Every needed communication with PDDLGym is managed using a Translator object, 
        which is used as a wrapper with PDDLGym.
    """

    def __init__(
        self,
        state_space,
        action_space,
        initial_planning_state,
        env,
        device='cpu'
    ) -> None:
        self.env = env
        self.state_space = state_space
        self.action_space = action_space+['timestep']

        self.action_space_repr = np.zeros((len(self.action_space),len(self.action_space)))
        for i,a in enumerate(action_space):
            self.action_space_repr[i] = np.in1d(list(self.action_space),[a]).astype(int)
        self.action_space_repr = torch.tensor(self.action_space_repr,dtype=torch.float).to(torch.device(device))

        self.curr_state = initial_planning_state
        self.curr_S = set()

        self.state_repr = np.in1d(list(self.state_space),list(self.curr_state.literals)).astype(int)
        self.action_repr = np.zeros(len(self.action_space), dtype=int)
    
    def getSizes(self):
        return len(self.state_space), len(self.action_space)
        
    def reset(self):
        state, _ = self.env.reset()
        self.curr_state = state
        self.curr_S = set()
        self.state_repr = np.in1d(list(self.state_space),list(self.curr_state.literals)).astype(int)
        self.action_repr = np.zeros(len(self.action_space), dtype=int)
        return self.return_current_repr()


    def return_current_repr(self):
        return np.concatenate((self.state_repr,self.action_repr))

    def get_all_current_actions(self):
        '''
        Returns all current available actions
        '''

        real_actions = []
        
        effs_S = []
        for a in self.curr_S:
            effs_S += core.get_effects_helper(self.curr_state,a,self.env.domain)
        effs_S = set(effs_S)

        precs_S = []
        for a in self.curr_S:
            precs_S += core.get_preconditions_helper(self.curr_state,a,self.env.domain)
        precs_S = set(precs_S)
        
        for a in self.env.action_space.all_ground_literals(self.curr_state):
            act = (self.env.action_space.sample(self.curr_state))

            if a not in self.curr_S:
                anti_effs_a = [e.inverted_anti for e in core.get_effects_helper(self.curr_state,a,self.env.domain)]
                anti_precs_a = [p.inverted_anti for p in core.get_preconditions_helper(self.curr_state,a,self.env.domain)]

                if len(set(anti_effs_a).intersection(set(effs_S))) != 0:
                    pass

                elif len(set(anti_effs_a).intersection(set(precs_S))) != 0:
                    pass

                elif len(set(anti_precs_a).intersection(set(effs_S))) != 0:
                    pass
                else:
                    real_actions.append(a)
        
        if self.curr_S:
            real_actions.append('timestep')

        return real_actions
    
    def getRandomAction(self):
        real_actions = self.get_all_current_actions()

        action = random.choice(real_actions)

        return list(self.action_space).index(action)

    def get_action(
        self,
        number): 
        
        return self.action_space[number]

    def get_number_of_action(
        self,
        action):
        return self.action_space.index(action)

    def step(
        self,
        action : int,
        ep_t : int,
        verbose : bool = True
        ):

        '''
        Makes a step in the process MDP
        '''

        action = self.action_space[action]
        info = ''
        done = False
        reward = -1
        if action != 'timestep':
            if core.isValid(self.curr_state,action,self.env.domain):
                self.curr_S = self.curr_S.union(set([action]))
            else:
                print('applying not valid action!')
        else:
            reward += len(self.curr_S) / 100 

            if self.curr_S:
                for a in self.curr_S:
                    new_state, rewardn, donen, info = self.env.step(a)  
                    reward += rewardn
                    done = done or donen  
            else:
                new_state = self.curr_state
            if done:
                reward += 100
            self.curr_state = new_state
            self.curr_S = set()


        self.state_repr = np.in1d(list(self.state_space),list(self.curr_state.literals)).astype(int)
        self.action_repr = np.in1d(list(self.action_space),list(self.curr_S)).astype(int)

        return self.return_current_repr(), reward, done, info

    def get_action_repr(
        self,
        action
        ):
        return self.action_space_repr[self.action_space.index(action)]
        

    
def createTranslator(env_id,device,num_problem):
    # Create the environment usign pddlgym
    env = pddlgym.make(env_id)   
    env.fix_problem_index(num_problem)
    initial_planning_state, debug_info = env.reset()

    # Retrieve all possible actions and literals
    literals_space = env.observation_space.all_ground_literals(initial_planning_state)
    env.action_space.all_ground_literals(initial_planning_state)
    actions = env.action_space._all_ground_literals

    # Create communication
    return Translator(literals_space,actions,initial_planning_state,env,device)

