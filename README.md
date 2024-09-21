This file contains all the necessary explanations to run our experiments. In "experiments" file, there are located all the problems and plans of our analysis. 
# Setup

We first need to install pddlgym. For this, unzip pddlgym and run inside pddlgym folder:

pip install -e .

This needs to be done this way because we slightly modified PDDLGym to fit our work.

Necessary extra installs:

- Pytorch, torchvision and torchaudio (refer to https://pytorch.org/)
- Pytorch-geometric (refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

We include all the packages that are installed in our runtime environment at the end of this document. Note that Pytorch and Pytorch-geometric can be installed directly on CPU by changing the device in the code.

# How to reproduce our experiments

Every train is already prepared inside PDDLGym. If one wants to train a policy for, say, problem 0 of multi-blocksworld domain, just run:

python main.py --env PDDLEnvBlocksmulti-v0 --num_problem 0

This will train the policy weights and save them inside train_models/PDDLEnvBlocksmulti-v0 with the name "5_actor.pth" and "5_critic.pth". A file "timesteps.txt" is also generated, in which we display the total number of iterations and the total time for training. 

To generate a plan with the trained policy, we run

python main.py --env PDDLEnvBlocksmulti-v0 --num_problem 0 --mode test --max_num 49 --actor_model trained_models/PDDLEnvBlocksmulti-v0/0_actor.pth

And the plan would be saved in the folder plans/PDDLEnvBlocksmulti-v0 as "0.pddl".

The max_num parameter specifies the maximum number of problems that are present for each domain. Values for each domain are as follows:

- PDDLEnvBlocksmulti-v0 (Multi-blocksworld): 49
- PDDLEnvFloortile-v0 (Floortile): 55
- PDDLEnvFree_openstacks-v0 (Free_openstacks): 34
- PDDLEnvLogpure-v0 (Transport): 63
- PDDLEnvOpenstacks-v0 (Openstacks): 34

As the reader can see, domains are addressed in a specific way. 

Note: problems in PDDLGym are addressed in lexical order. That is, problem number 3 is not 3.pddl, but 10.pddl (Because, in lexical order, '0.pddl' < '1.pddl' < '11.pddl').

# Process semantics validator

Here we briefly explain how the process semantics validator works:

The validator is based on a search where each node contains a list where states are stored, labeled with their timestamp, along with the set of actions that can be executed in that state simultaneously. The list of the initial node only contains the initial state, labeled with time zero, and no associated actions.Expanding a node means adding the first action in the plan that does not appear in that node. To add an action a to a node, all possible states are chosen from the list that satisfy the following:

- The preconditions of a are met in that state.
- There are no conflicts between a and the actions that must be executed in that state. A conflict occurs when actions have contradictory effects (one deletes what another adds) or when one action erases some precondition of another one.
- The effects of a do not prevent the preconditions of the actions in the list that must be executed later from being fulfilled.

The search branches according to the possible states where a can be inserted. If the action cannot be inserted into any state, the node is pruned. When a node containing all the actions of the plan is reached (final node), its makespan is taken into account to prune longer solutions. When there are no more nodes left to visit, the search ends and the final node with the best makespan found is returned. If the plan is invalid, the search will terminate without finding a plan and an error will be returned.

The process semantics checker is in file "process_checker", with a makefile attached for reproducibility purposes.

