import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from trainer import pi_mu
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import pybullet as p2
p2.connect(p2.UDP)
import pybullet_envs
import torch.optim as optim
l=torch.nn.MSELoss()
ll=torch.nn.PairwiseDistance(p=2,keepdim=True)
from autoencoder import autoencoder
from update_autoencoders import update_phi_pi, update_phi_clone
from utils import soft_update, hard_update

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetahBulletEnv-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')    
parser.add_argument('--policy', default="Deterministic",
                    help='Policy Type: Gaussian | Deterministic (default: Deterministic)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episodes (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.05, metavar='G', #0.05 hopper, 0.04 walker, 0.05 reacher, 0.26 Ant with normalization, 0.03 HalfCheetah
                    help='Temperature parameter α determines the relative importance of the intrinsic reward in agent pi\
                            term against the reward (default: 0.05)')
parser.add_argument('--alpha1', type=float, default=0.05, metavar='G', #0.05 hopper, 0.04 walker, 0.05 reacher, 0.26 Ant with normalization, 0.03 HalfCheetah
                    help='Temperature parameter α_1 determines the relative importance of the intrinsic reward in agent mu\
                            term against the reward (default: 0.2)')
parser.add_argument('--beta', type=float, default=0.1, metavar='G', 
                    help='Temperature parameter β determines the relative importance of the influence function\
                            against the objective function of agent pi (default: 0.1)')                        
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', 
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',    
                    help='hidden size (default: 256)')  
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', ####################4 for humanoid
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')  
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')   
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--autoencoder_update_frequency', type=int, default=5, metavar='N',
                    help='min number of episodes after which the autoencoder network is updated')
args = parser.parse_args()
     


# Environment  
env = gym.make(args.env_name)
env1=gym.make(args.env_name)


torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = pi_mu(env.observation_space.shape[0], env.action_space, args) #agent pi
enco=autoencoder(env.observation_space.shape[0]+env.action_space.shape[0]) #phi network
agentd=pi_mu(env.observation_space.shape[0], env.action_space, args) #agent mu
c=optim.Adam(enco.parameters(), lr=0.003, weight_decay=0.001) #Adam optimizer
#TensorboardX
writer = SummaryWriter(logdir='runs/{}_pi-mu_framework_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0
total_numstepsd = 0
updatesd = 0
x=0
for i_episode in itertools.count(1):
    ###############################################################
    #Agent pi execution and update
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample() # Sample random action
            
        else:
            action = agent.select_action(state) # Sample action from policy pi
            
        if len(memory) > args.batch_size:
            
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                if total_numsteps>args.start_steps:
                    #Define phi_clone
                    tenco=autoencoder(env.observation_space.shape[0]+env.action_space.shape[0])
                    hard_update(tenco, enco)
                    cc=optim.Adam(tenco.parameters(), lr=0.003, weight_decay=0.001)
                    #How the autoencoder (phi_clone) was affected by choosing (s,a)
                    num=5
                    update_phi_clone(num,state,action,env1,agent,tenco,l,cc)
                    
                    y=agentd.critic
                    yT=agentd.critic_target
                    ypi=agentd.policy
                    
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.X(state, action,memory,
                                                                                         args.batch_size, updates,env,enco, agentd.critic,
                                                                                         agentd.policy, agentd.critic_target,args,tenco)    
                    
                    agentd.critic=y
                    agentd.policy=yT
                    agentd.policy=ypi
                
                else:
                    
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parametersafter(memory, args.batch_size, updates,env,enco)
                    
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
        
        next_state, reward, done, _ = env.step(action.flatten()) # Step
        
        episode_steps += 1
        total_numsteps += 1
        x+=1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    

    writer.add_scalar('reward/train', episode_reward, i_episode)
    
    #Update the autoencoder network, phi
    if i_episode% 5 ==0 and total_numsteps>args.start_steps :
        episodes=15
        update_phi_pi( episodes, env, agent, c,l,enco)
    
    #################################################################################           
    #Agent mu execution and update
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            
            actiond = env.action_space.sample()# Sample random action
        else:
            
            actiond = agentd.select_action(state)# Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks of mu  
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agentd.update_parametersdeter(memory, args.batch_size, updatesd,env,enco)
                 
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updatesd += 1

        next_state, reward, done, _ = env.step(actiond.flatten()) # Step
        episode_steps += 1
        total_numstepsd += 1
        
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, actiond, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    

env.close()
