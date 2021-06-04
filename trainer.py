import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
import torch.optim as optim
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from autoencoder import autoencoder

import matplotlib.pyplot as plt

l=torch.nn.MSELoss()
ll=torch.nn.PairwiseDistance(p=2,keepdim=True)

class pi_mu(object):
    def __init__(self, num_inputs, action_space, args):
        self.args=args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.alpha1=args.alpha1
        
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.l=[]
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    def update_model1(self,model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param.to("cuda:0")+params.to("cuda:0"))
            index += params_length
    
    
    def update_parametersafter(self, memory, batch_size, updates,env,enco):
        '''
        Updates the parameters of pi w.r.t only the objective function J
        
        memory : stored (s,a,r,s') transitions
            
        batch_size : int
        
        updates : int
        
        env : The environment of interest
            
        enco : The corresponding autoencoder, phi
    

        '''
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            cat=torch.cat((next_state_batch,next_state_action),dim=-1)
            s=enco(cat)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            next_q_value = reward_batch +self.alpha* ll(cat,s) + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)  
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = (-min_qf_pi).mean() # policy_loss = -Jπ = - 𝔼st∼D,εt∼N[Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy_optim.step()
        
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def update_parametersdeter(self, memory, batch_size, updates,env,enco):
        '''
        Updates the parameters of the agent mu (critics and policy) w.r.t. external and intrinsic rewards.
        Parameters
        ----------
        memory : stored (s,a,r,s') transitions
            
        batch_size : int
        
        updates : int
        
        env : The environment of interest
            
        enco : The corresponding autoencoder, phi
        '''
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            cat=torch.cat((state_batch,action_batch),dim=-1)
            s=enco(cat)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            act,_,_=self.policy.sample(state_batch)
            
            
            next_q_value = reward_batch -self.alpha1* ((ll(cat,s)-torch.min(ll(cat,s)))/(torch.max(ll(cat,s))-torch.min(ll(cat,s)))
                                                      )*ll(act,action_batch) + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
    
        policy_loss = (-min_qf_pi).mean()

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy_optim.step()
        
        
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    # Save model parameters
    def X (self, ss,a,  memory, batch_size, updates,env,enco, Qdac, pidac, QTdac,args,tenco):
        '''
        Updates the parameters the agent pi (critics and policy) w.r.t. the objective function J - beta * F
        Parameters
        ----------
        ss : current state
        
        a : action taken in ss
        
        memory : stored (s,a,r,s') transitions
                         
        batch_size : int
        
        updates : int
        
        env : The environment of interest
        
        enco : The autoencoder, phi
        
        Qdac : mu's critic target network
        
        pidac : mu's policy
        
        QTdac : mu's critic target network
        
        tenco : The updated cloned autoencoder, phi_clone

        '''
        
        Qdac_optim = Adam(Qdac.parameters(), lr=args.lr)
        
        # Sample from the buffer B
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Update Q_pi parameters
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            cat=torch.cat((next_state_batch,next_state_action),dim=-1)
            s=enco(cat) # phi(s,a)
            
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) # min of two Q values
            
            
            next_q_value = reward_batch +self.alpha* ll(cat,s) + mask_batch * self.gamma * (min_qf_next_target) # target_value = r + psi+ gamma Q_tar
        
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        

        
        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        
                
        #Now update the Qs of the mu according to the updated phi_clone
        
        #Qdac, pidac, QTdac
        with torch.no_grad():
            next_state_action, _, _ = pidac.sample(next_state_batch)
            
            qf1_next_target, qf2_next_target = QTdac(next_state_batch, next_state_action)
            
            cat=torch.cat((state_batch,action_batch),dim=-1)
            s=tenco(cat)
            
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            act,_,_=pidac.sample(state_batch)
            
            next_q_value = reward_batch -self.alpha1* ((ll(cat,s)-torch.min(ll(cat,s)))/(torch.max(ll(cat,s))-torch.min(ll(cat,s)))
                                                      )*ll(act,action_batch) + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = Qdac(state_batch, action_batch)  
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        Qdac_optim.zero_grad()                              #For mu
        qf1_loss.backward()
        Qdac_optim.step()
        
        Qdac_optim.zero_grad()
        qf2_loss.backward()
        Qdac_optim.step()
        
        #########################################################
        #Now find the value of F (the influence function)
    
        pi_BAC,_,_=self.policy.sample(state_batch)
        
        with torch.no_grad():
            next_state_action, _, _ = pidac.sample(next_state_batch)
            qf1_next_target, qf2_next_target = QTdac(next_state_batch, next_state_action)
            
            cat=torch.cat((state_batch,pi_BAC),dim=-1)
            s=enco(cat)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            act,_,_=pidac.sample(state_batch)
            
            
            next_q_value = reward_batch -self.alpha1* ((ll(cat,s)-torch.min(ll(cat,s)))/(torch.max(ll(cat,s))-torch.min(ll(cat,s)))
                                                      )*ll(act,pi_BAC) + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = Qdac (state_batch, pi_BAC)
        
        
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        minlossinf=torch.min(qf1_loss,qf2_loss) # The value of F
        ##########################################################
        
        qf1_pi, qf2_pi = self.critic(state_batch, pi_BAC)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (-min_qf_pi).mean()
        
        policy_loss+=args.beta*minlossinf # Add F to -J as a regularizer
        self.policy_optim.zero_grad()
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy_optim.step() # Mininmize beta * F - J (or Maximize J - beta * F as in algorithm (1))
        
        
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    #save model parameters    
    def save_model(self, env_name, enco, suffix="", actor_path=None, critic_path=None, enco_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/actor/actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/critic/critic_{}_{}".format(env_name, suffix)
        if enco_path is None:
            enco_path = "models/enco/autoencoder_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(enco.state_dict(), enco_path)
        

    # Load model parameters
    def load_model(self, enco, actor_path, critic_path, enco_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if enco_path is not None:
            enco.load_state_dict(torch.load(enco_path))

