# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:46:09 2021

@author: lenovo
"""
import torch


def update_phi_pi (episodes,env,agent,c,l,enco):
    
    for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action.flatten())
                a=torch.Tensor(action.flatten()).unsqueeze(0)
                cat=torch.cat((torch.Tensor(state).unsqueeze(0),a),dim=-1)
                s=enco(cat)
                f=l(cat,s)
                #print(ll(cat,s).item()**2/ f.item(),"rrrrrrrrrrrrrrrrrrrrrr")
                c.zero_grad()
                f.backward(retain_graph=True)
                c.step()
                state = next_state
                
def update_phi_clone (num,state,action,env,agent,tenco,l,cc):
    for _ in range (num):
        _=env.reset()
        statee=state
        actione=action
        donee =False
        
        while not donee:
            
            next_statee, r, donee, _ = env.step(actione.flatten())
            donee=True
            #next_state=state
            #print(next_state)
            aa=torch.Tensor(actione.flatten()).unsqueeze(0)
            
            cat=torch.cat((torch.Tensor(statee).unsqueeze(0),aa),dim=-1)
            
            actione = agent.select_action(statee, evaluate=True)
            statee = next_statee
            #done=True
            s=tenco(cat)
            f=l(cat,s)  
            cc.zero_grad()
            
            f.backward(retain_graph=True)
            cc.step()