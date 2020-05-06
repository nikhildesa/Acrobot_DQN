# -*- coding: utf-8 -*-
"""
@author: nikhi
"""

import math
import random 
import numpy as np

from collections import namedtuple
from itertools import count         
import gym
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

batch_len = 256
gamma = 0.9888
target_update = 20
memory_capacity = 200000
number_of_episodes = 500
number_of_actions = 3


#<-------------------- Epsilon Greedy---------------->
    
class Epsilon_Greedy_Stratergy():
    def learning_rate(self,step):
        start = 1
        end = 0.01
        decrease = 0.001
        
        return end + (start - end) * \
            math.exp(-1. * step * decrease)



#<-------------------Experience ------------->
        
Instances = namedtuple('Instances',('state', 'action', 'next_state', 'reward') # subclass and fields
)
    
#<------------------- Replay memory----------->

class Experience_Memory():
    def __init__(self,depth):
        self.memory = []
        self.depth = depth
        self.count_in_memory = 0
        
    def push(self,instances):
        self.count_in_memory +=1 
        if(len(self.memory) < self.depth):
            self.memory.append(instances)
        
        else:
            push_at_start = self.count_in_memory % self.depth
            self.memory[push_at_start] = instances
  
    def sample_availability(self,batch_len):
        if(len(self.memory) >= batch_len):
            return True
        else:
            return False
        
    def sample(self,batch_len):
        rand_sample = random.sample(self.memory,batch_len)
        return rand_sample




#<------------------------------ DQN --------------------------->
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)     
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)


        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))





#<----------------Agent------------>

class Agent():
    def __init__(self,stratergy,device):
        
        self.stratergy = stratergy
        self.curr_step = 0
        self.device = device
    def choose_action(self,state,policy):
        rate = stratergy.learning_rate(self.curr_step)
        self.curr_step+=1
        
        if(rate < random.random()):
            with torch.no_grad():
                return policy(state).argmax(dim=1).to(self.device) # exploit
        else:
            return torch.tensor([random.randrange(3)]).to(self.device) # explore
            
#<------------------------------- Q value --------------------->
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def curr_q_value(policy, states, actions):
        q_val =  policy(states).gather(dim=1, index=actions.unsqueeze(-1))
        return q_val
        
    def next_q_value(target, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_len = next_states.shape[0]
        values = torch.zeros(batch_len).to(QValues.device)
        values[non_final_state_locations] = target(non_final_states).max(dim=1)[0].detach()
        return values    
        
#<------------ Environment---------------------->

class AcrobotEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('Acrobot-v1').unwrapped
        self.current_screen = None
        self.env.reset()
        self.done = False
        
        #Gym's functions        

    def close(self):
        self.env.close()
        
    def render(self, mode='human'):           # stop at the cureent screen
        return self.env.render(mode)
    
    
    def take_action(self, action):                                 #step executes the action taken by agent 
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device) 
    
    def starting(self):
        return self.current_screen is None
    
    def crop_screen(self, screen):                             # Crop_screen strips off top and bottom
        screen_height = screen.shape[1]
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.9)
        screen = screen[:, top:bottom, :]
        return screen
    
    def reset(self):
        self.env.reset()
        #Indicates the start of an episode
        self.current_screen = None 

    def get_state(self):                                           #Returns current state of the environment
        if self.starting() or self.done:
            self.current_screen = self.extracted_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen                               # Previous screen
            s2 = self.extracted_screen()                       
            self.current_screen = s2                               # Current screen
            return s2 - s1                                         # state is difference between two screens.
    
    def get_screen_height(self):
        screen = self.extracted_screen()                       # get height
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.extracted_screen()                       # get width
        return screen.shape[3]
    
    def extracted_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    
    
    
    def transform_screen_data(self, screen):                    # Convert to float, rescale, convert to tensor     
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255 #rescaling by dividing 255
        screen = torch.from_numpy(screen)
    
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])
    
        return resize(screen).unsqueeze(0).to(self.device)
    
    
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = AcrobotEnvManager(device)
screen = env.extracted_screen()
plt.figure()
plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu())
plt.title('Proccessed screen for CNN')
plt.show()



def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Instances(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)



#<----------- making objects of classes ---------------->

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stratergy = Epsilon_Greedy_Stratergy()
env = AcrobotEnvManager(device)
agent = Agent(stratergy,device)
memory = Experience_Memory(memory_capacity)


policy = DQN(env.get_screen_height(),env.get_screen_width(),number_of_actions)
target = DQN(env.get_screen_height(),env.get_screen_width(),number_of_actions)
target.load_state_dict(policy.state_dict())
target.eval()
optimize = optim.Adam(params = policy.parameters(),lr = 0.001)


#<---------------- Training------------------->
 
        

episodes_dur = [] 

for i in range(number_of_episodes):
    env.reset()
    state = env.get_state()
    
    for time in count():
        action = agent.choose_action(state,policy)
        reward = env.take_action(action)
        nextstate = env.get_state()
        memory.push(Instances(state,action,nextstate,reward))
        state = nextstate
        
        if(memory.sample_availability(batch_len)) == True:
            instance = memory.sample(batch_len)
            states,actions,rewards,nextstates = extract_tensors(instance)
        
            curr_q = QValues.curr_q_value(policy,states,actions)
            target_q = (QValues.next_q_value(target,nextstates) * gamma) + rewards

            loss = F.mse_loss(curr_q,target_q.unsqueeze(1))
            optimize.zero_grad()
            loss.backward()
            optimize.step()
        
        if (env.done == True):
            episodes_dur.append(time)
            #plt(episodes_dur, 100)
            print("duration is ",episodes_dur[-1],"for the episode number",len(episodes_dur))
            break

    if (i % target_update == 0):
        target.load_state_dict(policy.state_dict())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    