import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns

GAMMA = 0.9



# #of parkings:
J=2

# Parking capacity:
pcap_1=5000
pcap_2=4800

# Parking occupancy:
occ_1=0
occ_2=0

Parking1_occ={}
Parking2_occ={}

Parking1_occ[0,0,0] =0
Parking2_occ[0,0,0] =0

# Initial parking prices:
price1=12
price2=5

# Demand function parameters:
a=10
b=-0.2

# Demand function parameters:
mu=0
sigma=1
alpha=np.random.normal(mu,sigma,1)
#alpha=1
beta=-0.001

alphaa=1
betaa=--0.001

max_steps = 2



parking=[]

price_min=0

import random



def average(number1, number2):
  return (number1 + number2) / 2

T=4
M=3
i=0
k=0
iter=[]
iiter=[]
all1=[]
all2=[]
a1=[]
a2=[]
            
gt_1=[]


TG2Sum=[]
TG1Sum=[]

TG2=[]
TG1=[]
TotalG=[]
s=0
d=0
start=[]
duration=[]
time_step=[]

t_step=-1
#print("pcap_1", pcap_1)
#print("pcap_2", pcap_2)

for t in range(0,T+1):
    t_step=t_step+1
    print("t_step",t_step)
    time_step=[]
    reward_1=0
    reward_2=0
    discounted_rewards_1 = []
    discounted_rewards_2 = []
    pw=0
    
    
    avg=average(price1,price2)
    
    Gt_1=0
    Gt_2=0
    
    s=0
    
    pocc_1=[]
    pocc_2=[]
    
    New_occ_p1=[]
    New_occ_p2=[]
    

    
    for ts in range (t+1,M+2):
        s=s+1
        print("s",s)
        start.append(s)
        Start=np.array(start)
        d=0
        for Ta in range (1,M-ts+3):
            d+=1
            print("d",d)
            duration.append(d)
            Duration=np.array(duration)
            price=[]
            print("Ta",Ta)
            print("ts",ts)
            print("t",t)
            #### Demand:

            Demand=np.random.uniform(1,100,1)
            Demand=np.asscalar(np.array([Demand]))
            Demand=round(Demand)
            print("demand:",Demand)
            
            gt_2=[]
            gt_1=[]
            
            P1_occ=[]
            P2_occ=[]
            
            P1=[]
            P2=[]
            Pcap_2=[]
            H=[]
    
            for j in range(J):
                print("j",j)
                if j==0:
                    def trunc_gauss(mu, sigma, bottom, top):
                            a = random.gauss(mu,sigma)
                            while (bottom <= a <= top) == False:
                                a = random.gauss(mu,sigma)
                            return a
                    a=trunc_gauss(mu,sigma,0,1)
                    alpha=a
                    a1.append(alpha)
                    #print("a1",a1)
                    #print("a1",a1)
                    #print("alpha=",a)
                    price_1=alpha*np.exp(beta*Demand)
                    #print("price_1:",price_1)
                    if price_1<=5:
                        price.append(price_1)
                else:
                    def trunc_gauss(mu, sigma, bottom, top):
                        a = random.gauss(mu,sigma)
                        while (bottom <= a <= top) == False:
                            a = random.gauss(mu,sigma)
                        return a
                    a=trunc_gauss(mu,sigma,0,1)
                    alphaa=a
                    a2.append(alphaa)
                    
                    #print("a2",a2)
                    #print("alpha=",a)
                    price_2=alphaa*np.exp(betaa*Demand)
                    #print("price_2:",price_2)
                    if price_2<=5:
                        price.append(price_2)
            #print("price",price)
            parking=np.argmin(price)+1
            print("parking",parking)
            price_min=np.amin(price)
            #print("price_min",price_min)
    
            for customer in range (1,Demand+1):
                #print("customer",customer)
                if parking==1 and pcap_1>0:
                    #print("pcap_1",pcap_1)
                    occ_1=occ_1+1
                    #print("occ_1",occ_1)
                    P1_occ.append(occ_1)
                    #print("P1_occ",P1_occ)
                    #pocc_1=np.array(P1_occ)
                    #print("pocc_1",pocc_1)
                    pcap_1=pcap_1-1
                    #print("P1cap:",pcap_1)
                    Gt_1=Gt_1+price_min*1
                    #print("Gt1:",Gt_1)
                    gt_1.append(Gt_1)
                    TG_1=np.sum(gt_1)
                    #print("gt_1",len(gt_1))
                    #print("gt_1",len(gt_1))
                  
                    #print("Gt:",Gt_1)
                    #print("TG_1:",TG_1)
                elif parking==2 and pcap_2>0:
                    occ_2=occ_2+1
                    #print("occ_2",occ_2)
                    P2_occ.append(occ_2)
                    #pocc_2=np.array(P2_occ)
                    #p2_oocupancy=np.sum(P2_occ)
                    #print("pocc_1",pocc_1)
                    #print("p2_oocupancy:",p2_oocupancy)
                    pcap_2=pcap_2-1
                    Pcap_2.append(pcap_2)
                    #print("pcap_2",pcap_2)
                    Gt_2=Gt_2+price_min*1
                    #print("Gt2:",Gt_2)
                    gt_2.append(Gt_2)
                    TG_2=np.sum(gt_2)
                    #print("gt_2",gt_2)
                else:
                    break
                #print("occ_1",occ_1)
                #print("occ_2",occ_2)
                
                
            Parking1_occ[t_step,s,d] =occ_1
            Parking2_occ[t_step,s,d] =occ_2
            #print("Parking1_occ",Parking1_occ)
            #print("Parking2_occ",Parking2_occ)
            
            print("pcap_11:",pcap_1)
            print("pcap_22:",pcap_2)
            
            for k in range(0,t_step-1):
                print("k",k)
                for o in range(k+1,M+2):
                    print("o",o)
                    for p in range(1,M-o+3):
                        print("p",p)
                        if t_step+1==d+s:
                            #print("C1",Parking1_occ[k,o,p])
        
                            pcap_1=pcap_1+Parking1_occ[t_step,s,d]
                            pcap_2=pcap_2+Parking1_occ[t_step,s,d]
            print("pcap_11:",pcap_1)
            print("pcap_22:",pcap_2)
                  
                
            TG1Sum=np.sum(gt_1)
            TG1.append(TG1Sum)
          
           
            
            TG2Sum=np.sum(gt_2)
            TG2.append(TG2Sum)
            
            TotalG=[TG1,TG1]
            #print("TotalG",len(TotalG))
            Total=np.array(TotalG)
            #print("Total",Total.shape)
        
            #print("TG_2:",TG_2)
            #print("gt_2",len(gt_2))
                    #print("TG_2",TG_2.shape)
            #print("TG2Sum",TG2Sum)
            #print("TG1",TG1)
    
                    

    
    
    
    
#### Parking rewards at time t:

    reward_1 = Gt_1 + GAMMA**pw* reward_1
    #print("reward_1:",reward_1)
    pw = pw + 1
    discounted_rewards_1.append(reward_1)
    Rewards_1=np.array(discounted_rewards_1)
    all_rewards_1 = np.sum(Rewards_1)
    all1.append(all_rewards_1)
    #print("All1",all1)
    #print("all_rewards_1:",all_rewards_1)
           
        
    reward_2 = Gt_2 + GAMMA**pw* reward_2
    #print("reward-2:",reward_2)
    pw = pw + 1
    discounted_rewards_2.append(reward_2)
    #print("discounted_rewards_2:",discounted_rewards_2)
    Rewards_2=np.array(discounted_rewards_2)
    all_rewards_2 = np.sum(Rewards_2)
    all2.append(all_rewards_2)
    #print("All2",all2)
    #print("all_rewards_2:",all_rewards_2)
    
    i+=1
    iter.append(i)
    #print("Iter:",iter)
    
    
    
    
    
    


plt.figure(figsize = (8, 4))
plt.plot(duration,TG1)
plt.xlabel("duration")
plt.ylabel("TG1")
plt.title("TG1")
plt.show()
    
plt.figure(figsize = (8, 4))
plt.plot(duration,TG2)
plt.xlabel("duration")
plt.ylabel("TG2")
plt.title("TG2")
plt.show()


plt.figure(figsize = (8, 4))
plt.plot(a1,TG1)
plt.xlabel("a1")
plt.ylabel("TG1")
plt.title("TG1")
plt.show()
    
plt.figure(figsize = (8, 4))
plt.plot(a2,TG2)
plt.xlabel("a2")
plt.ylabel("TG2")
plt.title("TG2")
plt.show()





