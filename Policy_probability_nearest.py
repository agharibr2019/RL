#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:06:26 2021

@author: weiwenzhou
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#from keras.models import Model
#from keras.layers import Dense, Dropout, Flatten, Input, concatenate
#from keras.layers import Conv2D, MaxPooling2D, dot, Reshape
#from keras.optimizers import SGD, Adam
#from keras import regularizers, initializers
#from keras import backend as K
#import tensorflow as tf
#from tensorflow import set_random_seed
#from keras.callbacks import EarlyStopping
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time
from IPython import display
from Simulation import *
import param
import matplotlib.pyplot as plt
import matplotlib as mpl
from bisect import bisect_left
import pandas as pd
import numpy as np
import datetime as dt
import os
import math
import scipy.stats as stats
import random
import copy
import timeit
import statistics
import pickle
##########OR tool
from ortools.linear_solver import pywraplp


#===========================================================================================#
#============================== Simulator input to DQN =====================+++++++=========#
#===========================================================================================#
X_scale = param.J * param.delta
Y_scale = param.I * param.delta

nactions = 6 #['stay', 'up', 'down', 'left', 'right','accept']
action_possible =[0,1,2,3,4,5]              #['stay', 'up', 'down', 'left', 'right','accept']
#===========================================================================================#
#============================== Network & Learning Parameters ==============================#
#===========================================================================================#

inp_driver_geo_X = Input(shape=(X_scale,1)) #X location of the driver
inp_driver_geo_Y = Input(shape=(Y_scale,1)) #Y location of the driver
inp_t = Input(shape=(1,))# clock during that day, 1-d from 0 to 1
inp_t_remain = Input(shape=(1,))
inp_action = Input(shape=(nactions,1))    #['stay', 'up', 'down', 'left', 'right']

inp_driver_geo_X_flat = Flatten()(inp_driver_geo_X)
inp_driver_geo_Y_flat = Flatten()(inp_driver_geo_Y)
inp_action_flat = Flatten()(inp_action)


x = concatenate(inputs = [inp_driver_geo_X_flat,inp_driver_geo_Y_flat,inp_t,inp_t_remain,inp_action_flat])

x_v = Dense(200, activation = 'relu')(x) # fully connected layer
x_v = Dense(200, activation = 'relu')(x_v) # fully connected layer
x_v = Dense(200, activation = 'relu')(x_v) # fully connected layer
out_v = Dense(1, activation = 'linear')(x_v) # V function, 1-d output 

sgd_0 = Adam(lr=0.1)
sgd_1 = SGD(lr=0.05,momentum = 0.3)
Vnet = Model(inputs=[inp_driver_geo_X,inp_driver_geo_Y,inp_t,inp_t_remain,inp_action], outputs=out_v)

#option for loss function: 1. mean_squared_error, 2. mean_squared_logarithmic_error, 3. mean_absolute_error
#option for optimizer: Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,RMSprop,SGD,
#Vnet.compile(loss='mean_squared_error',optimizer=sgd)
Vnet.compile(loss='mean_squared_logarithmic_error',optimizer=sgd_1)



epsilon0 = .0 # initial noise 100%
epsilon1 = .99 # final noise 1%
buffer_batch=16 # batch size for training
buffer_memo_size = 720 # buffer memory size
episodes=500 # number of training episodes
gamma=1.0 # discount factor

#===========================================================================================#
#=========================+++++===== Simulator part ========================+++++++++++======#
#===========================================================================================#

def one_hot(x,scale):
    y = np.zeros((scale,1))
    for i in range(scale):
        if x == i:
            y[i,0] = 1
    return y






def Valid_actions(X,Y,offered,speed,action_possible,x_scale=X_scale,y_scale=Y_scale):
    
    actions = copy.deepcopy(action_possible)     #['stay', 'up', 'down', 'left', 'right','accept']
    
    if offered == 0:
        actions.remove(5)    #remove accept
   
        
    if X - speed/60 < 0:
        actions.remove(3)    #remove left
    elif X + speed/60 > x_scale:
        actions.remove(4)   #remove right

    if Y - speed/60 < 0:
        actions.remove(2)   #remove down
    elif Y + speed/60 > y_scale:
        actions.remove(1)     #remove up
    
    #creat one hot matrix
    action_onehot = np.zeros((len(actions), nactions,1),int)
    for i in range(len(actions)):
        action_onehot[i, actions[i]] = 1
    return action_onehot
       





start = timeit.default_timer()

#%matplotlib qt
#%matplotlib inline

def event_starter(order_set,driver_set):
    '''
    This function is used to create an initial set for events by reading from
    the order and vehicle files. The initial event set has an element for each
    new order and new delivery vehicle.

    Parameters
    ----------
    order_set : Pandas dataframe
        Orders dataframe.
    vehicle_set : Pandas dataframe
        Vehicles dataframe.

    Returns
    -------
    event_set : list
        Python list in which each element is an event. Each element is a list
        consists of event type, event id, and event time.

    '''
    def take_third(elem):
        return elem[2]
    
    event_set = []
    for index, row in order_set.iterrows():
        event_set.append(['order ready',row['id'],row['ready_time']])
    for index, row in driver_set.iterrows():
        event_set.append(['driver on',row['id'],row['on_time']])
        event_set.append(['driver off',row['id'],row['off_time']])
        
    event_set = sorted(event_set, key=take_third)
    return event_set

class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)

def event_updater(event_set,new_event):
    '''
    This function is used to update the event set. When an order is placed, it
    is used to make 1 future event associated with it: 1- when order is ready.
    Also after assinging order to a vehicle, this function is used to craete 2
    future event associated with it: 1- when order is picked up. 2- when order
    is delivered.
    This function will be edited to consider stochastic nature of the problem,
    i.e. stochastic travel times or ready times by sampling travel or ready times
    from their underlying distributions.

    Parameters
    ----------
    event_set : list
        Python list in which each element is an event. Each element is a list
        consists of event type, event id, and event time.
    new_event : list
        Python list which consists of event type, event id, and event time.

    Returns
    -------
    event_set : list
        Updated and sorted event_set.

    '''
    bslindex = bisect_left(KeyWrapper(event_set, key=lambda c: c[2]), new_event[2])
    event_set.insert(bslindex, new_event)
    return event_set

def travel_dist(X1, Y1, X2, Y2):
    dist = abs(X1-X2) + abs(Y1-Y2)
    return dist

#===========================================================================================#
#================================== Training ===============================================#
#===========================================================================================#
# replay buffer initialization	




Profit_runs = []
profit_average = []
profit_study = 0
restaurants = copy.deepcopy(pd.read_csv('C:/Users/wzhou/Documents/Meal Delivery Problem paper/Codes/Data/Data/data_ver20/restaurants.csv'))
Rst = np.zeros((len(restaurants),2),float)
for i in range(len(restaurants)):
    Rst[i,0] = restaurants.loc[i,'X']
    Rst[i,1] = restaurants.loc[i,'Y']


TRACE = []
buffer_count = 0
buffer_count_prim = 0
num_runs = param.num_runs
num_offer = 0
num_accept = 0
tip_thr = 5
for ep in range(1, episodes+1):
#for counter in range(1, param.num_runs+1):
    epsilon = np.max((np.min((epsilon0*(1-ep/(.05*episodes)) + epsilon1*(ep/(.05*episodes)),epsilon1)),0)) #noise %
    counter = random.randint(1,num_runs)  
    # initialize orders, vehicels, and event_set
    
    #orders = copy.deepcopy(pd.read_csv('/Users/weiwenzhou/Documents/research/Codes/Data/data_ver0/orders{}.csv'.format(counter)))
    orders = copy.deepcopy(pd.read_csv('Data/data_ver20/orders{}.csv'.format(counter)))
    drivers = copy.deepcopy(pd.read_csv('Data/data_ver20/drivers{}.csv'.format(counter)))
    ##########################
    ###choose the driver######
    ##########################
    #study_driver = random.randint(0,len(drivers)-1)       #choose the driver
    
    x_location = random.uniform(0,20)
    y_location = random.uniform(0,20)
    on_time = random.uniform(601,1320)
    off_time = min(on_time + np.random.normal(120,30),1440)
    on_time = 1080
    off_time = 1200
    
    study_driver_X = x_location
    study_driver_Y = y_location
    study_driver_on_time = on_time
    study_driver_off_time = off_time
    study_driver_index = 0
    while drivers.loc[study_driver_index,'on_time']<study_driver_on_time:
        study_driver_index +=1
    drivers.loc[study_driver_index] = [study_driver_index, study_driver_X, study_driver_Y, study_driver_on_time, study_driver_off_time]
    
    study_driver = study_driver_index
    

    #drivers.loc[study_driver,'X']=math.floor(drivers.loc[study_driver_id,'X'])   #round the location of driver for q-learning
    #drivers.loc[study_driver,'Y']=math.floor(drivers.loc[study_driver_id,'Y'])
    
    endtime = drivers.loc[study_driver,'off_time']      #when the driver end the work, we don't count

    ###########################################################################
    event_set = event_starter(orders,drivers)
    for i in range(math.ceil(event_set[0][2]),math.ceil(event_set[-1][2]),param.decision_interval):
        event_set = event_updater(event_set,['decision point','d',i])
    
    ####create competetor events
    
    #event_set.append(['driver on',twin['id'],twin['on_time']])
    
    #event_set.append(['driver off',twin['id'],twin['off_time']])
    
    # initialize sets
    drivers_set = {}
    signedout_drivers = {}
    pending_orders = {}
    assigned_orders = {}
    completed_orders = {}
    cul_profit = 0
    lock = 'on'
    nearest_rst = np.array([random.uniform(0,15),random.uniform(0,15)])
    # read event list one by one
    for y in event_set:
        #if x[2] > 1700:
        if y[2] > endtime+100:                   #end the simulation if the driver 
            break
        #print(x[0],x[1],x[2])
        # chaeck if event is 'decision point' event
        if y[0] == 'decision point':
            time = y[2]
            # first check if any driver is available
            available_drivers = {}
            for driver_id in drivers_set:
                driver = drivers_set[driver_id]
                if driver_id == study_driver:
                    drivers_set[study_driver].offered = 0
                if driver.busy == 0 and driver.switch == 'on':
                    driver.offered = 0
                    driver.waiting = driver.waiting + param.decision_interval
                    if driver.moving == 1:
                        if driver.driver_id == study_driver:
                            delta_t = time - driver.clock
                            driver.X = driver.X + driver.delta_X * delta_t
                            driver.Y = driver.Y + driver.delta_Y * delta_t
                            driver.travelled_empty = driver.travelled_empty + abs(driver.delta_X) + abs(driver.delta_Y)
                            
                        else:
                            driver.X = driver.X + driver.delta_X
                            driver.Y = driver.Y + driver.delta_Y
                            driver.travelled_empty = driver.travelled_empty + abs(driver.delta_X) + abs(driver.delta_Y)
                    available_drivers[driver_id] = drivers_set[driver_id]
                    
            if (not available_drivers) or (not pending_orders):
                do_nothing = 1
            else:
                ########call the solver#############
                
                solver = pywraplp.Solver.CreateSolver('SCIP')
                orders_id = list(pending_orders.keys())
                drivers_id = list(available_drivers.keys())
                
                # calculate parameters of the optimization model
                dist = {}
                r = {}
                f = {}
                for o in orders_id:
                    for d in drivers_id:
                        dist[o,d] = abs(pending_orders[o].X_rest-available_drivers[d].X) \
                        + abs(pending_orders[o].Y_rest-available_drivers[d].Y)
                
                for o in orders_id:
                    r[o] = time - pending_orders[o].ready_time
                
                for d in drivers_id:
                    f[d] = available_drivers[d].waiting    
                model_info = {}
                #create variable
                x = {}
                for o in orders_id:
                    for d in drivers_id:
                        x[o, d] = solver.IntVar(0, 1, '')
                        
                        
                # Add Constraints
                for d in drivers_id:
                    #model.addConstr(gp.quicksum(x[o,d] for o in orders_id) <= 1, name = "C3[%s]"%(d))
                    solver.Add(solver.Sum([x[o,d] for o in orders_id]) <= 1)
                    

                for o in orders_id:
                    #model.addConstr(gp.quicksum(x[o,d] for d in drivers_id) <= 1, name = "C4[%s]"%(o))
                    solver.Add(solver.Sum(x[o,d] for d in drivers_id) <= 1)
                  
                
                for o in orders_id:
                    for d in drivers_id:
                        #model.addConstr(dist[o,d]-param.max_dist <= (1-x[o,d])*10000, name = "C5[%s,%s]"%(o,d))
                        #model.addConstr((1-x[o,d])*10000 >=dist[o,d]-param.max_dist, name = "C5[%s,%s]"%(o,d))
                        solver.Add((1-x[o,d])*100 >= dist[o,d] - param.max_dist)
                
                        
                
                
                # Set the objective function
                solver.Minimize(solver.Sum([dist[o,d]*x[o,d] for o in orders_id for d in drivers_id])
                                -1000*solver.Sum([r[o]*x[o,d] for o in orders_id for d in drivers_id])
                                -100*solver.Sum([f[d]*x[o,d] for o in orders_id for d in drivers_id])
                                )
                status = solver.Solve()
                
                if status == pywraplp.Solver.OPTIMAL:
                    #print('---------------------------------------')
                    x_optimal = {}
                    for o in orders_id:
                        for d in drivers_id:
                            #opt_decision = model.getVarByName("x[%s,%s]"%(o,d)).x
                            opt_decision = round(x[(o,d)].solution_value())
                            if opt_decision == 1:
                                #print(o,d)
                                order = pending_orders[o]
                                driver = available_drivers[d]
                                
                                # update busy and waiting attribute of driver
                                driver.busy = 1
                                #driver.waiting = 0
                                #driver.offered = 'yes'
                                
                                # add order to set of assigned orders
                                assigned_orders[o] = order
                                
                                # remove order from pending order set
                                del pending_orders[o]
    
                                # create either a 'delivery accepted' or 'delivery rejected' event
                                rnd = random.random()
                                rnd_time = random.random()*param.max_resp_time
                                
                                #get information from order
                                res_x = order.X_rest
                                res_y = order.Y_rest
                                cus_x = order.X_cust
                                cus_y = order.Y_cust
                                fee = order.fee
                                tip = order.tip
                                
        
                                                               
                                if d == study_driver:
                                    driver.busy = 0
                                    #driver.offered_order = o
                                    drivers_set[study_driver].offered_order = o
                                    #driver.offered = 'yes'
                                    drivers_set[study_driver].offered = 1
                                    driver.moving = 0
                                    
                                    delta_t = time - driver.clock
                                    
                                    
                                    #cost = param.C_d * param.V / 60 * (np.absolute(driver.delta_X) + np.absolute(driver.delta_X))*delta_t
                                    cost = param.C_d * (np.absolute(driver.delta_X) + np.absolute(driver.delta_Y))*delta_t
                                    #buffer_profit[buffer_count] -= cost
                                    cul_profit-=cost
                                    #buffer_terminal_flag[buffer_count] = 0
                                    buffer_count += 1
                                    buffer_count_prim += 1
                                    #print('1_decision point')
                                    # next decision time
                                    next_decision_time = time + 0.01
                                    driver.move_id += 1
                                    driver.switch = "off"
                                    #driver.reloop = 1
                                    event_set = event_updater(event_set,['study driver decision point',[driver.last_order,study_driver],next_decision_time,[res_x,res_y,cus_x,cus_y,fee,tip,1]])
                               #####for the other drivers##########   
                                
                                elif rnd <= param.trip_prob:
                                    # deivery is accepted
                                    event_set = event_updater(event_set,
                                                              ['delivery accepted',[o,d],time+rnd_time])
                                    
                                    
                                else:
                                    #if rejected
                                    event_set = event_updater(event_set,
                                                              ['delivery rejected',[o,d],time+rnd_time])
                                    
                            else:
                                xxx = 1
                                
                
                    
            #check if the chosen driver is our chosen driver
            ######if the chosen driver is not assgned order, but are available
            
            if (study_driver in available_drivers.keys() and available_drivers[study_driver].offered==0):
                driver = drivers_set[study_driver]
                driver.X = driver.X - driver.delta_X * (time - driver.clock)
                driver.Y = driver.Y - driver.delta_Y * (time - driver.clock)
            
            # get the state vecotor of drivers
            #state = [time,X,Y,time_without_offer,Working={0,1},offered=[{0,1},length,compensation,restaurant,dropoff=[X,Y]]]






        # check if event x is 'order ready' event
        elif y[0] == 'order ready':
            # create an order object
            order_data = orders[orders['id'] == y[1]]
            order = Order(order_id=y[1],X_cust=order_data['X_cust'].tolist()[0],
                     Y_cust=order_data['Y_cust'].tolist()[0],
                     placement_time=order_data['placement_time'].tolist()[0],
                     ready_time=order_data['ready_time'].tolist()[0],
                     assignment_time=0, pickup_time=0, dropoff_time=0,
                     restaurant=order_data['restaurant'].tolist()[0],
                     X_rest=order_data['X_rest'].tolist()[0],
                     Y_rest=order_data['Y_rest'].tolist()[0],
                     tip=order_data['tip'].tolist()[0],
                     distance=order_data['distance'].tolist()[0],
                     fee=order_data['fee'].tolist()[0],
                     rejected_by=[])
            # add order to the set of pending orders
            pending_orders[y[1]] = order

        elif y[0] == 'driver on':
            
            # create a driver object
            driver_data = drivers[drivers['id'] == y[1]]
            driver = Driver(driver_id=y[1],
                            X=driver_data['X'].tolist()[0],
                            Y=driver_data['Y'].tolist()[0],
                            delta_X=0,delta_Y=0,busy=0,
                            on_time=driver_data['on_time'].tolist()[0],
                            off_time=0,moving=0,last_order=-1,num_orders=0,
                            tired=0,waiting=0,travelled_delivery=0,
                            travelled_empty=0,earnings=0,offered_order = -1, 
                            offered = 0, signal = 0, move_id = 0, clock = 0, switch = 'on')
            drivers_set[y[1]] = driver
            
            ####pick the study driver
            if y[1] == study_driver:
                #driver_id = y[1]
                next_driver_decision_time = y[2]+0.001 
                driver_id = study_driver
                driver.delta_X = 0
                driver.delta_Y = 0
                driver.switch = 'off'
                move_X = driver.delta_X
                move_Y = driver.delta_Y
                driver.clock = study_driver_on_time
                event_set = event_updater(event_set,['study driver first arrive',[move_X,move_Y],next_driver_decision_time,driver.move_id,driver_id])
                
            else:
                # determine how long driver will stay at current location
                idle_time = y[2] + np.random.normal(loc=param.idle_time_avg,
                                                scale=param.idle_time_std,size=1)[0]
                # create a 'driver moves' event for driver
                event_set = event_updater(event_set,['driver moves',[-1,y[1]],idle_time])

        elif y[0] == 'driver off':
            driver = drivers_set[y[1]]
            # if driver is not performing a delivery job remove it
            if driver.busy == 0:
                driver.off_time = y[2]
                signedout_drivers[y[1]] = driver
                if driver.driver_id == study_driver and driver.switch == 'on':

                    driver.switch = 'off'
                    #driver.move_id += 1
                
                del drivers_set[y[1]]
            # else if drver is peroforming a delivery job, change tired attribute
            # and remove it when the delivery is completed
            else:
                driver.tired = 1
                driver.signal = 1

        elif y[0] == 'delivery accepted':
            # get order and driver id associated with the accepted delivery
            order_id = y[1][0]
            driver_id = y[1][1]
            
            # obtain order and driver
            order = assigned_orders[order_id]
            driver = drivers_set[driver_id]
            
            # change availability of driver
            driver.busy = 1
            driver.moving = 0
            driver.num_orders = driver.num_orders + 1
            

            # create 'delivery completed' event
            pickup_time = 60*(abs(driver.X - order.X_rest) + abs(driver.Y - order.Y_rest))/param.V + param.pickup_service_time/2
            dropoff_time =  param.pickup_service_time/2 + 60*order.distance/param.V + param.dropoff_service_time
            
            # update travelled
            driver.travelled_delivery = driver.travelled_delivery + param.V*(pickup_time + dropoff_time)/60

            # order is assigned
            order.assignment_time = y[2]
            order.pickup_time = y[2] + pickup_time
            order.dropoff_time = y[2] + pickup_time + dropoff_time
            
            ##calculate the cost here, for drive to the restaurant then dropoff
            #some changes, only count the cost while driving
            #order_cost = (pickup_time+dropoff_time)*param.V/60*param.C_d
            order_cost = (pickup_time+dropoff_time-param.pickup_service_time-param.dropoff_service_time)*param.V/60*param.C_d
            
            event_set = event_updater(event_set,['delivery completed',[order_id,driver_id],order.dropoff_time,order_cost])
            
            
        elif y[0] == 'delivery rejected':
            # get order and driver id associated with the accepted delivery
            order_id = y[1][0]
            driver_id = y[1][1]

            # obtain order and driver
            order = assigned_orders[order_id]
            driver = drivers_set[driver_id]

            # delete order from assigned_order set and add it to pending_orders set
            del assigned_orders[order_id]
            pending_orders[order_id] = order
            
            # rejected by driver
            order.rejected_by.append(driver_id)
            
            # release driver
            driver.busy = 0
            if driver.tired == 1:
                driver.off_time = y[2]
                signedout_drivers[driver_id] = driver
                del drivers_set[driver_id]
            '''
            else:
                if driver_id == study_driver:
                    next_driver_decision_time = y[2] + 1
                    action = driver.action_before
                    event_set = event_updater(event_set,['study driver decision point',
                                                         [driver.last_order,driver_id],next_driver_decision_time,action])
            '''  
              
        elif y[0] == 'delivery completed':
            order_id = y[1][0]
            driver_id = y[1][1]
            
            order = assigned_orders[order_id]
            driver = drivers_set[driver_id]

            # add order to set of completed orders
            completed_orders[order_id] = order
            
            # remove order from accepted orders set
            del assigned_orders[order_id]
            
            # update X and Y of driver
            driver.X = order.X_cust
            driver.Y = order.Y_cust
            
            ##round for q table updating
            #x_axis = round(driver.X)
            #x_axis = round(driver.X * 10)/10
            #x_axis = math.floor(driver.X *2)/2
            #y_axis = round(driver.Y)
            #y_axis = round(driver.Y * 10)/10
            #y_axis = math.floor(driver.Y *2)/2
            # collect earnings
            driver.earnings = driver.earnings + order.fee + order.tip
            
            # update last order 
            driver.last_order = order_id
            #if driver_id == study_driver:
                #cost = y[3]
                #profit = order.fee + order.tip - cost
            lock = 'off'
            # change availability of driver or remove it if tired = 1
            if driver.tired == 1:
                driver.off_time = y[2]
                signedout_drivers[driver_id] = driver
                if driver_id == study_driver:
                    driver.switch = 'off'
                    cost = y[3]
                    profit = order.fee + order.tip - cost
                    cul_profit+=profit

                    #print('1_delivery_complete')
                del drivers_set[driver_id]
            else:
                driver.busy = 0
                if driver_id == study_driver:
                    driver.switch = 'off'
                    driver_decision_time = y[2]+0.001
                    cost = y[3]
                    profit = order.fee + order.tip - cost
                    cul_profit+=profit
                   

                    #print('1_delivery_complete')
                    event_set = event_updater(event_set,['study driver decision point',[driver.last_order,driver_id],driver_decision_time,[]])
                    
                else:
                    
                    # determine how long driver will stay at current location
                    idle_time = y[2] + np.random.normal(loc=param.idle_time_avg,
                                                    scale=param.idle_time_std,size=1)[0]
                    # create a 'driver moves' event for driver
                    event_set = event_updater(event_set,['driver moves',[order_id,driver_id],idle_time])

        elif y[0] == 'driver moves':
            # find the driver
            order_id = y[1][0]
            driver_id = y[1][1]

            if driver_id in list(drivers_set.keys()):
                driver = drivers_set[driver_id]

                if (driver.busy == 0 and driver.last_order == order_id):
                    new_X = stats.truncnorm.rvs(param.a[0],param.b[0],loc=param.mean[0],scale=param.std[0],size=1).tolist()[0]
                    new_Y = stats.truncnorm.rvs(param.a[1],param.b[1],loc=param.mean[1],scale=param.std[1],size=1).tolist()[0]

                    driving_time = y[2] + 60*(abs(driver.X-new_X)+abs(driver.Y-new_Y))/param.V
                    
                    driver.delta_Y = (param.V*param.decision_interval/60)/(1+(abs(new_X-driver.X)/abs(new_Y-driver.Y)))*np.sign(new_Y-driver.Y).tolist()
                    driver.delta_X = (abs(new_X-driver.X)/abs(new_Y-driver.Y))*abs(driver.delta_Y)*np.sign(new_X-driver.X).tolist()
                    
                    driver.moving = 1

                    # create a 'driver arrives' event for driver
                    event_set = event_updater(event_set,['driver arrives',[driver_id,new_X,new_Y],driving_time])
                    
        elif y[0] == 'driver arrives':
            driver_id = y[1][0]
            if driver_id in drivers_set:
                driver = drivers_set[y[1][0]]
                
                driver.X = y[1][1]
                driver.Y = y[1][2]
                driver.moving = 0
                
                # determine how long driver will stay at current location
                idle_time = y[2] + np.random.normal(loc=param.idle_time_avg,
                                                    scale=param.idle_time_std,size=1)[0]
                # create a 'driver moves' event for driver
                event_set = event_updater(event_set,['driver moves',[driver.last_order,driver_id],idle_time])

        elif y[0] == 'study driver arrives':
            driver_id = y[4]
            if driver_id in drivers_set:
                driver = drivers_set[driver_id]
                if driver.move_id == y[3]:
                #if driver == drivers_set[driver_id] and driver.move_id == x[3]:
                    driver.X = driver.X + y[1][0]
                    driver.Y = driver.Y + y[1][1]
                    driver.moving = 0
                    
                    #cost = param.C_d * param.V / 60 * (np.absolute(y[1][0]) + np.absolute(y[1][1]))
                    cost = param.C_d * (np.absolute(y[1][0]) + np.absolute(y[1][1]))
                    driver.travelled_empty = driver.travelled_empty + np.absolute(y[1][0]) + np.absolute(y[1][1])
                    cul_profit-=cost

                    #print('1_arrive')
                    # next decision time
                    next_decision_time = y[2] + 0.01
                    driver.move_id += 1
                    driver.switch = 'off'
                    # create a 'driver moves' event for driver
                    event_set = event_updater(event_set,['study driver decision point',[driver.last_order,driver_id],next_decision_time,[]])
        
        elif y[0] == 'study driver first arrive':
            
            driver_id = y[4]
            
            if driver_id in drivers_set:
                driver = drivers_set[driver_id]
                if driver.move_id == y[3]:
                #if driver == drivers_set[driver_id] and driver.move_id == y[3]:
                    driver.X = driver.X + y[1][0]
                    driver.Y = driver.Y + y[1][1]
                    driver.moving = 0
                   
                    # next decision time
                    next_decision_time = y[2] + 0.01
                    driver.move_id += 1
                    driver.switch = 'off'
                    # create a 'driver moves' event for driver
                    event_set = event_updater(event_set,['study driver decision point',[driver.last_order,driver_id],next_decision_time,[]])

        elif y[0] == 'study driver decision point':
            
            # find the driver
            if y[2] > 790:
                TRACE.append([driver.X, driver.Y])
            order_id = y[1][0]
            driver_id = y[1][1]
            driver_decision_time = y[2]
            if driver_id in list(drivers_set.keys()):
                
                driver = drivers_set[driver_id]
                driver.clock = y[2]
                #print(driver.clock)
                offered = driver.offered
                if (driver.busy == 0 and driver.last_order == order_id):
                    #if driver.moving == 1:
                    #    driver.X += driver.delta_X
                    #    driver.Y += driver.delta_Y
                    valid_action = Valid_actions(driver.X,driver.Y,driver.offered,param.V,action_possible,x_scale=20,y_scale=20)
                    num_valid = len(valid_action)
                    driver_x_onehot_set = np.array([one_hot(math.floor(driver.X), X_scale)]*num_valid)
                    driver_y_onehot_set = np.array([one_hot(math.floor(driver.Y), Y_scale)]*num_valid)
                   
                    
                    current_t = np.array([[y[2]]]*num_valid)
                    t_remain = np.array([[study_driver_off_time-y[2]]]*num_valid)
                    
                    rnd = random.random()
                    if offered == 1:
                        num_offer +=1
                        res_x, res_y, cus_x, cus_y, fee, tip, offered = y[3]
                        if tip >= tip_thr:
                            num_accept += 1 
                            action = np.array([[0],
                                               [0],
                                               [0],
                                               [0],
                                               [0],
                                               [1]])
                        
                    else:
                        #find the nearest restaurant
                        speed = param.V
                        if lock == 'off':
                            num_res = len(Rst)
                            dist_to_res = np.zeros(num_res)
                            for i in range(num_res):
                                dist_to_res [i] = travel_dist(driver.X, driver.Y, Rst[i,0], Rst[i,0])
                            rst_id = np.argmin(dist_to_res)
                            nearest_rst = Rst[rst_id]
                        #nearest_rst= np.array([4,3.5])
                        #determine how to move to this restaurant
                        x_dis = nearest_rst[0] - driver.X
                        y_dis = nearest_rst[1] - driver.Y
                        if abs(x_dis) >= abs(y_dis):
                            if x_dis >= 0 and driver.X + speed/60 < 20:
                                action = np.array([[0],
                                                   [0],
                                                   [0],
                                                   [0],
                                                   [1],
                                                   [0]])
                            elif x_dis <= 0 and driver.X - speed/60 > 0:
                                action = np.array([[0],
                                                   [0],
                                                   [0],
                                                   [1],
                                                   [0],
                                                   [0]])
                            else:
                                action = np.array([[1],
                                                   [0],
                                                   [0],
                                                   [0],
                                                   [0],
                                                   [0]])
                        else:
                            if y_dis >= 0 and driver.Y + speed/60 < 20:
                                action = np.array([[0],
                                                   [1],
                                                   [0],
                                                   [0],
                                                   [0],
                                                   [0]])
                            elif y_dis <= 0 and driver.Y - speed/60 > 0:
                                action = np.array([[0],
                                                   [0],
                                                   [1],
                                                   [0],
                                                   [0],
                                                   [0]])
                            else:
                                action = np.array([[1],
                                                   [0],
                                                   [0],
                                                   [0],
                                                   [0],
                                                   [0]])                                
                        
                    
                    if action[0,0] == 1:
                        driver.delta_X = 0
                        driver.delta_Y = 0
                        driver.moving = 1
                        driver.busy = 0
                        driver.switch = 'on'
                        next_driver_decision_time = driver_decision_time + 1 
                        move_X = driver.delta_X
                        move_Y = driver.delta_Y
                        
                        if offered == 1:
                            order_id = driver.offered_order
                            # delete order from assigned_order set and add it to pending_orders set
                            del assigned_orders[order_id]
                            pending_orders[order_id] = order
                            # rejected by driver
                            order.rejected_by.append(driver_id) 
                            
                            
                        if driver.tired == 1:
                            driver.off_time = y[2]
                            signedout_drivers[driver_id] = driver
                            del drivers_set[driver_id]
                        
                        event_set = event_updater(event_set,['study driver arrives',[move_X,move_Y],next_driver_decision_time,driver.move_id,driver_id])
                    if action[1,0] == 1:
                        driver.delta_X = 0
                        driver.delta_Y = param.V/60
                        driver.moving = 1
                        driver.busy = 0
                        driver.switch = 'on'
                        next_driver_decision_time = driver_decision_time + 1 
                        move_X = driver.delta_X
                        move_Y = driver.delta_Y
                        if offered == 1:
                            order_id = driver.offered_order
                            # delete order from assigned_order set and add it to pending_orders set
                            del assigned_orders[order_id]
                            pending_orders[order_id] = order
                            # rejected by driver
                            order.rejected_by.append(driver_id)   
                            
                        if driver.tired == 1:
                            driver.off_time = y[2]
                            signedout_drivers[driver_id] = driver
                            del drivers_set[driver_id]                            
                            
                        
                        event_set = event_updater(event_set,['study driver arrives',[move_X,move_Y],next_driver_decision_time,driver.move_id,driver_id])
                    if action[2,0] == 1:
                        driver.delta_X = 0
                        driver.delta_Y = - param.V/60
                        driver.moving = 1
                        driver.busy = 0 
                        driver.switch = 'on'
                        next_driver_decision_time = driver_decision_time + 1 
                        move_X = driver.delta_X
                        move_Y = driver.delta_Y
                        if offered == 1:
                            order_id = driver.offered_order
                            # delete order from assigned_order set and add it to pending_orders set
                            del assigned_orders[order_id]
                            pending_orders[order_id] = order
                            # rejected by driver
                            order.rejected_by.append(driver_id)   
                            
                            
                        if driver.tired == 1:
                            driver.off_time = y[2]
                            signedout_drivers[driver_id] = driver
                            del drivers_set[driver_id]        
                        
                        event_set = event_updater(event_set,['study driver arrives',[move_X,move_Y],next_driver_decision_time,driver.move_id,driver_id])
                    if action[3,0] == 1:
                        driver.delta_X = - param.V/60
                        driver.delta_Y = 0
                        driver.moving = 1
                        driver.busy = 0
                        driver.switch = 'on'
                        next_driver_decision_time = driver_decision_time + 1 
                        move_X = driver.delta_X
                        move_Y = driver.delta_Y
                        if offered == 1:
                            order_id = driver.offered_order
                            # delete order from assigned_order set and add it to pending_orders set
                            del assigned_orders[order_id]
                            pending_orders[order_id] = order
                            # rejected by driver
                            order.rejected_by.append(driver_id)  
                            
                        if driver.tired == 1:
                            driver.off_time = y[2]
                            signedout_drivers[driver_id] = driver
                            del drivers_set[driver_id]                            
                            
                            
                        
                        event_set = event_updater(event_set,['study driver arrives',[move_X,move_Y],next_driver_decision_time,driver.move_id,driver_id])
                    if action[4,0] == 1:
                        driver.delta_X = param.V/60
                        driver.delta_Y = 0
                        driver.moving = 1
                        driver.busy = 0
                        driver.switch = 'on'
                        next_driver_decision_time = driver_decision_time + 1 
                        move_X = driver.delta_X
                        move_Y = driver.delta_Y
                        if offered == 1:
                            order_id = driver.offered_order
                            # delete order from assigned_order set and add it to pending_orders set
                            del assigned_orders[order_id]
                            pending_orders[order_id] = order
                            # rejected by driver
                            order.rejected_by.append(driver_id)  
                            
                        if driver.tired == 1:
                            driver.off_time = y[2]
                            signedout_drivers[driver_id] = driver
                            del drivers_set[driver_id]                            
                            
                        
                        event_set = event_updater(event_set,['study driver arrives',[move_X,move_Y],next_driver_decision_time,driver.move_id,driver_id])
                        
                    if action[5,0] == 1:
                        driver.delta_X = 0
                        driver.delta_Y = 0
                        driver.switch = 'off'
                        o = driver.offered_order
                        d = driver.driver_id
                        driver.busy = 1
                        driver.moving = 0
                        rnd_time = 0.01
                        event_set = event_updater(event_set,
                                                  ['delivery accepted',[o,d],driver_decision_time+rnd_time])

    
    '''
    # save results of simulation
    orders_results = pd.DataFrame([[value for attr,value in order.__dict__.items()] for order in completed_orders.values()], \
                                    columns=[attr for attr,value in order.__dict__.items()])
    orders_results['ready_to_pickup'] = orders_results['pickup_time'] - orders_results['ready_time']
    
    drivers_results = pd.DataFrame([[value for attr,value in driver.__dict__.items()] for driver in signedout_drivers.values()], \
                                    columns=[attr for attr,value in driver.__dict__.items()])
    
    #drivers_results = pd.DataFrame([[value for attr,value in driver.__dict__.items()][0:26] for driver in signedout_drivers.values()], \
    #                                columns=['driver_id','X','Y','delta_X','delta_Y','on_time','off_time','busy','moving','last_order',
    #                                         'num_orders','tired','waiting','travelled_delivery','travelled_empty','earnings','X_before_order',
    #                                         'Y_before_order','action_before_order','t_before_order','last_distance_pickup','last_order_x',
    #                                         'last_order_y','last_order_action','last_profit','rejected'])  
        
    drivers_results['travelled'] = drivers_results['travelled_delivery'] + drivers_results['travelled_empty']
    drivers_results['work_duration'] = drivers_results['off_time'] - drivers_results['on_time']
    drivers_results['profit'] = drivers_results['earnings'] - param.C_d * drivers_results['travelled']
    
    orders_results.to_csv('/Users/weiwenzhou/Documents/research/Codes/Results/orders_results{}.csv'.format(counter),index=False)
    drivers_results.to_csv('/Users/weiwenzhou/Documents/research/Codes/Results/drivers_results{}.csv'.format(counter),index=False)
    profit_study += float(drivers_results.loc[drivers_results['driver_id']==study_driver]['profit'])
    '''
    #profit_twin += float(drivers_results.loc[drivers_results['driver_id']==999]['profit'])
    #if float(drivers_results.loc[drivers_results['driver_id']==study_driver]['profit']) > float(drivers_results.loc[drivers_results['driver_id']==999]['profit']):
        #num_wins+=1
    #Profit_runs.append(float(drivers_results.loc[drivers_results['driver_id']==study_driver]['profit']))
    Profit_runs.append(cul_profit)
    print(np.mean(Profit_runs))
    if num_offer != 0:
        print(num_accept / num_offer)
    
np.save('C:/Users/wzhou/Documents/Meal Delivery Problem paper/Codes/Result_data/Result_data/Layout1_3/Closest_rest/Profit_5.npy',Profit_runs)
  
   


    
    
    
    
    
    
    
    
    
ave_profit_study=profit_study
#ave_profit_twin = profit_twin/200
'''
fig = plt.figure()
plt.scatter(drivers_results['work_duration'],drivers_results['earnings'])
fig = plt.figure()
plt.scatter(drivers_results['travelled'],drivers_results['earnings'])
fig = plt.figure()
plt.scatter(drivers_results['X'],drivers_results['Y'],s=5,color='green',alpha=1)
fig = plt.figure()
plt.scatter(drivers_results['work_duration'],drivers_results['profit'])
fig = plt.figure()
plt.scatter(drivers_results['travelled_empty'],drivers_results['num_orders'])
'''



#print(num_wins/10)
print(ave_profit_study)
#result_set.to_csv('result_set.csv')

stop = timeit.default_timer() 
print("Run Time =", stop - start, "Seconds")
'''
plt.plot(Profit_runs[:400])
plt.plot(profit_average[:400])
plt.xlabel('episodes')
plt.ylabel('profits')
plt.show()
'''