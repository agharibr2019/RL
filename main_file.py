from Simulation import *
from gurobipy import *
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


for counter in range(1, param.num_runs+1):
    # initialize orders, vehicels, and event_set
    orders = pd.read_csv('Data/orders{}.csv'.format(counter))
    drivers = pd.read_csv('Data/drivers{}.csv'.format(counter))
    event_set = event_starter(orders,drivers)
    for i in range(math.ceil(event_set[0][2]),math.ceil(event_set[-1][2]),param.decision_interval):
        event_set = event_updater(event_set,['decision point','d',i])
    
    # initialize sets
    drivers_set = {}
    signedout_drivers = {}
    pending_orders = {}
    assigned_orders = {}
    completed_orders = {}

    # read event list one by one
    for x in event_set:
        if x[2] > 1700:
            break
        #print(x[0],x[1],x[2])
        # chaeck if event is 'decision point' event
        if x[0] == 'decision point':
            time = x[2]
            # first check if any driver is available
            available_drivers = {}
            for driver_id in drivers_set:
                driver = drivers_set[driver_id]
                if driver.busy == 0:
                    driver.waiting = driver.waiting + param.decision_interval
                    if driver.moving == 1:
                        driver.X = driver.X + driver.delta_X
                        driver.Y = driver.Y + driver.delta_Y
                        driver.travelled_empty = driver.travelled_empty + abs(driver.delta_X) + abs(driver.delta_Y)
                    available_drivers[driver_id] = drivers_set[driver_id]
            if (not available_drivers) or (not pending_orders):
                do_nothing = 1
            else:
                # create a new Gurobi Model
                model = Model("Driver-Order Assignment")
                model.setParam(GRB.Param.OutputFlag, 0)
                
                # create indeces for orders and drivers
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

                # create variables
                x = {}
                
                for o in orders_id:
                    for d in drivers_id:
                        x[o,d] = model.addVar(vtype = "B", name = "x[%s,%s]"%(o,d))

                # Add Constraints
                for d in drivers_id:
                    model.addConstr(quicksum(x[o,d] for o in orders_id) <= 1, name = "C3[%s]"%(d))

                for o in orders_id:
                    model.addConstr(quicksum(x[o,d] for d in drivers_id) <= 1, name = "C4[%s]"%(o))

                for o in orders_id:
                    for d in drivers_id:
                        model.addConstr(dist[o,d]-param.max_dist <= (1-x[o,d])*10000, name = "C5[%s,%s]"%(o,d))

                model.Params.OutputFlag = 0
                model.Params.Thread = 0
            
                # Set the objective function
                model.setObjective(quicksum(dist[o,d]*x[o,d] for o in orders_id for d in drivers_id)
                                   -1000000*quicksum(r[o]*x[o,d] for o in orders_id for d in drivers_id)
                                   -100000*quicksum(f[d]*x[o,d] for o in orders_id for d in drivers_id),GRB.MINIMIZE)
                
                # Solve the model
                model.optimize()
                
                # save model and solution
                #model.write("Model.lp")
                #model.write("Model.sol")
            
                if model.status == GRB.Status.OPTIMAL:
                    #print('---------------------------------------')
                    x_optimal = {}
                    for o in orders_id:
                        for d in drivers_id:
                            opt_decision = model.getVarByName("x[%s,%s]"%(o,d)).x
                            if opt_decision == 1:
                                #print(o,d)
                                order = pending_orders[o]
                                driver = available_drivers[d]
                                
                                # update busy and waiting attribute of driver
                                driver.busy = 1
                                driver.waiting = 0
                                
                                # add order to set of assigned orders
                                assigned_orders[o] = order
                                
                                # remove order from pending order set
                                del pending_orders[o]
    
                                # create either a 'delivery accepted' or 'delivery rejected' event
                                rnd = random.random()
                                rnd_time = random.random()*param.max_resp_time
                                if rnd <= param.trip_prob:
                                    # deivery is accepted
                                    event_set = event_updater(event_set,
                                                              ['delivery accepted',[o,d],time+rnd_time])
                                else:
                                    #if rejected
                                    event_set = event_updater(event_set,
                                                              ['delivery rejected',[o,d],time+rnd_time])
                            else:
                                xxx = 1
                else:
                    print(model.status)

        # check if event x is 'order ready' event
        elif x[0] == 'order ready':
            # create an order object
            order_data = orders[orders['id'] == x[1]]
            order = Order(order_id=x[1],X_cust=order_data['X_cust'].tolist()[0],
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
            pending_orders[x[1]] = order

        elif x[0] == 'driver on':
            # create a driver object
            driver_data = drivers[drivers['id'] == x[1]]
            driver = Driver(driver_id=x[1],
                            X=driver_data['X'].tolist()[0],
                            Y=driver_data['Y'].tolist()[0],
                            delta_X=0,delta_Y=0,busy=0,
                            on_time=driver_data['on_time'].tolist()[0],
                            off_time=0,moving=0,last_order=-1,num_orders=0,
                            tired=0,waiting=0,travelled_delivery=0,
                            travelled_empty=0,earnings=0)
            drivers_set[x[1]] = driver
            
            # determine how long driver will stay at current location
            idle_time = x[2] + np.random.normal(loc=param.idle_time_avg,
                                                scale=param.idle_time_std,size=1)[0]
            # create a 'driver moves' event for driver
            event_set = event_updater(event_set,['driver moves',[-1,x[1]],idle_time])

        elif x[0] == 'driver off':
            driver = drivers_set[x[1]]
            # if driver is not performing a delivery job remove it
            if driver.busy == 0:
                driver.off_time = x[2]
                signedout_drivers[x[1]] = driver
                del drivers_set[x[1]]
            # else if drver is peroforming a delivery job, change tired attribute
            # and remove it when the delivery is completed
            else:
                driver.tired = 1

        elif x[0] == 'delivery accepted':
            # get order and driver id associated with the accepted delivery
            order_id = x[1][0]
            driver_id = x[1][1]
            
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
            order.assignment_time = x[2]
            order.pickup_time = x[2] + pickup_time
            order.dropoff_time = x[2] + pickup_time + dropoff_time
            
            event_set = event_updater(event_set,['delivery completed',[order_id,driver_id],order.dropoff_time])

        elif x[0] == 'delivery rejected':
            # get order and driver id associated with the accepted delivery
            order_id = x[1][0]
            driver_id = x[1][1]

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
                driver.off_time = x[2]
                signedout_drivers[driver_id] = driver
                del drivers_set[driver_id]
                    
        elif x[0] == 'delivery completed':
            order_id = x[1][0]
            driver_id = x[1][1]
            
            order = assigned_orders[order_id]
            driver = drivers_set[driver_id]

            # add order to set of completed orders
            completed_orders[order_id] = order
            
            # remove order from accepted orders set
            del assigned_orders[order_id]
            
            # update X and Y of driver
            driver.X = order.X_cust
            driver.Y = order.Y_cust
            
            # collect earnings
            driver.earnings = driver.earnings + order.fee + order.tip

            # update last order 
            driver.last_order = order_id
            
            # change availability of driver or remove it if tired = 1
            if driver.tired == 1:
                driver.off_time = x[2]
                signedout_drivers[driver_id] = driver
                del drivers_set[driver_id]
            else:
                driver.busy = 0

                # determine how long driver will stay at current location
                idle_time = x[2] + np.random.normal(loc=param.idle_time_avg,
                                                    scale=param.idle_time_std,size=1)[0]
                # create a 'driver moves' event for driver
                event_set = event_updater(event_set,['driver moves',[order_id,driver_id],idle_time])

        elif x[0] == 'driver moves':
            # find the driver
            order_id = x[1][0]
            driver_id = x[1][1]

            if driver_id in list(drivers_set.keys()):
                driver = drivers_set[driver_id]

                if (driver.busy == 0 and driver.last_order == order_id):
                    new_X = stats.truncnorm.rvs(param.a[0],param.b[0],loc=param.mean[0],scale=param.std[0],size=1).tolist()[0]
                    new_Y = stats.truncnorm.rvs(param.a[1],param.b[1],loc=param.mean[1],scale=param.std[1],size=1).tolist()[0]

                    driving_time = x[2] + 60*(abs(driver.X-new_X)+abs(driver.Y-new_Y))/param.V
                    
                    driver.delta_Y = (param.V*param.decision_interval/60)/(1+(abs(new_X-driver.X)/abs(new_Y-driver.Y)))*np.sign(new_Y-driver.Y).tolist()
                    driver.delta_X = (abs(new_X-driver.X)/abs(new_Y-driver.Y))*abs(driver.delta_Y)*np.sign(new_X-driver.X).tolist()
                                        
                    driver.moving = 1

                    # create a 'driver arrives' event for driver
                    event_set = event_updater(event_set,['driver arrives',[driver_id,new_X,new_Y],driving_time])
                   
        elif x[0] == 'driver arrives':
            driver_id = x[1][0]
            if driver_id in drivers_set:
                driver = drivers_set[x[1][0]]
                
                driver.X = x[1][1]
                driver.Y = x[1][2]
                driver.moving = 0
                
                # determine how long driver will stay at current location
                idle_time = x[2] + np.random.normal(loc=param.idle_time_avg,
                                                    scale=param.idle_time_std,size=1)[0]
                # create a 'driver moves' event for driver
                event_set = event_updater(event_set,['driver moves',[driver.last_order,driver_id],idle_time])

    # save results of simulation
    orders_results = pd.DataFrame([[value for attr,value in order.__dict__.items()] for order in completed_orders.values()], \
                                    columns=[attr for attr,value in order.__dict__.items()])
    orders_results['ready_to_pickup'] = orders_results['pickup_time'] - orders_results['ready_time']
    
    drivers_results = pd.DataFrame([[value for attr,value in driver.__dict__.items()] for driver in signedout_drivers.values()], \
                                    columns=[attr for attr,value in driver.__dict__.items()])
    drivers_results['travelled'] = drivers_results['travelled_delivery'] + drivers_results['travelled_empty']
    drivers_results['work_duration'] = drivers_results['off_time'] - drivers_results['on_time']
    drivers_results['profit'] = drivers_results['earnings'] - param.C_d * drivers_results['travelled']
    
    orders_results.to_csv('Results/orders_results{}.csv'.format(counter),index=False)
    drivers_results.to_csv('Results/drivers_results{}.csv'.format(counter),index=False)

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

#result_set.to_csv('result_set.csv')

stop = timeit.default_timer() 
print("Run Time =", stop - start, "Seconds")