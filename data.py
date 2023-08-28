import param
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

################################# Data ########################################

### retaurnat data
restaurants = pd.DataFrame(columns=['id','X','Y','avg_ready_time','hourly_demand','avg_tip'])
restaurants['id'] = np.arange(0,param.R) # generate restaurant id's
restaurants['avg_ready_time'] = np.random.uniform(low=15,high=45,size=param.R) # generate avg order ready times
restaurants['X'] = stats.truncnorm.rvs(param.a[0],param.b[0],loc=param.mean[0],scale=param.std[0],size=param.R)
restaurants['Y'] = stats.truncnorm.rvs(param.a[1],param.b[1],loc=param.mean[1],scale=param.std[1],size=param.R)
restaurants['hourly_demand'] = np.random.uniform(low=param.demand_low,high=param.demand_high,size=param.R)
restaurants['avg_tip'] = np.random.uniform(low=param.tip_low,high=param.tip_high,size=param.R)
restaurants.to_csv('Data/restaurants.csv',index=False)

### order data
for i in range(1,param.num_runs+1):
    orders = pd.DataFrame(columns = ['id','X_cust','Y_cust','placement_time',
                                     'ready_time','restaurant','X_rest','Y_rest','tip'])
    index = 0
    for rest in restaurants['id']:
        for t in range(0,param.T):
            x = np.random.exponential(scale=1/(restaurants.iloc[int(rest)]['hourly_demand']*param.temporal_demand[t]), size=1)[0]*60
            if x < 60:
                orders.at[index,'id'] = index
                orders.at[index,'X_cust'] = np.random.uniform(low=0,high=param.I*param.delta) #stats.truncnorm.rvs(a_i,b_i,loc=mean_i,scale=std_i)
                orders.at[index,'Y_cust'] = np.random.uniform(low=0,high=param.J*param.delta) #stats.truncnorm.rvs(a_j,b_j,loc=mean_j,scale=std_j)
                orders.at[index,'placement_time'] = 60*(param.t_initial+t)+x
                orders.at[index,'ready_time'] = 60*(param.t_initial+t)+x + np.random.normal(loc=restaurants.iloc[int(rest)]['avg_ready_time'],
                                                                                      scale=restaurants.iloc[int(rest)]['avg_ready_time']/5,
                                                                                      size=1)[0]
                orders.at[index,'restaurant'] = rest
                orders.at[index,'X_rest'] = restaurants[restaurants['id'] == rest]['X'].tolist()[0]
                orders.at[index,'Y_rest'] = restaurants[restaurants['id'] == rest]['Y'].tolist()[0]
                avg_tip = restaurants.iloc[int(rest)]['avg_tip']
                orders.at[index,'tip'] = int(stats.truncnorm.rvs((0-avg_tip)/param.std_tip,10000,loc=avg_tip,scale=param.std_tip,size=1)[0])
                y = np.random.exponential(scale=1/(restaurants.iloc[int(rest)]['hourly_demand']*param.temporal_demand[t]), size=1)[0]*60
                index += 1
                while x+y < 60:
                    x += y
                    orders.at[index,'id'] = index
                    orders.at[index,'X_cust'] = np.random.uniform(low=0,high=param.I*param.delta) #stats.truncnorm.rvs(a_j,b_j,loc=mean_j,scale=std_j)
                    orders.at[index,'Y_cust'] = np.random.uniform(low=0,high=param.J*param.delta) #stats.truncnorm.rvs(a_j,b_j,loc=mean_j,scale=std_j)
                    orders.at[index,'placement_time'] = 60*(param.t_initial+t)+x
                    orders.at[index,'ready_time'] = 60*(param.t_initial+t)+x + np.random.normal(loc=restaurants.iloc[int(rest)]['avg_ready_time'],
                                                                                          scale=restaurants.iloc[int(rest)]['avg_ready_time']/5,
                                                                                          size=1)[0]
                    orders.at[index,'restaurant'] = rest
                    orders.at[index,'X_rest'] = restaurants[restaurants['id'] == rest]['X'].tolist()[0]
                    orders.at[index,'Y_rest'] = restaurants[restaurants['id'] == rest]['Y'].tolist()[0]
                    orders.at[index,'tip'] = np.random.poisson(restaurants.iloc[int(rest)]['avg_tip'])
                    avg_tip = restaurants.iloc[int(rest)]['avg_tip']
                    std_tip = 2
                    orders.at[index,'tip'] = int(stats.truncnorm.rvs((0-avg_tip)/std_tip,10000,loc=avg_tip,scale=std_tip,size=1)[0])
                    y = np.random.exponential(scale=1/(restaurants.iloc[int(rest)]['hourly_demand']*param.temporal_demand[t]), size=1)[0]*60
                    index += 1
    distance = abs(orders['X_cust'] - orders['X_rest']) + abs(orders['Y_cust'] - orders['Y_rest'])
    orders['distance'] = distance
    orders['fee'] = param.C_f + distance*param.C_a
    orders.to_csv('Data/orders{}.csv'.format(i),index=False)
    i += 1

### driver data
for i in range(1,param.num_runs+1):
    drivers = pd.DataFrame(columns = ['id','X','Y','on_time','off_time'])
    index = 0
    for t in range(0,param.T):
        x = np.random.exponential(scale=1/(param.drivers_rate*param.temporal_demand[t]), size=1)[0]*60
        if x < 60:
            drivers.at[index,'id'] = index
            drivers.at[index,'on_time'] = 60*(param.t_initial+t)+x
            drivers.at[index,'off_time'] = 60*(param.t_initial+t)+x + np.random.normal(loc=param.work_duration,scale=param.work_std,size=1)[0]
            y = np.random.exponential(scale=1/(param.drivers_rate*param.temporal_demand[t]), size=1)[0]*60
            index += 1
            while x+y < 60:
                x += y
                drivers.at[index,'id'] = index
                drivers.at[index,'on_time'] = 60*(param.t_initial+t)+x
                drivers.at[index,'off_time'] = 60*(param.t_initial+t)+x + np.random.normal(loc=120,scale=30,size=1)[0]
                y = np.random.exponential(scale=1/(param.drivers_rate*param.temporal_demand[t]), size=1)[0]*60
                index += 1
    drivers_num = len(drivers)
    drivers['X'] = stats.truncnorm.rvs(param.a[0],param.b[0],loc=param.mean[0],scale=param.std[0],size=drivers_num)
    drivers['Y'] = stats.truncnorm.rvs(param.a[1],param.b[1],loc=param.mean[1],scale=param.std[1],size=drivers_num)
    drivers.to_csv('Data/drivers{}.csv'.format(i),index=False)
    i += 1

