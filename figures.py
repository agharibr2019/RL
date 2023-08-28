import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
#%matplotlib qt
#%matplotlib inline

orders = pd.read_csv('orders1.csv')
drivers = pd.read_csv('drivers1.csv')
restaurants = pd.read_csv('restaurants.csv')
orders_results = pd.read_csv('orders1.csv')
drivers_results = pd.read_csv('drivers1.csv')

# heatmap for all data (pickups and dropoffs)
def Heat_Map(orders):
    heat_map = folium.Map(location=[orders['X_cust'].mean(),orders['Y_cust'].mean()], 
                          #tiles = 'Stamen Toner',
                          tiles = None,#"CartoDB dark_matter",
                          zoom_start = 20)
    data_pickup = orders[['X_rest','Y_rest']]
    data_dropoff = orders[['X_cust','Y_cust']]
    HeatMap(data_pickup,name='pickup', radius=10).add_to(heat_map)
    HeatMap(data_dropoff,name='dropoff', radius=10).add_to(heat_map)
    folium.LayerControl().add_to(heat_map)
    heat_map.save('Figures/heat_map.html')

Heat_Map(orders)

# visualize drivers working hour
def driver_hours(drivers):
    drivers = drivers.sort_values(by=['on_time','id']).reset_index(drop=True)
    i = 0
    fig = plt.figure(figsize=(10,12))
    for i in drivers['id'].unique():
        plt.plot([drivers.iloc[i]['on_time']/60,drivers.iloc[i]['off_time']/60],
                 [i,i],color='blue',alpha=1,ls='solid',linewidth=1)
        i += 1
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('Drivers',fontsize=14)
    plt.xticks(range(6,25),range(6,25))
    plt.savefig('Figures/Working Hours.png',dpi=300)

driver_hours(drivers)

# visualize orders, resaurant, drivers locations
def area(orders,restaurants,drivers):
    fig = plt.figure()
    plt.scatter(orders['X_cust'],orders['Y_cust'],s=5,color='red',alpha=0.7)
    plt.scatter(restaurants['X'],restaurants['Y'],s=5,color='blue',alpha=0.7)
    plt.scatter(drivers['X'],drivers['Y'],s=5,color='green',alpha=0.7)
    plt.title('Study Area',fontsize=14)
    plt.xlabel('X',fontsize=14)
    plt.ylabel('Y',fontsize=14)
    #for i in range(0,param.I+1):
    #    plt.hlines(i*param.delta,xmin=0,xmax=param.J*param.delta)
    #for j in range(0,param.J+1):
    #    plt.vlines(j*param.delta,ymin=0,ymax=param.I*param.delta)
    plt.savefig('Figures/Study Area.png',dpi=300)

area(orders,restaurants,drivers)

def customer_map(orders):
    fig = plt.figure()
    plt.scatter(orders['X_cust'],orders['Y_cust'],s=5,color='red',alpha=1)
    plt.title('Customers',fontsize=14)
    plt.xlabel('X',fontsize=14)
    plt.ylabel('Y',fontsize=14)
    #for i in range(0,param.I+1):
    #    plt.hlines(i*param.delta,xmin=0,xmax=param.J*param.delta)
    #for j in range(0,param.J+1):
    #    plt.vlines(j*param.delta,ymin=0,ymax=param.I*param.delta)
    plt.savefig('Figures/Customer Map.png',dpi=300)

customer_map(orders)

def restaurant_map(restaurants):
    fig = plt.figure()
    plt.scatter(restaurants['X'],restaurants['Y'],s=5,color='blue',alpha=1)
    plt.title('Restaurants',fontsize=14)
    plt.xlabel('X',fontsize=14)
    plt.ylabel('Y',fontsize=14)
    #for i in range(0,param.I+1):
    #    plt.hlines(i*param.delta,xmin=0,xmax=param.J*param.delta)
    #for j in range(0,param.J+1):
    #    plt.vlines(j*param.delta,ymin=0,ymax=param.I*param.delta)
    plt.savefig('Figures/Restaurant Map.png',dpi=300)

restaurant_map(restaurants)

def driver_map(drivers):
    fig = plt.figure()
    plt.scatter(drivers['X'],drivers['Y'],s=5,color='green',alpha=1)
    plt.title('Drivers Initial Location',fontsize=14)
    plt.xlabel('X',fontsize=14)
    plt.ylabel('Y',fontsize=14)
    #for i in range(0,param.I+1):
    #    plt.hlines(i*param.delta,xmin=0,xmax=param.J*param.delta)
    #for j in range(0,param.J+1):
    #    plt.vlines(j*param.delta,ymin=0,ymax=param.I*param.delta)
    plt.savefig('Figures/Driver Map.png',dpi=300)

def driver_earning1(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['work_duration'],drivers_results['earnings'],s=5,color='orange',alpha=1)
    #plt.title('Drivers Earning ($)',fontsize=14)
    plt.xlabel('Work Duration (minutes)',fontsize=14)
    plt.ylabel('Earning ($)',fontsize=14)
    plt.savefig('Figures/Driver Earning1.png',dpi=300)

driver_earning1(drivers_results)

def driver_earning2(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['travelled'],drivers_results['earnings'],s=5,color='orange',alpha=1)
    #plt.title('Drivers Earning ($)',fontsize=14)
    plt.xlabel('Distance Travelled (miles)',fontsize=14)
    plt.ylabel('Earning ($)',fontsize=14)
    plt.savefig('Figures/Driver Earning2.png',dpi=300)

driver_earning2(drivers_results)

def driver_earning3(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['travelled_empty'],drivers_results['earnings'],s=5,color='orange',alpha=1)
    #plt.title('Drivers Earning ($)',fontsize=14)
    plt.xlabel('Distance Travelled While Not Doing a Delivery (miles)',fontsize=14)
    plt.ylabel('Earning ($)',fontsize=14)
    plt.savefig('Figures/Driver Earning3.png',dpi=300)

driver_earning3(drivers_results)

def driver_profit1(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['travelled_empty'],drivers_results['profit'],s=5,color='orange',alpha=1)
    #plt.title('Drivers Profit ($)',fontsize=14)
    plt.xlabel('Distance Travelled While Not Doing a Delivery (miles)',fontsize=14)
    plt.ylabel('Profit ($)',fontsize=14)
    plt.savefig('Figures/Driver Profit1.png',dpi=300)

driver_profit1(drivers_results)

def driver_profit2(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['travelled'],drivers_results['profit'],s=5,color='orange',alpha=1)
    #plt.title('Drivers Profit ($)',fontsize=14)
    plt.xlabel('Distance Travelled (miles)',fontsize=14)
    plt.ylabel('Profit ($)',fontsize=14)
    plt.savefig('Figures/Driver Profit2.png',dpi=300)

driver_profit2(drivers_results)

def driver_num_order1(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['travelled_empty'],drivers_results['num_orders'],s=5,color='orange',alpha=1)
    #plt.title('Number of Orders  vs Travelled Empty',fontsize=14)
    plt.xlabel('Distance Travelled While Not Doing a Delivery (miles)',fontsize=14)
    plt.ylabel('Number of Orders',fontsize=14)
    plt.savefig('Figures/Number of Orders1.png',dpi=300)

driver_num_order1(drivers_results)

def driver_num_order2(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['num_orders'],drivers_results['profit'],s=5,color='orange',alpha=1)
    #plt.title(' Number of Orders vs Profit ($)',fontsize=14)
    plt.xlabel('Number of Orders',fontsize=14)
    plt.ylabel('Profit ($)',fontsize=14)
    plt.savefig('Figures/Number of Orders2.png',dpi=300)

driver_num_order2(drivers_results)

def driver_num_order3(drivers_results):
    fig = plt.figure()
    plt.scatter(drivers_results['work_duration'],drivers_results['num_orders'],s=5,color='orange',alpha=1)
    #plt.title(' Number of Orders vs Profit ($)',fontsize=14)
    plt.xlabel('Work Duration (minutes)',fontsize=14)
    plt.ylabel('Number of Orders',fontsize=14)
    plt.savefig('Figures/Number of Orders3.png',dpi=300)

driver_num_order3(drivers_results)