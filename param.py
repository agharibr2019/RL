import numpy as np
import math
import os
import pandas as pd

# parameters of simulation
num_runs = 1 #30 # number of runs
decision_interval = 1
trip_prob = 0.8 # probability of accepting trip by a driver
max_resp_time = 2 # max response time for a trip to be accepted by a driver in minutes
pickup_service_time = 1 # min pickup service time
dropoff_service_time = 1 # min dropoff service time
max_dist = 5 # max distance from a restuanrant to be assinged to an order from there (miles)

# parameters of area
I = 5 # number of rows in the area
J = 5 # number of columns in the area
delta = 5 # length of each zone in the area
T = 14 # number of time periods (hours)
t_initial = 10 # start time of operation (hour)

# parameters of restaurants
R = 500 # number of restaurants
demand_low = 0.02 # base demand rate (low)
demand_high = 0.10 # base demand rate (high)
tip_low = 1 # minimum for mean of tip 
tip_high = 5 # maximum for mean of tip
std_tip = 2 # std for tip distribution

# parameters of drivers
V = 35 # mph
drivers_rate = 20 # drivers sign in density
work_duration = 120 # mean of working duration (minutes)
work_std = 30 # std of working duration (minutes)
C_d = 0.2 # cost of drivering per mile
idle_time_avg = 10
idle_time_std = 2


# parameters of meal delivery compensation
C_f = 4 # fixed compensation for each delivery
C_a = 0.5 # additional pay for each delivery per mile

# parameters of restaurants and drivers geographical distributations
mean = np.array([np.random.uniform(0,I*delta),np.random.uniform(0,J*delta)]) # mean for restaurant distribution and drivers
std = np.array([5, 5]) # std for restaurant distribution and drivers
a = (0-mean)/std # standardized start point for truncated normal dist
b = (I*delta-mean)/std # standardized end point for truncated normal dist
temporal_demand = [1, 2, 3, 2.5, 2, # temporal pattern of order and drivers for 14 hours of operation
                   1.5, 2, 4, 7, 6,
                   4, 3, 2, 1]
