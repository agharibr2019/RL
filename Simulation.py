class Driver:
    def __init__(self,driver_id,X,Y,delta_X,delta_Y,on_time,off_time,busy=0,
                 moving=0,last_order=-1,num_orders=0,tired=0,waiting=0,
                 travelled_delivery=0,travelled_empty=0,earnings=0):
        self.driver_id = driver_id
        self.X = X
        self.Y = Y
        self.delta_X = delta_X
        self.delta_Y = delta_Y
        self.on_time = on_time
        self.off_time = off_time
        self.busy = busy
        self.moving = moving
        self.last_order = last_order
        self.num_orders = num_orders
        self.tired = tired
        self.waiting = waiting
        self.travelled_delivery = travelled_delivery
        self.travelled_empty = travelled_empty
        self.earnings = earnings
        
class Order:
    def __init__(self,order_id,X_cust,Y_cust,placement_time,ready_time,
                 assignment_time,pickup_time,dropoff_time,restaurant,
                 X_rest,Y_rest,tip,distance,fee,rejected_by=[]):
        self.order_id = order_id
        self.X_cust = X_cust
        self.Y_cust = Y_cust
        self.placement_time = placement_time
        self.ready_time = ready_time
        self.assignment_time = assignment_time
        self.pickup_time = pickup_time
        self.dropoff_time = dropoff_time
        self.restaurant = restaurant
        self.X_rest = X_rest
        self.Y_rest = Y_rest
        self.tip = tip
        self.distance = distance
        self.fee = fee
        self.rejected_by = rejected_by