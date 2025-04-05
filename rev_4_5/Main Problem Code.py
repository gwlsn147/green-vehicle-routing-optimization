import numpy as np
import pandas as pd
import math
import datetime

import gurobipy as gp
from gurobipy import GRB

model = gp.Model("multi_objective_milp")

#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#log_file = f"gurobi_log_{timestamp}.txt"
#model.setParam('LogFile', log_file)
#model.setParam('MIPFocus', 1)          # Emphasize finding feasibility
model.setParam('Heuristics', .3)       # Spend more effort on heuristic searches
#model.setParam('MIPGap', 0.01)         # 1% gap
#model.setParam('SolutionLimit', 1)     # Stop after the first feasible solution

# ===============================================================================================================================
# ===============================================================================================================================
#                                PROBLEM SET-UP
# ===============================================================================================================================
# ===============================================================================================================================

# =============================================================================
# 1. Define Sets
# =============================================================================

# Available vehicles:
K_c = {f"CFV{i}" for i in range (1,7)}   
K_e = {f"HGEV{i}" for i in range (1,7)}         
K = K_c.union(K_e)              

K_c_list = sorted(K_c)   # List of available CFVs
K_e_list = sorted(K_e)   # List of available HGEVs
K_list   = sorted(K)     # All vehicles

num_customers = len(pd.read_csv("Walmart Small/Demand.csv", header=None))-1
D = set(range(0,num_customers))
depot = num_customers

N_original = D.union([depot])

# -------------------------------
# Network Expansion Settings
# -------------------------------
Vmax = 2  # maximum allowed visits per customer

# Expanded customer set: each customer i is duplicated Vmax times.
D_expanded = {(i, v) for i in D for v in range(1, Vmax+1)}
# Expanded node set: depot is represented as (depot,0)
N_expanded = sorted({(depot, 0)}.union(D_expanded), key=lambda x: (x[0], x[1]))

# -------------------------------
# Tour Settings
# -------------------------------

Zmax = 2  # number of tours
Z = range(1, Zmax+1)  # set of tour indices

# =============================================================================
# 2. Define Parameters
# =============================================================================

# -------------------------------------------------------
# 2.1 Parameters Indexed by Vehicle (k in K)
# -------------------------------------------------------

# A^k: Acquisition cost for vehicle k ($USD)
A = {k: 165000 if k in K_c_list else 400000 for k in K_list}

# S^k: Subsidy amount for vehicle k ($USD)
S = {k: 0 if k in K_c_list else 0 for k in K_list}

# h^k: Energy consumption per distance by vehicle k (kWh/m)
h = {k: .004375 if k in K_c_list else .0011 for k in K_list}

# Q^k: The maximum load capacity for vehicle k  (kg)
Q = {k: 15000 for k in K_list}

# R^k: The battery capacity of vehicle k (kWh)
R = {k: 99999 if k in K_c_list else 900 for k in K_list}

# O^k: The operating costs of vehicle k per meter ($USD/m)
O = {k: .00018125 if k in K_c_list else .00015875 for k in K_list}

# -------------------------------------------------------
# 2.2 Parameters Indexed by Arc (i, j)
# -------------------------------------------------------

# d_ij: Distance between vertex i and vertex j
# t_ij: Travel time from vertex i to vertex j
# v_avg_ij: Average velocity between vertex i and vertex j

#Distance matrix:
Distance_Matrix = pd.read_csv("Walmart Small/Distances.csv", header=None)
distance_array = Distance_Matrix.values

#Travel time matrix:
Travel_Times = pd.read_csv("Walmart Small/Times.csv", header=None)
time_array = Travel_Times.values

d = {}
t = {}
for i in range(len(N_original)):
    for j in range(len(N_original)):
        if i != j:
            d[(i, j)] = distance_array[i, j]
            t[(i, j)] = time_array[i, j]

# υ: The average speed of the vehicle (m/s)
v_avg = {}

#copies of customers that could also be charging stations
ghost_pairs = {(0, 20), (11, 21), (20, 0), (21, 11)}

for i in range(len(N_original)):
    for j in range(len(N_original)):
        if i == j:
            continue
        dist_ij = distance_array[i, j]
        time_ij = time_array[i, j]
        if (i, j) in ghost_pairs:
            if time_ij == 0:
                time_ij = 0.1
            if time_ij != 0:
                v_avg[(i, j)] = dist_ij / time_ij        
        else:
            if dist_ij == 0 and time_ij == 0:
                continue
            if time_ij == 0 and dist_ij != 0:
                continue            
            v_avg[(i, j)] = dist_ij / time_ij

# CS: Set of customers that also could have recharging stations at them
CS = {20, 21}

# -------------------------------------------------------
# 2.3 Expanded Network Data
# -------------------------------------------------------
# For any two expanded nodes i_node and j_node in N_expanded, define:
#   - if either node is the depot (i.e. (depot,0)), use the original distance/time.
#   - if both nodes correspond to customers and have the same original customer index, set distance = 0.
#   - otherwise, use the original distance/time between the two customer indices.
d_exp = {}
t_exp = {}
v_avg_exp = {}
for i_node in N_expanded:
    for j_node in N_expanded:
        if i_node == j_node:
            continue
        # Determine original indices: for depot, the index is depot; for customers, use i_node[0]
        i_orig = i_node[0]
        j_orig = j_node[0]
        if i_orig == depot or j_orig == depot:
            d_exp[(i_node, j_node)] = d[(i_orig, j_orig)]
            t_exp[(i_node, j_node)] = t[(i_orig, j_orig)]
            v_avg_exp[(i_node, j_node)] = v_avg.get((i_orig, j_orig), 0)
        else:
            if i_orig == j_orig:
                d_exp[(i_node, j_node)] = 0  # no extra travel cost between copies
                t_exp[(i_node, j_node)] = 0
                v_avg_exp[(i_node, j_node)] = 0
            else:
                d_exp[(i_node, j_node)] = d[(i_orig, j_orig)]
                t_exp[(i_node, j_node)] = t[(i_orig, j_orig)]
                v_avg_exp[(i_node, j_node)] = v_avg[(i_orig, j_orig)]

# -------------------------------------------------------
# 2.4 Demand Parameter
# -------------------------------------------------------

Customer_demand = pd.read_csv("Walmart Small/Demand.csv", header=None)
Demand = Customer_demand.values

p = {i: float(Demand[i, 0]) for i in range(len(N_original))}

# -------------------------------------------------------
# 2.5 Emissions, Energy, and Fuel-Related Parameters
# -------------------------------------------------------

# β: Unit cost of GHG emissions (carbon tax rate, $USD/ metric ton of CO2)
Beta = 24  # The actual tax rate using 'beta' instead of 'β' 

# μ: Regional average grid emissions factor for electricity (kg CO2/kWh)
mu = .035

# Ω: GHG emissions coefficient for diesel fuel (kg CO2/L)
Omega = 2.56
#changed from 2.79

# r: Recharging costs ($USD/kWh)
r = .11

# F: cost of diesel fuel per liter ($USD/L)
F = 1.14

# M: Safety margin for HGEVs (kWh). HGEVs always reserve at least 3% charge for return trips.
Ms = 27

# ξ: Fuel-to-air mass ratio
xi = .055

# f: Engine friction factor
f_param = .25

# I: Engine speed (revolutions/s)
I_param = 23
#based off of Peterbilt 579

# E: Engine displacement
E_param = 12.9
#based off of Peterbilt 579 PACCAR MX-13

# m: The efficiency parameter for diesel engines (engine thermal efficiency)
m_param = .33

# n: The heating value of typical diesel fuel (kJ/g)
n_param = 42

# o: The mechanical drive-train efficiency of the vehicle
o_param = .85

# W_c: Curb weight (front + rear axle kg)
W_c = 21000
#value of Peterbilt 579

# τ: The average acceleration of the vehicle (m/s^2)
tau = .68
#Using value from Amiri et al

# g: Gravitational constant (m/s^2)
g_param = 9.81

# θ: Angle of the road
theta = math.radians(4.57)
#look up reference later, it is in my debugging notes

# C_d: Aerodynamic drag coefficient
C_d = .6
#look up new reference later

# C_r: Rolling resistance coefficient
C_r = .01
#look up reference later

# α: Effective frontal area of the vehicle (tractor + trailer combination, m^2)
alpha = 25

# ψ: Constant for converting g/s to L/s
psi = 840

# ρ: Air density (kg/m^3)
rho = 1.2041

# -------------------------------------------------------
# 2.6 Additional Terms
# -------------------------------------------------------

#HGEV_Initiation_Cost: Cost of infrastructure associated with being able to support HGEVs
HGEV_Initiation_Cost = 5000000
#HGEV_Expanded_Cost: Cost of infrastructure associated with being able to support HGEVs if the expanded network is to be activated
HGEV_Expanded_Cost = 6000000

#Tmax: The maximum amount of time a vehicle may service customers
Tmax = 22 * 60 * 60 #22 hours in seconds

#Tstop: The penalty time added for each stop at customers and the depot
Tstop = 2 * 60 * 60 #2 hours in seconds

# Terms used to simplify equations
lambda_param = xi/(n_param*psi)  # (λ)
phi_param    = 1/(1000*m_param*o_param)  # (φ)
sigma_param  = tau + g_param*math.sin(theta) + g_param*C_r*math.cos(theta)  # (σ)
epsilon_param = .5*C_d*rho*alpha  # (ϵ)

# =============================================================================
# 3. Define Variables
# =============================================================================

# -------------------------------------------------------
# 3.1  x_ij^kz: 1 if vehicle k travels arc (i, j) in tour z, 0 otherwise
# -------------------------------------------------------
#    Integer variable.
x = model.addVars(K_list, Z, N_expanded, N_expanded, vtype=GRB.BINARY, name="x")

# -------------------------------------------------------
# 3.2  q_ij^kz: The flow of load carried from vertex i to vertex j by vehicle k in tour z
# -------------------------------------------------------
#    Continuous variable.
q = model.addVars(K_list, Z, N_expanded, N_expanded, vtype=GRB.CONTINUOUS, lb=0.0, name="q")

# -------------------------------------------------------
# 3.3  y_i^kz: Cumulative charge amount variable 
# -------------------------------------------------------
#    Continuous variable.
#    The current energy level of vehicle k (HGEV) when arriving at vertex i in tour z.
y = model.addVars(K_e_list, Z, N_expanded, vtype=GRB.CONTINUOUS, lb=0.0, name="y")

# -------------------------------------------------------
# 3.6  Vehicle route time
# -------------------------------------------------------
#    Continuous variable.
T = model.addVars(K_list, Z, vtype=GRB.CONTINUOUS, lb=0.0, name="TotalTime")

# -------------------------------------------------------
# 3.7  Helper variable to determine if vehicle is used at all
# -------------------------------------------------------
#   Binary variable: used[k] = 1 if vehicle k is used, 0 otherwise
used = model.addVars(K_list, vtype=GRB.BINARY, name="used")

# -------------------------------------------------------
# 3.9  U_i^kz: Cumulative load variable
# -------------------------------------------------------
#   Continuous variable.
#   U[(k,z,i)] represents the remaining load on vehicle k upon arriving at node i in tour z.
U = model.addVars(K_list, Z, N_expanded, vtype=GRB.CONTINUOUS, lb=0.0, name="U")

# -------------------------------------------------------
# 3.10  HGEV_used: variable that indicates if any EV is used at all
# -------------------------------------------------------
#   Integer variable.
#   1 indicates that at least one HGEV is used, 0 indicates that none are used.
HGEV_used = model.addVar(vtype=GRB.BINARY, name="HGEV_used")

# -------------------------------------------------------
# 3.11  HGEV_exp: variable that indicates if any HGEV is used to traverse a distance greater than the trigger distance
# -------------------------------------------------------
#   Integer variable.
#   1 indicates that at least one HGEV is used to traverse a distance greater than the trigger distance in a single tour
#   0 indicates that none are used to go this distance.
HGEV_exp = model.addVar(vtype=GRB.BINARY, name="HGEV_exp")

# -------------------------------------------------------
# 3.12  Delivered_k^iz: the amount of load delivered at customer i by vehicle k in tour z
# -------------------------------------------------------
#   Continuous variable
#   Used for tracking deliveries at each customer
delivered = model.addVars(K_list, Z, D_expanded, vtype=GRB.CONTINUOUS, lb=0.0, name="delivered")

# ===============================================================================================================================
# ===============================================================================================================================
#                                OBJECTIVE FUNCTIONS
# ===============================================================================================================================
# ===============================================================================================================================

# =============================================================================
# 1. Minimize Total Costs
# =============================================================================

# -------------------------------------------------------
# 1.1 Recharging Costs
# -------------------------------------------------------
term1 = r * gp.quicksum(
    h[k] * d_exp[(i, j)] * x[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_e_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)

# -------------------------------------------------------
# 1.2 Diesel Fuel Costs Due to Curb Weight
# -------------------------------------------------------
term2 = F * gp.quicksum(
    (W_c * phi_param * sigma_param * d_exp[(i, j)] +
     f_param * I_param * E_param * t_exp[(i, j)] +
     epsilon_param * phi_param * d_exp[(i, j)] * (v_avg_exp[(i, j)])**2
    ) * lambda_param * x[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_c_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)

# -------------------------------------------------------
# 1.3 Diesel Fuel Costs Due to Load Effect
# -------------------------------------------------------
term3 = F * gp.quicksum(
    phi_param * sigma_param * lambda_param * d_exp[(i, j)] * q[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_c_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)

# -------------------------------------------------------
# 1.4 General Operating Costs
# -------------------------------------------------------
term4 = gp.quicksum(
    O[k] * d_exp[(i, j)] * x[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)

# -------------------------------------------------------
# 1.5 Acquisition Costs (Includes Subsidies)
# -------------------------------------------------------
term5 = gp.quicksum((A[k] - S[k]) * used[k] for k in K_list)

# -------------------------------------------------------
# 1.6 HGEV Infrastructure Costs
# -------------------------------------------------------

term6 = HGEV_Initiation_Cost*HGEV_used+HGEV_Expanded_Cost*HGEV_exp

# -------------------------------------------------------
# 1.7 Cost of Diesel Emissions
# -------------------------------------------------------
term7_temp = gp.quicksum(
    (W_c * phi_param * sigma_param * d_exp[(i, j)] +
     f_param * I_param * E_param * t_exp[(i, j)] +
     epsilon_param * phi_param * d_exp[(i, j)] * (v_avg_exp[(i, j)])**2
    ) * lambda_param * x[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_c_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)
term7_temp += gp.quicksum(
    phi_param * sigma_param * lambda_param * d_exp[(i, j)] * q[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_c_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)
term7 = Beta * Omega * term7_temp / 1000 #divided by a thousand as the price for carbon is per metric ton

# -------------------------------------------------------
# Complete Expression
# -------------------------------------------------------
obj_expr_cost = term1 + term2 + term3 + term4 + term5 + term6 + term7

# =============================================================================
# 2. Minimize GHG Emissions
# =============================================================================

# -------------------------------------------------------
# 2.1 Diesel Emissions
# -------------------------------------------------------
term8_temp = gp.quicksum(
    (W_c * phi_param * sigma_param * d_exp[(i, j)] +
     f_param * I_param * E_param * t_exp[(i, j)] +
     epsilon_param * phi_param * d_exp[(i, j)] * (v_avg_exp[(i, j)])**2
    ) * lambda_param * x[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_c_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)
term8_temp += gp.quicksum(
    phi_param * sigma_param * lambda_param * d_exp[(i, j)] * q[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_c_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)
term8 = Omega * term8_temp

# -------------------------------------------------------
# 2.2 Electricity-Related Emissions from Charging HGEVs
# -------------------------------------------------------
term9 = mu * gp.quicksum(
    h[k] * d_exp[(i, j)] * x[(k, z, i[0], i[1], j[0], j[1])]
    for k in K_e_list for z in Z
    for i in N_expanded for j in N_expanded if i != j
)

# -------------------------------------------------------
# Complete Expression
# -------------------------------------------------------
obj_expr_emissions = (term8 + term9)/1000 #divided by 1000 to output metric tons

# ===============================================================================================================================
# ===============================================================================================================================
#                                CONSTRAINTS
# ===============================================================================================================================
# ===============================================================================================================================

# =============================================================================
# 1. Routing Constraints
# =============================================================================

# -------------------------------------------------------
# 1.1 Vehicles Must Depart from the Depot
# -------------------------------------------------------
model.addConstrs(
    gp.quicksum(x[(k, z, depot, 0, j[0], j[1])] for z in Z for j in N_expanded if j != (depot, 0)) >= used[k]
    for k in K_list
)

# -------------------------------------------------------
# 1.2 Conservation of Routing
# -------------------------------------------------------
# For each vehicle, the number of arcs leaving the depot equal the number of arcs arriving at the depot
# Allow vehicles to leave and return to the depot multiple times,
# but enforce that the number of departures equals the number of returns.
model.addConstrs(
    gp.quicksum(x[(k, z, depot, 0, j[0], j[1])] for j in N_expanded if j != (depot, 0)) ==
    gp.quicksum(x[(k, z, i[0], i[1], depot, 0)] for i in N_expanded if i != (depot, 0))
    for k in K_list for z in Z
)

# -------------------------------------------------------
# 1.3 Connectivity of Tours
# -------------------------------------------------------
model.addConstrs(
    (gp.quicksum(x[(k, z, i[0], i[1], j[0], j[1])] for i in N_expanded if i != j) -
     gp.quicksum(x[(k, z, j[0], j[1], i[0], i[1])] for i in N_expanded if i != j)) == 0
    for k in K_list for z in Z for j in D_expanded
)

# -------------------------------------------------------
# 1.4 Maximum Route Time
# -------------------------------------------------------
# Each time a vehicle stops a 2-hour time penalty is incurred (for loading unloading)
# Define penalty nodes as all expanded nodes except those corresponding to the depot or a charging station.
penalty_nodes = {node for node in N_expanded if node[0] != depot and node[0] not in CS}

for k in K_list:
    # Compute the total travel time using the expanded travel time dictionary t_exp
    for z in Z:
        travel_time_expr = gp.quicksum(
            t_exp[(i, j)] * x[(k, z, i[0], i[1], j[0], j[1])]
            for i in N_expanded for j in N_expanded if i != j
        )
        # Only count arcs that start at a penalty node.
        penalty_arcs_expr = gp.quicksum(
            x[(k, z, i[0], i[1], j[0], j[1])]
            for i in penalty_nodes for j in N_expanded if i != j
        )
        # The first departure for each vehicle is still waived by subtracting used[k].
        model.addConstr(
            T[(k, z)] >= travel_time_expr + Tstop * (penalty_arcs_expr - used[k]),
            name=f"TimeLink_{k}_tour{z}"
        )

# -------------------------------------------------------
# 1.5 Driver shift time limit
# -------------------------------------------------------
# Each vehicle is limited to working Tmax amount of time at most (two shifts)
model.addConstrs(
    (gp.quicksum(T[(k, z)] for z in Z) <= Tmax for k in K_list),
    name="Maximum_service_time"
)

# -------------------------------------------------------
# 1.6 Used vehicle link
# -------------------------------------------------------
# Link used[k] to the arcs leaving the depot for vehicle k.
# If vehicle k uses any arc (depot -> j), then used[k] must be 1.
# This is linked to the vehicle acquisition cost
for k in K_list:
    for z in Z:
        for j in N_expanded:
            if j != (depot, 0):
                model.addConstr(used[k] >= x[(k, z, depot, 0, j[0], j[1])],
                                name=f"UsedLink_{k}_tour{z}_{j}")

# =============================================================================
# 2. Capacity and Flow Constraints
# =============================================================================

# -------------------------------------------------------
# 2.1 Prevent Flow on Unused Arcs
# -------------------------------------------------------
# Linking q and k
model.addConstrs(
    (q[(k, z, i[0], i[1], j[0], j[1])] <= Q[k] * x[(k, z, i[0], i[1], j[0], j[1])]
     for k in K_list for z in Z for i in N_expanded for j in N_expanded if i != j),
    name="link_q_x"
)

# -------------------------------------------------------
# 2.2 Customer demand is satisfied
# -------------------------------------------------------
for i in D:
    for z in Z:
        model.addConstr(
            gp.quicksum(delivered[(k, z, i, v)] for k in K_list for v in range(1, Vmax+1)) == p[i],
            name=f"demand_satisfaction_{i}_tour{z}"
        )

# -------------------------------------------------------
# 2.3 Per-Vehicle Flow Conservation at Customer Nodes
# -------------------------------------------------------
for k in K_list:
    for z in Z:
        for j in D_expanded:
            model.addConstr(
                gp.quicksum(q[(k, z, i[0], i[1], j[0], j[1])] for i in N_expanded if i != j) ==
                delivered[(k, z, j[0], j[1])] + gp.quicksum(q[(k, z, j[0], j[1], i[0], i[1])] for i in N_expanded if i != j),
                name=f"flow_conservation_{k}_tour{z}_{j}"
            )

# -------------------------------------------------------
# 2.4 The load each vehicle carries is reset at the depot and is updated between customers
# -------------------------------------------------------
for k in K_list:
    for z in Z:
        # At the depot, vehicles start with full capacity.
        model.addConstr(U[(k, z, depot, 0)] == Q[k], name=f"initial_load_{k}_tour{z}")
        for i in N_expanded:
            for j in N_expanded:
                if i == j:
                    continue
                if i == (depot, 0) and j in D_expanded:
                    # Leaving the depot: remaining load = full load minus what is delivered at the first stop.
                    model.addGenConstrIndicator(
                        x[(k, z, depot, 0, j[0], j[1])], True,
                        U[(k, z, j[0], j[1])] == Q[k] - delivered[(k, z, j[0], j[1])],
                        name=f"dep_depart_{k}_tour{z}_{j}"
                    )
                elif j == (depot, 0) and i in D_expanded:
                    # Arriving at the depot resets the load to full capacity.
                    model.addGenConstrIndicator(
                        x[(k, z, i[0], i[1], depot, 0)], True,
                        U[(k, z, depot, 0)] == Q[k],
                        name=f"arrive_dep_{k}_tour{z}_{i}"
                    )
                elif i in D_expanded and j in D_expanded:
                    # Between customers: subtract the delivery at j from the remaining load at i.
                    model.addGenConstrIndicator(
                        x[(k, z, i[0], i[1], j[0], j[1])], True,
                        U[(k, z, j[0], j[1])] == U[(k, z, i[0], i[1])] - delivered[(k, z, j[0], j[1])],
                        name=f"update_load_{k}_tour{z}_{i}_{j}"
                    )

# -------------------------------------------------------
# 2.5 Linking delivery variable with the route
# -------------------------------------------------------
# Ensures that if a vehicle does not actually visit a customer,
# then no delivery is recorded. Also caps the delivery at the customer’s demand
for k in K_list:
    for z in Z:
        for j in D_expanded:
            model.addConstr(
                delivered[(k, z, j[0], j[1])] <= p[j[0]] * gp.quicksum(x[(k, z, j[0], j[1], i[0], i[1])] for i in N_expanded if i != j),
                name=f"delivery_route_link_{k}_tour{z}_{j}"
            )

# -------------------------------------------------------
# 2.6 Ensure non-zero deliveries
# -------------------------------------------------------
# Q[k]/2 is there to incentivize the model to only all full or half deliveries
# (Demand is set to be in increments of half trucks)
for k in K_list:
    for z in Z:
        for j in D_expanded:
            model.addConstr(
                delivered[(k, z, j[0], j[1])] >= Q[k] / 2 * gp.quicksum(x[(k, z, j[0], j[1], i[0], i[1])] for i in N_expanded if i != j),
                name=f"min_delivery_{k}_tour{z}_{j}"
            )

# =============================================================================
# 3. HGEV Constraints
# =============================================================================

# -------------------------------------------------------
# 3.1 If any HGEV is used, "HGEV_Used" is set to 1
# -------------------------------------------------------
model.addConstrs(
    (HGEV_used >= used[k] for k in K_e_list),
    name="link_HGEV_used")

# -------------------------------------------------------
# 3.2 The battery level of vehicles is continuously updated
# -------------------------------------------------------
# For each arc (i,j) used by an HGEV in tour z:
for k in K_e_list:
    for z in Z:
        # At the depot, vehicles start with full charge.
        model.addConstr(y[(k, z, depot, 0)] == R[k], name=f"initial_charge_{k}_tour{z}")
        for i in N_expanded:
            for j in N_expanded:
                if i == j:
                    continue
                # If j is the depot, then recharge:
                if j == (depot, 0):
                    model.addGenConstrIndicator(
                        x[(k, z, i[0], i[1], depot, 0)], True,
                        y[(k, z, depot, 0)] == R[k],
                        name=f"recharge_at_{k}_tour{z}_{i}_{(depot, 0)}"
                    )
                # If j is a customer with a charging station (j[0] in CS) AND the expanded 
                # infrastructure network has been installed, then recharge:
                elif j[0] in CS:
                    model.addGenConstrIndicator(
                        HGEV_exp, True,
                        y[(k, z, j[0], j[1])] == R[k],
                        name=f"recharge_at_CS_{k}_tour{z}_{i}_{j}"
                    )
                # Otherwise, the battery level is reduced.
                else:
                    model.addGenConstrIndicator(
                        x[(k, z, i[0], i[1], j[0], j[1])], True,
                        y[(k, z, j[0], j[1])] == y[(k, z, i[0], i[1])] - h[k] * d_exp[(i, j)],
                        name=f"update_charge_{k}_tour{z}_{i}_{j}"
                    )

# -------------------------------------------------------
# 3.3 Reserve battery constraint
# -------------------------------------------------------
# HGEVs always reserve enough battery to return to the depot or a charging station
# (if the charging stations are active)
# Plus a 27kWh (Ms) margin of safety (3% of total charge)
# Compute d_star for each expanded customer node
d_star = {}
for i in N_expanded:
    # Only consider customer nodes (skip depot and charging stations)
    if i[0] == depot or i[0] in CS:
        continue
    d_to_depot = d[(i[0], depot)]
    # If CS is non-empty, compute the distance from i to each charging station and take the minimum.
    if len(CS) > 0:
        d_to_cs = min(d[(i[0], j)] for j in CS)
    else:
        d_to_cs = d_to_depot
    d_star[i] = min(d_to_depot, d_to_cs)
# Now add the reserve battery constraints for each HGEV at each node i (that is not depot)
# The depot is always active, so for any i not in the depot (and not itself a charging station):
for k in K_e_list:
    for z in Z:
        for i in N_expanded:
            if i[0] == depot or i[0] in CS:
                continue
            # When the expanded network is inactive (HGEV_exp==0), the only option is the depot.
            model.addGenConstrIndicator(
                HGEV_exp, False,
                y[(k, z, i[0], i[1])] >= h[k] * d[(i[0], depot)] + Ms,
                name=f"reserve_return_depot_{k}_tour{z}_{i}"
            )
            # When the expanded network is active (HGEV_exp==1), use the minimum distance (d_star).
            model.addGenConstrIndicator(
                HGEV_exp, True,
                y[(k, z, i[0], i[1])] >= h[k] * d_star[i] + Ms,
                name=f"reserve_return_expanded_{k}_tour{z}_{i}"
            )

# ===============================================================================================================================
# ===============================================================================================================================
#                                RUNNING THE MODEL
# ===============================================================================================================================
# ===============================================================================================================================

model.ModelSense = GRB.MINIMIZE

#model.setObjectiveN(obj_expr_cost, index=0, priority=1, name="Minimize_Total_Cost")
#model.setObjectiveN(obj_expr_emissions, index=1, priority=0, name="Minimize_GHG_Emissions")

model.setObjective(obj_expr_cost, GRB.MINIMIZE)

model.optimize()

# ===============================================================================================================================
# ===============================================================================================================================
#                                MODEL RESULTS
# ===============================================================================================================================
# ===============================================================================================================================

#Costs
recharging_cost = term1.getValue()
diesel_cost_curb = term2.getValue()
diesel_cost_load = term3.getValue()
operating_cost = term4.getValue()
acquisition_cost = term5.getValue()
hgev_infra_cost = term6.getValue()
diesel_emissions_cost = term7.getValue()
total_cost = recharging_cost + diesel_cost_curb + diesel_cost_load + operating_cost + acquisition_cost + hgev_infra_cost + diesel_emissions_cost

#Emissions
total_diesel_emissions = term8.getValue()
hgev_emissions = term9.getValue()

# -------------------------------------------------------
# Arc and vehicle selection
# -------------------------------------------------------
selected_arcs = [
    (k, z, (i0, i1), (j0, j1))
    for (k, z, i0, i1, j0, j1) in x.keys()
    if x[(k, z, i0, i1, j0, j1)].X > 0.5]
selected_vehicles = [k for k in K_list if used[k].X > 0.5]
print("Selected vehicles:", selected_vehicles)

# -------------------------------------------------------
# Objective function cost output values
# -------------------------------------------------------
print("Total cost:", total_cost)
print("Recharging costs:", recharging_cost)
print("Total diesel fuel costs:", diesel_cost_load + diesel_cost_curb)
print("Diesel fuel costs from curb weight:", diesel_cost_curb)
print("Diesel fuel costs from load weight:", diesel_cost_load)
print("Liters of diesel fuel used:", (diesel_cost_load + diesel_cost_curb)/F)
#print(f"Fuel efficiency of CFV trucks (mpg): {2.35215 * ((sum(d[(i[0], j[0])] for (k, z, i, j) in selected_arcs if k in K_c_list)/1000)/((diesel_cost_load + diesel_cost_curb)/F))}")
print("General operating costs:", operating_cost)
print("Acquisition cost:", acquisition_cost)
print("HGEV infrastructure cost:", hgev_infra_cost)
print("Diesel emissions cost:", diesel_emissions_cost)

#print(f"Total distance traveled: {sum(d[(i[0], j[0])] for (k, z, i, j) in selected_arcs)/1000}km")

# -------------------------------------------------------
# Objective function emissions output values
# -------------------------------------------------------
print("Total diesel emissions:", total_diesel_emissions)
print("Total HGEV emissions:", hgev_emissions)

# -------------------------------------------------------
# Partial delivery verification
# -------------------------------------------------------
print("Verifying Vehicle Load Reduction After Partial Deliveries:")
for k in K_list:
    for z in Z:
        print(f"\nVehicle {k}, Tour {z}:")
        # Find all arcs selected by this vehicle in tour z: note that x is now indexed as (k, z, i[0], i[1], j[0], j[1])
        selected_arcs = [
            (i, j) 
            for i in N_expanded 
            for j in N_expanded 
            if i != j and x[(k, z, i[0], i[1], j[0], j[1])].X > 0.5
        ]
        if not selected_arcs:
            print("  No arcs selected.")
            continue
        # For each selected arc, check the cumulative load update.
        for i, j in selected_arcs:
            if j == (depot, 0):
                # When arriving at the depot, the load resets to full capacity.
                print(f"  Arc ({i} -> {j}): Arriving at depot. U[{k}, Tour {z}, {j}] = {U[(k, z, j[0], j[1])].X:.2f} (should be Q[{k}] = {Q[k]:.2f}).")
            elif i == (depot, 0) and j != (depot, 0):
                # When leaving the depot, expected load at j = full capacity minus the delivery at j.
                expected_load = Q[k] - delivered[(k, z, j[0], j[1])].X
                print(f"  Arc ({i} -> {j}): U[{k}, Tour {z}, {j}] = {U[(k, z, j[0], j[1])].X:.2f}, Delivered[{k}, Tour {z}, {j}] = {delivered[(k, z, j[0], j[1])].X:.2f}, Expected U[{k}, Tour {z}, {j}] = {expected_load:.2f}")
            elif i != (depot, 0) and j != (depot, 0):
                # For arcs between customers: expected load at j = load at i minus delivery at j.
                expected_load = U[(k, z, i[0], i[1])].X - delivered[(k, z, j[0], j[1])].X
                print(f"  Arc ({i} -> {j}): U[{k}, Tour {z}, {i}] = {U[(k, z, i[0], i[1])].X:.2f}, Delivered[{k}, Tour {z}, {j}] = {delivered[(k, z, j[0], j[1])].X:.2f}, U[{k}, Tour {z}, {j}] = {U[(k, z, j[0], j[1])].X:.2f}, Expected U[{k}, Tour {z}, {j}] = {expected_load:.2f}")

# -------------------------------------------------------
# Charge updating verification
# -------------------------------------------------------
print("\n--- HGEV Battery Levels Along Their Routes ---")
# Loop over each HGEV that is used.
for k in K_e_list:
    if used[k].X < 0.5:
        continue  # Skip if vehicle k is not used.
    for z in Z:
        print(f"\nVehicle {k}, Tour {z}:")
        # Gather all arcs used by vehicle k in tour z.
        # Note: x is now indexed as (k, z, i[0], i[1], j[0], j[1])
        vehicle_arcs = [
            (i, j) 
            for i in N_expanded 
            for j in N_expanded 
            if i != j and x[(k, z, i[0], i[1], j[0], j[1])].X > 0.5
        ]
        if not vehicle_arcs:
            print("  No arcs selected.")
            continue
        # Reconstruct tours (each tour starts at the depot, i.e. (depot, 0)).
        remaining_arcs = vehicle_arcs.copy()
        tours = []
        while any(i == (depot, 0) for (i, j) in remaining_arcs):
            # Start a new tour with an arc leaving the depot.
            current_arc = None
            for arc in remaining_arcs:
                if arc[0] == (depot, 0):
                    current_arc = arc
                    break
            if current_arc is None:
                break
            tour = [(depot, 0), current_arc[1]]
            remaining_arcs.remove(current_arc)
            current_node = current_arc[1]
            # Follow arcs until return to depot.
            while current_node != (depot, 0):
                next_arc = None
                for arc in remaining_arcs:
                    if arc[0] == current_node:
                        next_arc = arc
                        break
                if next_arc is None:
                    break  # Incomplete tour, should not happen if model is feasible.
                tour.append(next_arc[1])
                remaining_arcs.remove(next_arc)
                current_node = next_arc[1]
            tours.append(tour)
        # Print out each tour with battery levels at each node.
        print(f"\nVehicle {k}, Tour {z} Routes and Battery Levels:")
        for tour in tours:
            route_str = " -> ".join(str(node) for node in tour)
            print(f"  Route: {route_str}")
            for node in tour:
                # Access y using the flattened key: (k, z, node[0], node[1])
                battery_level = y[(k, z, node[0], node[1])].X
                if node == (depot, 0):
                    station_type = "Depot (recharge expected)"
                elif node[0] in CS:
                    station_type = "Customer with charging station"
                else:
                    station_type = "Customer (no recharge)"
                print(f"    At node {node}: Battery level = {battery_level:.2f} ({station_type})")

# -------------------------------------------------------
# Exact vehicle routes
# -------------------------------------------------------
print("Vehicle Routes (Exact Paths):")
for k in K_list:
    if used[k].X < 0.5:
        continue  # Skip vehicles that are not used
    for z in Z:
        # Gather all selected arcs for this vehicle in tour z.
        # Note: x is defined over (k, z, i[0], i[1], j[0], j[1])
        vehicle_arcs = [
            (i, j)
            for i in N_expanded 
            for j in N_expanded 
            if i != j and x[(k, z, i[0], i[1], j[0], j[1])].X > 0.5
        ]
        # Create a copy to remove arcs as they're assigned to tours.
        remaining_arcs = vehicle_arcs.copy()
        tours = []
        # While there's an arc leaving the depot (i.e. (depot, 0)), start a new tour.
        while any(i == (depot, 0) for (i, j) in remaining_arcs):
            # Find an arc starting at the depot.
            current_arc = None
            for arc in remaining_arcs:
                if arc[0] == (depot, 0):
                    current_arc = arc
                    break
            if current_arc is None:
                break
            tour = [(depot, 0)]  # Start at the depot.
            tour.append(current_arc[1])  # Add the first customer.
            remaining_arcs.remove(current_arc)
            current_node = current_arc[1]
            # Follow the tour until the vehicle returns to the depot.
            while current_node != (depot, 0):
                next_arc = None
                for arc in remaining_arcs:
                    if arc[0] == current_node:
                        next_arc = arc
                        break
                if next_arc is None:
                    # In case the tour doesn't return to the depot (shouldn't happen in a valid solution).
                    break
                tour.append(next_arc[1])
                remaining_arcs.remove(next_arc)
                current_node = next_arc[1]
            tours.append(tour)
        # Print the tours for this vehicle and tour.
        print(f"\nVehicle {k}, Tour {z} Routes:")
        for idx, t in enumerate(tours, start=1):
            # Create a readable string of the tour.
            # (Optionally, you could print only the original node indices using, e.g., node[0].)
            path_str = " -> ".join(str(node) for node in t)
            print(f"  Trip {idx}: {path_str}")

# -------------------------------------------------------
# Other items of interest
# -------------------------------------------------------
#print("phi:", phi_param)
#print("sigma:", sigma_param)
#print("lambda:", lambda_param)
#print("epsilon:", epsilon_param)

#model.computeIIS()
#model.write("model.ilp")