import numpy as np
import pandas as pd
import math

import gurobipy as gp
from gurobipy import GRB

model = gp.Model("multi_objective_milp")
model.setParam('Threads', 12)

# ===============================================================================================================================
# ===============================================================================================================================
#                                PROBLEM SET-UP
# ===============================================================================================================================
# ===============================================================================================================================

# =============================================================================
# 1. Define Sets
# =============================================================================

# Available vehicles:
K_c = {f"CFV{i}" for i in range (1,21)}   
K_e = {f"HGEV{i}" for i in range (1,21)}         
K = K_c.union(K_e)              

K_c_list = sorted(K_c)   # List of available CFVs
K_e_list = sorted(K_e)   # List of available HGEVs
K_list   = sorted(K)     # All vehicles

num_customers = 20
D = set(range(0,num_customers))
depot = num_customers
num_nodes = num_customers + 1

N = D.union([depot])

N_0 = N
N_nplus1 = N
N_0_nplus1 = N

# =============================================================================
# 2. Define Parameters
# =============================================================================

# -------------------------------------------------------
# 2.1 Parameters Indexed by Vehicle (k in K)
# -------------------------------------------------------

# A^k: Acquisition cost for vehicle k
A = {k: 165000 if k in K_c_list else 400000 for k in K_list}

# S^k: Subsidy amount for vehicle k
S = {k: 0 if k in K_c_list else 50000 for k in K_list}

# h^k: Energy consumption per distance by vehicle k
h = {k: 4.375 if k in K_c_list else 1.1 for k in K_list}

# Q^k: The maximum load capacity for vehicle k 
Q = {k: 18000 for k in K_list}

# R^k: The battery capacity of vehicle k
R = {k: 99999 if k in K_c_list else 900 for k in K_list}

# L^k: Percent of the initial battery charge of vehicle k when leaving the depot
L = {k: 1 if k in K_c_list else .98 for k in K_list}

# -------------------------------------------------------
# 2.2 Parameters Indexed by Vehicle and Arc (i, j)
# -------------------------------------------------------
# c_ij^k: Cost of traveling from vertex i to vertex j by vehicle k
# For each vehicle and each arc (i, j) where i != j

CFV_Op_Cost = pd.read_csv("Walmart Small/CFV_Operating_Costs.csv", header=None)
HGEV_Op_Cost = pd.read_csv("Walmart Small/HGEV_Operating_Costs.csv", header=None)
CFV_cost_array = CFV_Op_Cost.values
HGEV_cost_array = HGEV_Op_Cost.values

c = {}

for k in K:
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if k.startswith("CFV"):
                    c[(k, i, j)] = CFV_cost_array[i, j]
                elif k.startswith("HGEV"):
                    c[(k, i, j)] = HGEV_cost_array[i, j]
                else:
                    c[(k, i, j)] = None

# -------------------------------------------------------
# 2.3 Parameters Indexed by Arc (i, j)
# -------------------------------------------------------

# d_ij: Distance between vertex i and vertex j
# t_ij: Travel time from vertex i to vertex j
# v_ij: Average velocity between vertex i and vertex j

#Distance matrix:
Distance_Matrix = pd.read_csv("Walmart Small/Distances.csv", header=None)
distance_array = Distance_Matrix.values
d = {}

#Travel time matrix:
Travel_Times = pd.read_csv("Walmart Small/Times.csv", header=None)
time_array = Travel_Times.values
t = {}

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            d[(i, j)] = distance_array[i, j]
            t[(i, j)] = time_array[i, j]

# υ: The average speed of the vehicle
v_avg = {}  # using 'v_avg' for υ

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            # Avoids division by zero by checking if the travel time is nonzero.
            if t[(i, j)] != 0:
                v_avg[(i, j)] = d[(i, j)] / t[(i, j)]
            else:
                v_avg[(i, j)] = float('inf')

# -------------------------------------------------------
# 2.4 Demand Parameter
# -------------------------------------------------------

Customer_demand = pd.read_csv("Walmart Small/Demand.csv", header=None)
Demand = Customer_demand.values

p = {i: float(Demand[i, 0]) for i in range(num_nodes)}

# -------------------------------------------------------
# 2.5 Emissions, Energy, and Fuel-Related Parameters
# -------------------------------------------------------

# β: Unit cost of GHG emissions (carbon tax rate)
Beta = 1  # using 'beta' instead of 'β' (((((!!!!!!!!!!!!!!!!!))))) come back and put in final value

# Υ: Price per carbon allowance
Y = 1  # (((((!!!!!!!!!!!!!!!!!))))) come back and put in final value

# μ: Regional average grid emissions factor for electricity
mu = .035

# Ω: GHG emissions coefficient for diesel fuel
Omega = 2.79

# G: Carbon cap for the firm
G = 1 #(((((!!!!!!!!!!!!!!!!!))))) come back and put in final value

# r: Recharging costs
r = .45

# ξ: Fuel-to-air mass ratio
xi = 1

# f: Engine friction factor
f_param = .2

# I: Engine speed
I_param = 22

# E: Engine displacement
E_param = 12.9

# m: The efficiency parameter for diesel engines
m_param = .9

# n: The heating value of typical diesel fuel
n_param = 44

# o: The drive-train efficiency of the vehicle
o_param = .4

# W_c: Curb weight
W_c = 12500

# τ: The average acceleration of the vehicle
tau = .68

# g: Gravitational constant
g_param = 9.81

# θ: Angle of the road
theta = 4.57
theta = math.radians(theta)

# C_d: Aerodynamic drag coefficient
C_d = .7

# C_r: Rolling resistance coefficient
C_r = .01

# α: Frontal area of the vehicle
alpha = 7.2

# ψ: Constant for converting gram/s to litre/s
psi = 737

# ρ: Air density
rho = 1.2041

# -------------------------------------------------------
# 2.6 Additional Terms
# -------------------------------------------------------

#EV_Initiation: Cost of infrastructure associated with being able to support HGEVs
HGEV_Initiation_Cost = 5000000
#EV_Expanded: Cost of infrastructure associated with being able to support HGEVs beyond 600km
#(((((!!!!!!!!!!!!!!!!!))))) come back and determine locations of 6 stations at various Supercenters
HGEV_Expanded_Cost = 6000000

# BigM: Used for Big-M formulation and constraint relaxation
BigM = 2000

# Terms used to simplify equations
lambda_param = xi/(n_param*psi)  # (λ)
phi_param    = 1/(1000*m_param*o_param)  # (φ)
sigma_param  = tau + g_param*math.sin(theta)+g_param*C_r*math.cos(theta)  # (σ)
epsilon_param = .5*C_d*rho*alpha  # (ϵ)

# =============================================================================
# 3. Define Variables
# =============================================================================

# -------------------------------------------------------
# 3.1  x_ij^k: 1 if vehicle k travels arc (i, j), 0 otherwise
# -------------------------------------------------------
#    Integer variable. Exclude i == j since an arc from a node to itself is not needed.
x = model.addVars(
    ((k, i, j) for k in K_list for i in range(num_nodes) for j in range(num_nodes) if i != j),
    vtype=GRB.BINARY,
    name="x"
)

# -------------------------------------------------------
# 3.2  q_ij^k: The flow of load carried from vertex i to vertex j by vehicle k
# -------------------------------------------------------
#    Continuous variable.
q = model.addVars(
    ((k, i, j) for k in K_list for i in range(num_nodes) for j in range(num_nodes) if i != j),
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    name="q"
)

# -------------------------------------------------------
# 3.3  y_i^k: The current energy level of vehicle k when arriving at vertex i
# -------------------------------------------------------
#    Continuous variable.
y = model.addVars(
    ((k, i) for k in K_list for i in range(num_nodes)),
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    name="y"
)

# -------------------------------------------------------
# 3.4  Emissions credits purchased
# -------------------------------------------------------
#    Continuous variable.
e_plus = model.addVar(vtype=GRB.CONTINUOUS,lb=0.0,name="e_plus")

# -------------------------------------------------------
# 3.5  Emissions credits sold
# -------------------------------------------------------
#    Continuous variable.
e_minus = model.addVar(vtype=GRB.CONTINUOUS,lb=0.0,name="e_minus")

# -------------------------------------------------------
# 3.6  Vehicle route time
# -------------------------------------------------------
#    Continuous variable.
T = model.addVars(K_list,vtype=GRB.CONTINUOUS,lb=0.0,name="TotalTime")

# -------------------------------------------------------
# 3.7  Helper variable to determine if vehicle is used at all
# -------------------------------------------------------
#   Binary variable: used[k] = 1 if vehicle k is used, 0 otherwise
used = model.addVars(K_list, vtype=GRB.BINARY, name="used")

# -------------------------------------------------------
# 3.8  z_ij^k: Half/full unload variable
# -------------------------------------------------------
#   Integer variable.
#   Define a single binary variable z for each arc: 1 indicates a half unload, 0 indicates full unload.
z = model.addVars(
    ((k, i, j) for k in K_list for i in N for j in N if i != j),
    vtype=GRB.BINARY,
    name="z"
)

# -------------------------------------------------------
# 3.9  U_i^k: Cumulative load variable
# -------------------------------------------------------
#   Continuous variable.
#   Define a cumulative load variable U[(k, i)] for each vehicle k and each node i in N.
#   U[(k, i)] represents the remaining load on vehicle k upon arriving at node i.
U = model.addVars(K_list, N, vtype=GRB.CONTINUOUS, lb=0.0, name="U")

# -------------------------------------------------------
# 3.10  HGEV_used: variable that indicates if any EV is used at all
# -------------------------------------------------------
#   Integer variable.
#   1 indicates that at least one HGEV is used, 0 indicates that none are used.
HGEV_used = model.addVar(vtype=GRB.BINARY, name="HGEV_used")

# -------------------------------------------------------
# 3.11  HGEV_exp: variable that indicates if any HGEV is used to traverse a distance of 600km or greater
# -------------------------------------------------------
#   Integer variable.
#   1 indicates that at least one HGEV is used to traverse a distance greater than 600km, 
#   0 indicates that none are used to go this distance.
HGEV_exp = model.addVar(vtype=GRB.BINARY, name="HGEV_exp")

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
    L[k] * R[k] *
    gp.quicksum(
        x[(k, depot, j)] - y[(k, depot)]
        for j in N
        if j != depot
    )
    for k in K_e_list
)

# -------------------------------------------------------
# 1.2 Acquisition Costs (Includes Subsidies)
# -------------------------------------------------------
term2 = gp.quicksum(
    (A[k] - S[k]) * x[(k, depot, j)]
    for k in K_list
    for j in N if j != depot
)

# -------------------------------------------------------
# 1.3 HGEV Infrastructure Costs
# -------------------------------------------------------

term3 = HGEV_Initiation_Cost*HGEV_used+HGEV_Expanded_Cost*HGEV_exp

# -------------------------------------------------------
# 1.4 Operating Costs
# -------------------------------------------------------
term4 = gp.quicksum(
    c[(k, i, j)] * x[(k, i, j)]
    for k in K_list
    for i in N_0
    for j in N_nplus1
    if i != j
)

# -------------------------------------------------------
# 1.5 Cost of Diesel Emissions
# -------------------------------------------------------
term5_temp = gp.quicksum(
    (
        W_c * phi_param * sigma_param * d[(i, j)]
        + f_param * I_param * E_param * t[(i, j)]
        + epsilon_param * phi_param * d[(i, j)] * (v_avg[(i, j)])**2
    ) * lambda_param * x[(k, i, j)]
    for k in K_c_list
    for i in N_0
    for j in N_nplus1
    if i != j
)
term5_temp += gp.quicksum(
    phi_param * sigma_param * lambda_param * d[(i, j)] * q[(k, i, j)]
    for k in K_c_list
    for i in N_0
    for j in N_nplus1
    if i != j
)

term5 = Beta*Omega*term5_temp

# -------------------------------------------------------
# 1.6 Cost of Electricity-Related Emissions from Charging HGEVs
# -------------------------------------------------------
term6 = Beta*mu * gp.quicksum(
    L[k] * R[k] *
    gp.quicksum(
        x[(k, depot, j)] - y[(k, depot)]
        for j in N  
        if j != depot
    )
    for k in K_e_list
)

# -------------------------------------------------------
# 1.7 Carbon Allowance Purchases/Sales
# -------------------------------------------------------
term7 = Y * (e_plus - e_minus)

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
    (
        W_c * phi_param * sigma_param * d[(i, j)]
        + f_param * I_param * E_param * t[(i, j)]
        + epsilon_param * phi_param * d[(i, j)] * (v_avg[(i, j)])**2
    ) * lambda_param * x[(k, i, j)]
    for k in K_c_list
    for i in N_0
    for j in N_nplus1
    if i != j
)
term8_temp += gp.quicksum(
    phi_param * sigma_param * lambda_param * d[(i, j)] * q[(k, i, j)]
    for k in K_c_list
    for i in N_0
    for j in N_nplus1
    if i != j
)

term8 = Omega*term8_temp

# -------------------------------------------------------
# 2.2 Electricity-Related Emissions from Charging HGEVs
# -------------------------------------------------------
term9 = mu * gp.quicksum(
    L[k] * R[k] *
    gp.quicksum(
        x[(k, depot, j)] - y[(k, depot)]
        for j in N  
        if j != depot
    )
    for k in K_e_list
)

# -------------------------------------------------------
# Complete Expression
# -------------------------------------------------------
obj_expr_emissions = term8 + term9

# ===============================================================================================================================
# ===============================================================================================================================
#                                CONSTRAINTS
# ===============================================================================================================================
# ===============================================================================================================================

# =============================================================================
# 1. Routing Constraints
# =============================================================================

# -------------------------------------------------------
# 1.1 Conservation of Routing
# -------------------------------------------------------
# For each vehicle, the number of arcs leaving the depot equal the number of arcs arriving at the depot
# Allow vehicles to leave and return to the depot multiple times,
# but enforce that the number of departures equals the number of returns.
model.addConstrs(
    (gp.quicksum(x[(k, depot, j)] for j in N if j != depot) ==
     gp.quicksum(x[(k, i, depot)] for i in N if i != depot)
     for k in K_list),
    name="depot_routing_conservation"
)

# -------------------------------------------------------
# 1.2 Connectivity of Tours
# -------------------------------------------------------
model.addConstrs(
    (gp.quicksum(x[(k, i, j)] for i in N if i != j) -
     gp.quicksum(x[(k, j, i)] for i in N if i != j) == 0
     for k in K_list for j in D),
    name="connectivity"
)

# -------------------------------------------------------
# 1.3 Maximum Route Time
# -------------------------------------------------------
# Each time a vehicle stops a 2-hour time penalty is incurred (for unloading)

# Link used[k] to the arcs leaving the depot for vehicle k.
# If vehicle k uses any arc (depot -> j), then used[k] must be 1.
for k in K_list:
    for j in N:
        if j != depot:
            model.addConstr(used[k] >= x[(k, depot, j)], name=f"UsedLink_{k}_{j}")


for k in K_list:
    travel_time_expr = gp.quicksum(t[(i, j)] * x[(k, i, j)]
                                   for i in N for j in N if i != j)
    # Total arcs used; for a used vehicle, the first departure is free of penalty.
    arc_count_expr = gp.quicksum(x[(k, i, j)]
                                 for i in N for j in N if i != j)
    model.addConstr(T[k] >= travel_time_expr + 2.0 * (arc_count_expr - used[k]),
                    name=f"TimeLink_{k}")

# -------------------------------------------------------
# 1.4 Driver shift time limit
# -------------------------------------------------------
# Each vehicle (driver) is limited to working 11 hours at most
model.addConstrs(
    (T[k] <= 11 for k in K_list),
    name="Max11Hours"
)

# =============================================================================
# 2. Capacity and Flow Constraints
# =============================================================================

# -------------------------------------------------------
# 2.1 Z is forced to 0 if an arc is unused
# -------------------------------------------------------
model.addConstrs(
    (z[(k, i, j)] <= x[(k, i, j)]
     for k in K_list for i in N for j in N if i != j),
    name="link_z_x"
)

# -------------------------------------------------------
# 2.2 Enforces that the load delivered is either a half or a full truck
# -------------------------------------------------------
model.addConstrs(
    (q[(k, i, j)] == Q[k]*x[(k, i, j)] - 0.5*Q[k]*z[(k, i, j)]
     for k in K_list for i in N for j in N if i != j),
    name="full_or_half"
)

# -------------------------------------------------------
# 2.3 Customer demand is satisfied
# -------------------------------------------------------
model.addConstrs(
    (gp.quicksum(q[(k, j, i)] for k in K_list for j in N if j != i) == p[i]
     for i in D),
    name="customer_demand_satisfaction"
)

# -------------------------------------------------------
# 2.4 At the depot vehicles begin fully loaded
# -------------------------------------------------------
model.addConstrs(
    (U[(k, depot)] == Q[k] for k in K_list),
    name="initial_load"
)

# -------------------------------------------------------
# 2.5 The load each vehicle carries is continuously updated
# -------------------------------------------------------
# For every arc (i,j) (with i != j) that is used by vehicle k,
# the remaining load at j equals the remaining load at i minus the load delivered on arc (i,j).
# If the arc is not used, the Big-M term relaxes the constraint.
model.addConstrs(
    (U[(k, j)] <= U[(k, i)] - q[(k, i, j)] + BigM*(1 - x[(k, i, j)])
     for k in K_list for i in N for j in N if i != j),
    name="update_load_upper"
)

model.addConstrs(
    (U[(k, j)] >= U[(k, i)] - q[(k, i, j)] - BigM*(1 - x[(k, i, j)])
     for k in K_list for i in N for j in N if i != j),
    name="update_load_lower"
)

# -------------------------------------------------------
# 2.6 Return to depot if less than half loaded
# -------------------------------------------------------
model.addConstrs(
    (x[(k, j, depot)] >= 1 - U[(k, j)] / (0.5 * Q[k])
     for k in K_list for j in N if j != depot),
    name="return_if_below_half"
)

# -------------------------------------------------------
# 2.7 Load is replenished to full upon returning to the depot
# -------------------------------------------------------
model.addConstrs(
    (U[(k, depot)] >= Q[k] - BigM*(1 - gp.quicksum(x[(k, i, depot)] for i in N if i != depot))
     for k in K_list),
    name="reset_load_lower"
)

model.addConstrs(
    (U[(k, depot)] <= Q[k] + BigM*(1 - gp.quicksum(x[(k, i, depot)] for i in N if i != depot))
     for k in K_list),
    name="reset_load_upper"
)

# =============================================================================
# 3. HGEV Constraints
# =============================================================================

# -------------------------------------------------------
# 3.1 If any HGEV is used, "HGEV_Used" is set to 1
# -------------------------------------------------------
model.addConstrs(
    (HGEV_used >= used[k] for k in K_e_list),
    name="link_HGEV_used"
)

# -------------------------------------------------------
# 3.2 If any HGEV is used to traverse >600km, "HGEV_Exp" is set to 1
# -------------------------------------------------------
model.addConstrs(
    (gp.quicksum(d[(i, j)] * x[(k, i, j)] for i in N for j in N if i != j)
     <= 600 + BigM * HGEV_exp
     for k in K_e_list),
    name="HGEV_total_distance"
)

# -------------------------------------------------------
# 3.3 The battery level of vehicles is continuously updated
# -------------------------------------------------------
model.addConstrs((
    (
        y[(k, j)] 
        <= y[(k, i)] - h[k]*d[(i, j)] * x[(k, i, j)] + R[k]*(1 - x[(k, i, j)])
    )
    for k in K_list
    for i in N
    for j in N
    if i != j),
    name="update_charge"
)

# -------------------------------------------------------
# 3.4 Initial charge of each vehicle
# -------------------------------------------------------
model.addConstrs(
    (y[(k, depot)] == L[k]*R[k]*used[k]
    for k in K_list),
    name="initial_charge"
)

# -------------------------------------------------------
# 3.5 Determine available energy when the previous Supercenter had a charging station
# -------------------------------------------------------
#model.addConstrs(
#    (y[(k, i)] >= R[k] - BigM * (1 - gp.quicksum(x[(k, j, i)] for j in N if j != i))
#     for k in K_e_list for i in F),
#    name="charge_lower"
#)

#model.addConstrs(
#    (y[(k, i)] <= R[k] + BigM * (1 - gp.quicksum(x[(k, j, i)] for j in N if j != i))
#     for k in K_e_list for i in F),
#    name="charge_upper"
#)

# =============================================================================
# 4. Carbon Cap Constraint
# =============================================================================

#Return to this 
# 
# 
# (((((!!!!!!!!!!!!!!!!!)))))


# ===============================================================================================================================
# ===============================================================================================================================
#                                RUNNING THE MODEL
# ===============================================================================================================================
# ===============================================================================================================================

model.ModelSense = GRB.MINIMIZE

model.setObjectiveN(obj_expr_cost, index=0, priority=1, name="Minimize_Total_Cost")
#model.setObjectiveN(obj_expr_emissions, index=1, priority=0, name="Minimize_GHG_Emissions")

model.optimize()

model.computeIIS()
model.write("model.ilp")
