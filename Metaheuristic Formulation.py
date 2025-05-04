import pandas as pd
import numpy as np
import random
import copy
import math

# ===============================================================================================================================
# ===============================================================================================================================
#                                PROBLEM SET-UP
# ===============================================================================================================================
# ===============================================================================================================================

# -------------------------------------------
# 1. Basic Sets and Parameters
# -------------------------------------------
# Available vehicles:
K_c = {f"CFV{i}" for i in range(1, 21)}
K_e = {f"HGEV{i}" for i in range(1, 21)}
K = K_c.union(K_e)

K_c_list = sorted(K_c)   # List of available CFVs
K_e_list = sorted(K_e)   # List of available HGEVs
K_list   = sorted(K)     # All vehicles

# Read demand data
demand_df = pd.read_csv("Walmart Small/Demand.csv", header=None)
num_customers = len(demand_df) - 1  # Assumes first row is header or depot not counted as customer
depot = num_customers              # Depot is set as the last node
num_nodes = num_customers + 1      # Total nodes: customers + depot

# Define customer set and full node set
D = set(range(0, num_customers))
N = D.union({depot})

# Customer demand dictionary
p = {i: float(demand_df.iloc[i, 0]) for i in range(num_nodes)}

# Time-related parameters
Tmax = 22 * 3600    # 22 hours in seconds
Tstop = 2 * 3600    # 2 hours in seconds

# Charging stations and ghost mapping
CS = {20, 21}           # Possible charging stations
ghost_mapping = {0: 20, 11: 21}  # Example: customer 0's ghost is 20, customer 11's ghost is 21

# Create ghost pairs for bidirectional treatment:
ghost_pairs = {(i, ghost_mapping[i]) for i in ghost_mapping}
ghost_pairs.update({(ghost_mapping[i], i) for i in ghost_mapping})

# -------------------------------------------
# 3. Distance, Time, and Velocity Dictionaries
# -------------------------------------------
# Read original distance and travel time matrices from CSV files.
distance_df = pd.read_csv("Walmart Small/Distances.csv", header=None)
time_df = pd.read_csv("Walmart Small/Times.csv", header=None)
distance_array = distance_df.values
time_array = time_df.values

# Initialize dictionaries for the original network.
d = {}       # Distance between nodes i and j
t = {}       # Travel time between nodes i and j
v_avg = {}   # Average velocity between nodes i and j

for i in range(num_nodes):
    for j in range(num_nodes):
        if i == j:
            continue  # No self-loop needed.
        d[(i, j)] = distance_array[i, j]
        t[(i, j)] = time_array[i, j]
        # Compute average velocity. For ghost pairs, if travel time is zero, use a small positive value to avoid division by zero.
        if (i, j) in ghost_pairs:
            effective_time = t[(i, j)] if t[(i, j)] > 0 else 0.1
            v_avg[(i, j)] = d[(i, j)] / effective_time
        else:
            v_avg[(i, j)] = d[(i, j)] / t[(i, j)] if t[(i, j)] > 0 else 0

# -------------------------------------------
# 4. Other parameters
# -------------------------------------------

# Acquisition cost and subsidy.
A = {k: 165000 if k in K_c_list else 400000 for k in K_list}

# S^k: Subsidy amount for vehicle k ($USD)
S = {k: 0 if k in K_c_list else 0 for k in K_list}

# Energy consumption per meter.
h = {k: 0.004375 if k in K_c_list else 0.0011 for k in K_list}

# Battery capacity (kWh); for CFVs, use a dummy high value.
R = {k: 99999 if k in K_c_list else 900 for k in K_list}

# Operating costs ($/m).
O = {k: 0.00018125 if k in K_c_list else 0.00015875 for k in K_list}

# Q^k: The maximum load capacity for vehicle k  (kg)
Q = {k: 15000 for k in K_list}

# Define the list of real customers: those that are not the depot and not ghost nodes, and have positive demand.
customers = [i for i in range(num_nodes) if i != depot and i not in CS and p.get(i, 0) > 0]

Beta = 24
mu = 0.035
Omega = 2.56
r = 0.11
F = 1.14
Ms = 27
xi = 0.055
f_param = 0.25
I_param = 23
E_param = 12.9
m_param = 0.33
n_param = 42
o_param = 0.85
W_c = 21000
tau = 0.68
g_param = 9.81
theta = math.radians(4.57)
C_d = 0.6
C_r = 0.01
alpha = 25
psi = 840
rho = 1.2041

HGEV_Initiation_Cost = 5000000
HGEV_Expanded_Cost = 6000000
trigger_distance = 793 * 1000 #trigger distance for HGEV_Expanded_Cost is 793km (* 1000 to be represented in meters, (900kWh-Ms)/h^k)

lambda_param = xi / (n_param * psi)
phi_param = 1 / (1000 * m_param * o_param)
sigma_param = tau + g_param * math.sin(theta) + g_param * C_r * math.cos(theta)
epsilon_param = 0.5 * C_d * rho * alpha

max_iterations = 1000
diversification_rate = .05

T0 = 1000
cooling_rate = .99

# Building the parameter dictionary
params = {
    'depot': depot,
    'Tmax': Tmax,
    'Tstop': Tstop,
    'CS': CS,
    'p': p,
    'A': A,
    'S': S,
    'h': h,
    'R': R,
    'O': O,
    'Q': Q,
    'Beta': Beta,
    'mu': mu,
    'Omega': Omega,
    'r': r,
    'F': F,
    'Ms': Ms,
    'xi': xi,
    'f_param': f_param,
    'I_param': I_param,
    'E_param': E_param,
    'm_param': m_param,
    'n_param': n_param,
    'o_param': o_param,
    'W_c': W_c,
    'tau': tau,
    'g_param': g_param,
    'theta': theta,
    'C_d': C_d,
    'C_r': C_r,
    'alpha': alpha,
    'psi': psi,
    'rho': rho,
    'HGEV_Initiation_Cost': HGEV_Initiation_Cost,
    'HGEV_Expanded_Cost': HGEV_Expanded_Cost,
    'trigger_distance': trigger_distance,
    'lambda_param': lambda_param,
    'phi_param': phi_param,
    'sigma_param': sigma_param,
    'epsilon_param': epsilon_param,
    'd': d,
    't': t,
    'v_avg': v_avg,
    'max_iterations': max_iterations,
    'diversification_rate': diversification_rate, # Chance to accept a dominated move for diversification.
    'T0': T0,
    'cooling_rate': cooling_rate}

# ===============================================================================================================================
# ===============================================================================================================================
#                                GREEDY SEARCH HEURISTIC
# ===============================================================================================================================
# ===============================================================================================================================

def greedy_search_heuristic(K_list, D, depot, d, p, Q):
    """
    Constructs vehicle routes greedily to satisfy customer demands.
    
    Parameters:
        K_list (list): List of vehicle identifiers.
        D (list): List of customer nodes (excluding the depot).
        depot (int): Depot node identifier.
        d (dict): Distance dictionary with keys (i, j) for i, j in the original node set.
        p (dict): Customer demand dictionary with keys as customer nodes.
        Q (dict): Vehicle capacity dictionary with keys as vehicle identifiers.
        
    Returns:
        routes (dict): Dictionary mapping each vehicle to a list of trips (each trip is a list of nodes).
        remaining_demand (dict): Updated demand for each customer after route construction.
    """

    # Create a copy of demand to modify as we assign deliveries.
    remaining_demand = p.copy()
    
    # Initialize the routes dictionary: each vehicle will have a list of trips.
    routes = {k: [] for k in K_list}
    
    # Continue building trips until all customer demands are satisfied.
    while any(remaining_demand[i] > 0 for i in D):
        for k in K_list:
            # Break if all demand has been met.
            if not any(remaining_demand[i] > 0 for i in D):
                break

            # Start a new trip for vehicle k: start at the depot.
            current_route = [depot]
            current_capacity = Q[k]
            current_node = depot

            # Construct the trip using a nearest neighbor rule.
            while True:
                # Identify customers with unsatisfied demand.
                feasible_customers = [i for i in D if remaining_demand[i] > 0]
                if not feasible_customers:
                    break

                # Choose the closest customer to the current node.
                next_customer = min(feasible_customers, key=lambda i: d.get((current_node, i), float('inf')))
                
                # Determine the amount to deliver: if the remaining demand exceeds current capacity,
                # deliver what you can and leave the rest for a future trip.
                if remaining_demand[next_customer] <= current_capacity:
                    delivered_amt = remaining_demand[next_customer]
                    remaining_demand[next_customer] = 0
                else:
                    delivered_amt = current_capacity
                    remaining_demand[next_customer] -= current_capacity

                current_capacity -= delivered_amt
                current_route.append(next_customer)
                current_node = next_customer

                # If the vehicle's capacity is exhausted, end the trip.
                if current_capacity <= 0:
                    break

            # Return to the depot to complete the trip.
            current_route.append(depot)
            routes[k].append(current_route)

    return routes, remaining_demand

# ===============================================================================================================================
# ===============================================================================================================================
#                                ALNS + SA CONSTRUCTION
# ===============================================================================================================================
# ===============================================================================================================================

# -------------------------------------------------------
# 1. Cost functions
# -------------------------------------------------------

def compute_cost(solution, params):
    total_cost = 0.0

    # Flags to track if any HGEV is used and if any HGEV travels a long distance.
    hgev_used_flag = False
    hgev_exp_flag = False
    
    for vehicle, routes in solution.items():
        is_cfv = vehicle.startswith("CFV")
        vehicle_arc_cost = 0.0
        Q_vehicle = params['Q'][vehicle]  # Maximum load for the vehicle
        
        # Determines the amount delivered at each customer
        for route in routes:
            n_customers = len(route) - 2  # number of customer visits in the route
            if n_customers == 1:
                # One-stop route: single customer receives full load.
                delivery = Q_vehicle
            elif n_customers == 2:
                # Two-stop route: each customer receives half the load.
                delivery = Q_vehicle / 2.0
            else:
                # If the route has no customer or more than two customers, skip or flag error.
                if n_customers <= 0:
                    continue
                else:
                    raise ValueError("Routes with more than two customers are not allowed.")
            
            # Determine the flow on each arc.
            # For a one-stop route: [depot, customer, depot]
            #   - Arc 0 (depot->customer): flow = Q_vehicle
            #   - Arc 1 (customer->depot): flow = Q_vehicle - delivery = 0
            # For a two-stop route: [depot, customer1, customer2, depot]
            #   - Arc 0 (depot->customer1): flow = Q_vehicle
            #   - Arc 1 (customer1->customer2): flow = Q_vehicle - delivery = Q_vehicle/2
            #   - Arc 2 (customer2->depot): flow = Q_vehicle - 2*delivery = 0
            flows = []
            for i in range(len(route) - 1):
                if i == 0:
                    flows.append(Q_vehicle)
                elif n_customers == 1:
                    flows.append(Q_vehicle - delivery)  # equals 0
                elif n_customers == 2:
                    if i == 1:
                        flows.append(Q_vehicle - delivery)  # equals Q_vehicle/2
                    else:
                        flows.append(Q_vehicle - 2 * delivery)  # equals 0
            
            # Iterate over each arc in the route, using the inferred flows.
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i+1]
                delivered_load = flows[i]
                
                # Retrieve arc metrics.
                arc_distance = params['d'].get((node_from, node_to), 0)
                arc_time = params['t'].get((node_from, node_to), 0)
                arc_velocity = params['v_avg'].get((node_from, node_to), 0)
                
                if not is_cfv:
                    # HGEV costs (recharging + operating costs): corresponds with term1 in Gurobi and integrates term4
                    arc_cost = (params['r'] * params['h'][vehicle] * arc_distance + params['O'][vehicle] * arc_distance)
                    # HGEV infrastructure initiation and expanded infrastructure costs: corresponds with term6 in Gurobi                    
                    hgev_used_flag = True
                    if arc_distance > params.get('trigger_distance', float('inf')):
                        hgev_exp_flag = True
                else:
                    # CFV diesel fuel costs due to curb weight: corresponds with term2 in Gurobi
                    curb_cost = params['F'] * params['lambda_param'] * (
                        params['W_c'] * params['phi_param'] * params['sigma_param'] * arc_distance +
                        params['f_param'] * params['I_param'] * params['E_param'] * arc_time +
                        params['epsilon_param'] * params['phi_param'] * arc_distance * (arc_velocity ** 2))
                    # CFV diesel fuel costs due to load effect: corresponds with term3 in Gurobi
                    load_cost = params['F'] * params['lambda_param'] * params['phi_param'] * params['sigma_param'] * arc_distance * delivered_load                    
                    fuel_costs = curb_cost + load_cost
                    # CFV cost due to carbon tax: corresponds with term7 in Gurobi
                    emissions_costs = (fuel_costs) * params['Beta'] * params['Omega'] / 1000
                    # Total CFV arc cost: integrates term2, term3, term4, and term7 from Gurobi
                    arc_cost = fuel_costs + emissions_costs + params['O'][vehicle] * arc_distance

                vehicle_arc_cost += arc_cost

        total_cost += vehicle_arc_cost
        
        # Vehicle acquisition costs: corresponds with term5 in Gurobi
        if routes and any(len(route) > 0 for route in routes):
            total_cost += (params['A'][vehicle] - params['S'][vehicle])
    
    # Add fixed HGEV infrastructure costs: term6 in Gurobi
    if hgev_used_flag:
        total_cost += params['HGEV_Initiation_Cost']
    if hgev_exp_flag:
        total_cost += params['HGEV_Expanded_Cost']
    
    return total_cost


def compute_emissions(solution, params):
    total_emissions = 0.0

    for vehicle, routes in solution.items():
        is_cfv = vehicle.startswith("CFV")
        Q_vehicle = params['Q'][vehicle]  # Maximum load for the vehicle
        
        # Determines the amount delivered at each customer
        for route in routes:
            # Determine number of customer visits in the route.
            n_customers = len(route) - 2  # exclude depot at start and end
            if n_customers == 1:
                # One-stop route: customer receives full load.
                delivery = Q_vehicle
            elif n_customers == 2:
                # Two-stop route: each customer receives half load.
                delivery = Q_vehicle / 2.0
            else:
                # If there are no customer visits or more than two, skip or flag error.
                if n_customers <= 0:
                    continue
                else:
                    raise ValueError("Routes with more than two customers are not allowed.")
            
            # Compute the delivered load (flow) for each arc.
            flows = []
            for i in range(len(route) - 1):
                if i == 0:
                    flows.append(Q_vehicle)
                elif n_customers == 1:
                    flows.append(Q_vehicle - delivery)
                elif n_customers == 2:
                    if i == 1:
                        flows.append(Q_vehicle - delivery)
                    else:
                        flows.append(Q_vehicle - 2 * delivery)
            
            # Now process each arc using the inferred delivered load.
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i+1]
                delivered_load = flows[i]
                
                d_arc = params['d'].get((node_from, node_to), 0)
                t_arc = params['t'].get((node_from, node_to), 0)
                v_arc = params['v_avg'].get((node_from, node_to), 0)
                
                if is_cfv:
                    # CFV emissions due to curb weight: corresponds to the first part of term8 in Gurobi
                    emissions_curb = params['Omega'] * params['lambda_param'] * (
                        params['W_c'] * params['phi_param'] * params['sigma_param'] * d_arc +
                        params['f_param'] * params['I_param'] * params['E_param'] * t_arc +
                        params['epsilon_param'] * params['phi_param'] * d_arc * (v_arc ** 2))
                    # CFV emissions due to vehicle load: corresponds to the second part of term8 in Gurobi
                    emissions_load = params['Omega'] * (params['phi_param'] * params['sigma_param'] * params['lambda_param'] * d_arc * delivered_load)
                    arc_emissions = emissions_curb + emissions_load
                else:
                    # HGEV emissions due to recharging: corresponds with term9 in Gurobi
                    arc_emissions = params['mu'] * params['h'][vehicle] * d_arc
                
                total_emissions += arc_emissions

    # Convert the sum to metric tons
    total_emissions /= 1000.0
    return total_emissions

def evaluate_solution(solution, params):
    cost = compute_cost(solution, params)
    emissions = compute_emissions(solution, params)
    return cost, emissions

# -------------------------------------------------------
# 2. Feasibility Function
# -------------------------------------------------------

def check_feasibility(solution, params):
    """
    Check if a solution satisfies the main constraints.
    
    Assumptions on solution representation:
      - solution: dict mapping vehicle id -> list of trips (each trip is a list of nodes).
        Example trips: [depot, customer, depot] (one-stop) or [depot, customer1, customer2, depot] (two-stop).
    
    Constraints:
      1. Each trip must start and end at the depot.
      2. Each trip must have a valid structure (3 nodes for one-stop, 4 nodes for two-stop).
      3. Delivered load is assigned per trip:
           - One-stop: the customer receives the full load Q.
           - Two-stop: each customer receives half of Q.
      4. For HGEVs, battery consumption is simulated along each trip. The battery is reset 
         when departing from the depot or a charging station, and it must not go negative nor fall below
         a required reserve.
      5. Overall customer demand (given in p) should ideally be met by the collection of trips.
      6. The cumulative service time for all trips assigned to a vehicle must not exceed Tmax.
      7. No vehicle should visit the same customer consecutively.
    
    Note: If partial deliveries are acceptable, we allow a solution to be feasible even if
          not 100% of the demand is met. This is controlled by a parameter 'min_delivery_fraction'.
    
    Returns:
      bool: True if the solution is feasible, False otherwise.
    """
    feasible = True
    errors = []
    depot = params['depot']
    Tmax = params['Tmax']
    Q = params['Q']
    p = params['p']
    CS = params.get('CS', set())
    tolerance = 1e-6
    demand_satisfied = True
    
    # Initialize delivered amounts for each customer (excluding the depot)
    delivered = {i: 0 for i in p if i != depot}
    
    # 1. Check each trip (for each vehicle).
    for vehicle, routes in solution.items():
        for route in routes:
            # Check for consecutive duplicate customer visits.
            for i in range(len(route) - 1):
                if route[i] != depot and route[i+1] != depot and route[i] == route[i+1]:
                    errors.append(f"Trip for vehicle {vehicle} has consecutive duplicate visits: {route}")
                    feasible = False
                    break  # Exit checking this route.
            if not feasible:
                continue
            
            # Each trip must have either 3 nodes ([depot, customer, depot]) or 4 nodes ([depot, customer1, customer2, depot]).
            if len(route) not in [3, 4]:
                errors.append(f"Trip for vehicle {vehicle} has invalid length: {route}")
                feasible = False
                continue
            
            # Ensure the trip starts and ends at the depot.
            if route[0] != depot or route[-1] != depot:
                errors.append(f"Trip for vehicle {vehicle} does not start/end at depot: {route}")
                feasible = False
            
            # Assign delivered load for the trip.
            n_customers = len(route) - 2
            if n_customers == 1:
                delivered[route[1]] += Q[vehicle]
            elif n_customers == 2:
                delivered[route[1]] += Q[vehicle] / 2.0
                delivered[route[2]] += Q[vehicle] / 2.0
            else:
                errors.append(f"Trip for vehicle {vehicle} has invalid number of stops: {route}")
                feasible = False
            
            # For HGEVs, simulate battery consumption along the trip.
            if not vehicle.startswith("CFV"):
                battery = params['R'][vehicle]
                for i in range(len(route) - 1):
                    node_from = route[i]
                    node_to = route[i+1]
                    if node_from == depot or node_from in CS:
                        battery = params['R'][vehicle]
                    arc_distance = params['d'].get((node_from, node_to), 0)
                    consumption = params['h'][vehicle] * arc_distance
                    battery -= consumption
                    if battery < 0:
                        errors.append(f"Battery depleted for vehicle {vehicle} on trip {route} from {node_from} to {node_to}.")
                        feasible = False
                    if node_to != depot and node_to not in CS:
                        d_to_depot = params['d'].get((node_to, depot), float('inf'))
                        d_to_cs = min([params['d'].get((node_to, cs), float('inf')) for cs in CS], default=float('inf'))
                        required = params['h'][vehicle] * min(d_to_depot, d_to_cs) + params['Ms']
                        if battery < required:
                            errors.append(f"Insufficient battery for vehicle {vehicle} at node {node_to} in trip {route}.")
                            feasible = False
    
    # 2. Check overall customer demand is met
    for customer, demand in p.items():
        if customer == depot:
            continue
        delivered_amount = delivered.get(customer, 0)
        if abs(delivered_amount - demand) > tolerance:
            errors.append(f"Demand for customer {customer} not fully met: delivered {delivered_amount} vs. demand {demand}.")
            feasible = False
            demand_satisfied = False

    # 3. Check cumulative service time for each vehicle.
    for vehicle in solution:
        total_time = compute_total_service_time(solution, params, vehicle)
        if total_time > Tmax:
            errors.append(f"Cumulative service time for vehicle {vehicle} exceeds Tmax: {total_time} > {Tmax}.")
            feasible = False
    
    # Uncomment for debugging:
    if not feasible:
        for err in errors:
            print("Feasibility error:", err)
    if demand_satisfied:
        print("Feasibility check: All customer demand requirements are fully satisfied.")

    return feasible

def compute_total_service_time(solution, params, vehicle):
    """
    Compute the cumulative service time for a given vehicle by summing the travel times 
    (plus Tstop penalties for each customer stop) over all trips (routes) assigned to that vehicle.
    
    Parameters:
      solution (dict): Mapping of vehicle IDs to lists of trips.
      params (dict): Contains keys 'depot', 't' (travel time dictionary), and 'Tstop'.
      vehicle: The vehicle ID.
    
    Returns:
      total_time (float): The cumulative service time for the vehicle.
    """
    depot = params['depot']
    Tstop = params['Tstop']
    total_time = 0.0
    for route in solution.get(vehicle, []):
        route_time = 0.0
        for i in range(len(route) - 1):
            node_from = route[i]
            node_to = route[i+1]
            travel_time = params['t'].get((node_from, node_to), 0)
            route_time += travel_time
            # Add Tstop penalty for each customer stop (i.e. when departing a node that is not the depot)
            if node_from != depot:
                route_time += Tstop
        total_time += route_time
    return total_time

# -------------------------------------------------------
# 3. Removal Functions
# -------------------------------------------------------

def random_removal(solution, params, removal_fraction=0.05):
    """
    Randomly removes a fraction of customer visits from the solution.
    The solution is represented as a dictionary mapping vehicle IDs to a list of routes,
    where each route is a list of nodes (with depot at both ends).
    
    Returns:
        partial_solution: a copy of the solution with selected customer visits removed.
        removed_customers: a list of removed customer nodes.
    """
    partial_solution = copy.deepcopy(solution)
    depot = params['depot']
    removed_customers = []

    # Count total customer visits
    visits = []
    for vehicle, routes in solution.items():
        for r_idx, route in enumerate(routes):
            for i in range(1, len(route)-1):  # skip depot at beginning and end
                visits.append((vehicle, r_idx, route[i]))
    
    num_to_remove = int(removal_fraction * len(visits))
    to_remove = random.sample(visits, num_to_remove)

    # Remove selected visits from the routes
    for vehicle, r_idx, customer in to_remove:
        route = partial_solution[vehicle][r_idx]
        # Remove the customer from the route if present
        if customer in route:
            route.remove(customer)
        removed_customers.append(customer)


    # Ensure routes still start and end with depot; if removal made route too short, adjust appropriately.
    for vehicle, routes in partial_solution.items():
        for r_idx, route in enumerate(routes):
            if route[0] != depot:
                route.insert(0, depot)
            if route[-1] != depot:
                route.append(depot)
    return partial_solution, removed_customers

def worst_removal_cost(solution, params, removal_fraction=0.05):
    """
    Worst removal operator based on cost.
    For each customer visit (non-depot node) in a route, we compute a cost penalty
    (the arc cost from the previous node to the customer) using the same formulas as in compute_cost.
    Then we remove the fraction of visits with the highest cost penalty.
    
    Parameters:
      solution (dict): Mapping of vehicle -> list of routes (each route is a list of nodes).
      params (dict): Parameter dictionary.
      removal_fraction (float): Fraction of customer visits to remove.
    
    Returns:
      partial_solution (dict): A copy of the solution with selected visits removed.
      removed_visits (list): List of removed customer nodes.
    """
    partial_solution = copy.deepcopy(solution)
    depot = params['depot']
    removed_visits = []
    visits = []  # List of tuples: (vehicle, route_index, customer, cost_penalty)

    for vehicle, routes in solution.items():
        is_cfv = vehicle.startswith("CFV")
        Q_vehicle = params['Q'][vehicle]
        for r_idx, route in enumerate(routes):
            n_customers = len(route) - 2  # exclude depot at start and end
            if n_customers == 1:
                delivery = Q_vehicle  # full load for one-stop route
            elif n_customers == 2:
                delivery = Q_vehicle / 2.0  # half load for two-stop route
            else:
                continue
            flows = []
            for i in range(len(route) - 1):
                if i == 0:
                    flows.append(Q_vehicle)
                elif n_customers == 1:
                    flows.append(Q_vehicle - delivery)  # should be 0
                elif n_customers == 2:
                    if i == 1:
                        flows.append(Q_vehicle - delivery)  # equals Q/2
                    else:
                        flows.append(Q_vehicle - 2 * delivery)  # equals 0

            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i+1]
                if node_to == depot:
                    continue
                delivered_load = flows[i]
                arc_distance = params['d'].get((node_from, node_to), 0)
                arc_time = params['t'].get((node_from, node_to), 0)
                arc_velocity = params['v_avg'].get((node_from, node_to), 0)
                
                if not is_cfv:
                    arc_cost = (params['r'] * params['h'][vehicle] * arc_distance +
                                params['O'][vehicle] * arc_distance)
                else:
                    curb_cost = params['F'] * params['lambda_param'] * (
                        params['W_c'] * params['phi_param'] * params['sigma_param'] * arc_distance +
                        params['f_param'] * params['I_param'] * params['E_param'] * arc_time +
                        params['epsilon_param'] * params['phi_param'] * arc_distance * (arc_velocity ** 2)
                    )
                    load_cost = params['F'] * params['lambda_param'] * params['phi_param'] * params['sigma_param'] * arc_distance * delivered_load
                    fuel_costs = curb_cost + load_cost
                    emissions_costs = fuel_costs * params['Beta'] * params['Omega'] / 1000
                    arc_cost = fuel_costs + emissions_costs + params['O'][vehicle] * arc_distance
                
                visits.append((vehicle, r_idx, node_to, arc_cost))
    
    visits.sort(key=lambda x: x[3], reverse=True)
    num_to_remove = int(removal_fraction * len(visits))
    to_remove = visits[:num_to_remove]
    
    for vehicle, r_idx, customer, _ in to_remove:
        route = partial_solution[vehicle][r_idx]
        if customer in route:
            route.remove(customer)
        removed_visits.append(customer)
    
    # Ensure each route still starts and ends with depot.
    for vehicle, routes in partial_solution.items():
        for r_idx, route in enumerate(routes):
            if not route:
                continue
            if route[0] != depot:
                route.insert(0, depot)
            if route[-1] != depot:
                route.append(depot)
    
    return partial_solution, removed_visits

def worst_removal_emissions(solution, params, removal_fraction=0.05):
    """
    Worst removal operator based on emissions.
    For each customer visit in a route, we compute an emissions penalty
    (the emissions from the arc arriving at that customer) using the same formulas as in compute_emissions.
    Then we remove the fraction of visits with the highest emissions penalty.
    
    Parameters:
      solution (dict): Mapping of vehicle -> list of routes (each route is a list of nodes).
      params (dict): Parameter dictionary.
      removal_fraction (float): Fraction of customer visits to remove.
    
    Returns:
      partial_solution (dict): A copy of the solution with selected visits removed.
      removed_visits (list): List of removed customer nodes.
    """
    partial_solution = copy.deepcopy(solution)
    depot = params['depot']
    removed_visits = []
    visits = []  # List of tuples: (vehicle, route_index, customer, emissions_penalty)

    for vehicle, routes in solution.items():
        is_cfv = vehicle.startswith("CFV")
        Q_vehicle = params['Q'][vehicle]
        for r_idx, route in enumerate(routes):
            n_customers = len(route) - 2
            if n_customers == 1:
                delivery = Q_vehicle
            elif n_customers == 2:
                delivery = Q_vehicle / 2.0
            else:
                continue
            flows = []
            for i in range(len(route) - 1):
                if i == 0:
                    flows.append(Q_vehicle)
                elif n_customers == 1:
                    flows.append(Q_vehicle - delivery)
                elif n_customers == 2:
                    if i == 1:
                        flows.append(Q_vehicle - delivery)
                    else:
                        flows.append(Q_vehicle - 2 * delivery)
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i+1]
                if node_to == depot:
                    continue
                delivered_load = flows[i]
                d_arc = params['d'].get((node_from, node_to), 0)
                t_arc = params['t'].get((node_from, node_to), 0)
                v_arc = params['v_avg'].get((node_from, node_to), 0)
                
                if is_cfv:
                    emissions_curb = params['Omega'] * params['lambda_param'] * (
                        params['W_c'] * params['phi_param'] * params['sigma_param'] * d_arc +
                        params['f_param'] * params['I_param'] * params['E_param'] * t_arc +
                        params['epsilon_param'] * params['phi_param'] * d_arc * (v_arc ** 2)
                    )
                    emissions_load = params['Omega'] * (params['phi_param'] * params['sigma_param'] *
                                                        params['lambda_param'] * d_arc * delivered_load)
                    arc_emissions = emissions_curb + emissions_load
                else:
                    arc_emissions = params['mu'] * params['h'][vehicle] * d_arc
                visits.append((vehicle, r_idx, node_to, arc_emissions))
    
    visits.sort(key=lambda x: x[3], reverse=True)
    num_to_remove = int(removal_fraction * len(visits))
    to_remove = visits[:num_to_remove]
    
    for vehicle, r_idx, customer, _ in to_remove:
        route = partial_solution[vehicle][r_idx]
        if customer in route:
            route.remove(customer)
        removed_visits.append(customer)
    
    for vehicle, routes in partial_solution.items():
        for r_idx, route in enumerate(routes):
            if not route:
                continue
            if route[0] != depot:
                route.insert(0, depot)
            if route[-1] != depot:
                route.append(depot)
    
    return partial_solution, removed_visits

# -------------------------------------------------------
# 4. Insertion Functions
# -------------------------------------------------------

def route_cost(route, vehicle, params):
    """
    Compute the cost of a single route (list of nodes) for a given vehicle.
    Uses the same formulas as in the compute_cost function.
    """
    cost = 0.0
    Q_vehicle = params['Q'][vehicle]
    n_customers = len(route) - 2  # exclude depot at start and end
    if n_customers == 1:
        delivery = Q_vehicle
    elif n_customers == 2:
        delivery = Q_vehicle / 2.0
    else:
        # If the route is empty or invalid, cost is 0.
        return 0.0

    # Inferred flow on each arc:
    flows = []
    for i in range(len(route) - 1):
        if i == 0:
            flows.append(Q_vehicle)
        elif n_customers == 1:
            flows.append(Q_vehicle - delivery)  # should be 0
        elif n_customers == 2:
            if i == 1:
                flows.append(Q_vehicle - delivery)  # equals Q_vehicle/2
            else:
                flows.append(Q_vehicle - 2 * delivery)  # equals 0

    # Sum cost over arcs.
    for i in range(len(route) - 1):
        node_from = route[i]
        node_to = route[i+1]
        arc_distance = params['d'].get((node_from, node_to), 0)
        arc_time = params['t'].get((node_from, node_to), 0)
        arc_velocity = params['v_avg'].get((node_from, node_to), 0)
        delivered_load = flows[i]
        if vehicle.startswith("CFV"):
            curb_cost = params['F'] * params['lambda_param'] * (
                params['W_c'] * params['phi_param'] * params['sigma_param'] * arc_distance +
                params['f_param'] * params['I_param'] * params['E_param'] * arc_time +
                params['epsilon_param'] * params['phi_param'] * arc_distance * (arc_velocity ** 2)
            )
            load_cost = params['F'] * params['lambda_param'] * params['phi_param'] * params['sigma_param'] * arc_distance * delivered_load
            fuel_costs = curb_cost + load_cost
            emissions_costs = fuel_costs * params['Beta'] * params['Omega'] / 1000
            arc_cost = fuel_costs + emissions_costs + params['O'][vehicle] * arc_distance
        else:
            arc_cost = params['r'] * params['h'][vehicle] * arc_distance + params['O'][vehicle] * arc_distance
        cost += arc_cost
    return cost

def route_emissions(route, vehicle, params):
    """
    Compute the emissions (in metric tons) of a single route (list of nodes) for a given vehicle.
    Uses the same formulas as in the compute_emissions function.
    """
    emissions = 0.0
    Q_vehicle = params['Q'][vehicle]
    n_customers = len(route) - 2
    if n_customers == 1:
        delivery = Q_vehicle
    elif n_customers == 2:
        delivery = Q_vehicle / 2.0
    else:
        return 0.0

    flows = []
    for i in range(len(route)-1):
        if i == 0:
            flows.append(Q_vehicle)
        elif n_customers == 1:
            flows.append(Q_vehicle - delivery)
        elif n_customers == 2:
            if i == 1:
                flows.append(Q_vehicle - delivery)
            else:
                flows.append(Q_vehicle - 2*delivery)

    for i in range(len(route)-1):
        node_from = route[i]
        node_to = route[i+1]
        d_arc = params['d'].get((node_from, node_to), 0)
        t_arc = params['t'].get((node_from, node_to), 0)
        v_arc = params['v_avg'].get((node_from, node_to), 0)
        delivered_load = flows[i]
        if vehicle.startswith("CFV"):
            emissions_curb = params['Omega'] * params['lambda_param'] * (
                params['W_c'] * params['phi_param'] * params['sigma_param'] * d_arc +
                params['f_param'] * params['I_param'] * params['E_param'] * t_arc +
                params['epsilon_param'] * params['phi_param'] * d_arc * (v_arc ** 2)
            )
            emissions_load = params['Omega'] * (params['phi_param'] * params['sigma_param'] *
                                                params['lambda_param'] * d_arc * delivered_load)
            arc_emissions = emissions_curb + emissions_load
        else:
            arc_emissions = params['mu'] * params['h'][vehicle] * d_arc
        emissions += arc_emissions
    return emissions / 1000.0  # convert kg to metric tons

def clean_solution(solution):
    """
    Remove trips that have no customer visits (i.e., trips with length < 3).
    """
    new_solution = {}
    for vehicle, routes in solution.items():
        valid_routes = [route for route in routes if len(route) >= 3]
        new_solution[vehicle] = valid_routes
    return new_solution

def greedy_insertion_cost(solution, removed_customers, params):
    """
    Greedy insertion operator based on cost.
    Attempts to insert removed customers into existing routes or creates a new route.
    Uses check_feasibility to filter out candidate insertions that would violate constraints.
    
    Returns:
      new_solution (dict): Updated solution with removed customers reinserted.
    """
    new_solution = copy.deepcopy(solution)
    depot = params['depot']
    
    for customer in removed_customers:
        best_increase = float('inf')
        best_vehicle = None
        best_route_index = None
        best_position = None
        
        # Try inserting into each existing route.
        for vehicle, routes in new_solution.items():
            for r_idx, route in enumerate(routes):
                # Try each possible insertion position.
                for pos in range(1, len(route)):
                    candidate_route = route[:pos] + [customer] + route[pos:]
                    # Ensure the candidate route has a valid length (e.g., 3 or 4 nodes).
                    if len(candidate_route) not in [3, 4]:
                        continue
                    # Create a candidate solution with this insertion.
                    candidate_solution = copy.deepcopy(new_solution)
                    candidate_solution[vehicle][r_idx] = candidate_route
                    # Check feasibility of the candidate.
                    if not check_feasibility(candidate_solution, params):
                        continue
                    old_cost = route_cost(route, vehicle, params)
                    new_cost = route_cost(candidate_route, vehicle, params)
                    increase = new_cost - old_cost
                    if increase < best_increase:
                        best_increase = increase
                        best_vehicle = vehicle
                        best_route_index = r_idx
                        best_position = pos
        
        # Also consider creating a new route.
        for vehicle in new_solution.keys():
            candidate_route = [depot, customer, depot]
            candidate_solution = copy.deepcopy(new_solution)
            candidate_solution[vehicle].append(candidate_route)
            if not check_feasibility(candidate_solution, params):
            #    print(f"New route candidate for customer {customer} on vehicle {vehicle} failed feasibility.")
                continue
            candidate_cost = route_cost(candidate_route, vehicle, params)
            #print(f"New route candidate for customer {customer} on vehicle {vehicle} cost: {candidate_cost}")
            if candidate_cost < best_increase:
                best_increase = candidate_cost
                best_vehicle = vehicle
                best_route_index = None
                best_position = None
        
        # Insert the customer using the best feasible option.
        if best_vehicle is not None:
            if best_route_index is None:
                new_solution[best_vehicle].append([depot, customer, depot])
            else:
                new_solution[best_vehicle][best_route_index].insert(best_position, customer)

    new_solution = clean_solution(new_solution)

    return new_solution

def greedy_insertion_emissions(solution, removed_customers, params):
    """
    Greedy insertion operator based on emissions.
    For each removed customer, it attempts to insert the customer into an existing route
    or create a new route [depot, customer, depot]. Candidate insertions are only considered
    if the resulting candidate solution is feasible.
    
    Returns:
      new_solution (dict): Updated solution with removed customers reinserted.
    """
    new_solution = copy.deepcopy(solution)
    depot = params['depot']
    
    for customer in removed_customers:
        best_increase = float('inf')
        best_vehicle = None
        best_route_index = None
        best_position = None
        
        # Try inserting into each existing route.
        for vehicle, routes in new_solution.items():
            for r_idx, route in enumerate(routes):
                if len(route) < 4:  # if route is not full (assumed limit: 1 or 2 stops)
                    for pos in range(1, len(route)):
                        candidate_route = route[:pos] + [customer] + route[pos:]
                        if len(candidate_route) not in [3, 4]:
                            continue
                        # Create a candidate solution with this modified route.
                        candidate_solution = copy.deepcopy(new_solution)
                        candidate_solution[vehicle][r_idx] = candidate_route
                        if not check_feasibility(candidate_solution, params):
                            continue
                        old_emissions = route_emissions(route, vehicle, params)
                        new_emissions = route_emissions(candidate_route, vehicle, params)
                        increase = new_emissions - old_emissions
                        if increase < best_increase:
                            best_increase = increase
                            best_vehicle = vehicle
                            best_route_index = r_idx
                            best_position = pos
        
        # Also consider creating a new route.
        for vehicle in new_solution.keys():
            candidate_route = [depot, customer, depot]
            candidate_solution = copy.deepcopy(new_solution)
            candidate_solution[vehicle].append(candidate_route)
            if not check_feasibility(candidate_solution, params):
                continue
            candidate_emissions = route_emissions(candidate_route, vehicle, params)
            if candidate_emissions < best_increase:
                best_increase = candidate_emissions
                best_vehicle = vehicle
                best_route_index = None  # signal new route creation
                best_position = None
        
        # Insert the customer using the best feasible option found.
        if best_vehicle is not None:
            if best_route_index is None:
                new_solution[best_vehicle].append([depot, customer, depot])
            else:
                new_solution[best_vehicle][best_route_index].insert(best_position, customer)
    
    new_solution = clean_solution(new_solution)
    
    return new_solution

def regret2_insertion_cost(solution, removed_customers, params):
    """
    Regret-2 insertion operator based on cost.
    For each removed customer, computes the best and second-best cost increases from candidate insertions,
    either into an existing route or as a new route, and selects the customer with the maximum regret.
    Only feasible candidate insertions are considered.
    
    Returns:
      new_solution (dict): Updated solution with all removed customers reinserted.
    """
    new_solution = copy.deepcopy(solution)
    depot = params['depot']
    remaining = list(removed_customers)
    
    while remaining:
        candidate_info = []  # Each tuple: (customer, regret, best_increase, best_info)
        for customer in remaining:
            best_increase = float('inf')
            second_best = float('inf')
            best_info = None  # Expected to be a tuple: (vehicle, route_index, position) or (vehicle, None, None) for new route.
            
            # Check insertion into existing routes.
            for vehicle, routes in new_solution.items():
                for r_idx, route in enumerate(routes):
                    if len(route) < 4:
                        for pos in range(1, len(route)):
                            candidate_route = route[:pos] + [customer] + route[pos:]
                            if len(candidate_route) not in [3, 4]:
                                continue
                            candidate_solution = copy.deepcopy(new_solution)
                            candidate_solution[vehicle][r_idx] = candidate_route
                            if not check_feasibility(candidate_solution, params):
                                continue
                            old_cost = route_cost(route, vehicle, params)
                            new_cost = route_cost(candidate_route, vehicle, params)
                            increase = new_cost - old_cost
                            if increase < best_increase:
                                second_best = best_increase
                                best_increase = increase
                                best_info = (vehicle, r_idx, pos)
                            elif increase < second_best:
                                second_best = increase
            # Consider creating a new route.
            for vehicle in new_solution.keys():
                candidate_route = [depot, customer, depot]
                candidate_solution = copy.deepcopy(new_solution)
                candidate_solution[vehicle].append(candidate_route)
                if not check_feasibility(candidate_solution, params):
                    continue
                candidate_cost = route_cost(candidate_route, vehicle, params)
                if candidate_cost < best_increase:
                    second_best = best_increase
                    best_increase = candidate_cost
                    best_info = (vehicle, None, None)
                elif candidate_cost < second_best:
                    second_best = candidate_cost
            
            # Fallback if no feasible insertion was found.
            if best_info is None:
                fallback_insertion = None
                fallback_cost = float('inf')
                for vehicle in new_solution.keys():
                    candidate_route = [depot, customer, depot]
                    candidate_solution = copy.deepcopy(new_solution)
                    candidate_solution[vehicle].append(candidate_route)
                    if not check_feasibility(candidate_solution, params):
                        continue
                    cost_candidate = route_cost(candidate_route, vehicle, params)
                    if cost_candidate < fallback_cost:
                        fallback_cost = cost_candidate
                        fallback_insertion = (vehicle, None, None)
                if fallback_insertion is None:
                    # Skip the customer if no insertion is feasible.
                    candidate_info.append((customer, 0, float('inf'), None))
                else:
                    best_info = fallback_insertion
                    best_increase = fallback_cost
                    second_best = fallback_cost  # Not used further in this case.
                    candidate_info.append((customer, 0, best_increase, best_info))
            else:
                regret = second_best - best_increase
                candidate_info.append((customer, regret, best_increase, best_info))
        
        candidate_info.sort(key=lambda x: x[1], reverse=True)
        chosen_customer, regret, best_increase, best_info = candidate_info[0]
        if best_info is None:
            # If still no candidate, skip this customer.
            remaining.remove(chosen_customer)
            continue
        vehicle, r_idx, pos = best_info
        if r_idx is None:
            new_solution[vehicle].append([depot, chosen_customer, depot])
        else:
            new_solution[vehicle][r_idx].insert(pos, chosen_customer)
        remaining.remove(chosen_customer)
    
    new_solution = clean_solution(new_solution)
    return new_solution

def regret2_insertion_emissions(solution, removed_customers, params):
    """
    Regret-2 insertion operator based on emissions.
    For each removed customer, calculates the best and second-best marginal emissions increases from candidate insertions,
    either into an existing route or as a new route, and selects the customer with the maximum regret.
    Only feasible candidate insertions are considered.
    
    Returns:
      new_solution (dict): Updated solution with all removed customers reinserted.
    """
    new_solution = copy.deepcopy(solution)
    depot = params['depot']
    remaining = list(removed_customers)
    
    while remaining:
        candidate_info = []  # Each tuple: (customer, regret, best_increase, best_info)
        for customer in remaining:
            best_increase = float('inf')
            second_best = float('inf')
            best_info = None  # (vehicle, route_index, position) or (vehicle, None, None) for new route.
            
            for vehicle, routes in new_solution.items():
                for r_idx, route in enumerate(routes):
                    if len(route) < 4:
                        for pos in range(1, len(route)):
                            candidate_route = route[:pos] + [customer] + route[pos:]
                            if len(candidate_route) not in [3, 4]:
                                continue
                            candidate_solution = copy.deepcopy(new_solution)
                            candidate_solution[vehicle][r_idx] = candidate_route
                            if not check_feasibility(candidate_solution, params):
                                continue
                            old_emissions = route_emissions(route, vehicle, params)
                            new_emissions = route_emissions(candidate_route, vehicle, params)
                            increase = new_emissions - old_emissions
                            if increase < best_increase:
                                second_best = best_increase
                                best_increase = increase
                                best_info = (vehicle, r_idx, pos)
                            elif increase < second_best:
                                second_best = increase
            for vehicle in new_solution.keys():
                candidate_route = [depot, customer, depot]
                candidate_solution = copy.deepcopy(new_solution)
                candidate_solution[vehicle].append(candidate_route)
                if not check_feasibility(candidate_solution, params):
                    continue
                candidate_emissions = route_emissions(candidate_route, vehicle, params)
                if candidate_emissions < best_increase:
                    second_best = best_increase
                    best_increase = candidate_emissions
                    best_info = (vehicle, None, None)
                elif candidate_emissions < second_best:
                    second_best = candidate_emissions
            
            # Fallback if no candidate was found.
            if best_info is None:
                fallback_insertion = None
                fallback_emissions = float('inf')
                for vehicle in new_solution.keys():
                    candidate_route = [depot, customer, depot]
                    candidate_solution = copy.deepcopy(new_solution)
                    candidate_solution[vehicle].append(candidate_route)
                    if not check_feasibility(candidate_solution, params):
                        continue
                    emissions_candidate = route_emissions(candidate_route, vehicle, params)
                    if emissions_candidate < fallback_emissions:
                        fallback_emissions = emissions_candidate
                        fallback_insertion = (vehicle, None, None)
                if fallback_insertion is None:
                    candidate_info.append((customer, 0, float('inf'), None))
                else:
                    best_info = fallback_insertion
                    best_increase = fallback_emissions
                    second_best = fallback_emissions
                    candidate_info.append((customer, 0, best_increase, best_info))
            else:
                regret = second_best - best_increase
                candidate_info.append((customer, regret, best_increase, best_info))
        
        candidate_info.sort(key=lambda x: x[1], reverse=True)
        chosen_customer, regret, best_increase, best_info = candidate_info[0]
        if best_info is None:
            remaining.remove(chosen_customer)
            continue
        vehicle, r_idx, pos = best_info
        if r_idx is None:
            new_solution[vehicle].append([depot, chosen_customer, depot])
        else:
            new_solution[vehicle][r_idx].insert(pos, chosen_customer)
        remaining.remove(chosen_customer)
    
    new_solution = clean_solution(new_solution)
    return new_solution

# -------------------------------------------------------
# 5. Operator Manager and Selection Functions
# -------------------------------------------------------

# Establishing the Operator Manager class
class OperatorManager:
    def __init__(self, removal_ops, insertion_ops, phi=0.1):
        """
        Initialize the operator manager with removal and insertion operator dictionaries.
        
        Parameters:
          removal_ops (dict): Dictionary mapping removal operator names to dicts with keys:
                              'func': the removal function,
                              'weight': initial weight,
                              'score_history': list for tracking performance scores.
          insertion_ops (dict): Similar dictionary for insertion operators.
          phi (float): Reaction factor used for weight updates.
        """
        self.removal_ops = removal_ops
        self.insertion_ops = insertion_ops
        self.phi = phi

    def select_operator(self, ops_dict):
        """
        Select an operator from the provided dictionary with probability proportional to its weight.
        Returns:
          The key (name) of the selected operator.
        """
        total_weight = sum(op["weight"] for op in ops_dict.values())
        if total_weight == 0:
            return random.choice(list(ops_dict.keys()))
        pick = random.uniform(0, total_weight)
        cumulative = 0.0
        for op_name, op_data in ops_dict.items():
            cumulative += op_data["weight"]
            if pick <= cumulative:
                return op_name
        return list(ops_dict.keys())[-1]  # Fallback

    def select_removal_operator(self):
        """Select a removal operator based on current weights."""
        return self.select_operator(self.removal_ops)

    def select_insertion_operator(self):
        """Select an insertion operator based on current weights."""
        return self.select_operator(self.insertion_ops)

    def update_operator_weight(self, ops_dict, operator_name, score):
        """
        Update the weight for a given operator using:
            W_new = W_old * (1 - phi) + phi * score
        and record the score in the operator's score history.
        """
        old_weight = ops_dict[operator_name]["weight"]
        new_weight = old_weight * (1 - self.phi) + self.phi * score
        ops_dict[operator_name]["weight"] = new_weight
        ops_dict[operator_name]["score_history"].append(score)

    def update_removal_operator(self, operator_name, score):
        """Update the weight for a removal operator."""
        self.update_operator_weight(self.removal_ops, operator_name, score)

    def update_insertion_operator(self, operator_name, score):
        """Update the weight for an insertion operator."""
        self.update_operator_weight(self.insertion_ops, operator_name, score)

    def get_removal_operator_func(self, operator_name):
        """Return the removal operator function corresponding to operator_name."""
        return self.removal_ops[operator_name]["func"]

    def get_insertion_operator_func(self, operator_name):
        """Return the insertion operator function corresponding to operator_name."""
        return self.insertion_ops[operator_name]["func"]

# Initialize removal and insertion operator dictionaries
# Each operator entry stores its function, initial weight, and score history.
removal_ops = {
    "random_removal": {
        "func": random_removal,
        "weight": 1.0,
        "score_history": [],
    },
    "worst_cost_removal": {
        "func": worst_removal_cost,
        "weight": 1.0,
        "score_history": [],
    },
    "worst_emissions_removal": {
        "func": worst_removal_emissions,
        "weight": 1.0,
        "score_history": [],
    },
}

insertion_ops = {
    "greedy_insertion_cost": {
        "func": greedy_insertion_cost,
        "weight": 1.0,
        "score_history": [],
    },
    "greedy_insertion_emissions": {
        "func": greedy_insertion_emissions,
        "weight": 1.0,
        "score_history": [],
    },
    "regret2_insertion_cost": {
        "func": regret2_insertion_cost,
        "weight": 1.0,
        "score_history": [],
    },
    "regret2_insertion_emissions": {
        "func": regret2_insertion_emissions,
        "weight": 1.0,
        "score_history": [],
    },
}

operator_manager = OperatorManager(removal_ops, insertion_ops, phi=0.1)

# -------------------------------------------------------
# 6. Pareto Function
# -------------------------------------------------------

def dominates(sol1_obj, sol2_obj):
    """
    Returns True if sol1_obj dominates sol2_obj.
    Each objective tuple is (cost, emissions) and lower values are better.
    """
    cost1, emis1 = sol1_obj
    cost2, emis2 = sol2_obj
    # sol1 dominates sol2 if it's no worse in both objectives and strictly better in one.
    return (cost1 <= cost2 and emis1 <= emis2) and (cost1 < cost2 or emis1 < emis2)

# -------------------------------------------------------
# 7. ALNS + SA Functions
# -------------------------------------------------------

def acceptance_criterion_SA(new_solution, current_solution, temperature, params):
    """
    Simulated Annealing acceptance criterion based on cost difference.
    This version ensures that infeasible solutions are never accepted.
    
    Parameters:
      new_solution: Candidate solution.
      current_solution: Current solution.
      temperature (float): Current temperature.
      params (dict): Parameter dictionary (used by evaluate_solution and check_feasibility).
    
    Returns:
      accepted (bool): True if new_solution is accepted.
      improvement_type (str): "improved" if the candidate is strictly better, otherwise None.
    """
    # Reject new_solution outright if it is infeasible.
    if not check_feasibility(new_solution, params):
        return False, None

    new_cost, _ = evaluate_solution(new_solution, params)
    current_cost, _ = evaluate_solution(current_solution, params)
    delta = new_cost - current_cost

    if delta < 0:
        return True, "improved"
    else:
        probability = math.exp(-delta / temperature)
        if random.random() < probability:
            return True, None
        else:
            return False, None

def alns_iteration_SA(current_solution, params, operator_manager, temperature, removal_fraction=0.05):
    # Select removal and insertion operator names.
    rem_op_name = operator_manager.select_removal_operator()
    ins_op_name = operator_manager.select_insertion_operator()
    
    # Get the operator functions.
    removal_func = operator_manager.get_removal_operator_func(rem_op_name)
    insertion_func = operator_manager.get_insertion_operator_func(ins_op_name)
    
    # Apply removal and then insertion.
    partial_solution, removed_customers = removal_func(current_solution, params, removal_fraction)
    candidate_solution = insertion_func(partial_solution, removed_customers, params)
    
    # Apply SA acceptance criterion.
    accepted, improvement_type = acceptance_criterion_SA(candidate_solution, current_solution, temperature, params)
    
    # Determine a score for weight updates.
    if improvement_type == "new_best":
        score = 5.0
    elif improvement_type == "improved":
        score = 2.0
    elif accepted:
        score = 1.0
    else:
        score = 0.0
    
    # Update operator weights.
    operator_manager.update_removal_operator(rem_op_name, score)
    operator_manager.update_insertion_operator(ins_op_name, score)
    
    return candidate_solution if accepted else current_solution

def run_alns(num_iterations, initial_solution, params, operator_manager):
    """
    Runs the ALNS+SA algorithm for a given number of iterations.
    
    Temperature is updated according to a cooling schedule.
    """
    current_solution = copy.deepcopy(initial_solution)
    T0 = params.get('T0')
    cooling_rate = params.get('cooling_rate')
    temperature = T0

    for iteration in range(num_iterations):
        current_solution = alns_iteration_SA(current_solution, params, operator_manager, temperature, removal_fraction=0.05)
        temperature *= cooling_rate  # Cool down
        # Optionally log current status.
    
    return current_solution

# ===============================================================================================================================
# ===============================================================================================================================
#                                ALNS OUTPUT
# ===============================================================================================================================
# ===============================================================================================================================
"""
def print_objective_breakdown(solution, params):
    #Given an ALNS solution (a dict mapping vehicle to a list of trips),
    #prints a breakdown of the objective function terms for each arc and trip,
    #and then sums up per-vehicle and overall costs and emissions.
    
    #For CFVs, per arc:
    #  - curb_cost = F * lambda_param * (W_c * phi_param * sigma_param * arc_distance +
    #                                      f_param * I_param * E_param * arc_time +
    #                                      epsilon_param * phi_param * arc_distance * (arc_velocity ** 2))
    #  - load_cost = F * lambda_param * phi_param * sigma_param * arc_distance * delivered_load
    #  - fuel_costs = curb_cost + load_cost
    #  - emissions_costs = (fuel_costs * Beta * Omega) / 1000
    #  - operating_cost = O[vehicle] * arc_distance
    #  - arc_cost = fuel_costs + emissions_costs + operating_cost
      
    #For HGEVs, per arc:
    #  - recharge_cost = r * h[vehicle] * arc_distance
    #  - operating_cost = O[vehicle] * arc_distance
    #  - arc_cost = recharge_cost + operating_cost
    #  - Additionally, for HGEVs, we compute arc emissions as:
    #        arc_emissions = (mu * h[vehicle] * arc_distance) / 1000
    #Acquisition cost is added once per used vehicle: (A[vehicle] - S[vehicle]).
    depot = params['depot']
    overall_total_cost = 0.0
    overall_total_emissions = 0.0
    
    # Iterate over vehicles.
    for vehicle, trips in solution.items():
        vehicle_total_cost = 0.0
        vehicle_total_emissions = 0.0
        print(f"Vehicle {vehicle}:")
        
        # Determine vehicle capacity.
        Q_vehicle = params['Q'][vehicle]
        
        # Iterate over each trip.
        for trip_index, route in enumerate(trips):
            # Skip empty trips (or trips that haven't been filled, e.g., [depot, depot])
            if len(route) < 3:
                continue
            print(f"  Trip {trip_index}: {route}")
            
            # Determine delivered amount per stop for this trip.
            n_customers = len(route) - 2  # exclude depot at start and end
            if n_customers == 1:
                delivery = Q_vehicle
            elif n_customers == 2:
                delivery = Q_vehicle / 2.0
            else:
                delivery = 0  # Should not occur
            
            # Compute inferred flows for each arc.
            flows = []
            for i in range(len(route) - 1):
                if i == 0:
                    flows.append(Q_vehicle)
                elif n_customers == 1:
                    flows.append(Q_vehicle - delivery)  # should be 0
                elif n_customers == 2:
                    if i == 1:
                        flows.append(Q_vehicle - delivery)  # equals Q_vehicle/2
                    else:
                        flows.append(Q_vehicle - 2 * delivery)  # equals 0
            
            # Iterate over arcs.
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i+1]
                arc_distance = params['d'].get((node_from, node_to), 0)
                arc_time = params['t'].get((node_from, node_to), 0)
                arc_velocity = params['v_avg'].get((node_from, node_to), 0)
                delivered_load = flows[i]
                
                if vehicle.startswith("CFV"):
                    curb_cost = params['F'] * params['lambda_param'] * (
                        params['W_c'] * params['phi_param'] * params['sigma_param'] * arc_distance +
                        params['f_param'] * params['I_param'] * params['E_param'] * arc_time +
                        params['epsilon_param'] * params['phi_param'] * arc_distance * (arc_velocity ** 2)
                    )
                    load_cost = params['F'] * params['lambda_param'] * params['phi_param'] * params['sigma_param'] * arc_distance * delivered_load
                    fuel_costs = curb_cost + load_cost
                    emissions_costs = (fuel_costs * params['Beta'] * params['Omega']) / 1000
                    operating_cost = params['O'][vehicle] * arc_distance
                    arc_cost = fuel_costs + emissions_costs + operating_cost
                    arc_emissions = emissions_costs  # For CFVs, we use the computed emissions_costs as the emissions contribution.
                    print(f"    Arc {node_from} -> {node_to}:")
                    print(f"      Distance: {arc_distance}, Time: {arc_time}, Velocity: {arc_velocity}")
                    print(f"      Delivered load: {delivered_load}")
                    print(f"      CFV Costs -> Curb: {curb_cost:.2f}, Load: {load_cost:.2f}, Fuel: {fuel_costs:.2f}")
                    print(f"      Emissions Cost: {emissions_costs:.2f}, Operating: {operating_cost:.2f}")
                else:
                    operating_cost = params['O'][vehicle] * arc_distance
                    recharge_cost = params['r'] * params['h'][vehicle] * arc_distance
                    arc_cost = recharge_cost + operating_cost
                    # Compute emissions for HGEVs as: (mu * h[vehicle] * arc_distance) / 1000
                    arc_emissions = (params['mu'] * params['h'][vehicle] * arc_distance) / 1000.0
                    print(f"    Arc {node_from} -> {node_to}:")
                    print(f"      Distance: {arc_distance}, Time: {arc_time}, Velocity: {arc_velocity}")
                    print(f"      HGEV Costs -> Recharge: {recharge_cost:.2f}, Operating: {operating_cost:.2f}")
                    print(f"      Arc cost: {arc_cost:.2f}")
                
                print(f"      Arc cost: {arc_cost:.2f}")
                vehicle_total_cost += arc_cost
                vehicle_total_emissions += arc_emissions
            
        # If the vehicle has at least one trip (i.e., is used), add acquisition cost.
        if any(len(trip) >= 3 for trip in trips):
            acquisition_cost = params['A'][vehicle] - params['S'][vehicle]
        else:
            acquisition_cost = 0
        vehicle_total_cost += acquisition_cost
        
        print(f"  Acquisition cost for vehicle {vehicle}: {acquisition_cost:.2f}")
        print(f"  Total cost for vehicle {vehicle}: {vehicle_total_cost:.2f}")
        print(f"  Total emissions for vehicle {vehicle}: {vehicle_total_emissions:.2f}\n")
        overall_total_cost += vehicle_total_cost
        overall_total_emissions += vehicle_total_emissions
        
    print(f"Overall total cost: {overall_total_cost:.2f}")
    print(f"Overall total emissions: {overall_total_emissions:.2f}")
"""

if __name__ == "__main__":

    # --- Generate an initial solution using the greedy heuristic ---
    initial_solution, rem_demand = greedy_search_heuristic(K_list, list(D), depot, d, p, Q)
    
    print("Initial Greedy Solution (routes):")
    for vehicle, routes in initial_solution.items():
        print(f"{vehicle}: {routes}")
    print("Remaining demand after greedy:", rem_demand)

    # --- Pass the initial solution into the ALNS loop ---
    final_solution = run_alns(num_iterations=max_iterations, 
                              initial_solution=initial_solution, 
                              params=params, 
                              operator_manager=operator_manager)
    
    print("\nALNS finished. Final solution:")
    for vehicle, routes in final_solution.items():
        print(f"{vehicle}: {routes}")
    
    # --- Operator weight output ---
    print("Final removal operator weights:")
    for op_name, op_data in removal_ops.items():
        print(f"  {op_name} -> weight={op_data['weight']:.2f}, score history={op_data['score_history']}")
    
    print("Final insertion operator weights:")
    for op_name, op_data in insertion_ops.items():
        print(f"  {op_name} -> weight={op_data['weight']:.2f}, score history={op_data['score_history']}")
    
    # --- Print objective function breakdown ---
    #print("\nObjective Function Breakdown:")
    #print_objective_breakdown(final_solution, params)


