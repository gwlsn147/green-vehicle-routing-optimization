import numpy as np
import pandas as pd
import math
import gurobipy as gp
from gurobipy import GRB
import os
import re # For parsing potentially problematic numbers

# --- Configuration ---
# File Paths (Update these to your actual file locations)
DATA_DIR = "./data" # Directory containing the CSV files - Assuming current directory
GENERAL_PARAMS_FILE = os.path.join(DATA_DIR, "General.csv")
DISTANCE_FILE = os.path.join(DATA_DIR, "Distances.csv")
TIME_FILE = os.path.join(DATA_DIR, "Times.csv")
DEMAND_FILE = os.path.join(DATA_DIR, "Demand.csv")

# Operational Parameters (Can be overridden by General.csv if present)
ANNUAL_OPERATING_DAYS = 250
DEPOT_NODE = 20 # Index of the depot (0 to num_customers)
HGEV_MIN_SOC_PERCENT = 0.20 # Default Min allowed state of charge
HGEV_MAX_SOC_PERCENT = 0.98 # Default Max allowed state of charge

# Fleet Configuration (Can be overridden by General.csv if present)
NUM_CFV = 20 # Default number
NUM_HGEV = 20 # Default number

# --- Helper Functions ---

def check_file(path):
    """Checks if a file exists at the given path."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return False
    return True

def parse_numeric(value):
    """Safely parses a value to float, handling potential errors or specific strings."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ['-', '????', '']:
            return None # Represent missing/undefined values as None
        try:
            # Remove commas used as thousands separators
            value = value.replace(',', '')
            return float(value)
        except ValueError:
            print(f"Warning: Could not parse '{value}' as numeric.")
            return None
    return None # Handle other types if necessary

def load_general_params(filepath):
    """Loads parameters from the General.csv file."""
    if not check_file(filepath):
        return None

    try:
        # Read the CSV, skipping initial rows and using the 'Parameter' column
        # Need to handle the specific structure where Parameter name is in column 1 (index 1)
        # and Value is in column 4 (index 4)
        df = pd.read_csv(filepath, skiprows=1) # Skip header row

        params = {}
        # Define mapping from CSV Parameter column to dictionary keys
        # Using more descriptive keys internally
        param_mapping = {
            'Ak (HGEV)': 'HGEV_ACQUISITION_COST',
            'Sk': 'HGEV_SUBSIDY_BASE', # Base value from CSV
            'hk': 'HGEV_ENERGY_CONSUMPTION',
            'Qk': 'VEHICLE_CAPACITY', # Assuming same for both for now
            'Rk': 'HGEV_BATTERY_CAPACITY',
            'Lk': 'HGEV_INITIAL_CHARGE_PERCENT',
            'mu': 'HGEV_GRID_EMISSION_FACTOR',
            'Ak (CFV)': 'CFV_ACQUISITION_COST',
            'xi': 'CFV_xi',
            'f': 'CFV_f_param',
            'I': 'CFV_I_param',
            'E': 'CFV_E_param',
            'm': 'CFV_m_param',
            'n': 'CFV_n_param',
            'o': 'CFV_o_param',
            'rho': 'CFV_rho',
            'Wc': 'CFV_W_c',
            'tau': 'CFV_tau',
            'g': 'CFV_g_param',
            'Cd': 'CFV_C_d',
            'Cr': 'CFV_C_r',
            'alpha': 'CFV_alpha',
            'psi': 'CFV_psi',
            'Theta': 'CFV_theta_deg',
            'Omega': 'CFV_OMEGA',
            'r': 'HGEV_CHARGE_COST_PER_KWH',
            'Beta': 'BETA_BASE', # Base value from CSV ($/tonne)
            'Upsilon': 'UPSILON_BASE', # Base value from CSV ($/tonne)
            'G': 'CARBON_CAP_BASE' # Base value from CSV
        }
        
        # Sensitivity range keys
        range_mapping = {
            'Sk': 'HGEV_SUBSIDY', # Parameter name for range
            'Beta': 'BETA',       # Parameter name for range
            'Upsilon': 'UPSILON'  # Parameter name for range
        }

        # Iterate through rows to find parameters
        for index, row in df.iterrows():
            param_name = row.iloc[1] # Parameter name in column index 1
            if pd.isna(param_name): continue # Skip empty rows

            param_name = param_name.strip()

            # Check if it's a parameter we recognize
            if param_name in param_mapping:
                dict_key = param_mapping[param_name]
                value = parse_numeric(row.iloc[4]) # Value in column index 4
                if value is not None:
                    params[dict_key] = value
                    # print(f"Loaded {dict_key}: {value}") # Debug print

            # Check if it's a parameter with sensitivity ranges
            if param_name in range_mapping:
                 range_key = range_mapping[param_name]
                 low = parse_numeric(row.iloc[5]) # Low range in column index 5
                 high = parse_numeric(row.iloc[6]) # High range in column index 6
                 if low is not None:
                     params[f'{range_key}_LOW'] = low
                 if high is not None:
                     params[f'{range_key}_HIGH'] = high
                 # print(f"Loaded Range {range_key}: Low={low}, High={high}") # Debug print


        # --- Unit Conversions and Defaults ---
        # Convert Beta and Upsilon from $/tonne to $/kg
        if 'BETA_BASE' in params:
            params['BETA_PER_KG'] = params['BETA_BASE'] / 1000.0
        else:
            params['BETA_PER_KG'] = 0.0 # Default if not found
            print("Warning: BETA_BASE not found in General.csv, using 0.")

        if 'UPSILON_BASE' in params:
            params['UPSILON_PER_KG'] = params['UPSILON_BASE'] / 1000.0
        else:
            params['UPSILON_PER_KG'] = 0.0 # Default if not found
            print("Warning: UPSILON_BASE not found in General.csv, using 0.")

        # Set default Carbon Cap if missing or invalid
        if params.get('CARBON_CAP_BASE', None) is None:
            params['CARBON_CAP'] = 500000.0 # Default value
            print(f"Warning: CARBON_CAP_BASE not found or invalid in General.csv, using default: {params['CARBON_CAP']}")
        else:
             params['CARBON_CAP'] = params['CARBON_CAP_BASE']
             
        # Set default subsidy if base is missing
        if 'HGEV_SUBSIDY_BASE' not in params:
            params['HGEV_SUBSIDY_BASE'] = 0.0
            print("Warning: HGEV_SUBSIDY_BASE not found in General.csv, using 0.")

        # Ensure essential parameters have defaults if missing from CSV
        default_params = {
            'HGEV_ACQUISITION_COST': 400000.0, 'HGEV_ENERGY_CONSUMPTION': 1.1,
            'VEHICLE_CAPACITY': 18000.0, 'HGEV_BATTERY_CAPACITY': 900.0,
            'HGEV_INITIAL_CHARGE_PERCENT': 0.98, 'HGEV_GRID_EMISSION_FACTOR': 0.350,
            'CFV_ACQUISITION_COST': 165000.0, 'CFV_xi': 1.0, 'CFV_f_param': 0.2,
            'CFV_I_param': 22.0, 'CFV_E_param': 12.9, 'CFV_m_param': 0.9,
            'CFV_n_param': 44000.0, 'CFV_o_param': 0.4, 'CFV_rho': 1.2041,
            'CFV_W_c': 12500.0, 'CFV_tau': 0.68, 'CFV_g_param': 9.81,
            'CFV_C_d': 0.7, 'CFV_C_r': 0.01, 'CFV_alpha': 7.2, 'CFV_psi': 835.0,
            'CFV_theta_deg': 0.0, 'CFV_OMEGA': 2.68, 'HGEV_CHARGE_COST_PER_KWH': 0.45
        }
        for key, default_value in default_params.items():
            if key not in params:
                print(f"Warning: Parameter {key} not found in General.csv. Using default value: {default_value}")
                params[key] = default_value
                
        # Calculate radians for theta
        params['CFV_theta_rad'] = math.radians(params['CFV_theta_deg'])

        return params

    except Exception as e:
        print(f"Error loading parameters from {filepath}: {e}")
        return None

def calculate_cfv_fuel_ghg(dist_km, time_hr, avg_speed_kmh, load_kg, params):
    """Calculates CFV fuel consumption (L) and GHG emissions (kg CO2e) for an arc using loaded params."""
    if dist_km <= 0 or time_hr <= 0 or avg_speed_kmh <= 0:
        return 0.0, 0.0

    # Get parameters from the loaded dictionary
    Q_PARAM = params['VEHICLE_CAPACITY']
    CFV_OMEGA = params['CFV_OMEGA']
    CFV_W_c = params['CFV_W_c']
    CFV_tau = params['CFV_tau']
    CFV_g_param = params['CFV_g_param']
    CFV_theta_rad = params['CFV_theta_rad']
    CFV_C_r = params['CFV_C_r']
    CFV_C_d = params['CFV_C_d']
    CFV_rho = params['CFV_rho']
    CFV_alpha = params['CFV_alpha']
    CFV_m_param = params['CFV_m_param']
    CFV_o_param = params['CFV_o_param']
    CFV_f_param = params['CFV_f_param']
    CFV_I_param = params['CFV_I_param']
    CFV_E_param = params['CFV_E_param']
    # CFV_xi = params['CFV_xi']
    # CFV_n_param = params['CFV_n_param'] # kJ/g
    # CFV_psi = params['CFV_psi'] # g/L

    # Convert units for physics calculations
    dist_m = dist_km * 1000.0
    time_s = time_hr * 3600.0
    avg_speed_mps = avg_speed_kmh / 3.6
    total_weight_kg = CFV_W_c + load_kg # W_T = W_c + current load

    # Using the placeholder model as the Appendix A derivation needs unit review/clarification
    base_fuel_per_100km = 30.0 # Litres per 100km (example for heavy truck - consider making this a param)
    load_factor_effect = 10.0 # Additional L/100km at full load (example - consider making this a param)

    fuel_rate_L_per_100km = base_fuel_per_100km + load_factor_effect * (load_kg / Q_PARAM if Q_PARAM > 0 else 0)
    fuel_consumed_L = (fuel_rate_L_per_100km / 100.0) * dist_km

    ghg_emitted_kg = fuel_consumed_L * CFV_OMEGA

    return fuel_consumed_L, ghg_emitted_kg

# --- Main Model Building ---

def build_and_solve_gvrp(params, current_beta_kg, current_upsilon_kg, current_subsidy):
    """Loads data, builds, and solves the GVRP model using loaded parameters."""

    # 1. Load Location/Demand Data
    if not all(check_file(f) for f in [DISTANCE_FILE, TIME_FILE, DEMAND_FILE]):
        return

    distances_df = pd.read_csv(DISTANCE_FILE, header=None)
    times_df = pd.read_csv(TIME_FILE, header=None)
    demand_df = pd.read_csv(DEMAND_FILE, header=None)

    num_nodes = len(distances_df)
    if num_nodes != DEPOT_NODE + 1:
         print(f"Warning: Depot node index {DEPOT_NODE} might be inconsistent with data size {num_nodes}.")

    customers = set(range(DEPOT_NODE))
    nodes = customers.union({DEPOT_NODE})

    # 2. Define Sets (Using defaults, consider making NUM_CFV/HGEV params)
    K_c = {f"CFV{i}" for i in range(NUM_CFV)}
    K_e = {f"HGEV{i}" for i in range(NUM_HGEV)}
    K = K_c.union(K_e)

    # 3. Prepare Parameters using the loaded 'params' dict
    A = {k: (params['CFV_ACQUISITION_COST'] if k in K_c else params['HGEV_ACQUISITION_COST']) for k in K}
    # Use the current subsidy value passed for sensitivity analysis
    S = {k: (0.0 if k in K_c else current_subsidy) for k in K}
    h = {k: params['HGEV_ENERGY_CONSUMPTION'] for k in K_e}
    Q = {k: params['VEHICLE_CAPACITY'] for k in K}
    R = {k: params['HGEV_BATTERY_CAPACITY'] for k in K_e}
    L = {k: params['HGEV_INITIAL_CHARGE_PERCENT'] for k in K_e} # Initial charge Lk
    HGEV_mu = params['HGEV_GRID_EMISSION_FACTOR'] # kg CO2e / kWh
    HGEV_CHARGE_COST_PER_KWH = params['HGEV_CHARGE_COST_PER_KWH']
    CARBON_CAP = params['CARBON_CAP']

    d = {(i, j): distances_df.iloc[i, j] for i in nodes for j in nodes if i != j}
    t = {(i, j): times_df.iloc[i, j] for i in nodes for j in nodes if i != j}
    p = {i: demand_df.iloc[i, 0] for i in customers}
    p[DEPOT_NODE] = 0 # Depot demand is 0

    # Calculate average speed
    v_avg = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                dist = d.get((i, j), 0)
                time = t.get((i, j), 0)
                if time > 1e-6: v_avg[(i, j)] = dist / time
                else: v_avg[(i, j)] = 0

    # Pre-calculate arc costs and emissions
    avg_load_proxy = Q[list(K)[0]] / 2.0 # Use capacity from params

    arc_cfv_fuel_L = {}
    arc_cfv_ghg_kg = {}
    arc_hgev_energy_kwh = {}
    arc_hgev_grid_ghg_kg = {}
    arc_cfv_fuel_cost = {}

    for i in nodes:
        for j in nodes:
            if i == j: continue
            dist = d.get((i, j), 0)
            time = t.get((i, j), 0)
            speed = v_avg.get((i, j), 0)

            # CFV calculations
            fuel_L, ghg_kg = calculate_cfv_fuel_ghg(dist, time, speed, avg_load_proxy, params)
            arc_cfv_fuel_L[(i, j)] = fuel_L
            arc_cfv_ghg_kg[(i, j)] = ghg_kg
            # Add fuel cost calculation (assuming cost per Litre is constant)
            # Consider making fuel cost a parameter in General.csv
            fuel_cost_per_L = 1.50 # Example, make this a param
            arc_cfv_fuel_cost[(i,j)] = fuel_L * fuel_cost_per_L


            # HGEV calculations
            energy_kwh = h[list(K_e)[0]] * dist # h is same for all HGEVs here
            grid_ghg_kg = energy_kwh * HGEV_mu
            arc_hgev_energy_kwh[(i, j)] = energy_kwh
            arc_hgev_grid_ghg_kg[(i, j)] = grid_ghg_kg

    # 4. Create Gurobi Model
    model = gp.Model("GVRP_BiObjective")

    # 5. Define Variables (same as before)
    x = model.addVars(K, nodes, nodes, vtype=GRB.BINARY, name="x") # x[k,i,j]
    u = model.addVars(K, nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="u") # u[k,i] - MTZ load variable
    y = model.addVars(K_e, nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="y") # y[k,i] - HGEV energy state
    used = model.addVars(K, vtype=GRB.BINARY, name="used") # used[k]
    e_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="e_plus")
    e_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="e_minus")

    # Remove arcs to self
    for k in K:
        for i in nodes:
            x[k, i, i].ub = 0

    # 6. Define Constraints (Mostly same as before, using loaded params)

    # --- Routing Constraints ---
    model.addConstrs((gp.quicksum(x[k, i, j] for k in K for i in nodes if i != j) == 1
                      for j in customers), name="AssignCust")
    model.addConstrs((gp.quicksum(x[k, DEPOT_NODE, j] for j in customers) == used[k]
                      for k in K), name="DepotStart")
    model.addConstrs((gp.quicksum(x[k, i, DEPOT_NODE] for i in customers) == used[k]
                      for k in K), name="DepotEnd")
    model.addConstrs((gp.quicksum(x[k, i, j] for i in nodes if i != j) -
                      gp.quicksum(x[k, j, i] for i in nodes if i != j) == 0
                      for k in K for j in customers), name="FlowCons")

    # --- Capacity and Subtour Elimination (MTZ) ---
    # u[k,i] = load delivered by vehicle k *up to and including* node i
    model.addConstrs((u[k, DEPOT_NODE] == 0 for k in K), name="MTZ_Depot_Load")
    for k in K:
        q_k = Q[k] # Capacity for vehicle k
        for i in nodes:
            # Load bounds at customer nodes
            if i in customers:
                model.addConstr(u[k, i] >= p[i], name=f"MTZ_LowerBound_{k}_{i}")
                model.addConstr(u[k, i] <= q_k, name=f"MTZ_UpperBound_{k}_{i}")

            # MTZ core constraint
            for j in customers: # Only need for customer nodes j
                if i != j:
                    model.addConstr(u[k, i] - u[k, j] + q_k * x[k, i, j] <= q_k - p[j],
                                    name=f"MTZ_{k}_{i}_{j}")


    # --- HGEV Energy Constraints ---
    for k in K_e:
        R_k = R[k] # Battery capacity for vehicle k
        L_k = L[k] # Initial charge fraction
        # Initial Charge (State on leaving depot)
        # We need state on *arrival* at depot for y[k, DEPOT_NODE]
        # Let's set the state *leaving* the depot implicitly via the first arc constraint
        # y[k, DEPOT_NODE] will represent state upon *arrival* back at depot

        # Energy Update & Bounds
        for i in nodes:
            # State on arrival at node i must be sufficient if used
            model.addConstr(y[k, i] >= HGEV_MIN_SOC_PERCENT * R_k * gp.quicksum(x[k,jj,i] for jj in nodes if jj!=i), name=f"HGEV_MinSOC_{k}_{i}")
            model.addConstr(y[k, i] <= HGEV_MAX_SOC_PERCENT * R_k, name=f"HGEV_MaxSOC_{k}_{i}")

            for j in nodes:
                if i != j:
                    # Energy level upon arrival at j depends on state at i
                    # If i is the depot, the state *leaving* depot is L_k * R_k
                    y_leaving_i = y[k, i] if i != DEPOT_NODE else L_k * R_k
                    
                    # Constraint using Big M to handle the initial charge case
                    M_energy = R_k # A suitable large number
                    model.addConstr(y[k, j] <= y_leaving_i - arc_hgev_energy_kwh.get((i,j),0) + M_energy * (1 - x[k, i, j]),
                                    name=f"HGEV_EnergyUpdate_{k}_{i}_{j}")


    # --- Carbon Cap Constraint ---
    total_daily_ghg = gp.LinExpr()
    for k in K_c:
        total_daily_ghg += gp.quicksum(arc_cfv_ghg_kg.get((i, j), 0) * x[k, i, j] for i in nodes for j in nodes if i != j)
    for k in K_e:
        total_daily_ghg += gp.quicksum(arc_hgev_grid_ghg_kg.get((i, j), 0) * x[k, i, j] for i in nodes for j in nodes if i != j)

    model.addConstr(total_daily_ghg * ANNUAL_OPERATING_DAYS <= CARBON_CAP + e_plus - e_minus, name="CarbonCap")

    # 7. Define Objective Functions (using current policy params)

    # --- Objective 1: Minimize Total Cost ---
    cost_acquisition = gp.quicksum((A[k] - S[k]) * used[k] for k in K) # Subsidy applied here

    cost_cfv_op = gp.quicksum(arc_cfv_fuel_cost.get((i, j), 0) * x[k, i, j]
                                for k in K_c for i in nodes for j in nodes if i != j)

    cost_hgev_op = gp.quicksum(arc_hgev_energy_kwh.get((i, j), 0) * HGEV_CHARGE_COST_PER_KWH * x[k, i, j]
                                   for k in K_e for i in nodes for j in nodes if i != j)

    cost_carbon_tax = gp.quicksum(arc_cfv_ghg_kg.get((i, j), 0) * current_beta_kg * x[k, i, j]
                                  for k in K_c for i in nodes for j in nodes if i != j) \
                    + gp.quicksum(arc_hgev_grid_ghg_kg.get((i, j), 0) * current_beta_kg * x[k, i, j]
                                  for k in K_e for i in nodes for j in nodes if i != j)

    cost_ets = current_upsilon_kg * (e_plus - e_minus) # Use current ETS price

    total_cost_objective = cost_acquisition + cost_cfv_op + cost_hgev_op + cost_carbon_tax + cost_ets

    # --- Objective 2: Minimize Total GHG Emissions ---
    total_ghg_objective = total_daily_ghg # Already calculated

    # 8. Set Objectives in Gurobi
    model.setObjectiveN(total_cost_objective, index=0, priority=1, weight=1.0, name="Total_Cost")
    model.setObjectiveN(total_ghg_objective, index=1, priority=0, weight=1.0, name="Total_GHG")
    model.ModelSense = GRB.MINIMIZE

    # 9. Optimize Model
    print("\n--- Running Scenario ---")
    print(f"Beta (Carbon Tax): {current_beta_kg*1000:.2f} $/tonne")
    print(f"Upsilon (ETS Price): {current_upsilon_kg*1000:.2f} $/tonne")
    print(f"HGEV Subsidy: {current_subsidy:.2f} $")
    print("Starting Gurobi optimization...")
    # model.setParam('TimeLimit', 300) # Optional: 5 minutes
    model.optimize()

    # 10. Process Results
    results = {
        "status": model.Status,
        "beta": current_beta_kg * 1000,
        "upsilon": current_upsilon_kg * 1000,
        "subsidy": current_subsidy,
        "cost_objective": None,
        "ghg_objective_daily": None,
        "ghg_objective_annual": None,
        "e_plus": None,
        "e_minus": None,
        "vehicles_used_count": 0,
        "cfv_used_count": 0,
        "hgev_used_count": 0,
        "total_dist": 0,
        "total_time": 0,
        "routes": {}
    }

    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        print(f"Solution Status: {model.Status} (Optimal or Suboptimal)")
        obj_cost = model.getObjective(0).getValue()
        obj_ghg = model.getObjective(1).getValue()
        results["cost_objective"] = obj_cost
        results["ghg_objective_daily"] = obj_ghg
        results["ghg_objective_annual"] = obj_ghg * ANNUAL_OPERATING_DAYS
        results["e_plus"] = e_plus.X
        results["e_minus"] = e_minus.X

        print(f"  - Cost Objective Value: {obj_cost:.2f}")
        print(f"  - GHG Objective Value: {obj_ghg:.2f} kg CO2e (Daily)")
        print(f"  - Annual GHG Emissions: {results['ghg_objective_annual']:.2f} kg CO2e")
        print(f"  - Allowances Purchased (e+): {e_plus.X:.2f} kg")
        print(f"  - Allowances Sold (e-): {e_minus.X:.2f} kg")

        # Reconstruct routes
        for k in sorted(list(K)):
            if used[k].X > 0.5:
                results["vehicles_used_count"] += 1
                route_dist = 0
                route_time = 0
                route_nodes_seq = [DEPOT_NODE]
                current_node = DEPOT_NODE
                vehicle_type = "CFV" if k in K_c else "HGEV"
                
                if vehicle_type == "CFV": results["cfv_used_count"] += 1
                else: results["hgev_used_count"] += 1

                # Loop to find path segments
                visited_in_route = {DEPOT_NODE}
                while True:
                    found_next = False
                    for j in nodes:
                        # Check if arc is used and destination not already in this path segment (basic loop prevention)
                        if j not in visited_in_route and x[k, current_node, j].X > 0.5:
                            dist_segment = d.get((current_node, j), 0)
                            time_segment = t.get((current_node, j), 0)
                            route_dist += dist_segment
                            route_time += time_segment
                            route_nodes_seq.append(j)
                            visited_in_route.add(j)
                            current_node = j
                            found_next = True
                            break
                    
                    # Break conditions
                    if not found_next: # No outgoing arc found
                         if current_node != DEPOT_NODE:
                             print(f"Warning: Route for {k} seems incomplete, stopping at {current_node}.")
                         break # Exit loop if stuck or returned to depot implicitly
                    if current_node == DEPOT_NODE: # Returned to depot explicitly
                        break

                if len(route_nodes_seq) > 2: # Valid route visiting customers
                    results["routes"][k] = {
                        "type": vehicle_type,
                        "sequence": route_nodes_seq,
                        "distance_km": route_dist,
                        "time_hr": route_time
                    }
                    results["total_dist"] += route_dist
                    results["total_time"] += route_time
                # else: # Vehicle marked used but no valid customer route found
                #     results["vehicles_used_count"] -= 1
                #     if vehicle_type == "CFV": results["cfv_used_count"] -= 1
                #     else: results["hgev_used_count"] -= 1


    elif model.Status == GRB.INFEASIBLE:
        print("Model is Infeasible. Computing IIS...")
        model.computeIIS()
        iis_file = "gvrp_infeasible.ilp"
        model.write(iis_file)
        print(f"IIS written to {iis_file}")
    else:
        print(f"Optimization ended with status code: {model.Status}")

    return results

# --- Run the Model ---
if __name__ == "__main__":
    print("Loading parameters from General.csv...")
    params = load_general_params(GENERAL_PARAMS_FILE)

    if params:
        print("Parameters loaded successfully.")
        # --- Run a single scenario using base parameters ---
        # Use base values from CSV for the first run
        beta_run = params.get('BETA_PER_KG', 0.0)
        upsilon_run = params.get('UPSILON_PER_KG', 0.0)
        subsidy_run = params.get('HGEV_SUBSIDY_BASE', 0.0)

        results = build_and_solve_gvrp(params, beta_run, upsilon_run, subsidy_run)

        # --- Print Results Summary ---
        if results:
             print("\n" + "="*30 + " RESULTS SUMMARY " + "="*30)
             print(f"Status: {results['status']}")
             print(f"Policy Scenario: Beta={results['beta']:.2f}, Upsilon={results['upsilon']:.2f}, Subsidy={results['subsidy']:.2f}")
             if results['status'] == GRB.OPTIMAL or results['status'] == GRB.SUBOPTIMAL:
                 print("\nObjectives:")
                 print(f"  Total Cost: {results['cost_objective']:.2f}")
                 print(f"  Daily GHG: {results['ghg_objective_daily']:.2f} kg CO2e")
                 print(f"  Annual GHG: {results['ghg_objective_annual']:.2f} kg CO2e")
                 print("\nETS:")
                 print(f"  Allowances Purchased: {results['e_plus']:.2f}")
                 print(f"  Allowances Sold: {results['e_minus']:.2f}")
                 print("\nFleet & Routing:")
                 print(f"  Vehicles Used: {results['vehicles_used_count']} ({results['cfv_used_count']} CFV, {results['hgev_used_count']} HGEV)")
                 print(f"  Total Distance: {results['total_dist']:.2f} km")
                 print(f"  Total Time: {results['total_time']:.2f} hr")
                 print("\nRoutes:")
                 if results['routes']:
                     for k, route_info in results['routes'].items():
                         print(f"  {k} ({route_info['type']}): Dist={route_info['distance_km']:.1f}km, Time={route_info['time_hr']:.2f}hr")
                         # print(f"    Sequence: {route_info['sequence']}") # Uncomment for full sequence
                 else:
                     print("  No routes involving customers were found.")
             print("="*77)

        # --- Placeholder for Sensitivity Analysis ---
        # To run sensitivity analysis, you would loop here,
        # changing beta_run, upsilon_run, subsidy_run based on
        # params['BETA_LOW'], params['BETA_HIGH'], etc.
        # and call build_and_solve_gvrp inside the loop, collecting results.
        print("\nSensitivity Analysis Placeholder:")
        print("Modify the script to loop through parameter ranges from General.csv")
        print(f"  Beta Range: [{params.get('BETA_LOW', 'N/A')}, {params.get('BETA_HIGH', 'N/A')}] $/tonne")
        print(f"  Upsilon Range: [{params.get('UPSILON_LOW', 'N/A')}, {params.get('UPSILON_HIGH', 'N/A')}] $/tonne")
        print(f"  Subsidy Range: [{params.get('HGEV_SUBSIDY_LOW', 'N/A')}, {params.get('HGEV_SUBSIDY_HIGH', 'N/A')}] $")

    else:
        print("Failed to load parameters. Exiting.")