"""Resilient_Supply_Chain_Optimizer.ipynb"""
# ==========================================
# 1. SETUP & LIBRARIES
# ==========================================
import pulp
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

np.random.seed(42)

print("Libraries loaded successfully.")

# ==========================================
# 2. DATA GENERATION
# ==========================================

data = {
    "Supplier": ["S1_Germany", "S2_Vietnam", "S3_China", "S4_SKorea",
                 "S5_USA", "S6_UAE", "S7_Brazil", "S8_Indonesia",
                 "S9_Japan", "S10_Thailand"],
    "Country": ["Germany", "Vietnam", "China", "South Korea",
                "USA", "UAE", "Brazil", "Indonesia",
                "Japan", "Thailand"],
    "Unit_Cost": [12, 9, 8, 11, 13, 10, 9.5, 8.5, 12.5, 9],
    "Lead_Time_Days": [30, 18, 15, 20, 35, 12, 28, 16, 22, 17],
    "Capacity": [3000, 2000, 2500, 1800, 2200, 1500, 1700, 2000, 1600, 2100],
    "Distance_km": [6500, 3000, 4500, 5000, 12000, 2500, 14000, 4000, 6000, 3500],
    "Carbon_per_km": [0.02, 0.018, 0.021, 0.019, 0.025, 0.017, 0.026, 0.02, 0.022, 0.018],
    "Geopolitical_Risk": [2, 4, 6, 3, 3, 2, 5, 4, 2, 3],
    "Port_Risk": [3, 5, 7, 4, 4, 3, 6, 5, 3, 4],
    "Reliability_Rate": [0.95, 0.9, 0.85, 0.92, 0.93, 0.96, 0.88, 0.89, 0.94, 0.91]
}

df = pd.DataFrame(data)

# Feature Engineering
df["Carbon_Impact_Per_Unit"] = df["Distance_km"] * df["Carbon_per_km"]
df["Composite_Risk"] = (0.6 * df["Geopolitical_Risk"] + 0.4 * df["Port_Risk"])
df["Risk_Adjusted_Cost"] = df["Unit_Cost"] * (1 + df["Composite_Risk"]/10)

# Define Warehouses and Demand
warehouses = ["W1_North", "W2_South", "W3_East", "W4_West"]
warehouse_demand = {w: 1000 for w in warehouses} # Total demand 4000

# Create Global Lookup Dictionaries
suppliers = df["Supplier"].tolist()
supplier_capacity_dict = df.set_index("Supplier")["Capacity"].to_dict()

# Simulate Cost, Risk, and Carbon Matrices
cost_matrix = pd.DataFrame(
    np.random.rand(len(suppliers), len(warehouses)) * 5 + 5,
    index=suppliers, columns=warehouses
)

risk_matrix = pd.DataFrame(
    np.random.rand(len(suppliers), len(warehouses)) * 3 + 1,
    index=suppliers, columns=warehouses
)

carbon_matrix = pd.DataFrame(
    np.random.rand(len(suppliers), len(warehouses)) * 0.1 + 0.05,
    index=suppliers, columns=warehouses
)

print("Data generation complete.")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

fig1 = px.scatter(
    df, x="Unit_Cost", y="Lead_Time_Days", size="Composite_Risk", color="Composite_Risk",
    hover_name="Supplier", title="Supplier Analysis: Cost vs Speed (Size = Risk)",
    color_continuous_scale="RdYlGn_r"
)
fig1.show()

# ==========================================
# 4. THE OPTIMIZATION ENGINE
# ==========================================

def solve_supply_chain(w_cost, w_risk, w_carbon, custom_capacity=None, custom_risk_matrix=None):
    """
    Solves the LP problem.
    Inputs:
        custom_capacity: Dictionary to override default capacities (for stress testing)
        custom_risk_matrix: DataFrame to override default risk (for Monte Carlo)
    """
    # Use default data if no custom overrides are provided
    current_capacity = custom_capacity if custom_capacity else supplier_capacity_dict
    current_risk_matrix = custom_risk_matrix if custom_risk_matrix is not None else risk_matrix

    model = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)

    shipment = pulp.LpVariable.dicts(
        "Ship", [(i, j) for i in suppliers for j in warehouses],
        lowBound=0, cat='Continuous'
    )

    # Objective Function
    model += pulp.lpSum([
        shipment[(i,j)] * (
            w_cost * cost_matrix.loc[i,j] +
            w_risk * current_risk_matrix.loc[i,j] +
            w_carbon * carbon_matrix.loc[i,j]
        )
        for i in suppliers for j in warehouses
    ])

    # Constraints
    for i in suppliers:
        model += pulp.lpSum([shipment[(i,j)] for j in warehouses]) <= current_capacity[i]

    for j in warehouses:
        model += pulp.lpSum([shipment[(i,j)] for i in suppliers]) == warehouse_demand[j]

    # Risk Diversification (Max 60%)
    for i in suppliers:
        for j in warehouses:
            model += shipment[(i,j)] <= 0.6 * warehouse_demand[j]

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[model.status] != 'Optimal':
        return None, None, None

    total_cost = sum(shipment[(i,j)].varValue * cost_matrix.loc[i,j] for i in suppliers for j in warehouses)
    total_risk = sum(shipment[(i,j)].varValue * current_risk_matrix.loc[i,j] for i in suppliers for j in warehouses)
    total_carbon = sum(shipment[(i,j)].varValue * carbon_matrix.loc[i,j] for i in suppliers for j in warehouses)

    return total_cost, total_risk, total_carbon

print("Optimization Engine Ready.")

# ==========================================
# 5. SCENARIO ANALYSIS
# ==========================================

scenarios = {
    "Cost Focused": (0.8, 0.1, 0.1),
    "Risk Focused": (0.3, 0.6, 0.1),
    "Green Focused": (0.3, 0.1, 0.6)
}

results = {}
for name, (wc, wr, wcarb) in scenarios.items():
    results[name] = solve_supply_chain(wc, wr, wcarb)

results_df = pd.DataFrame(results, index=["Total Cost", "Total Risk", "Total Carbon"]).T
normalized_df = (results_df - results_df.min()) / (results_df.max() - results_df.min())

normalized_df.plot(kind='bar', figsize=(10, 6))
plt.title("Scenario Trade-offs (Normalized 0-1)")
plt.xticks(rotation=0)
plt.show()

# ==========================================
# 6. MONTE CARLO SIMULATION (FIXED)
# ==========================================

def monte_carlo_simulation(w_cost, w_risk, w_carbon, simulations=100):
    """
    Runs simulations with fluctuating risk levels.
    RETURNS: The list of all simulated costs.
    """
    simulated_costs_list = []  # Rename to avoid confusion
    
    for _ in range(simulations):
        # Create a randomized risk matrix based on the original
        noise = np.random.normal(0, 0.5, risk_matrix.shape)
        simulated_risk = risk_matrix + noise
        simulated_risk = simulated_risk.clip(lower=0) # Ensure no negative risk
        
        # Pass this NEW matrix to the solver
        result = solve_supply_chain(w_cost, w_risk, w_carbon, custom_risk_matrix=simulated_risk)
        
        if result[0] is not None:
            simulated_costs_list.append(result[0])

    return simulated_costs_list

print("\n--- Running Monte Carlo Simulation ---")
# Run 100 simulations and CAPTURE the result
all_costs = monte_carlo_simulation(0.5, 0.3, 0.2, simulations=100)

# Calculate Stats
avg_cost = np.mean(all_costs)
cost_std = np.std(all_costs)

print(f"Average Cost: ${avg_cost:,.2f}")
print(f"Volatility (Std Dev): ${cost_std:,.2f}")

# Plot Histogram using the CAPTURED variable 'all_costs'
plt.figure(figsize=(8,5))
plt.hist(all_costs, bins=20, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Simulation: Distribution of Total Costs")
plt.xlabel("Total Cost")
plt.ylabel("Frequency")
plt.axvline(avg_cost, color='red', linestyle='dashed', linewidth=1, label=f'Mean: ${avg_cost:,.0f}')
plt.legend()
plt.show()

# ==========================================
# 7. RESILIENCE STRESS TESTING
# ==========================================
print("\n--- Running Stress Test: 'Vietnam Lockdown' ---")

shocked_capacity = supplier_capacity_dict.copy()
shocked_capacity["S2_Vietnam"] *= 0.3  # Reduce capacity by 70%

normal_res = solve_supply_chain(0.5, 0.3, 0.2)
shock_res = solve_supply_chain(0.5, 0.3, 0.2, custom_capacity=shocked_capacity)

cost_increase = ((shock_res[0] - normal_res[0]) / normal_res[0]) * 100
resilience_score = 1 / (1 + (cost_increase/100))

print(f"Normal Cost: ${normal_res[0]:,.2f}")
print(f"Shock Cost:  ${shock_res[0]:,.2f}")
print(f"Cost Surge:  {cost_increase:.2f}%")
print(f"Resilience Score: {resilience_score:.4f}")

# ==========================================
# 8. PARETO OPTIMIZATION
# ==========================================

cost_weights = np.linspace(0, 1, 20)
pareto_points = []

for wc in cost_weights:
    remaining = 1 - wc
    wr = remaining / 2
    wcarb = remaining / 2
    res = solve_supply_chain(wc, wr, wcarb)
    if res[0] is not None:
        pareto_points.append(res)

costs = np.array([p[0] for p in pareto_points])
risks = np.array([p[1] for p in pareto_points])

# Knee Point Detection
c_norm = (costs - costs.min()) / (costs.max() - costs.min())
r_norm = (risks - risks.min()) / (risks.max() - risks.min())
distances = np.sqrt(c_norm**2 + r_norm**2)
knee_idx = np.argmin(distances)

plt.figure(figsize=(10, 6))
plt.plot(costs, risks, marker='o', label='Pareto Frontier')
plt.scatter(costs[knee_idx], risks[knee_idx], color='red', s=150, label='Optimal Knee Point', zorder=5)
plt.title("Pareto Frontier: Cost vs. Risk")
plt.xlabel("Total Cost ($)")
plt.ylabel("Tot`al Risk Score")
plt.legend()
plt.show()

print(f"Optimal Strategy Found at Cost: ${costs[knee_idx]:,.2f}")
