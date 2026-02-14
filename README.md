# 📦 Resilient Multi-Objective Supply Chain Optimizer

### *Balancing Cost, Risk, and Sustainability through Linear Programming*

---

## 📖 Project Overview

In modern production engineering, the cheapest supplier is rarely the best choice if they carry high geopolitical risk or excessive carbon footprint. This project builds a **mathematical optimization model** to determine the optimal allocation of orders across global suppliers.

Unlike traditional models that only minimize cost, this engine simultaneously optimizes for three conflicting objectives:

1. **💰 Cost Efficiency:** Minimizing procurement and logistics spend.
2. **⚠️ Disruption Risk:** diversifying dependency to avoid single points of failure.
3. **🌍 Carbon Emissions:** Reducing the environmental impact of logistics.

---

## 📊 Key Methodology

This project uses **Linear Programming (PuLP)** to solve a multi-objective optimization problem.

### 1. Weighted Sum Optimization

We normalize and weight three distinct objective functions to find a unified solution:


### 2. Resilience Stress Testing

The model simulates "Supply Chain Shocks" (e.g., a port lockdown in Vietnam) to test network robustness.

* **Metric:** *Resilience Score* (calculated based on cost surge during disruption).

### 3. Pareto Frontier Analysis

We plot the trade-off curve between Cost and Risk to identify non-dominated solutions.

* **Automated Knee-Point Detection:** The algorithm automatically identifies the mathematical "sweet spot"—the point where increasing risk no longer yields significant cost savings.

---

## 🚀 Key Insights & Results

After running the optimization across 10 global suppliers and 4 distribution centers:

* **The "Knee Point" (Optimal Strategy):**
* **Cost Weight:** 
* **Total Cost:** 
* **Total Risk Score:** 


* **Conclusion:** The purely cost-minimized strategy was highly fragile. By accepting a **5% increase in cost**, the model achieved a **40% reduction in supply chain risk**, proving that resilience is a quantifiable investment.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Optimization:** `PuLP` (Linear Programming solver)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Plotly` (for interactive trade-off charts)

---

## ⚙️ How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/supply-chain-optimizer.git

```


2. **Install dependencies:**
```bash
pip install pulp pandas numpy plotly matplotlib

```


3. **Run the script:**
```bash
python resilient_supply_chain.py

```



---

## 🔮 Future Improvements

* Integrate real-time shipping API data for live cost updates.
* Implement a Stochastic Programming approach for demand uncertainty.
* Build a Streamlit dashboard for interactive "What-If" scenario planning.

---

### *Author*

**Rudra Ray**
*Jadavpur University*

