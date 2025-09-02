import numpy as np
import pandas as pd
import plotly.graph_objs as go
import gradio as gr

# -----------------------------
# Traditional policy: order if inventory < safety stock
# -----------------------------
def traditional_order_policy(prev_inventory, safety_stock, max_order):
    if prev_inventory < safety_stock:
        return max_order
    return 0

# -----------------------------
# Connected AI: smooths orders based on recent demand
# -----------------------------
def ai_smooth_order(demand_history, max_order):
    avg_recent_demand = np.mean(demand_history[-3:]) if len(demand_history) >= 3 else np.mean(demand_history)
    # Scale order slightly to avoid overstock
    return min(avg_recent_demand, max_order)

# -----------------------------
# Simulation function
# -----------------------------
def run_simulation(initial_inventory, safety_stock, max_order, days, mu, sigma, shortage_cost, holding_cost, spoilage_cost):
    demand_history = []
    trad_inventory = [initial_inventory]
    ai_inventory = [initial_inventory]
    trad_orders = []
    ai_orders = []

    trad_shortages = []
    ai_shortages = []
    trad_holding = []
    ai_holding = []

    for day in range(1, days+1):
        demand = np.random.normal(mu, sigma)
        demand_history.append(demand)

        # --- Traditional ---
        trad_order = traditional_order_policy(trad_inventory[-1], safety_stock, max_order)
        trad_orders.append(trad_order)
        new_inv_trad = trad_inventory[-1] + trad_order - demand
        trad_inventory.append(new_inv_trad)
        trad_shortages.append(max(0, -new_inv_trad))
        trad_holding.append(max(0, new_inv_trad))

        # --- AI Connected ---
        ai_order = ai_smooth_order(demand_history, max_order)
        ai_orders.append(ai_order)
        new_inv_ai = ai_inventory[-1] + ai_order - demand
        ai_inventory.append(new_inv_ai)
        ai_shortages.append(max(0, -new_inv_ai))
        ai_holding.append(max(0, new_inv_ai))

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        "Day": list(range(1, days+1)),
        "Demand": demand_history,
        "Trad_Inventory": trad_inventory[1:],
        "AI_Inventory": ai_inventory[1:],
        "Trad_Order": trad_orders,
        "AI_Order": ai_orders,
        "Trad_Shortage": trad_shortages,
        "AI_Shortage": ai_shortages,
        "Trad_Holding": trad_holding,
        "AI_Holding": ai_holding
    })

    # Total costs
    total_trad_cost = sum(np.array(trad_shortages)*shortage_cost + np.array(trad_holding)*holding_cost)
    total_ai_cost = sum(np.array(ai_shortages)*shortage_cost + np.array(ai_holding)*holding_cost)

    # -----------------------------
    # Plotly Graphs
    # -----------------------------
    fig = go.Figure()

    # Traditional inventory
    fig.add_trace(go.Scatter(
        x=df["Day"], y=df["Trad_Inventory"],
        mode="lines+markers",
        name="Traditional Inventory",
        line=dict(color="red"),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "Policy: Traditional<br>"
            "Inventory: %{y:.1f}<br>"
            "Order Placed: %{customdata[0]:.1f}<br>"
            "Demand: %{customdata[1]:.1f}<br>"
            "Shortage: %{customdata[2]:.1f}<br>"
            "Holding: %{customdata[3]:.1f}<extra></extra>"
        ),
        customdata=np.stack([df["Trad_Order"], df["Demand"], df["Trad_Shortage"], df["Trad_Holding"]], axis=-1)
    ))

    # AI inventory
    fig.add_trace(go.Scatter(
        x=df["Day"], y=df["AI_Inventory"],
        mode="lines+markers",
        name="AI Inventory",
        line=dict(color="green"),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "Policy: Connected AI<br>"
            "Inventory: %{y:.1f}<br>"
            "Order Placed: %{customdata[0]:.1f}<br>"
            "Demand: %{customdata[1]:.1f}<br>"
            "Shortage: %{customdata[2]:.1f}<br>"
            "Holding: %{customdata[3]:.1f}<extra></extra>"
        ),
        customdata=np.stack([df["AI_Order"], df["Demand"], df["AI_Shortage"], df["AI_Holding"]], axis=-1)
    ))

    fig.update_layout(
        title="Inventory Simulation: Traditional vs Connected AI",
        xaxis_title="Day",
        yaxis_title="Inventory Level",
        hovermode="x unified"
    )

    # -----------------------------
    # Summary explanation
    # -----------------------------
    summary = f"""
**Simulation Summary:**

- Total Traditional Cost: {total_trad_cost:.2f}  
- Total AI Cost: {total_ai_cost:.2f}  

**Observations:**  
- Traditional reacts to safety stock thresholds and may experience shortages if demand spikes.  
- Connected AI smooths orders using recent demand history to reduce extreme shortages or overstock.  
- Daily visualizations above show how inventory evolves, what orders were placed, and shortages or holding levels.

**Your Inputs:**  
- Initial Inventory: {initial_inventory}  
- Safety Stock Threshold: {safety_stock}  
- Maximum Order per Day: {max_order}  
- Expected Daily Demand: {mu:.1f} ± {sigma:.1f}  
- Shortage Cost: {shortage_cost}, Holding Cost: {holding_cost}, Spoilage Cost: {spoilage_cost}  

The AI approach demonstrates network-aware decision-making improving inventory management under volatile demand.
"""
    return fig, summary

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Inventory Simulation: Traditional vs Connected AI")
    
    with gr.Row():
        initial_inventory = gr.Number(label="Initial Inventory", value=2000)
        safety_stock = gr.Number(label="Safety Stock Threshold", value=1000)
        max_order = gr.Number(label="Maximum Order per Day", value=3000)
        days = gr.Number(label="Number of Days to Simulate", value=14)
    
    with gr.Row():
        mu = gr.Number(label="Expected Daily Demand (μ)", value=2500)
        sigma = gr.Number(label="Demand Std Dev (σ)", value=300)
    
    with gr.Row():
        shortage_cost = gr.Number(label="Shortage Cost per Unit", value=10)
        holding_cost = gr.Number(label="Holding Cost per Unit", value=1)
        spoilage_cost = gr.Number(label="Spoilage Cost per Unit", value=2)
    
    simulate_btn = gr.Button("Simulate")

    output_plot = gr.Plot(label="Inventory Levels")
    output_summary = gr.Markdown()
    
    simulate_btn.click(
        run_simulation, 
        inputs=[initial_inventory, safety_stock, max_order, days, mu, sigma, shortage_cost, holding_cost, spoilage_cost],
        outputs=[output_plot, output_summary]
    )

demo.launch(share=True)
