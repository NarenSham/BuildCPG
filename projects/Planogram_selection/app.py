import gradio as gr
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# --- SKU setup ---
skus = ["Coke 500ml", "Coke 1L", "Pepsi 500ml", "Fanta 500ml", "Sprite 500ml", "Mtn Dew 500ml"]
base_prices = np.array([1.50, 2.50, 1.40, 1.30, 1.20, 1.60])
base_quantities = np.array([1000, 600, 900, 700, 500, 400], dtype=float)
demo_elasticities = np.array([-0.8, -0.6, -0.2, -0.7, -0.5, -0.6])
cross_elasticities = np.array([
    [-0.8, -0.1, -0.05, -0.02, -0.02, -0.01],
    [-0.1, -0.6, -0.05, -0.02, -0.02, -0.01],
    [-0.05, -0.05, -0.2, -0.02, -0.01, -0.01],
    [-0.02, -0.02, -0.01, -0.7, -0.02, -0.01],
    [-0.02, -0.02, -0.01, -0.02, -0.5, -0.01],
    [-0.01, -0.01, -0.01, -0.01, -0.01, -0.6]
])
price_grid = np.linspace(0.5, 3.0, 50)

# Map SKU to emoji
sku_emojis = {
    "Coke 500ml": "🍹",
    "Coke 1L": "🧋",
    "Pepsi 500ml": "🥤",
    "Fanta 500ml": "🍊",
    "Sprite 500ml": "🍋",
    "Mtn Dew 500ml": "🧃"
}

# --- Demand Generation ---
def generate_smooth_demand(prices):
    quantities = np.zeros_like(prices)
    for i in range(len(skus)):
        qty = base_quantities[i]
        for j, p in enumerate(prices):
            qty *= (p / base_prices[j]) ** cross_elasticities[i, j]
        quantities[i] = qty
    return quantities

# --- Optimization ---
def optimize_prices(prices, controllable_flags):
    optimized_prices = prices.copy()
    for i, ctrl in enumerate(controllable_flags):
        if ctrl and demo_elasticities[i] < -0.5:
            optimized_prices[i] *= 1.1  # Simple demo logic
    return optimized_prices

# --- Simulation Function ---
def run_simulation(*inputs):
    prices = np.array(inputs[:len(skus)])
    controllable_flags = np.array(inputs[len(skus):])
    
    quantities = generate_smooth_demand(prices)
    optimized_prices = optimize_prices(prices, controllable_flags)
    optimized_quantities = generate_smooth_demand(optimized_prices)
    revenue = prices * quantities
    optimized_revenue = optimized_prices * optimized_quantities
    revenue_diff = optimized_revenue - revenue

    # --- Demand Curves ---
    fig_demand = go.Figure()
    for i, sku in enumerate(skus):
        qty_curve = base_quantities[i] * (price_grid / base_prices[i]) ** demo_elasticities[i]
        fig_demand.add_trace(go.Scatter(
            x=price_grid, y=qty_curve,
            mode='lines',
            name=sku,
            hovertemplate=f"{sku}<br>Price: %{{x:.2f}}<br>Quantity: %{{y:.0f}}<extra></extra>"
        ))
    fig_demand.update_layout(title="SKU Demand Curves", xaxis_title="Price", yaxis_title="Quantity")

    # --- Top 3 SKUs Selection ---
    top_indices = np.argsort(optimized_revenue)[-3:]  # Select top 3 by revenue
    selected_mask = np.zeros(len(skus), dtype=bool)
    selected_mask[top_indices] = True

    # --- Optimized Planogram ---
    fig_shelf = go.Figure()
    shelf_width = 1
    spacing = 0.5
    x_offset = 0
    y_level = 0
    for i, sku in enumerate(skus):
        # Highlight selected SKUs with green outline (no fill)
        if selected_mask[i]:
            fig_shelf.add_shape(
                type="rect",
                x0=x_offset - 0.35, x1=x_offset + 0.35,
                y0=y_level - 0.35, y1=y_level + 0.35,
                line=dict(color='green', width=3),
                fillcolor='rgba(0,0,0,0)'
            )
        # Add emoji
        fig_shelf.add_trace(go.Scatter(
            x=[x_offset],
            y=[y_level],
            text=[sku_emojis[sku]],
            mode="text",
            textfont=dict(size=50),
            showlegend=False,
            hovertemplate=f"{sku}<br>Price: ${prices[i]:.2f}<br>Qty: {int(quantities[i])}<br>Revenue: ${int(revenue[i])}<extra></extra>"
        ))
        x_offset += shelf_width + spacing

    fig_shelf.update_layout(
        title="Optimized Planogram (Top 3 SKUs Highlighted in Green Outline)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=250
    )

    # --- Demand Table ---
    table_df = pd.DataFrame({
        "SKU": skus,
        "Price": np.round(prices, 2),
        "Quantity": np.round(quantities, 0).astype(int),
        "Revenue": np.round(revenue, 0).astype(int),
        "Optimized Price": np.round(optimized_prices, 2),
        "Optimized Quantity": np.round(optimized_quantities, 0).astype(int),
        "Optimized Revenue": np.round(optimized_revenue, 0).astype(int),
        "Revenue Δ": np.round(revenue_diff, 0).astype(int)
    })

    # --- Summary ---
    summary = "Simulation Summary:\n"
    summary += f"Top 3 SKUs by revenue: {', '.join([skus[i] for i in top_indices])}.\n"
    summary += "Prices and volumes were optimized to maximize revenue given SKU elasticity.\n"
    summary += "Optimized prices differ from your input because highly elastic, controllable SKUs were adjusted to improve overall revenue.\n"
    summary += "Revenue Δ column shows the change in revenue after optimization: positive means revenue increased, negative means it decreased."

    return fig_demand, fig_shelf, table_df, summary

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Planogram & Pricing Simulator")
    gr.Markdown(
        "Select prices for each SKU and check whether each SKU is **controllable** (price can be optimized). "
        "Only space for **3 SKUs** in the shelf. Watch how the optimized planogram changes based on demand and revenue."
    )

    with gr.Row():
        sliders = [gr.Slider(0.5, 3.0, value=base_prices[i], label=f"{sku} Price") for i, sku in enumerate(skus)]
    
    with gr.Row():
        checkboxes = [gr.Checkbox(label=f"{sku} Controllable", value=True) for sku in skus]

    run_button = gr.Button("Run Simulation")

    with gr.Row():
        demand_plot = gr.Plot(label="Demand Curves")
        planogram_plot = gr.Plot(label="Optimized Planogram")

    demand_table = gr.Dataframe(
        headers=["SKU","Price","Quantity","Revenue","Optimized Price","Optimized Quantity","Optimized Revenue","Revenue Δ"], 
        interactive=False
    )
    summary_box = gr.Textbox(label="Simulation Summary", interactive=False, lines=6)

    run_button.click(
        run_simulation,
        inputs=sliders + checkboxes,
        outputs=[demand_plot, planogram_plot, demand_table, summary_box]
    )

demo.launch()
