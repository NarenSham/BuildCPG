import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --- SKU setup ---
skus = ["Coke 500ml", "Coke 1L", "Pepsi 500ml"]
base_prices = np.array([1.50, 2.50, 1.40])
base_quantities = np.array([1000, 600, 900], dtype=float)
controllable = [True, True, False]

# Direct elasticities
demo_elasticities = np.array([-0.8, -0.6, -0.2])

# Cross-elasticities (how each SKU reacts to every other SKU)
cross_elasticities = np.array([
    [-0.8, -0.1, -0.05],
    [-0.1, -0.6, -0.05],
    [-0.05, -0.05, -0.2]
])

price_grid = np.linspace(0.5, 3.0, 50)

# --- Demand Generation ---
def generate_smooth_demand(sku_index, prices, model_type):
    base_qty = base_quantities[sku_index]
    base_price = base_prices[sku_index]

    # Compute cross-elasticity effect at each point in the price grid
    quantities = np.zeros_like(price_grid)
    for i, p_grid in enumerate(price_grid):
        qty = base_qty
        temp_prices = prices.copy()
        temp_prices[sku_index] = p_grid  # vary current SKU
        for j, price_j in enumerate(temp_prices):
            qty *= (price_j / base_prices[j]) ** cross_elasticities[sku_index, j]
        quantities[i] = qty

    # Optional smoothing for Polynomial / Random Forest
    if model_type in ["Polynomial", "Random Forest"]:
        X_sample = price_grid.reshape(-1, 1)
        y_sample = quantities
        if model_type == "Polynomial":
            pf = PolynomialFeatures(degree=3)
            model = LinearRegression()
            model.fit(pf.fit_transform(X_sample), y_sample)
            quantities = model.predict(pf.transform(X_sample))
        else:  # Random Forest
            rf = RandomForestRegressor(n_estimators=50)
            rf.fit(X_sample, y_sample)
            quantities = rf.predict(X_sample)

    # Interpolate to get quantity at the exact input price
    qty_at_price = np.interp(prices[sku_index], price_grid, quantities)
    return qty_at_price, quantities

# --- Simulation Function ---
def run_demo(price1, price2, price3, model_type):
    prices = np.array([price1, price2, price3])
    quantities = np.zeros_like(prices)
    elasticity_curves = {}

    # Compute current quantities and curves
    for i, sku in enumerate(skus):
        qty_at_price, curve = generate_smooth_demand(i, prices, model_type)
        quantities[i] = qty_at_price
        elasticity_curves[sku] = curve

    # --- Optimization: maximize total revenue including cross-elasticities ---
    optimized_prices = prices.copy()
    min_ratio, max_ratio = 0.8, 1.2  # keep prices reasonable

    for i, ctrl in enumerate(controllable):
        if ctrl:
            candidate_prices = price_grid[(price_grid >= base_prices[i]*min_ratio) & 
                                          (price_grid <= base_prices[i]*max_ratio)]
            total_revenues = []
            for p in candidate_prices:
                temp_prices = optimized_prices.copy()
                temp_prices[i] = p
                temp_qty = np.zeros_like(prices)
                for j, sku_j in enumerate(skus):
                    temp_qty[j], _ = generate_smooth_demand(j, temp_prices, model_type)
                total_revenues.append(np.sum(temp_qty * temp_prices))
            optimized_prices[i] = candidate_prices[np.argmax(total_revenues)]

    # Compute optimized quantities
    optimized_quantities = np.zeros_like(prices)
    for i, sku in enumerate(skus):
        qty_at_opt, _ = generate_smooth_demand(i, optimized_prices, model_type)
        optimized_quantities[i] = qty_at_opt

    # Revenues
    old_revenue = prices * quantities
    new_revenue = optimized_prices * optimized_quantities
    revenue_diff = new_revenue - old_revenue

    # --- Elasticity Curves Plot ---
    fig_elasticity = go.Figure()
    for i, sku in enumerate(skus):
        fig_elasticity.add_trace(go.Scatter(
            x=price_grid, y=elasticity_curves[sku],
            mode='lines',
            name=f"{sku} Demand Curve",
            hovertemplate=f"{sku}<br>Price: %{{x:.2f}}<br>Quantity: %{{y:.0f}}"
        ))
    fig_elasticity.update_layout(
        title="SKU Demand vs Price (Elasticity Curves)",
        xaxis_title="Price",
        yaxis_title="Quantity"
    )

    # --- Quantity vs Revenue Plot ---
    fig_quantity = go.Figure()
    fig_quantity.add_trace(go.Bar(
        x=skus, y=quantities, name="Previous Quantity",
        marker_color='#1f77b4',
        hovertemplate="%{x}<br>Qty: %{y:.0f}<br>Revenue: $%{customdata[0]:.0f}",
        customdata=np.stack([old_revenue], axis=1)
    ))
    fig_quantity.add_trace(go.Bar(
        x=skus, y=optimized_quantities, name="Optimized Quantity",
        marker_color='#ff7f0e',
        hovertemplate="%{x}<br>Qty: %{y:.0f}<br>Revenue: $%{customdata[0]:.0f}",
        customdata=np.stack([new_revenue], axis=1)
    ))
    fig_quantity.update_layout(
        barmode='group',
        title="Previous vs Optimized Quantity & Revenue",
        yaxis_title="Quantity"
    )

    # --- Price Comparison ---
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Bar(
        x=skus, y=prices, name="Current Prices",
        marker_color='#2C5F2D',
        hovertemplate="%{x}<br>Price: %{y:.2f}"
    ))
    fig_prices.add_trace(go.Bar(
        x=skus, y=optimized_prices, name="Optimized Prices",
        marker_color='#B85042',
        hovertemplate="%{x}<br>Optimized Price: %{y:.2f}"
    ))
    fig_prices.update_layout(
        barmode='group',
        title="Price Comparison"
    )

    # --- Comparison Table ---
    comparison_df = pd.DataFrame({
        "SKU": skus,
        "Old Price": np.round(prices, 2),
        "Old Qty": np.round(quantities, 0).astype(int),
        "Old Revenue": np.round(old_revenue, 0).astype(int),
        "New Price": np.round(optimized_prices, 2),
        "New Qty": np.round(optimized_quantities, 0).astype(int),
        "New Revenue": np.round(new_revenue, 0).astype(int),
        "Revenue Δ": np.round(revenue_diff, 0).astype(int)
    })

    # --- Description Text ---
    desc_text = "*Simulation Summary with Old vs Optimized Values*\n\n"
    for i, sku in enumerate(skus):
        change_pct = (optimized_prices[i]-prices[i])/prices[i]*100
        desc_text += f"{sku} | Price Change: {change_pct:+.1f}% | Qty: {quantities[i]:.0f} | Revenue: ${old_revenue[i]:.0f}\n"
    desc_text += "\nOptimized Quantities and Revenue:\n"
    for i, sku in enumerate(skus):
        desc_text += f"{sku}: Qty: {optimized_quantities[i]:.0f} | Revenue: ${new_revenue[i]:.0f}\n"
    desc_text += "\n*Optimization now accounts for cross-elasticities, so Pepsi volume responds to Coke price changes.*"

    return fig_elasticity, fig_quantity, fig_prices, comparison_df, desc_text

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Price Elasticity Simulator Demo")

    # Top controls
    with gr.Row():
        price1 = gr.Slider(0.5, 3.0, value=1.5, label="Coke 500ml Price", info="Adjust price to see demand effect")
        price2 = gr.Slider(0.5, 3.0, value=2.5, label="Coke 1L Price", info="Adjust price to see demand effect")
        price3 = gr.Slider(0.5, 3.0, value=1.4, label="Pepsi 500ml Price", info="Pepsi responds to Coke prices")
        elasticity_model = gr.Dropdown(["Linear", "Polynomial", "Random Forest"], value="Linear", label="Elasticity Model")

    run_button = gr.Button("Run Simulation")

    # Plots
    with gr.Row():
        elasticity_plot = gr.Plot(label="Elasticity Curves")
        quantity_plot = gr.Plot(label="Previous vs Optimized Quantity & Revenue")
        price_plot = gr.Plot(label="Price Comparison")

    comparison_table = gr.Dataframe(headers=["SKU","Old Price","Old Qty","Old Revenue","New Price","New Qty","New Revenue","Revenue Δ"], interactive=False)
    desc_box = gr.Textbox(label="Simulation Summary", interactive=False, lines=10)

    run_button.click(
        run_demo,
        inputs=[price1, price2, price3, elasticity_model],
        outputs=[elasticity_plot, quantity_plot, price_plot, comparison_table, desc_box]
    )

demo.launch()
