import gradio as gr
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ----- Hidden true demand function -----
TRUE_A = 50
TRUE_B = 2

OPT_PRICE = TRUE_A / (2 * TRUE_B)
OPT_REVENUE_PER_DAY = OPT_PRICE * (TRUE_A - TRUE_B * OPT_PRICE)

# Storage
history_prices = []
history_demands = []
history_revenues = []
history_regrets = []

def demand_function(price):
    demand = max(0, TRUE_A - TRUE_B * price + np.random.normal(0, 2))
    return int(round(demand))

def simulate_day(price):
    global history_prices, history_demands, history_revenues, history_regrets

    demand = demand_function(price)
    revenue = price * demand

    history_prices.append(price)
    history_demands.append(demand)
    history_revenues.append(revenue)

    daily_regret = OPT_REVENUE_PER_DAY - revenue
    history_regrets.append(daily_regret)

    # Fit regression line if enough data
    reg_line = None
    if len(history_prices) >= 2:
        X = np.array(history_prices).reshape(-1,1)
        y = np.array(history_demands)
        model = LinearRegression().fit(X, y)
        reg_line = (model.intercept_, model.coef_[0])

    # ----- Plot 1: Demand curve -----
    prices = np.linspace(0, 30, 100)
    true_demands = TRUE_A - TRUE_B * prices

    demand_fig = go.Figure()
    demand_fig.add_trace(go.Scatter(x=prices, y=true_demands, mode='lines', 
                                    line=dict(dash='dash', color='green'),
                                    name='True Demand',
                                    hovertemplate='Price: %{x}<br>Demand: %{y}<extra></extra>'))
    demand_fig.add_trace(go.Scatter(x=history_prices, y=history_demands, mode='markers+text', 
                                    marker=dict(color='blue', size=10),
                                    text=[f"${p}" for p in history_prices],
                                    textposition="top center",
                                    name='Observed Data',
                                    hovertemplate='Price: %{x}<br>Demand: %{y}<extra></extra>'))
    if reg_line:
        fitted = reg_line[0] + reg_line[1]*prices
        demand_fig.add_trace(go.Scatter(x=prices, y=fitted, mode='lines', 
                                        line=dict(color='red'), name='Estimated Demand',
                                        hovertemplate='Price: %{x}<br>Est. Demand: %{y}<extra></extra>'))
    demand_fig.update_layout(title="Demand Curve Learning",
                             xaxis_title="Price",
                             yaxis_title="Demand",
                             template='plotly_white')

    # ----- Plot 2: Cumulative revenue -----
    days = np.arange(1, len(history_revenues)+1)
    cum_revenue = np.cumsum(history_revenues)
    cum_opt = OPT_REVENUE_PER_DAY * days

    revenue_fig = go.Figure()
    revenue_fig.add_trace(go.Scatter(x=days, y=cum_revenue, mode='lines+markers', 
                                     name='Your cumulative revenue',
                                     line=dict(color='blue'),
                                     hovertemplate='Day %{x}<br>Revenue: %{y}<extra></extra>'))
    revenue_fig.add_trace(go.Scatter(x=days, y=cum_opt, mode='lines', 
                                     name='Optimal cumulative revenue',
                                     line=dict(color='green', dash='dash'),
                                     hovertemplate='Day %{x}<br>Optimal Revenue: %{y}<extra></extra>'))
    revenue_fig.update_layout(title="Cumulative Revenue vs Optimal",
                               xaxis_title="Day",
                               yaxis_title="Revenue",
                               template='plotly_white')

    # ----- Plot 3: Daily regret -----
    regret_fig = go.Figure()
    regret_fig.add_trace(go.Bar(x=days, y=history_regrets, marker_color='lightcoral', 
                                name='Daily Regret',
                                hovertemplate='Day %{x}<br>Regret: %{y}<extra></extra>'))
    regret_fig.update_layout(title="Daily Retrospective Regret",
                             xaxis_title="Day",
                             yaxis_title="Regret",
                             template='plotly_white')

    # ----- Explanatory text -----
    explanation = f"### Day {len(history_prices)} Results\n"
    explanation += f"- You set price **${price:.2f}**.\n"

    if len(history_prices) < 2:
        explanation += "- We need at least **2 price points** to start estimating the demand curve.\n"
        explanation += f"- Observed demand today: **{demand} units**.\n"
    else:
        explanation += f"- Observed demand: **{demand} units** → Revenue: **${revenue:.2f}**.\n"
        expected_demand = max(0, TRUE_A - TRUE_B*price)
        if demand < expected_demand - 3:
            explanation += "- Demand was **lower than expected** → noise or price too high.\n"
        elif demand > expected_demand + 3:
            explanation += "- Demand was **higher than expected** → noise or price lower than optimal.\n"
        else:
            explanation += "- Demand was close to expected given this price.\n"

        total_rev = np.sum(history_revenues)
        total_opt = OPT_REVENUE_PER_DAY * len(history_revenues)
        explanation += f"\n📊 **Cumulative revenue so far:** ${total_rev:.2f}\n"
        explanation += f"📈 **Optimal cumulative revenue:** ${total_opt:.2f}\n"
        explanation += f"❌ **Cumulative regret:** ${total_opt - total_rev:.2f}\n\n"

        if price < OPT_PRICE - 2:
            explanation += f"👉 Price is **too low** compared to optimal (~${OPT_PRICE:.2f}). Try raising it.\n"
        elif price > OPT_PRICE + 2:
            explanation += f"👉 Price is **too high** compared to optimal (~${OPT_PRICE:.2f}). Try lowering it.\n"
        else:
            explanation += f"✅ Price is **close to optimal** (~${OPT_PRICE:.2f}). Keep experimenting nearby.\n"

        explanation += "\n*Note: Regret is calculated retrospectively using the hidden true demand curve.*"

    return explanation, demand_fig, revenue_fig, regret_fig

def reset():
    global history_prices, history_demands, history_revenues, history_regrets
    history_prices, history_demands, history_revenues, history_regrets = [], [], [], []
    return "Game reset! Enter a price to start again.", None, None, None

# ----- Gradio UI -----
with gr.Blocks() as demo:
    gr.Markdown("## 🛒 Dynamic Pricing Simulation - Modern Interactive Plots\n"
                "Enter a price each day to see demand, revenue, regret, and recommendations in three interactive plots.")

    with gr.Row():
        price_in = gr.Number(label="Enter today's price", value=10)
        submit_btn = gr.Button("Simulate Day")
        reset_btn = gr.Button("Reset Simulation")

    demand_out = gr.Markdown()
    demand_fig_out = gr.Plot()
    revenue_fig_out = gr.Plot()
    regret_fig_out = gr.Plot()

    submit_btn.click(simulate_day, inputs=price_in, outputs=[demand_out, demand_fig_out, revenue_fig_out, regret_fig_out])
    reset_btn.click(reset, inputs=None, outputs=[demand_out, demand_fig_out, revenue_fig_out, regret_fig_out])

demo.launch()
