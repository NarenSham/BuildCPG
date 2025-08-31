import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def optimize_inventory(demand_mean, demand_std, unit_cost, shortage_cost, spoilage_cost):
    """
    Monte Carlo simulation to find optimal inventory order quantity.
    """
    # Generate simulated demand scenarios
    demands = np.random.normal(demand_mean, demand_std, 5000).astype(int)
    
    best_q = None
    best_profit = -1e9
    profits = []

    # Loop over possible order quantities
    for q in range(max(0, demand_mean-50), demand_mean+50):
        total_profit = 0
        for d in demands:
            sales = min(q, d)
            spoil = max(0, q-d)
            shortage = max(0, d-q)
            
            revenue = sales * unit_cost
            cost_spoil = spoil * spoilage_cost
            cost_short = shortage * shortage_cost
            total_profit += revenue - cost_spoil - cost_short
        
        avg_profit = total_profit / len(demands)
        profits.append((q, avg_profit))
        
        if avg_profit > best_profit:
            best_profit = avg_profit
            best_q = q

    # Prepare modern styled plot
    qs, ps = zip(*profits)
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8,5))
    plt.plot(qs, ps, color='blue', linewidth=2, label="Expected Profit")
    plt.axvline(best_q, color='red', linestyle='--', linewidth=2, label="Optimal Order")
    plt.scatter(best_q, best_profit, color='red', zorder=5)
    plt.text(best_q, best_profit*0.95, f"{best_q} units", color='red', fontsize=12, ha='center')
    plt.xlabel("Order Quantity", fontsize=14)
    plt.ylabel("Expected Profit", fontsize=14)
    plt.title("Profit vs Order Quantity", fontsize=16, weight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    return f"Optimal Order: {best_q} units\nExpected Profit: {round(best_profit,2)}", plt

# Build the interactive Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Fresh Inventory Optimization Demo")
    gr.Markdown(
        "Adjust the sliders to set demand, costs, and penalties. "
        "Click **Optimize** to see the recommended order quantity and profit curve."
    )
    
    with gr.Row():
        demand_mean = gr.Slider(50, 200, value=100, label="Mean Demand")
        demand_std = gr.Slider(1, 50, value=20, label="Demand Std Dev")
        unit_cost = gr.Slider(1, 20, value=5, label="Unit Price")
        shortage_cost = gr.Slider(1, 50, value=10, label="Shortage Penalty per Unit")
        spoilage_cost = gr.Slider(1, 20, value=5, label="Spoilage Cost per Unit")
    
    output_text = gr.Textbox(label="Result")
    output_plot = gr.Plot(label="Profit Curve")
    
    btn = gr.Button("Optimize")
    btn.click(
        optimize_inventory, 
        inputs=[demand_mean, demand_std, unit_cost, shortage_cost, spoilage_cost],
        outputs=[output_text, output_plot]
    )

# Launch the app
demo.launch()
