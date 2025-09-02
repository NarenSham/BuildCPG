---
title: Inventory Simulation: Traditional vs Connected AI
emoji: 📦
colorFrom: "blue"
colorTo: "green"
sdk: gradio
sdk_version: "3.50.0"
app_file: app.py
pinned: true
---

# Inventory Simulation: Traditional vs Connected AI

This interactive app demonstrates how traditional inventory ordering compares to a connected AI approach (network-aware) under variable demand scenarios.  

**Features:**

- Adjust key parameters such as:
  - Initial Inventory
  - Safety Stock Threshold
  - Maximum Order per Day
  - Expected Daily Demand
  - Holding, Shortage, and Spoilage Costs
- Simulate both **Traditional** and **AI-based Connected** ordering policies.
- Dynamic **Plotly visualizations** with hover info for:
  - Inventory levels
  - Orders placed
  - Daily demand
- Detailed **simulation summary** highlighting total costs, shortages, and AI benefits.

**How it works:**

1. **Traditional Policy:**  
   Orders are triggered if inventory falls below safety stock, without network-wide insight.
   
2. **Connected AI Policy:**  
   Smooths orders over time using recent demand history and simulated network visibility, reducing extreme shortages and overstock situations.

**Use Cases:**

- Supply chain and inventory optimization
- Commercial operations planning
- Demonstrating AI-enhanced decision-making in volatile environments

**Instructions:**

1. Adjust input parameters on the left panel.
2. Click **Simulate**.
3. Explore the interactive Plotly graphs and insights.
4. Hover over data points for daily breakdowns of orders, inventory, and demand.

**Learn more:**  
This simulation is inspired by recent research on **Scenario Predict then Optimize (SPO)** and **AI-enabled networked inventory optimization**.
