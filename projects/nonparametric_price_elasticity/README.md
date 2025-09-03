---
title: Price Elasticity Simulation
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Price Elasticity Simulation

This Space demonstrates SKU-level demand, cross-price elasticity, and revenue optimization 
using linear, polynomial, and random forest models. Built with Gradio + Plotly.

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```

### Push your code to Hugging Face
From terminal, inside your project folder:

```bash
git init
git lfs install
git remote add origin https://huggingface.co/spaces/your-username/price-elasticity-sim
git add app.py requirements.txt README.md
git commit -m "Initial commit: Price Elasticity Simulator"
git push origin main
```