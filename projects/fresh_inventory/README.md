# Fresh Inventory Optimization Demo

This demo simulates a simple inventory optimization problem using Monte Carlo demand simulations.

## How It Works

- You can adjust the following inputs:
  - **Mean Demand**: expected average demand per period.
  - **Demand Std Dev**: variability in demand.
  - **Unit Price**: revenue per sold unit.
  - **Shortage Penalty per Unit**: cost for not meeting demand.
  - **Spoilage Cost per Unit**: cost for unsold inventory.

- When you click **Optimize**, the app:
  1. Simulates thousands of demand scenarios.
  2. Computes expected profit for each possible order quantity.
  3. Finds the **order quantity that maximizes expected profit**.
  4. Plots the **profit curve** with the optimal order highlighted.

## Features

- Interactive sliders for all input parameters.
- Modern, clean visualization of profit vs. order quantity.
- Highlights optimal order quantity and expected profit.
- Ready to deploy on Hugging Face Spaces using Gradio.

## How to Run Locally

1. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # Mac/Linux
env\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```
### **Steps to Deploy on Hugging Face Spaces**

1. Go to: [https://huggingface.co/spaces](https://huggingface.co/spaces)  
2. Click **“Create new Space”**  
   - Name it `fresh-inventory-demo` (or anything you like)  
   - Choose **Gradio** as SDK  
3. Upload the three files: `app.py`, `requirements.txt`, `README.md`  
4. Commit changes → Hugging Face automatically deploys the app.  
5. Share the public link with anyone.
