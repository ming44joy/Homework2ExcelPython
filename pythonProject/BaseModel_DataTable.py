# Analysis 1 - Basic Break Even Analysis

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.odr import Model
from whatif import Model, get_sim_results_df



#order_cost = unit_cost * order_quantity
#sales_revenue = np.minimum(order_quantity, demand) * selling_price
#refund_revenue = np.maximum(0, order_quantity - demand) * unit_refund
#profit = sales_revenue + refund_revenue - order_cost


# D = 0.06S^2 - 35S + 4900
# D = monthly demand
# S = selling price SPF

# 1a Base Model
# Check - Demand 1668, Profit =$20028



class SingleProductSPF():
    """Base model"""
    def __init__(self, fixed_cost=5000, variable_cost_unit=100, selling_price=115,
                 spf_constant=4900, spf_linear=-35, spf_quadratic=0.06):
        self.fixed_cost = fixed_cost
        self.variable_cost_unit = variable_cost_unit
        self.selling_price = selling_price
        self.spf_constant = spf_constant
        self.spf_linear = spf_linear
        self.spf_quadratic = spf_quadratic

    def demand(self, selling_price):
        """Compute Demand"""
        demand = self.spf_quadratic * selling_price **2 + self.spf_linear * selling_price + self.spf_constant
        return demand

    def profit(self, selling_price):
        demand = self.demand(selling_price)
        revenue = self.selling_price * demand
        total_cost = self.fixed_cost + (demand * self.variable_cost_unit)
        profit = revenue - total_cost
        return profit



# One-way data table

    def data_table(self, selling_price_range):
        """ Create Data table"""
        results = {
            'Selling_price': [],
            'Demand': [],
            'Profit': []
        }

        for price in selling_price_range:
            demand = self.demand(price)
            profit = self.profit(price)
            results['Selling_price'].append(price)
            results['Demand'].append(demand)
            results['Profit'].append(profit)

        return results

    def plot_data_table(self, data):
        """Plot Data table"""
        fig, ax1 = plt.subplots()

        color = 'tab:pink'
        ax1.set_xlabel('Selling Price ($)')
        ax1.set_ylabel('Profit ($)', color=color)
        ax1.plot(data['Selling_price'], data['Profit'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:purple'
        ax2.set_ylabel('Demand (units)', color=color)
        ax2.plot(data['Selling_price'], data['Demand'], color=color, marker='x')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Relationship between Selling Price, Profit, and Demand')
        fig.tight_layout()
        plt.show()

    def __str__(self):
        """String representation"""
        return str(vars(self))

model = SingleProductSPF()

# Define selling price range
selling_price_range = np.arange(80, 141, 10)

# Create data table
data = model.data_table(selling_price_range)

# Print data table
for i in range(len(data['Selling_price'])):
    print(f"Selling Price: {data['Selling_price'][i]}, Demand: {data['Demand'][i]:.2f}, Profit: {data['Profit'][i]:.2f}")

    model.plot_data_table(data)

# Discussion - the linear downward trend observed in the graph suggests a negative linear relationship
# between Selling Price and Profit, which could be influenced by factors such as demand elasticity,
# production costs, and pricing strategies. As price increases, demand and profit decrease.