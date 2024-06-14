# Analysis 1 - Basic Break Even Analysis

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Optional, Union, Tuple, List
from whatif import Model, get_sim_results_df
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection._search import ParameterGrid


# D = 0.06S^2 - 35S + 4900
# D = monthly demand
# S = selling price SPF

# 1a Base Model
# Check - Demand 1668, Profit =$20028


class SingleProductSPF(Model):
    """Base model for single product sales and profit"""
    def __init__(self, fixed_cost=5000, variable_cost_unit=100, selling_price=115,
                 spf_constant=4900, spf_linear=-35, spf_quadratic=0.06):
        """Initialize model with default parameters"""
        self.fixed_cost = fixed_cost
        self.variable_cost_unit = variable_cost_unit
        self.selling_price = selling_price
        self.spf_constant = spf_constant
        self.spf_linear = spf_linear
        self.spf_quadratic = spf_quadratic

    def update(self, params):
        """Update the model with new parameters.

        Args:
            params (dict): Dictionary of parameters to update.
        """
        for key, value in params.items():
            setattr(self, key, value)

    def demand(self):
        """Compute Demand based on the quadratic formula.

        Returns:
            float: Computed demand.
        """
        demand = self.spf_quadratic * self.selling_price ** 2 + self.spf_linear * self.selling_price + self.spf_constant
        return demand

    def total_cost(self):
        """Compute Total Cost.

        Returns:
            float: Total cost.
        """
        demand = self.demand()
        total_cost = self.fixed_cost + (demand * self.variable_cost_unit)
        return total_cost

    def profit(self):
        """Calculate profit.

        Returns:
            float: Computed profit.
        """
        demand = self.demand()
        revenue = self.selling_price * demand
        total_cost = self.fixed_cost + (demand * self.variable_cost_unit)
        profit = revenue - total_cost
        return profit

    def break_even(self, target, by_changing, a, b):
        """Find break-even point by adjusting parameter.

        Args:
            target (float): Target profit to achieve.
            by_changing (str): Parameter to adjust.
            a (float): Start value of parameter.
            b (float): End value of parameter.

        Returns:
            float: Value of parameter that achieves the target profit.
        """
        return self.goal_seek('profit', target, by_changing, a, b)

    def plot_data_table(self,
                        data: Any,
                        param1: str,
                        param2: Optional[str] = None,
                        fig: Optional[plt.Figure] = None,
                        ax1: Optional[plt.Axes] = None,
                        ax2: Optional[Union[plt.Axes, Axes3D]] = None,
                        color: Optional[str] = 'tab:pink') -> None:
        """Plot one-way or two-way data table.

        Parameters
        ----------
        data : Any
            DataFrame containing the data table.
        param1 : str
            The first parameter in the data table.
        param2 : str, optional
            The second parameter in the data table (for two-way data table).
        fig : matplotlib.figure.Figure, optional
            Matplotlib figure object.
        ax1 : matplotlib.axes.Axes, optional
            Matplotlib axes object for the first plot.
        ax2 : Union[matplotlib.axes.Axes, mpl_toolkits.mplot3d.axes3d.Axes3D], optional
            Matplotlib axes object for the second plot (for two-way data table).
        color : str, optional
            Color for plotting.

        """
        if param2 is None:
            # One-way data table
            # Conditional statement that checks if any of these variables are none
            if fig is None or ax1 is None or color is None:
                fig, ax1 = plt.subplots()

            ax1.set_xlabel(f'{param1.capitalize()}')
            ax1.set_ylabel('Profit ($)', color=color)
            ax1.plot(data[param1], data['profit'], color=color, marker='o')
            ax1.tick_params(axis='y', labelcolor=color)

            if ax2 is None:
                ax2 = ax1.twinx()

            color2 = 'tab:purple'
            ax2.set_ylabel('Demand (units)', color=color)
            ax2.plot(data[param1], data['demand'], color=color, marker='x')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(f'Relationship between {param1.capitalize()}, Profit, and Demand')
            fig.tight_layout()
            plt.show()
        else:
            # Two-way data table
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d') # 111 is shorthand for 1x1 grid and index of subplot within grid

            ax.set_xlabel(f'{param1.capitalize()}')
            ax.set_ylabel(f'{param2.capitalize()}')
            ax.set_zlabel('Profit ($)')

            # Plot profit
            ax.scatter(data[param1], data[param2], data['profit'], color='tab:pink', marker='o', label='Profit')

            # Plot demand
            ax.scatter(data[param1], data[param2], data['demand'], color='tab:purple', marker='x', label='Demand')

            plt.title(f'Relationship between {param1.capitalize()}, {param2.capitalize()}, Profit, Demand')
            ax.legend()
            plt.show()

    def __str__(self):
        """String representation"""
        return str(vars(self))


# Initialize model
model = SingleProductSPF()

# Create one-way data table

# Defne selling price range
selling_price_range = np.arange(80, 141, 10)

# Create scenario inputs dictionary
scenario_inputs1 = {'selling_price': selling_price_range}
outputs1 = ['demand', 'profit']

# Create data table using imported data
data_one_way = model.data_table(scenario_inputs1, outputs1)

# Print data table
print("One-Way Data Table:")
print(data_one_way)

# Plot data table
fig, ax1 = plt.subplots()

# Plot profit on ax1
color1 = 'tab:purple'
ax1.set_xlabel('Selling Price ($)')
ax1.set_ylabel('Profit ($)', color=color1)
ax1.plot(data_one_way['selling_price'], data_one_way['profit'], color=color1, marker='o', label='Profit')
ax1.tick_params(axis='y', labelcolor=color1)

# Create twin axes for demand
ax2 = ax1.twinx()
color2 = 'tab:pink'
ax2.set_ylabel('Demand (units)', color=color2)
ax2.plot(data_one_way['selling_price'], data_one_way['demand'], color=color2, marker='x', label='Demand')
ax2.tick_params(axis='y', labelcolor=color2)

# Plot details
fig.tight_layout()
plt.title('Relationship between Selling Price, Profit, and Demand')
plt.show()

# print("\nDiscussion:")
# print("The demand curve appears nonlinear, as it decreases from left to right.")
# print("In contrast, the profit curve increases, indicating a positive relationship with selling price.")

# Discussion - the demand line starts in the top left corner and
# decreases as it moves towards the bottom right corner.
# This indicates an inverse relationship between selling price and demand,
# meaning as the selling price increases, the demand decreases.
# the profit line starts at the bottom left corner and increases as it moves
# towards the top right corner. This suggests a positive relationship between
# selling price and profit, meaning as the selling price increases,
# the profit also increases. The relationship between selling price and demand is nonlinear,
# as indicated by the downward sloping demand curve.


# 1c Break-even

# Define target profit
target_profit = 0

# Define range of selling prices
selling_price_range = np.arange(80, 141, 10)

# Find break-even selling price with goal_seek
break_even_prices = []
for price in selling_price_range:
    break_even_price = model.break_even(target_profit, 'selling_price', price, price + 10)
    break_even_prices.append(break_even_price)

# Print
for price, break_even_price in zip(selling_price_range, break_even_prices):
    if break_even_price is not None:
        print(f"Break-even Selling Price for {price}: {break_even_price}")
    else:
        print(f"Break-even price not found in range for Selling Price: {price}")

# 1d Two-way data table

# Define range of selling prices
selling_price_range = np.arange(80, 141, 10)

# Define variable cost range
variable_cost_range = np.arange(85, 111, 5)

# Create scenario inputs dictionary for two-way data table
scenario_inputs2 = {'selling_price': selling_price_range, 'variable_cost_unit': variable_cost_range}
outputs2 = ['demand', 'profit']

# Create two-way data table using data_table function
data_two_way = model.data_table(scenario_inputs2, outputs2)

# Print
print("\nTwo-Way Data Table:")
print(data_two_way)

# DO I NEED THIS PLOT? HARD TO SEE AND HARD TO UNDERSTAND
# Plot sensitivity of profit by selling price and variable cost
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') # Shorthand for 1x1 grid

# Plot profit
ax.scatter(data_two_way['selling_price'], data_two_way['variable_cost_unit'], data_two_way['profit'],
           color='tab:purple', marker='o', label='Profit')

# Plot demand
ax.scatter(data_two_way['selling_price'], data_two_way['variable_cost_unit'], data_two_way['demand'],
           color='tab:pink', marker='x', label='Demand')

# Set title and labels
ax.set_xlabel('Selling Price ($)')
ax.set_ylabel('Variable Cost ($)')
ax.set_zlabel('Values ($)')
plt.title('Sensitivity of Profit to Selling Price and Variable Cost')
ax.legend()
plt.show()

# KEEP THIS PLOT
# Plot sensitivity of profit by selling price and variable cost
pivot_table = data_two_way.pivot(index='variable_cost_unit', columns='selling_price', values='profit')
sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='.0f')
plt.xlabel('Selling Price ($)')
plt.ylabel('Variable Cost ($)')
plt.title('Profit Heatmap by Selling Price and Variable Cost')
plt.show()


# Discussion (based on the plot)
# print("\nDiscussion:")
# print("The plot shows how profit and demand change with variations in selling price and variable cost.")
# print("Both profit and demand tend to increase with higher selling prices and variable costs,"
#      "but the relationship is not strictly linear.")
# print("This suggests that optimizing selling price and variable cost together is crucial for maximizing profit.")



# 1e Digging Deeper

# Redo goal seek
# Discussion -
# Computational Limitations: Depending on how goal_seek is implemented, it might face challenges when the range is too wide.
# It could be due to the method used for numerical optimization or the step size chosen.

#Model Assumptions: The model (SingleProductSPF) might have assumptions or constraints that lead to undefined or
# non-converging solutions outside the original range. For example, the demand or profit equation might not be
# well-defined or behave unexpectedly outside certain boundaries.

# Precision and Iterations: Wider ranges may require more iterations or finer steps for numerical methods to converge accurately.
# If the method does not handle this effectively, it might fail to find a solution.

# Redo one-way table


# 1f Simulate

# Set seed


# Define uniform distribution for variable cost
#variable_cost_distribution = np.random.uniform(low=80, high=120, size=1000)

# Define random inputs dictionary
#random_inputs1 = {'variable_cost_unit': variable_cost_distribution}

# Define outputs
#outputs1 = ['profit']

# Perform simulation
# with pd.option_context('mode.use_inf_as_na', True): # Added this line to resolve FutureWarning
#simulation_results = model.simulate(random_inputs1, outputs1)

# Create data frame
#simulation_df = get_sim_results_df(simulation_results)

# Create histogram for profit in simulation
#plt.figure(figsize=(10, 6))
#sns.histplot(simulation_df['profit'], bins=50, kde=True)
#plt.xlabel('Profit ($)')
#plt.ylabel('Frequency')
#plt.title('Distribution of Profit from Simulation')
#plt.show()

# Calculate probability that profit is negative in simulation
#neg_profit_probability = (simulation_df['profit'] < 0).mean()
#print(f'Negative Profit Probability: {neg_profit_probability:.2%}')


# SIMULATION REDO
# Define range of selling prices
#selling_price_range = np.arange(80, 141, 10)

# Define variable cost range
#variable_cost_range = np.arange(85, 111, 5)

#np.random.seed(4470)

# Number of simulations for each selling price
#num_sims = 1000

# Create lists to store profits and demand in for each simulation run
#profits = []
#demands = []
#selling_prices = []

# Run simulation
#for selling_price in selling_price_range:
    # Prepare random inputs
#    random_inputs = {
#        'variable_cost_unit': np.random.uniform(80, 120, num_sims)
#    }
    # Prepare scenario inputs
#    scenario_inputs = {
#        'selling_price': [selling_price]
#    }
    # Specify outputs
#    outputs = ['demand', 'profit']

    # Call simulate function
#    results = model.simulate(random_inputs, outputs, scenario_inputs=scenario_inputs)

    # Process results
#    for result in results:
#        profit_array = result['output']['profit']
#        demand_array = result['output']['demand']

        # Ensure profit_array and demand_array are lists
#        if not isinstance(profit_array, list):
#            profit_array = [profit_array]
#        if not isinstance(demand_array, list):
#            demand_array = [demand_array]

        # Extend lists
#        profits.extend(profit_array)
#        demands.extend(demand_array)

        # Extend selling prices
#        selling_prices.extend(['selling_price'] * len(profit_array))


# Create dataframe
#results_df = pd.DataFrame({
#    'selling_price': selling_prices,
#    'profit': profits,
#    'demand': demands
#})


#results_df['profit'] = results_df['profit'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x).astype(float)

#print(results_df.head())
#print(results_df.dtypes)


# Plot histogram
#plt.figure(figsize=(10, 6))
#sns.histplot(results_df['profit'], bins=10, kde=True)
#plt.xlabel('Profit ($)')
#plt.ylabel('Frequency')
#plt.title('Histogram of Profit with uncertainty in Variable Cost')
#plt.show()





# Initialize
fixed_cost = 5000
variable_cost_unit = 100
selling_price = 115
spf_constant = 4900
spf_linear = -35
spf_quadratic = 0.06

model2 = SingleProductSPF(fixed_cost=fixed_cost,
                          variable_cost_unit=variable_cost_unit,
                          selling_price=selling_price,
                          spf_constant=spf_constant,
                          spf_linear=spf_linear,
                          spf_quadratic=spf_quadratic)

num_reps = 100
variable_cost_unit_sim = np.random.uniform(85, 110, num_reps)
selling_price_sim = np.random.uniform(80, 140, num_reps)

random_inputs = {'variable_cost_unit': variable_cost_unit_sim,
                 'selling_price': selling_price_sim}

scenario_inputs = {'selling_price': np.arange(80, 140, 10)}
#list(ParameterGrid(scenario_inputs))

sim_outputs = ['profit']

model2_results = model2.simulate(random_inputs, sim_outputs, scenario_inputs)

which_scenario = 4
print(model2_results[which_scenario].keys())


for scenario in model2_results:
    print(scenario['scenario_num'], scenario['scenario_vals'], scenario['output']['profit'].mean())

model2_results_df = get_sim_results_df(model2_results)
print(model2_results_df.head())

# Plot
sns.boxplot(x="selling_price", y="profit", data=model2_results_df)

profit_histo_g2 = sns.FacetGrid(model2_results_df, col='selling_price', sharey=True, col_wrap=3)
profit_histo_g2 = profit_histo_g2.map(plt.hist, "profit")

model2_results_df.groupby(['scenario_num'])['profit'].describe()

plt.show()

# Show summary statistics
summary_stats = model2_results_df.groupby(['scenario_num'])['profit'].describe()
print(summary_stats)

# Calculate probability that profit is negative
neg_profit_probability = (model2_results_df['profit'] < 0).mean()
print(f'Negative Profit Probability: {neg_profit_probability:.2%}')