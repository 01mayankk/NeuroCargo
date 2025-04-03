import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from matplotlib.patches import Wedge, Circle
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_histogram(df, column, title, save_path):
    """Generates a histogram for a given column in a DataFrame."""
    try:
        plt.figure(figsize=(10, 6))                    
        sns.histplot(df[column], bins=20, kde=True)
        plt.title(title)
        plt.xlabel(column.replace('_', ' ').title())  # Format the x-axis label
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Histogram generated and saved to {save_path}")
        return True  # Indicate success
    except Exception as e:
        logging.error(f"Error generating histogram: {e}")
        return False  # Indicate failure

def generate_histogram_by_category(df, value_col, category_col, title, save_path):
    """Generates a histogram for a value column grouped by a category column."""
    try:
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df, x=value_col, hue=category_col, kde=True, multiple="stack")
        plt.title(title)
        plt.xlabel(value_col.replace('_', ' ').title())
        plt.ylabel("Frequency")
        plt.legend(title=category_col.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Histogram by category generated and saved to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating histogram by category: {e}")
        return False

def generate_boxplot(df, column, title, save_path):
    """Generates a boxplot for a given column in a DataFrame."""
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(title)
        plt.xlabel(column.replace('_', ' ').title())  # Format the x-axis label
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Boxplot generated and saved to {save_path}")
        return True  # Indicate success
    except Exception as e:
        logging.error(f"Error generating boxplot: {e}")
        return False  # Indicate failure

def generate_boxplot_by_category(df, value_col, category_col, title, save_path):
    """Generates a boxplot comparing a value column across different categories."""
    try:
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=category_col, y=value_col, data=df)
        plt.title(title)
        plt.xlabel(category_col.replace('_', ' ').title())
        plt.ylabel(value_col.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Boxplot by category generated and saved to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating boxplot by category: {e}")
        return False

def generate_scatter_plot(df, x_col, y_col, title, path):
    """Generates a scatter plot and saves it to a file."""
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_col, y=y_col, data=df)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logging.info(f"Scatter plot generated and saved to {path}")
        return True
    except Exception as e:
        logging.error(f"Error generating scatter plot: {e}")
        return False

def generate_scatter_plot_with_hue(df, x_col, y_col, hue_col, title, path):
    """Generates a scatter plot with points colored by a categorical variable."""
    try:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.legend(title=hue_col.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logging.info(f"Scatter plot with hue generated and saved to {path}")
        return True
    except Exception as e:
        logging.error(f"Error generating scatter plot with hue: {e}")
        return False

def generate_heatmap(df, title, path):
    """Generates a heatmap and saves it to a file."""
    try:
        plt.figure(figsize=(12, 10))
        
        # Check if df is pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            logging.error(f"Expected DataFrame for heatmap, got {type(df)}")
            return False
        
        # Make sure we have numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logging.error("No numeric columns found for correlation heatmap")
            return False
            
        # Calculate correlation matrix
        corr = numeric_df.corr(numeric_only=True)
        
        # Use a mask to hide the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Generate heatmap with triangle mask and improved aesthetics
        sns.heatmap(corr, mask=mask, annot=True, cmap='viridis', 
                   fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logging.info(f"Heatmap generated and saved to {path}")
        return True
    except Exception as e:
        logging.error(f"Error generating heatmap: {e}")
        logging.exception(e)
        return False

def generate_pair_plot(df, title, save_path):
    """Generates a pair plot for a DataFrame."""
    try:
        # Limit to numerical columns to avoid errors
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # If too many columns, select a subset
        if len(numerical_cols) > 5:
            numerical_cols = numerical_cols[:5]
            
        sns.pairplot(df[numerical_cols])
        plt.suptitle(title, y=1.02)  # Adjust title position
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Pair plot generated and saved to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating pair plot: {e}")
        return False

def generate_count_plot(df, column, title, save_path):
    """Generates a count plot for a given column in a DataFrame."""
    try:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=df[column])
        
        # Add count labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', 
                       xytext = (0, 5), textcoords = 'offset points')
        
        plt.title(title)
        plt.xlabel(column.replace('_', ' ').title())  # Format the x-axis label
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Count plot generated and saved to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating count plot: {e}")
        return False

def generate_pie_chart(values, labels, title, save_path):
    """Generates a pie chart of values with custom styling."""
    try:
        # Use a custom color palette
        colors = sns.color_palette('viridis', len(values))
        
        plt.figure(figsize=(10, 8))
        
        # Create pie chart with styling
        wedges, texts, autotexts = plt.pie(
            values, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05] * len(values),  # Slightly explode all slices
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Pie chart generated and saved to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating pie chart: {e}")
        return False

def generate_gauge_chart(percentage, title, save_path):
    """Generates a gauge chart showing capacity utilization percentage."""
    try:
        # Ensure percentage is between 0 and 100
        percentage = max(0, min(100, percentage))
        
        # Set up the figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # Draw the gauge background (gray semicircle)
        wedge = Wedge(center=(0.5, 0), r=0.4, theta1=0, theta2=180, 
                     fc='#EEEEEE', linewidth=2, edgecolor='#666666')
        ax.add_patch(wedge)
        
        # Calculate angle for the gauge needle
        angle = np.deg2rad(180 * (1 - percentage / 100))
        
        # Draw the colored section based on percentage
        if percentage <= 75:
            color = 'green'
        elif percentage <= 90:
            color = 'orange'
        else:
            color = 'red'
            
        wedge = Wedge(center=(0.5, 0), r=0.4, theta1=0, theta2=180 * (percentage / 100), 
                     fc=color, linewidth=0)
        ax.add_patch(wedge)
        
        # Draw the center point and needle
        circle = Circle((0.5, 0), 0.02, fc='black', zorder=10)
        ax.add_patch(circle)
        
        # Draw needle
        needle_x = 0.5 + 0.35 * np.cos(angle)
        needle_y = 0.35 * np.sin(angle)
        ax.plot([0.5, needle_x], [0, needle_y], 'k-', linewidth=3, zorder=11)
        
        # Add percentage text
        ax.text(0.5, -0.1, f"{percentage:.1f}%", ha='center', va='center', 
               fontsize=24, fontweight='bold', color=color)
        
        # Add min and max labels
        ax.text(0.1, 0, "0%", ha='center', va='center', fontsize=14)
        ax.text(0.9, 0, "100%", ha='center', va='center', fontsize=14)
        
        # Add gauge ticks
        for i in range(0, 101, 10):
            tick_angle = np.deg2rad(180 * (1 - i / 100))
            tick_start_r = 0.35
            tick_end_r = 0.4
            
            tick_start_x = 0.5 + tick_start_r * np.cos(tick_angle)
            tick_start_y = tick_start_r * np.sin(tick_angle)
            
            tick_end_x = 0.5 + tick_end_r * np.cos(tick_angle)
            tick_end_y = tick_end_r * np.sin(tick_angle)
            
            ax.plot([tick_start_x, tick_end_x], [tick_start_y, tick_end_y], 'k-', linewidth=2)
            
            # Add label for major ticks (every 25%)
            if i % 25 == 0 and i > 0 and i < 100:
                label_r = 0.3
                label_x = 0.5 + label_r * np.cos(tick_angle)
                label_y = label_r * np.sin(tick_angle)
                ax.text(label_x, label_y, f"{i}%", ha='center', va='center', fontsize=12)
        
        # Set up the plot
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.45, title, ha='center', va='center', fontsize=16, fontweight='bold')
        
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Gauge chart generated and saved to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating gauge chart: {e}")
        return False

def generate_line_chart(data, x_column, y_column, hue_column=None, title="Line Chart", figsize=(10, 6), save_path=None):
    """Generate a line chart with optional grouping by a categorical variable."""
    try:
        plt.figure(figsize=figsize)
        sns.set_style("whitegrid")
        
        # Use a custom palette
        custom_palette = sns.color_palette("husl", 8)
        
        # Create the line plot
        if hue_column:
            ax = sns.lineplot(data=data, x=x_column, y=y_column, hue=hue_column, palette=custom_palette, marker='o')
            # Add a legend with a title
            plt.legend(title=hue_column, loc='best', frameon=True, framealpha=0.9)
        else:
            ax = sns.lineplot(data=data, x=x_column, y=y_column, color=custom_palette[0], marker='o')
        
        # Set title and labels
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        
        # Rotate x-axis labels if there are many categories
        if len(data[x_column].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Customize the grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a light background color
        ax.set_facecolor('#f8f9fa')
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Line chart generated and saved to {save_path}")
        
        return save_path
    except Exception as e:
        logging.error(f"Error generating line chart: {e}")
        return None

def generate_bar_chart(data, x_column, y_column, hue_column=None, title="Bar Chart", figsize=(10, 6), save_path=None):
    """Generate a bar chart with optional grouping by a categorical variable."""
    try:
        plt.figure(figsize=figsize)
        sns.set_style("whitegrid")
        
        # Use a custom palette
        custom_palette = sns.color_palette("husl", 8)
        
        # Create the bar plot
        if hue_column:
            ax = sns.barplot(data=data, x=x_column, y=y_column, hue=hue_column, palette=custom_palette)
            # Add a legend with a title
            plt.legend(title=hue_column, loc='best', frameon=True, framealpha=0.9)
        else:
            ax = sns.barplot(data=data, x=x_column, y=y_column, color=custom_palette[0])
        
        # Set title and labels
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        
        # Rotate x-axis labels if there are many categories
        if len(data[x_column].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Customize the grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a light background color
        ax.set_facecolor('#f8f9fa')
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Bar chart generated and saved to {save_path}")
        
        return save_path
    except Exception as e:
        logging.error(f"Error generating bar chart: {e}")
        return None

def generate_radar_chart(data, categories, values, title="Radar Chart", figsize=(10, 8), save_path=None):
    """Generate a radar chart for visualizing multivariate data."""
    try:
        # Convert data to numpy arrays for calculations
        categories = np.array(categories)
        values = np.array(values)
        
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot (divide the plot / number of variables)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        
        # Values need to be repeated to close the polygon as well
        values = np.concatenate((values, [values[0]]))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw the plot
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#5CB85C')
        ax.fill(angles, values, color='#5CB85C', alpha=0.4)
        
        # Add a title
        plt.title(title, size=16, fontweight='bold', y=1.1)
        
        # Set y-axis limits
        plt.ylim(0, max(values) * 1.1)
        
        # Add grid lines and styling
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Radar chart generated and saved to {save_path}")
        
        return save_path
    except Exception as e:
        logging.error(f"Error generating radar chart: {e}")
        return None