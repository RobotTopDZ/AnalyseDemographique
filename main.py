# Carrier and Farag Demographic Smoothing Technique
# This script implements the Carrier and Farag Demographic Smoothing Technique
# for analyzing and smoothing demographic data from Afghanistan.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import os
from matplotlib.backends.backend_pdf import PdfPages

# Set visualization styles
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Function to load and clean data
def load_and_clean_data(file_path='./data/population.csv'):
    # Load the population data
    population_data = pd.read_csv(file_path)
    
    # Clean the dataset by dropping unnecessary columns
    columns_to_keep = ['Location', 'Time', 'Sex', 'AgeStart', 'AgeEnd', 'Age', 'Value']
    df_clean = population_data[columns_to_keep].copy()
    
    # Filter for Afghanistan data
    df_afghanistan = df_clean[df_clean['Location'] == 'Afghanistan'].copy()
    
    return df_afghanistan

# Function to group data into five-year age groups
def group_into_five_year_ages(df, year, sex_category='Both sexes'):
    # Print available years for debugging
    available_years = sorted(df['Time'].unique())
    print(f"Available years in dataset: {available_years}")
    
    # Filter data for the specified year and sex category
    year_data = df[(df['Time'].astype(str) == str(year)) & (df['Sex'] == sex_category)].copy()
    
    # Print debug information
    print(f"Found {len(year_data)} records for year {year} and sex {sex_category}")
    if len(year_data) > 0:
        print(f"Sample data: \n{year_data[['Age', 'AgeStart', 'Value']].head()}")
        print(f"Unique age groups: {year_data['Age'].unique()}")
    else:
        print(f"WARNING: No data found for year {year}. Checking for closest available year...")
        # If no data for the specified year, use the closest available year
        available_years_int = [int(y) for y in available_years]
        if available_years_int:
            closest_year = min(available_years_int, key=lambda x: abs(x - int(year)))
            print(f"Using closest available year: {closest_year}")
            year_data = df[(df['Time'].astype(str) == str(closest_year)) & (df['Sex'] == sex_category)].copy()
            print(f"Found {len(year_data)} records for year {closest_year}")
            
            # If still no data found, return empty DataFrame with expected columns
            if len(year_data) == 0:
                print(f"ERROR: No data available for year {year} or closest year {closest_year}")
                return pd.DataFrame(columns=['Age_Group', 'Population', 'AgeStart', 'Year', 'Sex'])
    
    # Sort by age start to ensure proper ordering
    year_data = year_data.sort_values('AgeStart')
    
    # Create result DataFrame directly from the existing age groups
    result_df = pd.DataFrame({
        'Age_Group': year_data['Age'],
        'Population': year_data['Value'],
        'AgeStart': year_data['AgeStart']
    })
    
    # Add year and sex information
    result_df['Year'] = year
    result_df['Sex'] = sex_category
    
    return result_df

# Function to apply the Carrier-Farag smoothing technique
def apply_carrier_farag_smoothing(df_five_year):
    # Compute the cumulative population C(x)
    df_cumulative = df_five_year.sort_values('AgeStart').reset_index(drop=True).copy()
    df_cumulative['Cumulative_Population'] = df_cumulative['Population'].cumsum()
    
    # Filter to include only ages up to 60
    df_cumulative_60 = df_cumulative[df_cumulative['AgeStart'] <= 60].copy()
    
    # Compute the K coefficient using K = C(60)/60
    if len(df_cumulative_60) == 0:
        print("ERROR: No data available for ages up to 60")
        return pd.DataFrame(), None, None
        
    try:
        c_60 = df_cumulative_60[df_cumulative_60['AgeStart'] == 60]['Cumulative_Population'].values[0]
        k_coefficient = c_60 / 60
    except (IndexError, KeyError):
        print("ERROR: Could not calculate K coefficient - missing data for age 60")
        return pd.DataFrame(), None, None
    
    # Generate a new dataset with C(x) - Kx
    df_adjusted = df_cumulative_60.copy()
    df_adjusted['Kx'] = df_adjusted['AgeStart'] * k_coefficient
    df_adjusted['C(x) - Kx'] = df_adjusted['Cumulative_Population'] - df_adjusted['Kx']
    
    # Apply smoothing to create a bell-shaped curve
    # Fit a polynomial to the C(x) - Kx values
    x = df_adjusted['AgeStart']
    y = df_adjusted['C(x) - Kx']
    
    # Fit the polynomial
    coefficients = np.polyfit(x, y, 4)
    polynomial = np.poly1d(coefficients)
    
    # Calculate the smoothed values
    df_adjusted['Smoothed C(x) - Kx'] = polynomial(x)
    df_adjusted['Smoothed C(x)'] = df_adjusted['Smoothed C(x) - Kx'] + df_adjusted['Kx']
    
    # Calculate the smoothed population for each age group
    # For the first age group, the smoothed population is the same as the smoothed cumulative population
    df_adjusted.loc[0, 'Smoothed Population'] = df_adjusted.loc[0, 'Smoothed C(x)']
    
    # For subsequent age groups, calculate the difference in smoothed cumulative population
    for i in range(1, len(df_adjusted)):
        df_adjusted.loc[i, 'Smoothed Population'] = df_adjusted.loc[i, 'Smoothed C(x)'] - df_adjusted.loc[i-1, 'Smoothed C(x)']
    
    return df_adjusted, polynomial, k_coefficient

# Function to create population bar chart
def plot_population_comparison(df_smoothed, year, sex_category):
    plt.figure(figsize=(14, 8))
    
    # Create a bar chart for raw and smoothed population
    x = df_smoothed['Age_Group']
    y1 = df_smoothed['Population']
    y2 = df_smoothed['Smoothed Population']
    
    # Set the width of the bars
    width = 0.35
    
    # Set the positions of the bars on the x-axis
    x_pos = np.arange(len(x))
    
    # Create the bars
    plt.bar(x_pos - width/2, y1, width, label='Raw Population', color='#3498db', alpha=0.7)
    plt.bar(x_pos + width/2, y2, width, label='Smoothed Population', color='#e74c3c', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Age Group')
    plt.ylabel('Population')
    plt.title(f'Raw vs. Smoothed Population by Age Group (Afghanistan, {year}, {sex_category})')
    
    # Add xticks on the middle of the group bars
    plt.xticks(x_pos, x, rotation=45)
    
    # Add a legend
    plt.legend(loc='upper right')
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()  # Return the current figure

# Function to create C(x) - Kx plot
def plot_cx_kx(df_smoothed, polynomial, year, sex_category):
    plt.figure(figsize=(14, 8))
    
    # Plot the raw C(x) - Kx values
    plt.scatter(df_smoothed['AgeStart'], df_smoothed['C(x) - Kx'], 
                label='Raw C(x) - Kx', color='#3498db', s=80, alpha=0.7)
    
    # Plot the smoothed C(x) - Kx values
    plt.scatter(df_smoothed['AgeStart'], df_smoothed['Smoothed C(x) - Kx'], 
                label='Smoothed C(x) - Kx', color='#e74c3c', s=80, alpha=0.7)
    
    # Plot the polynomial fit
    x_smooth = np.linspace(0, 60, 1000)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, label='Polynomial Fit', color='#2ecc71', linewidth=2, linestyle='--')
    
    # Add labels and title
    plt.xlabel('Age')
    plt.ylabel('C(x) - Kx')
    plt.title(f'Raw vs. Smoothed C(x) - Kx (Afghanistan, {year}, {sex_category})')
    
    # Add a legend
    plt.legend(loc='upper right')
    
    # Add grid lines
    plt.grid(linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()  # Return the current figure

# Function to create interactive population bar chart with Plotly
def create_interactive_population_chart(df_smoothed):
    fig = go.Figure()
    
    # Add raw population trace
    fig.add_trace(go.Bar(
        x=df_smoothed['Age_Group'],
        y=df_smoothed['Population'],
        name='Raw Population',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    # Add smoothed population trace
    fig.add_trace(go.Bar(
        x=df_smoothed['Age_Group'],
        y=df_smoothed['Smoothed Population'],
        name='Smoothed Population',
        marker_color='rgba(26, 118, 255, 0.7)'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Raw vs. Smoothed Population by Age Group (Afghanistan, {df_smoothed["Year"].iloc[0]}, {df_smoothed["Sex"].iloc[0]})',
        xaxis_title='Age Group',
        yaxis_title='Population',
        barmode='group',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.1)'
        ),
        hovermode='closest'
    )
    
    return fig

# Function to create interactive C(x) - Kx plot with Plotly
def create_interactive_cx_kx_plot(df_smoothed, polynomial):
    fig = go.Figure()
    
    # Add raw C(x) - Kx trace
    fig.add_trace(go.Scatter(
        x=df_smoothed['AgeStart'],
        y=df_smoothed['C(x) - Kx'],
        mode='lines+markers',
        name='Raw C(x) - Kx',
        line=dict(color='rgba(55, 83, 109, 1)', width=2),
        marker=dict(size=8)
    ))
    
    # Add smoothed C(x) - Kx trace
    fig.add_trace(go.Scatter(
        x=df_smoothed['AgeStart'],
        y=df_smoothed['Smoothed C(x) - Kx'],
        mode='lines+markers',
        name='Smoothed C(x) - Kx',
        line=dict(color='rgba(26, 118, 255, 1)', width=2),
        marker=dict(size=8)
    ))
    
    # Add polynomial fit trace
    x_smooth = np.linspace(0, 60, 1000)
    y_smooth = polynomial(x_smooth)
    
    fig.add_trace(go.Scatter(
        x=x_smooth,
        y=y_smooth,
        mode='lines',
        name='Polynomial Fit',
        line=dict(color='rgba(46, 204, 113, 1)', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Raw vs. Smoothed C(x) - Kx (Afghanistan, {df_smoothed["Year"].iloc[0]}, {df_smoothed["Sex"].iloc[0]})',
        xaxis_title='Age',
        yaxis_title='C(x) - Kx',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.1)'
        ),
        hovermode='closest'
    )
    
    return fig

# Function to process multiple years and sex categories
def process_multiple_years_and_sexes(df, years, sex_categories):
    results = []
    
    for year in years:
        for sex in sex_categories:
            # Group into five-year age groups
            df_five_year = group_into_five_year_ages(df, year, sex)
            
            # Apply Carrier-Farag smoothing
            df_smoothed, _, k_coefficient = apply_carrier_farag_smoothing(df_five_year)
            
            # Add to results
            results.append(df_smoothed)
    
    # Combine all results into a single DataFrame
    combined_results = pd.concat(results, ignore_index=True)
    
    return combined_results

# Function to save results to CSV
def save_results_to_csv(df, output_file='afghanistan_demographic_data.csv'):
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Function to save plots to PDF
def save_plots_to_pdf(figures, output_file='demographic_analysis.pdf'):
    with PdfPages(output_file) as pdf:
        for fig in figures:
            pdf.savefig(fig)
    print(f"Plots saved to {output_file}")

# Main function to run the analysis
def main(year=2020, sex='Both sexes', output_dir='./output'):
    # Load and clean the data
    df_afghanistan = load_and_clean_data()
    
    # Get the list of available years and sex categories
    available_years = sorted(df_afghanistan['Time'].unique())
    sex_categories = sorted(df_afghanistan['Sex'].unique())
    
    print(f'Available years: {available_years}')
    print(f'Sex categories: {sex_categories}')
    
    # Use the provided year and sex category for analysis
    year_to_analyze = year
    sex_to_analyze = sex
    
    print(f'Analyzing year: {year_to_analyze}, Sex: {sex_to_analyze}')
    
    # Group into five-year age groups
    df_five_year = group_into_five_year_ages(df_afghanistan, year_to_analyze, sex_to_analyze)
    
    # Apply the Carrier-Farag smoothing technique
    df_smoothed, polynomial, k_coefficient = apply_carrier_farag_smoothing(df_five_year)
    
    # Display the results
    print(f'K coefficient: {k_coefficient:.2f}')
    print(df_smoothed.head(10))
    
    # Create visualizations
    fig_population = plot_population_comparison(df_smoothed, year_to_analyze, sex_to_analyze)
    fig_cx_kx = plot_cx_kx(df_smoothed, polynomial, year_to_analyze, sex_to_analyze)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, f'afghanistan_demographic_data_{year_to_analyze}.csv')
    save_results_to_csv(df_smoothed, csv_path)
    
    # Save plots to PDF
    pdf_path = os.path.join(output_dir, f'demographic_analysis_{year_to_analyze}.pdf')
    save_plots_to_pdf([fig_population, fig_cx_kx], pdf_path)
    
    # Process multiple years and sex categories
    years_to_analyze = [2015, 2020, 2022, 2024, 2030]  # Expanded to include more recent years
    # Use only the sex categories that are actually available in the data
    sexes_to_analyze = sex_categories  # This will use the sex categories found in the data
    
    combined_results = process_multiple_years_and_sexes(df_afghanistan, years_to_analyze, sexes_to_analyze)
    
    # Save combined results to CSV
    combined_csv_path = os.path.join(output_dir, 'afghanistan_demographic_data_combined.csv')
    save_results_to_csv(combined_results, combined_csv_path)
    
    print("Analysis completed successfully!")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Carrier and Farag Demographic Smoothing Analysis')
    parser.add_argument('--year', type=int, default=2020, help='Year to analyze (default: 2020)')
    parser.add_argument('--sex', type=str, default='Both sexes', 
                        choices=['Both sexes', 'Male', 'Female'], 
                        help='Sex category to analyze (default: Both sexes)')
    parser.add_argument('--output', type=str, default='./output', help='Output directory for results (default: ./output)')
    
    args = parser.parse_args()
    
    # Run the main function with the provided arguments
    main(year=args.year, sex=args.sex, output_dir=args.output)