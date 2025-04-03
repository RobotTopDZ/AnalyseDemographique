# Run Dashboard Application
# This script launches the interactive dashboard for exploring the Carrier and Farag
# Demographic Smoothing Technique results

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io
import json

# Load the population data
population_data = pd.read_csv('./data/population.csv')

# Clean the dataset by dropping unnecessary columns
columns_to_keep = ['Location', 'Time', 'Sex', 'AgeStart', 'AgeEnd', 'Age', 'Value']
df_clean = population_data[columns_to_keep].copy()

# Filter for Afghanistan data
df_afghanistan = df_clean[df_clean['Location'] == 'Afghanistan'].copy()

# Get the list of available years
available_years = sorted(df_afghanistan['Time'].unique())

# Get the list of available sex categories
sex_categories = sorted(df_afghanistan['Sex'].unique())

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
    
    # Create a dictionary to store the five-year age groups
    five_year_groups = {}
    
    # Group ages into five-year intervals
    for i in range(0, 100, 5):
        age_group = f"{i}-{i+4}"
        # Filter for the current age group
        group_data = year_data[(year_data['AgeStart'] >= i) & (year_data['AgeStart'] < i+5)]
        
        # Sum the population values for this age group
        if not group_data.empty:
            five_year_groups[age_group] = group_data['Value'].sum()
        else:
            five_year_groups[age_group] = 0
    
    # Convert to DataFrame
    result_df = pd.DataFrame({
        'Age_Group': list(five_year_groups.keys()),
        'Population': list(five_year_groups.values()),
        'Value': list(five_year_groups.values()),  # Add Value column for compatibility
        'AgeStart': [int(ag.split('-')[0]) for ag in five_year_groups.keys()],
        'Year': year,
        'Sex': sex_category
    })
    
    # Create Population column directly from Value
    result_df['Population'] = result_df['Value']
    
    # Sort by age start
    result_df = result_df.sort_values('AgeStart').reset_index(drop=True)
    
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
    
    # Check for empty dataset
    if len(df_cumulative_60) == 0:
        print("ERROR: No data available for ages up to 60")
        return pd.DataFrame(), None, None
        
    try:
        # Compute the K coefficient using K = C(60)/60
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
def create_population_chart(df):
    fig = go.Figure()
    
    # Add raw population trace
    fig.add_trace(go.Bar(
        x=df['Age_Group'],
        y=df['Population'],
        name='Raw Population',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    # Add smoothed population trace
    fig.add_trace(go.Bar(
        x=df['Age_Group'],
        y=df['Smoothed Population'],
        name='Smoothed Population',
        marker_color='rgba(26, 118, 255, 0.7)'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Raw vs. Smoothed Population by Age Group (Afghanistan, {df["Year"].iloc[0]})',
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

# Function to create C(x) - Kx plot
def create_cx_kx_plot(df, polynomial):
    fig = go.Figure()
    
    # Add raw C(x) - Kx trace
    fig.add_trace(go.Scatter(
        x=df['AgeStart'],
        y=df['C(x) - Kx'],
        mode='lines+markers',
        name='Raw C(x) - Kx',
        line=dict(color='rgba(55, 83, 109, 1)', width=2),
        marker=dict(size=8)
    ))
    
    # Add smoothed C(x) - Kx trace
    fig.add_trace(go.Scatter(
        x=df['AgeStart'],
        y=df['Smoothed C(x) - Kx'],
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
        title=f'Raw vs. Smoothed C(x) - Kx (Afghanistan, {df["Year"].iloc[0]})',
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

# Function to create population pyramid
def create_population_pyramid(df_male, df_female):
    # Validate input data
    if df_male.empty or df_female.empty:
        print("Warning: Empty dataframe passed to create_population_pyramid")
        return go.Figure()
    
    # Create a figure with two subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_yaxes=True,
                        column_widths=[0.5, 0.5])
    
    # Add male population (left side, negative values)
    if 'Smoothed Population' in df_male.columns:
        fig.add_trace(go.Bar(
            y=df_male['Age_Group'],
            x=-df_male['Smoothed Population'],  # Negative values for left side
            name='Male (Smoothed)',
            orientation='h',
            marker_color='rgba(58, 71, 80, 0.8)'
        ), row=1, col=1)
    
    # Add female population (right side)
    if 'Smoothed Population' in df_female.columns:
        fig.add_trace(go.Bar(
            y=df_female['Age_Group'],
            x=df_female['Smoothed Population'],
            name='Female (Smoothed)',
            orientation='h',
            marker_color='rgba(246, 78, 139, 0.8)'
        ), row=1, col=2)
    
    # Update the layout with dynamic ranges
    fig.update_layout(
        title=f'Population Pyramid - Smoothed Data (Afghanistan, {df_male["Year"].iloc[0]})',
        barmode='overlay',
        bargap=0.1,
        xaxis=dict(
            title='Male Population',
            tickvals=[-5000000, -4000000, -3000000, -2000000, -1000000, 0],
            ticktext=['5M', '4M', '3M', '2M', '1M', '0'],
            range=[-5000000, 0]
        ),
        xaxis2=dict(
            title='Female Population',
            tickvals=[0, 1000000, 2000000, 3000000, 4000000, 5000000],
            ticktext=['0', '1M', '2M', '3M', '4M', '5M'],
            range=[0, 5000000]
        ),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h'
        ),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the app layout
app.layout = html.Div([
    html.H1('Carrier and Farag Demographic Smoothing Dashboard', 
           style={'textAlign': 'center', 'marginBottom': 30, 'fontFamily': 'Arial', 'color': '#2c3e50'}),
    
    # Controls section
    html.Div([
        html.Div([
            html.Label('Select Year:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in available_years],
                value=available_years[-1],  # Default to most recent year
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Label('Select Visualization:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='viz-dropdown',
                options=[
                    {'label': 'Population by Age Group', 'value': 'population'},
                    {'label': 'C(x) - Kx Analysis', 'value': 'cx_kx'},
                    {'label': 'Population Pyramid', 'value': 'pyramid'}
                ],
                value='population',  # Default visualization
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Button('Export Data', id='export-button', n_clicks=0,
                       style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px',
                              'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer',
                              'marginTop': '20px'})
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'})
    ], style={'marginBottom': 30, 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
    
    # Information section
    html.Div(id='info-section', style={'marginBottom': 20, 'padding': '15px', 'backgroundColor': '#e3f2fd', 'borderRadius': '5px'}),
    
    # Visualization section
    html.Div([
        dcc.Graph(id='main-visualization', style={'height': '600px'})
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '5px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
    
    # Hidden div for storing data
    html.Div(id='processed-data', style={'display': 'none'}),
    
    # Download component
    dcc.Download(id='download-data')
    
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'fontFamily': 'Arial'})

# Callback to process data based on selected year
@app.callback(
    Output('processed-data', 'children'),
    Input('year-dropdown', 'value')
)
def process_data(selected_year):
    # Process data for both sexes
    df_both = group_into_five_year_ages(df_afghanistan, selected_year, 'Both sexes')
    df_smoothed_both, polynomial_both, k_both = apply_carrier_farag_smoothing(df_both)
    
    # Process data for males
    df_male = group_into_five_year_ages(df_afghanistan, selected_year, 'Male')
    df_smoothed_male, polynomial_male, k_male = apply_carrier_farag_smoothing(df_male)
    
    # Process data for females
    df_female = group_into_five_year_ages(df_afghanistan, selected_year, 'Female')
    df_smoothed_female, polynomial_female, k_female = apply_carrier_farag_smoothing(df_female)
    
    # Ensure all required columns are present before JSON conversion
    for df in [df_smoothed_both, df_smoothed_male, df_smoothed_female]:
        if df is not None and not df.empty and 'Smoothed Population' not in df.columns:
            df['Smoothed Population'] = df['Population']  # Default to raw population if smoothed not available
    
    # Convert DataFrames to JSON for storage
    data_json = {
        'both': df_smoothed_both.to_json(date_format='iso', orient='split') if df_smoothed_both is not None else None,
        'male': df_smoothed_male.to_json(date_format='iso', orient='split') if df_smoothed_male is not None else None,
        'female': df_smoothed_female.to_json(date_format='iso', orient='split') if df_smoothed_female is not None else None,
        'k_both': k_both,
        'k_male': k_male,
        'k_female': k_female,
        'year': selected_year
    }
    
    return json.dumps(data_json)  # Convert to JSON string for storage in the hidden div

# Callback to update information section
@app.callback(
    Output('info-section', 'children'),
    Input('processed-data', 'children'),
    Input('viz-dropdown', 'value')
)
def update_info_section(data_json_str, selected_viz):
    if not data_json_str:
        raise PreventUpdate
    
    # Parse the data JSON
    data_json = json.loads(data_json_str)
    
    # Extract K values and year
    k_both = data_json['k_both']
    year = data_json['year']
    
    # Create information content based on selected visualization
    if selected_viz == 'population':
        return html.Div([
            html.H4(f'Population Analysis for Afghanistan ({year})', style={'marginBottom': '10px'}),
            html.P([
                'The Carrier and Farag smoothing technique adjusts demographic data to correct for age heaping and other reporting errors. ',
                f'For {year}, the K coefficient (C(60)/60) is {k_both:,.2f}, which represents the average annual population growth.'
            ])
        ])
    elif selected_viz == 'cx_kx':
        return html.Div([
            html.H4(f'C(x) - Kx Analysis for Afghanistan ({year})', style={'marginBottom': '10px'}),
            html.P([
                'This visualization shows the C(x) - Kx values, which represent the deviation from linear growth. ',
                'The smoothed curve (polynomial fit) helps identify the true demographic pattern by removing irregularities.'
            ])
        ])
    elif selected_viz == 'pyramid':
        return html.Div([
            html.H4(f'Population Pyramid for Afghanistan ({year})', style={'marginBottom': '10px'}),
            html.P([
                'The population pyramid shows the age and sex structure of the population after smoothing. ',
                'This visualization helps identify demographic transitions and potential future population changes.'
            ])
        ])

# Callback to update main visualization
@app.callback(
    Output('main-visualization', 'figure'),
    Input('processed-data', 'children'),
    Input('viz-dropdown', 'value')
)
def update_visualization(data_json_str, selected_viz):
    if not data_json_str:
        raise PreventUpdate
    
    # Parse the data JSON
    data_json = json.loads(data_json_str)
    
    # Convert JSON back to DataFrames
    df_both = pd.read_json(data_json['both'], orient='split')
    df_male = pd.read_json(data_json['male'], orient='split')
    df_female = pd.read_json(data_json['female'], orient='split')
    
    # Create the selected visualization
    if selected_viz == 'population':
        return create_population_chart(df_both)
    elif selected_viz == 'cx_kx':
        # Recreate polynomial from the data
        x = df_both['AgeStart']
        y = df_both['Smoothed C(x) - Kx']
        coefficients = np.polyfit(x, y, 4)
        polynomial = np.poly1d(coefficients)
        return create_cx_kx_plot(df_both, polynomial)
    elif selected_viz == 'pyramid':
        return create_population_pyramid(df_male, df_female)

# Callback for data export
@app.callback(
    Output('download-data', 'data'),
    Input('export-button', 'n_clicks'),
    State('processed-data', 'children'),
    prevent_initial_call=True
)
def export_data(n_clicks, data_json_str):
    if not data_json_str or n_clicks == 0:
        raise PreventUpdate
    
    # Parse the data JSON
    data_json = json.loads(data_json_str)
    
    # Convert JSON back to DataFrames
    df_both = pd.read_json(data_json['both'], orient='split')
    df_male = pd.read_json(data_json['male'], orient='split')
    df_female = pd.read_json(data_json['female'], orient='split')
    year = data_json['year']
    
    # Create a combined DataFrame for export
    # Ensure all required columns exist in the DataFrames
    for df in [df_both, df_male, df_female]:
        if 'Population' not in df.columns:
            # Check if 'Value' exists before trying to use it
            if 'Value' in df.columns:
                df['Population'] = df['Value']
            else:
                # If neither Population nor Value exists, set a default value
                print("WARNING: Neither 'Population' nor 'Value' column found in dataframe")
                df['Population'] = 0  # Default to zero
            
    export_df = pd.DataFrame({
        'Age_Group': df_both['Age_Group'],
        'AgeStart': df_both['AgeStart'],
        'Raw_Population_Both': df_both['Population'],
        'Smoothed_Population_Both': df_both['Smoothed Population'],
        'Raw_Population_Male': df_male['Population'],
        'Smoothed_Population_Male': df_male['Smoothed Population'],
        'Raw_Population_Female': df_female['Population'],
        'Smoothed_Population_Female': df_female['Smoothed Population']
    })
    
    # Return the data for download
    return dcc.send_data_frame(export_df.to_csv, f'afghanistan_demographic_data_{year}.csv', index=False)
    
    # Return the data for download
    return dcc.send_data_frame(export_df.to_csv, f'afghanistan_demographic_data_{year}.csv', index=False)

# Run the app
if __name__ == '__main__':
    print("Starting the Demographic Smoothing Dashboard...")
    print("Open your web browser and navigate to http://127.0.0.1:8050/")
    app.run_server(debug=True)