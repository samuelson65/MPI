import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
#  YOUR DATA LOADING SECTION
#  IMPORTANT: Place your code to load the DataFrame 'df' here.
#  The rest of the script assumes 'df' exists with columns:
#  - 'Diag soi' (dictionary of diag codes and soi values)
#  - 'aprdrg' (4-digit code)
#  - 'sta' ('I' for overpayment, 'Z' for no findings)
# ==============================================================================
# Example: df = pd.read_csv('your_data.csv')
# Example: df = pd.read_excel('your_data.xlsx')


# ==============================================================================
#  PREPROCESSING: FEATURE EXTRACTION
#  This section transforms your raw data into features useful for analysis.
# ==============================================================================

# Robustly extract features from 'Diag soi' dictionary
def extract_diag_features(diag_dict):
    """Safely extracts number of diagnoses and max SOI from a dictionary."""
    try:
        if not isinstance(diag_dict, dict) or not diag_dict:
            return 0, 0, 0
        soi_values = list(diag_dict.values())
        return len(soi_values), max(soi_values), np.mean(soi_values)
    except Exception:
        return 0, 0, 0

df[['num_diags', 'max_soi', 'avg_soi']] = df['Diag soi'].apply(
    lambda x: pd.Series(extract_diag_features(x))
)

# Extract features from the 'aprdrg' code
df['aprdrg'] = df['aprdrg'].astype(str).str.zfill(4)
df['aprdrg_drg'] = df['aprdrg'].str[:3]
df['aprdrg_soi'] = df['aprdrg'].str[3].astype(int)

# Create a critical feature: SOI Discrepancy
# This flags claims where the billed SOI does not match the max clinical SOI.
df['soi_discrepancy'] = (df['aprdrg_soi'] != df['max_soi'])


# ==============================================================================
#  CREATIVE PLOTLY VISUALIZATIONS
# ==============================================================================

### Plot 1: Hierarchical Overpayment Analysis (Sunburst Chart)
# This chart provides an interactive, hierarchical view of your data,
# showing how the overpayment status is distributed across DRG codes and SOI values.
fig_sunburst = px.sunburst(
    df,
    path=['sta', 'aprdrg_drg', 'aprdrg_soi'],
    title='Hierarchical Analysis: Overpayment Status -> DRG -> SOI',
    color='sta',
    color_discrete_map={'I': 'firebrick', 'Z': 'limegreen', '(?)': 'gray'}
)
fig_sunburst.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>')
fig_sunburst.show()


### Plot 2: Severity Discrepancy Scatter Plot
# This is a powerful, custom scatter plot to visualize the relationship between
# clinical severity (`max_soi`) and billed severity (`aprdrg_soi`).
# It highlights claims with mismatched SOI values, which may be a strong indicator of overpayment.
fig_scatter = go.Figure()

# Plot points where SOI values match
df_match = df[~df['soi_discrepancy']]
fig_scatter.add_trace(go.Scatter(
    x=df_match['max_soi'],
    y=df_match['aprdrg_soi'],
    mode='markers',
    name='SOI Match',
    marker=dict(symbol='circle', color='royalblue', size=6),
    hovertemplate='<b>Clinical SOI: %{x}</b><br><b>APRDRG SOI: %{y}</b><br>Status: %{customdata[0]}<extra></extra>',
    customdata=df_match[['sta']]
))

# Plot points where SOI values mismatch
df_mismatch = df[df['soi_discrepancy']]
fig_scatter.add_trace(go.Scatter(
    x=df_mismatch['max_soi'],
    y=df_mismatch['aprdrg_soi'],
    mode='markers',
    name='SOI Mismatch',
    marker=dict(symbol='diamond', color='firebrick', size=8, line=dict(width=1, color='black')),
    hovertemplate='<b>Clinical SOI: %{x}</b><br><b>APRDRG SOI: %{y}</b><br>Status: %{customdata[0]}<extra></extra>',
    customdata=df_mismatch[['sta']]
))

# Add the line of equality (perfect match) for visual comparison
max_val = max(df['max_soi'].max(), df['aprdrg_soi'].max())
fig_scatter.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color='black', width=2, dash='dash'))

fig_scatter.update_layout(
    title='Clinical vs. APRDRG Severity: Highlighting Discrepancies',
    xaxis_title='Maximum Clinical SOI (from "Diag soi")',
    yaxis_title='APRDRG SOI (4th digit)',
    showlegend=True
)
fig_scatter.show()


### Plot 3: Overpayment Rate by DRG, Facetted by Discrepancy Status
# This chart directly compares the overpayment rate for each DRG,
# broken down by whether a severity discrepancy exists.
drg_agg = df.groupby(['aprdrg_drg', 'soi_discrepancy'])['sta'].apply(
    lambda x: (x == 'I').mean()
).reset_index(name='overpayment_rate')

fig_facet = px.bar(
    drg_agg,
    x='aprdrg_drg',
    y='overpayment_rate',
    color='soi_discrepancy',
    barmode='group',
    labels={'aprdrg_drg': 'APRDRG Code', 'overpayment_rate': 'Overpayment Rate'},
    title='Overpayment Rate by DRG, Separated by SOI Discrepancy Status'
)
fig_facet.update_traces(hovertemplate='<b>DRG: %{x}</b><br>Overpayment Rate: %{y:.2%}<extra></extra>')
fig_facet.show()
