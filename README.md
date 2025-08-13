import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Create a sample DataFrame to simulate your data ---
# This part is for demonstration; you would replace this with your actual data loading
np.random.seed(42)
data = []
drg_codes = [123, 456, 789, 101]
diag_codes = ['A01.0', 'B02.1', 'C03.2', 'D04.5', 'E05.6']
for _ in range(500):
    num_diags = np.random.choice([0, 1, 2, 3])
    diag_soi_dict = {
        np.random.choice(diag_codes): np.random.randint(1, 5) 
        for _ in range(num_diags)
    }
    aprdrg_drg = np.random.choice(drg_codes)
    aprdrg_soi = np.random.randint(1, 5)
    aprdrg_full = int(f"{aprdrg_drg}{aprdrg_soi}")
    sta_value = np.random.choice(['I', 'Z'], p=[0.25, 0.75])
    
    # Introduce some creative data anomalies
    # High SOI claims are more likely to be overpaid
    if max(diag_soi_dict.values()) > 3 and sta_value == 'Z':
        if np.random.rand() > 0.5:
            sta_value = 'I'
            
    # Claims with discrepancies between APRDRG SOI and clinical SOI are more likely to be overpaid
    if diag_soi_dict and aprdrg_soi != max(diag_soi_dict.values()):
        if np.random.rand() > 0.7:
            sta_value = 'I'
            
    data.append({
        'Diag soi': diag_soi_dict,
        'aprdrg': aprdrg_full,
        'sta': sta_value
    })

df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df.head())
print("-" * 50)

# --- 2. Feature Creation for EDA ---
# Unpacking the dictionary and the 4-digit code for analysis

# Unpack 'Diag soi'
df['num_diags'] = df['Diag soi'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
df['max_soi'] = df['Diag soi'].apply(lambda x: max(x.values()) if x else 0)
df['avg_soi'] = df['Diag soi'].apply(lambda x: np.mean(list(x.values())) if x else 0)

# Decompose 'aprdrg'
df['aprdrg'] = df['aprdrg'].astype(str)
df['aprdrg_drg'] = df['aprdrg'].str[:3]
df['aprdrg_soi'] = df['aprdrg'].str[3].astype(int)

# --- 3. Creative EDA with Plotly Visualizations ---

# Plot 1: Overpayment Status Distribution (Pie Chart)
# Shows the overall balance of your target variable
fig_pie = px.pie(
    df,
    names='sta',
    title='Distribution of Overpayment Status ("I" vs "Z")',
    color_discrete_map={'I': 'red', 'Z': 'green'}
)
fig_pie.show()

# Plot 2: Relationship between Number of Diagnoses and Overpayment (Box Plot)
# Visualizes if claims with more diagnoses are more likely to be overpaid
fig_box1 = px.box(
    df, 
    x='sta', 
    y='num_diags',
    color='sta',
    title='Number of Diagnoses by Overpayment Status',
    labels={'sta': 'Status', 'num_diags': 'Number of Diagnoses'}
)
fig_box1.show()

# Plot 3: Clinical Severity (max_soi) vs. Overpayment (Violin Plot)
# A violin plot is a more creative and detailed alternative to a box plot, showing the density of the data.
fig_violin = px.violin(
    df, 
    x='sta', 
    y='max_soi',
    color='sta',
    box=True, # Show a box plot inside the violin
    points='all', # Show all data points
    title='Distribution of Clinical Severity (Max SOI) by Overpayment Status'
)
fig_violin.show()

# Plot 4: Overpayment Rate by APRDRG Code (Bar Chart)
# Identifies which Diagnosis-Related Groups have a higher incidence of overpayment.
drg_overpayment_rate = df.groupby('aprdrg_drg')['sta'].apply(
    lambda x: (x == 'I').mean()
).reset_index(name='overpayment_rate')

fig_bar = px.bar(
    drg_overpayment_rate.sort_values('overpayment_rate', ascending=False),
    x='aprdrg_drg',
    y='overpayment_rate',
    title='Overpayment Rate by APRDRG Code'
)
fig_bar.show()

# Plot 5: Severity Discrepancy Analysis (Scatter Plot)
# This is a creative plot to check if there is a mismatch between the billed SOI (aprdrg_soi)
# and the clinical SOI (max_soi), and if that mismatch is related to overpayment.
fig_scatter = px.scatter(
    df,
    x='max_soi',
    y='aprdrg_soi',
    color='sta',
    hover_data=['num_diags', 'aprdrg_drg'],
    title='Clinical vs. APRDRG Severity, Colored by Overpayment Status',
    labels={'max_soi': 'Clinical Max SOI', 'aprdrg_soi': 'APRDRG SOI'}
)
fig_scatter.add_trace(go.Scatter(
    x=df['max_soi'],
    y=df['max_soi'],
    mode='lines',
    name='Perfect Match',
    line=dict(color='black', dash='dash')
))
fig_scatter.show()

# Plot 6: Hierarchical View of Overpayment (Sunburst Chart)
# A highly creative way to visualize the nested relationship between DRG, SOI, and the outcome.
# This shows the percentage of I/Z for each combination.
fig_sunburst = px.sunburst(
    df,
    path=['aprdrg_drg', 'aprdrg_soi', 'sta'],
    title='Hierarchical Overpayment Analysis: DRG -> Severity -> Status',
    color='sta',
    color_discrete_map={'I': 'red', 'Z': 'green', '(?)': 'gray'}
)
fig_sunburst.show()


