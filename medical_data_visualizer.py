import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def is_overweight(bmi_list):
    result = []
    for x in bmi_list:
        if x > 25:
            result.append(1)
        else:
            result.append(0)
    return result


# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = (df['weight'] / (df['height']**2) * 10000).tolist()
df['overweight'] = is_overweight(bmi)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df_normalized = df.copy()

for col in df_normalized.columns:
    if (col == 'cholesterol' or col == 'gluc'):
        decimal = (df_normalized[col] - df_normalized[col].min()) / (
            df_normalized[col].max() - df_normalized[col].min())
        df_normalized[col] = round(decimal).astype(int)

# Draw Categorical Plot


def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df_normalized, id_vars=['cardio'], value_vars=[
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat['total'] = 1
    df_cat = df_cat.groupby(
        ['cardio', 'variable', 'value'], as_index=False).count()
    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x="variable", y="total", col="cardio",
                      hue="value", kind="bar", data=df_cat).figure
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr(method="pearson")

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, ax=ax, annot=True, mask=mask, linewidths=1, vmin=-0.12, vmax=0.28,
                fmt='.1f', center=0, cbar_kws={"ticks": [0.24, 0.16, 0.08, 0.00, -0.08], "shrink": 0.5})
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
