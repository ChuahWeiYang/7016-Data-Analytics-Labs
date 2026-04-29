# =============================================================================
# Lab 2 – Exploratory Data Analysis (EDA)
# Dataset  : WorldEnergy.csv
# Scenario : Smart Energy Transition – Renewable vs Fossil Fuel Trends
# Scope    : World + Malaysia, China, United States, Germany, India
# Period   : 1965–2024 (rows with sufficient data)
# Author   : [GUO XUANZHUO & CHUAH WEI YANG]
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')               # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Colour palette ────────────────────────────────────────────────────────────
COUNTRY_COLORS = {
    'World'         : '#2C3E50',
    'Malaysia'      : '#27AE60',
    'China'         : '#E74C3C',
    'United States' : '#3498DB',
    'Germany'       : '#F39C12',
    'India'         : '#9B59B6',
}
COUNTRIES = list(COUNTRY_COLORS.keys())
SELECTED  = [c for c in COUNTRIES if c != 'World']   # 5 individual countries

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight'})

# =============================================================================
# 0. LOAD & PREPARE DATA
# =============================================================================
df_raw = pd.read_csv('C:/Users/TK/Desktop/7016/Assignment1/WorldEnergy.csv')

# Core variables used throughout
CORE_VARS = [
    'country', 'year', 'population', 'gdp',
    'primary_energy_consumption',
    'renewables_consumption', 'fossil_fuel_consumption',
    'renewables_share_energy', 'fossil_share_energy',
    'renewables_share_elec',  'fossil_share_elec',
    'solar_consumption', 'wind_consumption',
    'hydro_consumption', 'nuclear_consumption',
]

df = df_raw[df_raw['country'].isin(COUNTRIES)][CORE_VARS].copy()

# Keep years from 1965 onward (data become reliable after this point)
df = df[df['year'] >= 1965].reset_index(drop=True)

# =============================================================================
# 1. TABULAR SUMMARY – dataset overview
# =============================================================================
print("=" * 65)
print("  DATASET OVERVIEW")
print("=" * 65)
print(f"  Shape after filtering  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Countries              : {COUNTRIES}")
print(f"  Year range             : {df['year'].min()} – {df['year'].max()}")
print()

# --- 1a. Descriptive statistics for key numeric variables ---
desc_cols = [
    'primary_energy_consumption',
    'renewables_share_energy',
    'fossil_share_energy',
    'renewables_consumption',
    'fossil_fuel_consumption',
]
print("  Descriptive Statistics (all selected countries combined)")
print(df[desc_cols].describe().round(2).to_string())
print()

# --- 1b. Missing-value analysis ---
print("  Missing Values per Column")
mv = df[CORE_VARS[2:]].isnull().sum()
mv_pct = (mv / len(df) * 100).round(1)
mv_df = pd.DataFrame({'Missing Count': mv, 'Missing %': mv_pct})
print(mv_df[mv_df['Missing Count'] > 0].to_string())
print()

# =============================================================================
# 2. GRAPH 1 – World Primary Energy Consumption over Time (Line Plot)
# =============================================================================
world = df[df['country'] == 'World'].dropna(subset=['primary_energy_consumption'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(world['year'], world['primary_energy_consumption'],
        color=COUNTRY_COLORS['World'], linewidth=2.5, marker='o', markersize=3)

# Rolling mean (window = 5 years) to highlight trend
rolling_mean = world.set_index('year')['primary_energy_consumption'].rolling(5, center=True).mean()
ax.plot(rolling_mean.index, rolling_mean.values,
        color='#E74C3C', linewidth=2, linestyle='--', label='5-yr Rolling Mean')

ax.set_title('World Primary Energy Consumption (1965–2024)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Primary Energy Consumption (TWh)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.savefig('graph1_world_primary_energy.png')
plt.close()
print("  [Graph 1] Saved: graph1_world_primary_energy.png")

# =============================================================================
# 3. GRAPH 2 – World: Renewable Share vs Fossil Share over Time (Area Chart)
# =============================================================================
world2 = df[df['country'] == 'World'].dropna(
    subset=['renewables_share_energy', 'fossil_share_energy'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(world2['year'], world2['fossil_share_energy'],
                alpha=0.55, color='#E74C3C', label='Fossil Fuel Share (%)')
ax.fill_between(world2['year'], world2['renewables_share_energy'],
                alpha=0.70, color='#27AE60', label='Renewables Share (%)')
ax.plot(world2['year'], world2['fossil_share_energy'],
        color='#C0392B', linewidth=1.5)
ax.plot(world2['year'], world2['renewables_share_energy'],
        color='#1E8449', linewidth=1.5)

ax.set_title('World: Renewable vs Fossil Share of Primary Energy (1965–2024)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Share of Primary Energy (%)')
ax.legend(loc='center right')
plt.tight_layout()
plt.savefig('graph2_world_renewable_vs_fossil_share.png')
plt.close()
print("  [Graph 2] Saved: graph2_world_renewable_vs_fossil_share.png")

# =============================================================================
# 4. GRAPH 3 – Renewable Share over Time per Selected Country (Multi-line)
# =============================================================================
sel = df[df['country'].isin(SELECTED)].dropna(subset=['renewables_share_energy'])

fig, ax = plt.subplots(figsize=(11, 6))
for country in SELECTED:
    cdf = sel[sel['country'] == country]
    ax.plot(cdf['year'], cdf['renewables_share_energy'],
            label=country, color=COUNTRY_COLORS[country],
            linewidth=2, marker='o', markersize=2.5)

ax.set_title('Renewable Energy Share of Primary Energy – Selected Countries (1965–2024)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Renewables Share (%)')
ax.legend(title='Country', framealpha=0.9)
plt.tight_layout()
plt.savefig('graph3_renewable_share_countries.png')
plt.close()
print("  [Graph 3] Saved: graph3_renewable_share_countries.png")

# =============================================================================
# 5. GRAPH 4 – Fossil Fuel Share over Time per Selected Country (Multi-line)
# =============================================================================
sel4 = df[df['country'].isin(SELECTED)].dropna(subset=['fossil_share_energy'])

fig, ax = plt.subplots(figsize=(11, 6))
for country in SELECTED:
    cdf = sel4[sel4['country'] == country]
    ax.plot(cdf['year'], cdf['fossil_share_energy'],
            label=country, color=COUNTRY_COLORS[country],
            linewidth=2, marker='o', markersize=2.5)

ax.set_title('Fossil Fuel Share of Primary Energy – Selected Countries (1965–2024)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Fossil Share (%)')
ax.legend(title='Country', framealpha=0.9)
plt.tight_layout()
plt.savefig('graph4_fossil_share_countries.png')
plt.close()
print("  [Graph 4] Saved: graph4_fossil_share_countries.png")

# =============================================================================
# 6. GRAPH 5 – Latest Year: Renewable Consumption Bar Chart (Selected Countries)
# =============================================================================
latest_year = df.dropna(subset=['renewables_consumption'])['year'].max()
latest = df[(df['year'] == latest_year) & (df['country'].isin(SELECTED))].dropna(
    subset=['renewables_consumption']).sort_values('renewables_consumption', ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(latest['country'],
               latest['renewables_consumption'],
               color=[COUNTRY_COLORS[c] for c in latest['country']],
               edgecolor='white', height=0.6)
for bar, val in zip(bars, latest['renewables_consumption']):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f} TWh', va='center', fontsize=9)

ax.set_title(f'Renewable Energy Consumption – {latest_year}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Renewables Consumption (TWh)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig('graph5_renewable_consumption_bar.png')
plt.close()
print("  [Graph 5] Saved: graph5_renewable_consumption_bar.png")

# =============================================================================
# 7. GRAPH 6 – Correlation Heatmap of Key Energy Variables (World only)
# =============================================================================
heatmap_cols = [
    'primary_energy_consumption',
    'renewables_consumption',
    'fossil_fuel_consumption',
    'renewables_share_energy',
    'fossil_share_energy',
    'solar_consumption',
    'wind_consumption',
    'hydro_consumption',
]
world_hm = df[df['country'] == 'World'][heatmap_cols].dropna()
corr = world_hm.corr()

short_labels = {
    'primary_energy_consumption' : 'Primary Energy',
    'renewables_consumption'     : 'Renewables',
    'fossil_fuel_consumption'    : 'Fossil Fuels',
    'renewables_share_energy'    : 'Renew. Share %',
    'fossil_share_energy'        : 'Fossil Share %',
    'solar_consumption'          : 'Solar',
    'wind_consumption'           : 'Wind',
    'hydro_consumption'          : 'Hydro',
}
corr.index   = corr.index.map(short_labels)
corr.columns = corr.columns.map(short_labels)

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=-1, vmax=1, linewidths=0.5,
            mask=mask, ax=ax, square=True,
            annot_kws={'size': 9})
ax.set_title('Correlation Heatmap – World Energy Variables',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('graph6_correlation_heatmap.png')
plt.close()
print("  [Graph 6] Saved: graph6_correlation_heatmap.png")

# =============================================================================
# 8. GRAPH 7 – Renewable Mix Breakdown: Solar / Wind / Hydro (World stacked area)
# =============================================================================
mix_cols = ['solar_consumption', 'wind_consumption', 'hydro_consumption']
world_mix = df[df['country'] == 'World'][['year'] + mix_cols].dropna()

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(world_mix['year'],
             world_mix['hydro_consumption'],
             world_mix['wind_consumption'],
             world_mix['solar_consumption'],
             labels=['Hydro', 'Wind', 'Solar'],
             colors=['#3498DB', '#27AE60', '#F39C12'],
             alpha=0.85)
ax.set_title('World Renewable Mix: Hydro / Wind / Solar (1965–2024)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (TWh)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('graph7_renewable_mix_stacked.png')
plt.close()
print("  [Graph 7] Saved: graph7_renewable_mix_stacked.png")

# =============================================================================
# 9. GRAPH 8 – Boxplot: Distribution of Renewables Share by Country
# =============================================================================
box_df = df[df['country'].isin(SELECTED)].dropna(subset=['renewables_share_energy'])

fig, ax = plt.subplots(figsize=(9, 5))
order = box_df.groupby('country')['renewables_share_energy'].median().sort_values(
    ascending=False).index.tolist()
palette = [COUNTRY_COLORS[c] for c in order]
sns.boxplot(data=box_df, x='country', y='renewables_share_energy',
            order=order, palette=palette, width=0.5, linewidth=1.2, ax=ax)

ax.set_title('Distribution of Renewables Share (%) – 1965 to 2024',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Country')
ax.set_ylabel('Renewables Share of Primary Energy (%)')
plt.tight_layout()
plt.savefig('graph8_renewable_share_boxplot.png')
plt.close()
print("  [Graph 8] Saved: graph8_renewable_share_boxplot.png")

# =============================================================================
# 10. UNIVARIATE ANALYSIS – Histogram of Renewables Share (World, all years)
# =============================================================================
world_u = df[df['country'] == 'World'].dropna(subset=['renewables_share_energy'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(world_u['renewables_share_energy'], bins=20,
             color='#27AE60', edgecolor='white', alpha=0.85)
axes[0].axvline(world_u['renewables_share_energy'].mean(),
                color='red', linestyle='--', linewidth=1.5, label='Mean')
axes[0].axvline(world_u['renewables_share_energy'].median(),
                color='navy', linestyle=':', linewidth=1.5, label='Median')
axes[0].set_title('Histogram – World Renewables Share (%)')
axes[0].set_xlabel('Renewables Share (%)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Density plot (KDE)
sns.kdeplot(world_u['renewables_share_energy'],
            color='#27AE60', fill=True, alpha=0.5, ax=axes[1])
axes[1].set_title('Density Plot – World Renewables Share (%)')
axes[1].set_xlabel('Renewables Share (%)')
axes[1].set_ylabel('Density')

skew_val = world_u['renewables_share_energy'].skew()
kurt_val = world_u['renewables_share_energy'].kurtosis()
axes[1].text(0.05, 0.90, f'Skewness : {skew_val:.2f}\nKurtosis : {kurt_val:.2f}',
             transform=axes[1].transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7))

plt.suptitle('Univariate Analysis – World Renewables Share', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('graph9_univariate_renewables_share.png')
plt.close()
print("  [Graph 9] Saved: graph9_univariate_renewables_share.png")

# =============================================================================
# 11. PRINT SUMMARY STATISTICS per COUNTRY (latest available year)
# =============================================================================
print()
print("=" * 65)
print("  PER-COUNTRY SUMMARY – Most Recent Available Year")
print("=" * 65)
summary_cols = ['country', 'year',
                'primary_energy_consumption',
                'renewables_share_energy',
                'fossil_share_energy',
                'renewables_consumption',
                'fossil_fuel_consumption']
latest_all = []
for c in COUNTRIES:
    cdf = df[df['country'] == c].dropna(subset=['renewables_share_energy'])
    if not cdf.empty:
        latest_all.append(cdf.iloc[-1][summary_cols])
summary_df = pd.DataFrame(latest_all).set_index('country')
summary_df['year'] = summary_df['year'].astype(int)
summary_df = summary_df.round(2)
print(summary_df.to_string())
print()
print("  All graphs have been saved successfully.")
print("=" * 65)
