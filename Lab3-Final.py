import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE   = os.path.join(_SCRIPT_DIR, 'WorldEnergy.csv')
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, 'lab3_output')

TARGET_COL  = 'renewables_share_elec'   # % of electricity from renewables
COUNTRY_COL = 'country'
YEAR_COL    = 'year'
DECADE_COL  = 'decade'

ALPHA = 0.05

# Five ASEAN nations chosen for regional relevance
ASEAN_COUNTRIES = ['Malaysia', 'Indonesia', 'Thailand', 'Vietnam', 'Philippines']

# Minimum observations per cell to flag balance concern
MIN_CELL_N = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD DATASET

print('=' * 64)
print('LAB 3: ANOVA ANALYSIS - RENEWABLE ENERGY IN ASEAN NATIONS')
print('=' * 64)

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"\nERROR: Dataset not found at:\n  {DATA_FILE}")
    print("Ensure WorldEnergy.csv is in the same folder as this script.")
    sys.exit(1)

print(f"\nDataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

required_cols = [COUNTRY_COL, YEAR_COL, TARGET_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"ERROR: Required column(s) not found: {missing}")
    sys.exit(1)

print(f"Required columns: {required_cols} - all present")
print(f"Year range (full dataset): {int(df[YEAR_COL].min())} to {int(df[YEAR_COL].max())}")
print(f"Unique entities in dataset: {df[COUNTRY_COL].nunique()}")

# Confirm ASEAN countries are present
found = [c for c in ASEAN_COUNTRIES if c in df[COUNTRY_COL].values]
missing_countries = [c for c in ASEAN_COUNTRIES if c not in df[COUNTRY_COL].values]
if missing_countries:
    print(f"\nWARNING: These countries were not found in the dataset: {missing_countries}")
    print("Proceeding with available countries only.")
    ASEAN_COUNTRIES = found
print(f"ASEAN countries confirmed: {ASEAN_COUNTRIES}")


# INITIAL INSPECTION


print('\n' + '=' * 64)
print('SECTION 2: INITIAL INSPECTION')
print('=' * 64)

print("\nColumn data types (working columns only):")
print(df[required_cols].dtypes.to_string())

null_count = df[TARGET_COL].isna().sum()
total_rows  = len(df)
print(f"\nMissing values in '{TARGET_COL}': {null_count} / {total_rows} "
      f"({null_count / total_rows * 100:.1f}%)")

print(f"\nDescriptive statistics for '{TARGET_COL}' (full dataset, all countries):")
print(df[TARGET_COL].describe().round(4).to_string())


# DATA CLEANING

print('\n' + '=' * 64)
print('SECTION 3: DATA SELECTION & CLEANING')
print('=' * 64)

# Five ASEAN countries
df_asean = df[required_cols].copy()
df_asean = df_asean[df_asean[COUNTRY_COL].isin(ASEAN_COUNTRIES)]
df_asean.dropna(subset=[TARGET_COL], inplace=True)
df_asean[YEAR_COL] = df_asean[YEAR_COL].astype(int)

# Year coverage inspection
print("\nYear coverage per country (non-null observations for renewables_share_elec):")
coverage = df_asean.groupby(COUNTRY_COL)[YEAR_COL].agg(
    first_year='min', last_year='max', n_obs='count'
).reindex(ASEAN_COUNTRIES)
print(coverage.to_string())

latest_start = int(coverage['first_year'].max())
earliest_end  = int(coverage['last_year'].min())

if coverage['first_year'].nunique() > 1:
    print(f"\nCoverage start years differ across countries.")
    print(f"  -> Applying year filter: {latest_start} to {earliest_end}")
    print(f"  Justification: Aligns all countries to the same temporal window,")
    print(f"  ensuring the decade groups are balanced and comparable across nations.")
    df_asean = df_asean[
        (df_asean[YEAR_COL] >= latest_start) &
        (df_asean[YEAR_COL] <= earliest_end)
    ]
else:
    print(f"\nAll countries share the same start year ({latest_start}). No year filter needed.")

print(f"\nWorking dataset after cleaning: {df_asean.shape[0]} rows")

# Create decade column
df_asean[DECADE_COL] = (df_asean[YEAR_COL] // 10 * 10).astype(str) + 's'

# Treat factors as categorical
df_asean[COUNTRY_COL] = pd.Categorical(df_asean[COUNTRY_COL], categories=ASEAN_COUNTRIES, ordered=False)
decade_order = sorted(df_asean[DECADE_COL].unique())
df_asean[DECADE_COL] = pd.Categorical(df_asean[DECADE_COL], categories=decade_order, ordered=True)

df_asean.reset_index(drop=True, inplace=True)

df_work = df_asean.rename(columns={
    TARGET_COL:  'renewables_share_elec',
    COUNTRY_COL: 'country',
    DECADE_COL:  'decade',
})

# Descriptive statistics by country
print("\nDescriptive statistics by country:")
desc = df_work.groupby('country', observed=True)['renewables_share_elec'].agg(
    count='count', mean='mean', std='std', min='min', max='max'
).round(2)
print(desc.to_string())

# Balance check: country x decade cell sizes 
print("\nCell counts (country x decade) - balance check:")
balance = df_work.groupby(['country', 'decade'], observed=True)['renewables_share_elec'] \
                 .count().unstack(fill_value=0)
print(balance.to_string())

sparse_cells = (balance < MIN_CELL_N) & (balance > 0)
empty_cells  = balance == 0
has_sparse   = sparse_cells.any().any()
has_empty    = empty_cells.any().any()

if has_empty:
    print(f"\nWARNING: Some country-decade cells have ZERO observations.")
    print("  This can cause rank-deficiency in the two-way ANOVA model.")
if has_sparse:
    print(f"\nWARNING: Some cells have fewer than {MIN_CELL_N} observations.")
    print("  Interpret two-way ANOVA interaction results with caution.")
if not has_sparse and not has_empty:
    print(f"\nBalance check: All cells have >= {MIN_CELL_N} observations. Proceeding.")

if '2020s' in decade_order:
    n_2020s = int(balance['2020s'].sum()) if '2020s' in balance.columns else 0
    print(f"\nNote: The 2020s decade contains only {n_2020s} observations "
          "(partial decade, 2020-2023). Results for this period should be interpreted cautiously.")

# ASSUMPTION CHECKS

print('\n' + '=' * 64)
print('SECTION 4: ASSUMPTION CHECKS')
print('=' * 64)

# Normality (Shapiro-Wilk per country group) 
print("\n4a. Normality Test - Shapiro-Wilk (per country group):")
print(f"    H0: Data within each group is normally distributed")
print(f"    {'Country':<15} {'n':>5}  {'W-stat':>8}  {'p-value':>8}  Result")
print(f"    {'-'*55}")

normality_results = {}
for country in ASEAN_COUNTRIES:
    vals = df_work[df_work['country'] == country]['renewables_share_elec'].values
    n = len(vals)
    if n < 3:
        print(f"    {country:<15} {n:>5}  {'N/A':>8}  {'N/A':>8}  SKIP (n < 3)")
        continue
    w_stat, p_val = stats.shapiro(vals)
    result = 'PASS' if p_val >= ALPHA else 'FAIL'
    normality_results[country] = p_val
    print(f"    {country:<15} {n:>5}  {w_stat:>8.4f}  {p_val:>8.4f}  {result}")

n_fail_norm = sum(1 for p in normality_results.values() if p < ALPHA)
print(f"\n    {n_fail_norm}/{len(normality_results)} groups fail normality (p < {ALPHA}).")

# Levene's Test for homogeneity of variance
print("\n4b. Levene's Test - Homogeneity of Variance (across all countries):")
print(f"    H0: Variances are equal across all country groups")

country_groups = [
    df_work[df_work['country'] == c]['renewables_share_elec'].values
    for c in ASEAN_COUNTRIES
    if len(df_work[df_work['country'] == c]) >= 2
]
lev_stat, lev_p = stats.levene(*country_groups, center='mean')
lev_result = 'PASS' if lev_p >= ALPHA else 'FAIL'
print(f"    Levene statistic: {lev_stat:.4f} | p-value: {lev_p:.4f} | Result: {lev_result}")
if lev_p < ALPHA:
    print("    Variances differ across groups.")
else:
    print("    Equal variance assumption holds.")

print("\n    NOTE: Proceeding with ANOVA as a lab demonstration regardless of")
print("    assumption results. ANOVA is reasonably robust to moderate violations")
print("    when group sizes are similar.")

# ONE-WAY ANOVA

print('\n' + '=' * 64)
print('SECTION 5: ONE-WAY ANOVA')
print('=' * 64)
print("\nResearch question:")
print("  Does country identity significantly affect renewable electricity share")
print("  among ASEAN nations?")
print(f"\n  H0: Mean renewables_share_elec is equal across all ASEAN countries")
print(f"  H1: At least one country has a significantly different mean")

groups_by_country = [
    df_work[df_work['country'] == c]['renewables_share_elec'].values
    for c in ASEAN_COUNTRIES
    if len(df_work[df_work['country'] == c]) >= 2
]

f_stat, p_val = stats.f_oneway(*groups_by_country)

print(f"\nOne-Way ANOVA Results:")
print(f"  Factor (IV) : Country ({len(groups_by_country)} levels: {', '.join(ASEAN_COUNTRIES)})")
print(f"  Dependent   : renewables_share_elec (%)")
print(f"  F-statistic : {f_stat:.4f}")
print(f"  p-value     : {p_val:.4f}")
print(f"  Alpha       : {ALPHA}")

if p_val < ALPHA:
    print(f"\n  Decision: REJECT H0")
    print(f"  Interpretation: There is a statistically significant difference in mean")
    print(f"  renewable electricity share across ASEAN countries")
    print(f"  (F = {f_stat:.4f}, p = {p_val:.4f} < alpha = {ALPHA}).")
    print(f"  Country identity is a significant factor in explaining variation.")
else:
    print(f"\n  Decision: FAIL TO REJECT H0")
    print(f"  Interpretation: No statistically significant difference detected across")
    print(f"  countries at alpha = {ALPHA} (F = {f_stat:.4f}, p = {p_val:.4f}).")

# Tukey HSD
if p_val < ALPHA:
    print("\n  Post-hoc Analysis: Tukey HSD (pairwise comparisons)")
    print("  Pairs marked 'True' under 'reject' have significantly different means.\n")
    tukey = pairwise_tukeyhsd(
        endog=df_work['renewables_share_elec'],
        groups=df_work['country'],
        alpha=ALPHA
    )
    print(tukey.summary())


# TWO-WAY ANOVA 

print('\n' + '=' * 64)
print('SECTION 6: TWO-WAY ANOVA (WITH INTERACTION)')
print('=' * 64)
print("\nResearch question:")
print("  Do country identity, decade, and their interaction significantly affect")
print("  renewable electricity share among ASEAN nations?")

formula = 'renewables_share_elec ~ C(country) + C(decade) + C(country):C(decade)'
print(f"\n  Formula : {formula}")
print(f"  SS Type : III (tests each effect after accounting for all others,")
print(f"            recommended when the model includes an interaction term)")

model = ols(formula, data=df_work).fit()
anova_table = anova_lm(model, typ=3)

print("\nTwo-Way ANOVA Table (Type III SS):")
print(anova_table.round(4).to_string())

print(f"\nEffect-by-effect interpretation (alpha = {ALPHA:.2f}):")
effect_labels = {
    'C(country)':            'Country effect            ',
    'C(decade)':             'Decade effect             ',
    'C(country):C(decade)':  'Country x Decade interaction',
}
for key, label in effect_labels.items():
    if key in anova_table.index:
        row   = anova_table.loc[key]
        f_val = row.get('F', float('nan'))
        p_row = row.get('PR(>F)', float('nan'))
        sig   = 'SIGNIFICANT' if p_row < ALPHA else 'NOT SIGNIFICANT'
        print(f"  {label}: F = {f_val:>8.4f}, p = {p_row:.4f}  =>  {sig}")

# Interaction interpretation
interaction_p = anova_table.loc['C(country):C(decade)', 'PR(>F)'] \
    if 'C(country):C(decade)' in anova_table.index else float('nan')

print()
if interaction_p < ALPHA:
    print("  Interaction is SIGNIFICANT: The rate of change in renewable electricity")
    print("  share over decades differs across ASEAN countries. Some nations")
    print("  accelerated their transition while others stagnated -- this divergence")
    print("  cannot be detected from either main effect alone.")
else:
    print("  Interaction is NOT SIGNIFICANT: The trajectory of change in renewable")
    print("  electricity share over decades is broadly similar across ASEAN countries.")
    print("  Main effects can be interpreted independently.")

if has_sparse or has_empty:
    print("\n  CAUTION: Sparse or empty country-decade cells were detected (Section 3).")
    print("  The interaction term estimates may be unreliable. Interpret with caution.")


# VISUALISATIONS

print('\n' + '=' * 64)
print('SECTION 7: VISUALISATIONS')
print('=' * 64)

# Plot 1: Boxplot (one-way ANOVA)
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=df_work,
    x='country',
    y='renewables_share_elec',
    order=ASEAN_COUNTRIES,
    palette='Set2',
    ax=ax
)
ax.set_title(
    'Renewable Electricity Share by ASEAN Country\n'
    '(One-Way ANOVA: Country as Factor)',
    fontsize=13
)
ax.set_xlabel('Country', fontsize=11)
ax.set_ylabel('Renewable Electricity Share (%)', fontsize=11)
ax.tick_params(axis='x', labelsize=10)
plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, 'plot1_oneway_anova_boxplot.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot 1 saved: {plot1_path}")

# Plot 2: Interaction line plot (two-way ANOVA)
interaction_data = (
    df_work.groupby(['decade', 'country'], observed=True)['renewables_share_elec']
           .mean()
           .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=interaction_data,
    x='decade',
    y='renewables_share_elec',
    hue='country',
    hue_order=ASEAN_COUNTRIES,
    marker='o',
    linewidth=2,
    ax=ax
)
ax.set_title(
    'Mean Renewable Electricity Share by Decade and Country\n'
    '(Two-Way ANOVA: Interaction Plot)',
    fontsize=13
)
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Mean Renewable Electricity Share (%)', fontsize=11)
ax.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plot2_path = os.path.join(OUTPUT_DIR, 'plot2_twoway_anova_interaction.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 2 saved: {plot2_path}")

print('=' * 64)
print('Analysis complete. Outputs saved to:', OUTPUT_DIR)
print('=' * 64)
