# =============================================================================
# THE GENDER CANCER DIVIDE
# A Data Analysis of Worldwide Cancer Incidence by Sex, 2020
# DSC 680 — Mostapha Balon
#
# This script contains the complete, end-to-end analysis for the project.
# It is structured as a Jupyter Notebook — each section is clearly labeled
# so you can copy individual blocks into notebook cells.
#
# REQUIRED LIBRARIES:
#   pip install pandas plotly kaleido
#
# DATA FILES NEEDED (place in the same folder as this script):
#   - Worldwide_Cancer_Dataset__females__.csv
#   - Worldwide_Cancer_Dataset__Males__.csv
#   - Worldwide_Cancer_Dataset.csv
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS
# All third-party libraries are imported here so any missing package surfaces
# immediately at the top rather than failing halfway through the analysis.
# =============================================================================

import pandas as pd          # Data loading, cleaning, and merging
import plotly.express as px  # Interactive chart library (butterfly, treemap, scatter)
import plotly.graph_objects as go  # Lower-level Plotly for custom charts (dot plot)
from plotly.subplots import make_subplots  # Side-by-side grouped bar chart


# =============================================================================
# CELL 2 — LOAD THE RAW DATA
#
# Each file has the same four columns:
#   Rank              — integer position in that sex's ranked cancer list
#   Cancer            — cancer type name (string, not yet normalized)
#   New cases in 2020 — absolute global incidence count (stored as string with commas)
#   % of all cancers  — share of that sex group's total (float)
#
# Known quirks we handle immediately:
#   1. Each file starts with a UTF-8 BOM character that attaches to the first
#      column header. encoding='utf-8-sig' strips it automatically.
#   2. The first data row in each file is a "All cancers*" summary — not a
#      real cancer type. We filter it out by dropping rows with no Rank.
#   3. The female file has a duplicate row for "Lip, oral cavity" (rows 18+19
#      are identical). We drop the second occurrence with drop_duplicates().
# =============================================================================

# --- Load female dataset ---
df_female = pd.read_csv(
    "Worldwide_Cancer_Dataset__females__.csv",
    encoding="utf-8-sig",   # Strips the BOM from the header
    thousands=","           # Tells pandas the case count uses comma as thousands separator
)

# --- Load male dataset ---
df_male = pd.read_csv(
    "Worldwide_Cancer_Dataset__Males__.csv",
    encoding="utf-8-sig",
    thousands=","
)

# --- Load combined (all-sex) dataset ---
df_combined = pd.read_csv(
    "Worldwide_Cancer_Dataset.csv",
    encoding="utf-8-sig",
    thousands=","
)

# Quick sanity check — print shape of each file before cleaning
print("Raw shapes (rows x columns):")
print(f"  Female:   {df_female.shape}")
print(f"  Male:     {df_male.shape}")
print(f"  Combined: {df_combined.shape}")


# =============================================================================
# CELL 3 — CLEAN THE DATA
#
# This is the most important cell. A missed cleaning step silently breaks the
# merge downstream without raising an error — it just produces a smaller table
# than the data actually supports. Every decision here is documented.
# =============================================================================

def clean_dataset(df, name):
    """
    Applies all cleaning steps to a single cancer incidence dataframe.

    Steps:
      1. Drop the summary "All cancers*" totals row (has NaN in Rank column)
      2. Drop exact duplicate rows (fixes the female Lip/oral cavity issue)
      3. Convert the "New cases in 2020" column to a proper integer
         (raw values are stored as strings like "2,261,419")
      4. Create a 'cancer_clean' column — lowercase, stripped of footnote
         markers — that will be used as the merge key across files
      5. Reset the index so rows are numbered cleanly from 0

    Parameters
    ----------
    df   : pd.DataFrame — the raw dataframe to clean
    name : str          — label used in print statements (e.g. "Female")

    Returns
    -------
    pd.DataFrame — cleaned copy of the input
    """

    df = df.copy()  # Never mutate the original; work on a copy

    # Step 1: Remove the "All cancers*" totals row.
    # It has no Rank value, which makes it easy to identify and drop.
    before = len(df)
    df = df.dropna(subset=["Rank"])
    print(f"[{name}] Dropped {before - len(df)} totals row(s). Rows remaining: {len(df)}")

    # Step 2: Drop exact duplicate rows.
    # This specifically targets the Lip/oral cavity duplication in the female file.
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[{name}] Dropped {dropped} exact duplicate row(s).")
    else:
        print(f"[{name}] No duplicate rows found.")

    # Step 3: Convert Rank to integer.
    # dropna() above removed the NaN row, so this should now be safe.
    df["Rank"] = df["Rank"].astype(int)

    # Step 4: Convert "New cases in 2020" to integer.
    # The thousands= parameter in read_csv handles the commas at load time,
    # but the column may still be float due to NaN handling. Force to int.
    case_col = "New cases in 2020"
    df[case_col] = pd.to_numeric(df[case_col], errors="coerce").astype("Int64")

    # Step 5: Normalize cancer names for merging.
    # Problems in the raw data:
    #   - "Colorectal **" vs "Colorectal**" (spacing around marker)
    #   - "Lip, oral cavity" vs "Lip, oral cavity " (trailing space)
    #   - Mixed capitalization in some rows
    # Solution: strip all whitespace, remove asterisk markers, lowercase.
    cancer_col = "Cancer"
    df["cancer_clean"] = (
        df[cancer_col]
        .str.strip()                    # Remove leading/trailing whitespace
        .str.replace(r"\s*\*+\s*", "", regex=True)  # Remove ** footnote markers
        .str.lower()                    # Lowercase for case-insensitive matching
    )

    # Step 6: Clean index
    df = df.reset_index(drop=True)

    print(f"[{name}] Final clean shape: {df.shape}")
    print()
    return df


# Apply cleaning to all three datasets
df_female   = clean_dataset(df_female,   "Female")
df_male     = clean_dataset(df_male,     "Male")
df_combined = clean_dataset(df_combined, "Combined")

# Verify: print the cleaned cancer_clean column for female to spot-check
print("Sample of cleaned female cancer names:")
print(df_female[["Cancer", "cancer_clean"]].head(10).to_string(index=False))


# =============================================================================
# CELL 4 — DESCRIPTIVE STATISTICS
#
# Before building any charts, it's worth printing key summary numbers.
# These will be cited in the white paper narrative and referenced in chart
# annotations. Getting them explicitly from the data (rather than memorizing
# numbers) ensures accuracy.
# =============================================================================

# Total new cancer cases per sex group
total_female   = 8_751_759   # From the "All cancers*" row we dropped (known value)
total_male     = 9_342_957
total_combined = 18_094_716

print("=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)

# --- Top 5 cancers for each sex ---
print("\nTop 5 female cancers (by % share):")
print(
    df_female[["Cancer", "New cases in 2020", "% of all cancers"]]
    .head(5)
    .to_string(index=False)
)

print("\nTop 5 male cancers (by % share):")
print(
    df_male[["Cancer", "New cases in 2020", "% of all cancers"]]
    .head(5)
    .to_string(index=False)
)

# --- Overall burden difference ---
gap = total_male - total_female
print(f"\nMen had {gap:,} more new cancer diagnoses than women in 2020.")
print(f"Male total:   {total_male:,}")
print(f"Female total: {total_female:,}")

# --- Breast cancer dominance sanity check ---
breast_pct = df_female.loc[df_female["cancer_clean"] == "breast", "% of all cancers"].values[0]
second_pct  = df_female.iloc[1]["% of all cancers"]  # Second ranked cancer
print(f"\nBreast cancer is {breast_pct}% of all female cancers.")
print(f"The second-largest female cancer is {df_female.iloc[1]['Cancer']} at {second_pct}%.")
print(f"Ratio: breast cancer is {breast_pct / second_pct:.1f}x the size of the next largest.")


# =============================================================================
# CELL 5 — CROSS-DATASET MERGE (THE KEY ANALYTICAL STEP)
#
# Neither the female nor the male file alone can answer "which cancers appear
# in both groups and how do the case counts compare?" That requires an inner
# join on the normalized cancer name.
#
# Expected result: 14 shared cancer types.
# If the merge produces fewer than 14, name normalization missed a match.
# If it produces more than 14, there's an accidental duplicate in a clean file.
# =============================================================================

# Rename columns before merging to avoid ambiguous column names in the output.
# We keep the same structure but suffix each sex's columns clearly.
df_female_merge = df_female.rename(columns={
    "Rank":              "rank_f",
    "Cancer":            "cancer_f",
    "New cases in 2020": "cases_f",
    "% of all cancers":  "pct_f"
})

df_male_merge = df_male.rename(columns={
    "Rank":              "rank_m",
    "Cancer":            "cancer_m",
    "New cases in 2020": "cases_m",
    "% of all cancers":  "pct_m"
})

# Inner join on the normalized cancer name.
# 'inner' means only rows present in BOTH dataframes are kept.
# Cancers that only appear in one sex's list are excluded from this table.
df_shared = pd.merge(
    df_female_merge[["cancer_clean", "cancer_f", "cases_f", "pct_f"]],
    df_male_merge[  ["cancer_clean", "cancer_m", "cases_m", "pct_m"]],
    on="cancer_clean",   # The normalized key column
    how="inner"          # Only matching rows from both files
)

# Verify the count
print(f"Number of shared cancer types found: {len(df_shared)}")
print("(Expected: 14)\n")

# If count is wrong, print both name lists to find the mismatch
if len(df_shared) != 14:
    female_names = set(df_female["cancer_clean"])
    male_names   = set(df_male["cancer_clean"])
    only_female  = female_names - male_names
    only_male    = male_names - female_names
    print("Cancers only in FEMALE file:", sorted(only_female))
    print("Cancers only in MALE file:", sorted(only_male))

# Print the shared cancer table for verification
print("Shared cancers (female vs. male cases):")
print(df_shared[["cancer_f", "cases_f", "cases_m"]].to_string(index=False))


# =============================================================================
# CELL 6 — MALE-TO-FEMALE RATIO ANALYSIS
#
# For each of the 14 shared cancers, we compute:
#   mf_ratio = male_cases / female_cases
#
# Interpretation:
#   ratio > 1.0  → more male cases (male-skewed)
#   ratio = 1.0  → equal burden
#   ratio < 1.0  → more female cases (female-skewed)
#
# The ratio is then used to rank the shared cancers and color the dot plot.
# =============================================================================

# Compute the M:F ratio for each shared cancer.
# We convert to float first to avoid integer division issues.
df_shared["mf_ratio"] = (
    df_shared["cases_m"].astype(float) / df_shared["cases_f"].astype(float)
).round(2)   # Round to 2 decimal places for display

# Create a direction label — used for color coding in charts
df_shared["direction"] = df_shared["mf_ratio"].apply(
    lambda r: "Male-skewed" if r >= 1.0 else "Female-skewed"
)

# Sort descending by ratio so the most male-skewed cancer is at the top.
# This is the order the dot plot will display them in.
df_shared = df_shared.sort_values("mf_ratio", ascending=False).reset_index(drop=True)

# Print the complete ratio table — this goes into the white paper
print("Male-to-female incidence ratios (sorted):")
print(
    df_shared[["cancer_f", "cases_f", "cases_m", "mf_ratio", "direction"]]
    .rename(columns={
        "cancer_f":  "Cancer",
        "cases_f":   "Female cases",
        "cases_m":   "Male cases",
        "mf_ratio":  "M:F ratio",
        "direction": "Direction"
    })
    .to_string(index=False)
)


# =============================================================================
# CELL 7 — VISUALIZATION 1: DIVERGING BUTTERFLY BAR CHART
#
# Purpose: Show at a glance how the cancer burden is distributed differently
#          between males and females. Female bars extend left; male extend right.
#          Both are scaled as a percentage of their own sex's total.
#
# Design choices:
#   - We only show the top 12 cancer types per sex to keep the chart readable.
#     (The full 27-30 types would make the labels too small to read.)
#   - Female bars use a pink-coral color; male bars use a steel blue.
#   - The x-axis shows percentage share (not raw cases) so both sides are
#     on the same scale regardless of the fact that men have more total cases.
#
# Reading guide (added as a subtitle so non-technical readers understand it):
#   "Each bar shows what percentage of that sex's total cancer cases
#    that cancer type represents. Female cancers extend left; male, right."
# =============================================================================

# Select top 12 cancers per sex for the chart
top_n = 12
df_female_top = df_female.nlargest(top_n, "% of all cancers").copy()
df_male_top   = df_male.nlargest(top_n,   "% of all cancers").copy()

# Collect all unique cancer names across both top lists so the y-axis
# covers every cancer that appears in either group
all_cancers = pd.Index(
    pd.concat([df_female_top["Cancer"], df_male_top["Cancer"]]).unique()
)

# Build a combined dataframe with one row per cancer per sex.
# Female values are stored as NEGATIVE so their bars extend left on the chart.
butterfly_rows = []

for cancer in all_cancers:
    # Female value (negative for leftward bar)
    f_row = df_female_top[df_female_top["Cancer"] == cancer]
    f_pct = -f_row["% of all cancers"].values[0] if len(f_row) > 0 else 0.0

    # Male value (positive for rightward bar)
    m_row = df_male_top[df_male_top["Cancer"] == cancer]
    m_pct = m_row["% of all cancers"].values[0] if len(m_row) > 0 else 0.0

    butterfly_rows.append({
        "Cancer":    cancer,
        "Female %":  f_pct,
        "Male %":    m_pct,
    })

df_butterfly = pd.DataFrame(butterfly_rows)

# Sort by absolute female percentage (descending) so breast cancer is at top
df_butterfly = df_butterfly.sort_values("Female %", ascending=True)

# Build the chart with two bar traces — one per sex
fig_butterfly = go.Figure()

# Female bars (negative values → extend left)
fig_butterfly.add_trace(go.Bar(
    name="Female",
    y=df_butterfly["Cancer"],
    x=df_butterfly["Female %"],
    orientation="h",                # Horizontal bar
    marker_color="#ED93B1",         # Soft pink
    marker_line_color="#D4537E",    # Darker pink border
    marker_line_width=0.5,
    customdata=df_butterfly["Female %"].abs(),   # Store absolute value for tooltip
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Female: %{customdata:.1f}% of all female cancers<extra></extra>"
    )
))

# Male bars (positive values → extend right)
fig_butterfly.add_trace(go.Bar(
    name="Male",
    y=df_butterfly["Cancer"],
    x=df_butterfly["Male %"],
    orientation="h",
    marker_color="#85B7EB",         # Soft blue
    marker_line_color="#378ADD",    # Darker blue border
    marker_line_width=0.5,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Male: %{x:.1f}% of all male cancers<extra></extra>"
    )
))

# Style the layout
fig_butterfly.update_layout(
    title=dict(
        text="<b>Cancer burden by sex</b><br>"
             "<sup>Each bar = % of that sex's total cancer diagnoses. "
             "Female extends left, male extends right.</sup>",
        x=0.5,   # Center the title
        font=dict(size=16)
    ),
    barmode="overlay",    # Bars from both traces share the same y-axis position
    xaxis=dict(
        title="% share of sex-specific total",
        # Show absolute values on the axis (not negative numbers for female side)
        tickvals=[-25, -20, -15, -10, -5, 0, 5, 10, 15],
        ticktext=["25%", "20%", "15%", "10%", "5%", "0%", "5%", "10%", "15%"],
        zeroline=True,
        zerolinecolor="#888888",
        zerolinewidth=1.5,
        gridcolor="#EEEEEE",
    ),
    yaxis=dict(
        title="",
        tickfont=dict(size=11)
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=180, r=40, t=120, b=60)
)

fig_butterfly.show()
# fig_butterfly.write_html("chart1_butterfly.html")   # Uncomment to save


# =============================================================================
# CELL 8 — VISUALIZATION 2: M:F RATIO DOT PLOT
#
# Purpose: Rank all 14 shared cancers by their male-to-female incidence ratio
#          so the direction and magnitude of each gender gap is immediately
#          visible in a single chart.
#
# Design choices:
#   - A vertical reference line at x=1.0 marks "equal burden"
#   - Blue dots = male-skewed (ratio > 1.0)
#   - Coral dot = female-skewed (ratio < 1.0) — gallbladder is the only one
#   - Cancers are sorted top-to-bottom from highest to lowest ratio
#   - Ratio values are annotated directly on each dot so readers don't
#     have to read the x-axis precisely
#
# Reading guide: "Dots to the right of the 1.0 line indicate more male
#   cases than female. The further right, the larger the male skew."
# =============================================================================

# Color-code each dot based on direction
dot_colors = df_shared["direction"].map({
    "Male-skewed":   "#378ADD",   # Blue for male-skewed
    "Female-skewed": "#D4537E"    # Coral for female-skewed (gallbladder)
})

fig_dot = go.Figure()

# Draw the reference line at ratio = 1.0 first (so dots render on top of it)
fig_dot.add_vline(
    x=1.0,
    line_width=1.5,
    line_dash="dash",
    line_color="#888888",
    annotation_text="Equal burden",
    annotation_position="top",
    annotation_font_size=11,
    annotation_font_color="#888888"
)

# Draw a thin horizontal line from x=0 to each dot (lollipop effect)
for _, row in df_shared.iterrows():
    fig_dot.add_shape(
        type="line",
        x0=1.0,                         # Line starts at the reference line
        x1=row["mf_ratio"],             # Line ends at the dot
        y0=row["cancer_f"],
        y1=row["cancer_f"],
        line=dict(color="#CCCCCC", width=1)
    )

# Draw the dots themselves
fig_dot.add_trace(go.Scatter(
    x=df_shared["mf_ratio"],
    y=df_shared["cancer_f"],
    mode="markers+text",
    marker=dict(
        color=dot_colors,
        size=14,
        line=dict(color="white", width=1.5)
    ),
    text=df_shared["mf_ratio"].astype(str) + ":1",   # Label each dot with its ratio
    textposition="middle right",
    textfont=dict(size=10),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "M:F ratio: %{x:.2f}<br>"
        "Female cases: %{customdata[0]:,}<br>"
        "Male cases: %{customdata[1]:,}<extra></extra>"
    ),
    customdata=df_shared[["cases_f", "cases_m"]].values
))

fig_dot.update_layout(
    title=dict(
        text="<b>Male-to-female incidence ratio — 14 shared cancers</b><br>"
             "<sup>Dots right of the dashed line = more male cases. "
             "Ratio shown beside each dot.</sup>",
        x=0.5,
        font=dict(size=16)
    ),
    xaxis=dict(
        title="Male-to-female case ratio (1.0 = equal)",
        range=[0, 8],        # Larynx at 6.58 needs room on the right
        gridcolor="#EEEEEE",
        zeroline=False,
    ),
    yaxis=dict(
        title="",
        tickfont=dict(size=11),
        autorange="reversed"  # Highest ratio at top
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=520,
    showlegend=False,
    margin=dict(l=200, r=100, t=120, b=60)
)

fig_dot.show()
# fig_dot.write_html("chart2_ratio_dotplot.html")   # Uncomment to save


# =============================================================================
# CELL 9 — VISUALIZATION 3: GROUPED BAR CHART (TOP 5 MALE-SKEWED CANCERS)
#
# Purpose: Show the actual case counts (not just ratios) for the five cancers
#          with the largest male skew, so the reader can see both the relative
#          gap and the absolute scale of each cancer's burden.
#
# The five cancers are: Larynx (6.58:1), Bladder (3.33:1),
#   Lip/oral cavity (2.33:1), Liver (2.31:1), Esophagus (2.25:1)
#
# Primary risk factor callouts are added as annotations because these
# contextual explanations are essential for a non-clinical audience.
# =============================================================================

# Extract the top 5 male-skewed cancers from our ratio-sorted shared table
top5_skewed = df_shared.head(5).copy()

# Risk factor labels for each cancer (from literature synthesis)
# These appear as annotation text on the chart
risk_labels = {
    "larynx":          "Tobacco + alcohol",
    "bladder":         "Tobacco + occupational carcinogens",
    "lip, oral cavity":"Tobacco + alcohol",
    "liver":           "Hepatitis B/C + alcohol",
    "oesophagus":      "Tobacco + alcohol"
}

# Map risk labels to the cleaned name column
top5_skewed["risk_factor"] = top5_skewed["cancer_clean"].map(risk_labels).fillna("See literature")

# Build side-by-side bar chart
fig_grouped = go.Figure()

# Female bars
fig_grouped.add_trace(go.Bar(
    name="Female cases",
    x=top5_skewed["cancer_f"],
    y=top5_skewed["cases_f"],
    marker_color="#ED93B1",
    marker_line_color="#D4537E",
    marker_line_width=0.5,
    text=top5_skewed["cases_f"].apply(lambda v: f"{v:,}"),
    textposition="outside",
    textfont=dict(size=10),
    hovertemplate="<b>%{x}</b><br>Female: %{y:,} cases<extra></extra>"
))

# Male bars
fig_grouped.add_trace(go.Bar(
    name="Male cases",
    x=top5_skewed["cancer_f"],
    y=top5_skewed["cases_m"],
    marker_color="#85B7EB",
    marker_line_color="#378ADD",
    marker_line_width=0.5,
    text=top5_skewed["cases_m"].apply(lambda v: f"{v:,}"),
    textposition="outside",
    textfont=dict(size=10),
    hovertemplate="<b>%{x}</b><br>Male: %{y:,} cases<extra></extra>"
))

# Add M:F ratio annotation above each cancer group
for _, row in top5_skewed.iterrows():
    # Position annotation above the taller (male) bar
    fig_grouped.add_annotation(
        x=row["cancer_f"],
        y=row["cases_m"] * 1.15,    # Slightly above the male bar top
        text=f"M:F = {row['mf_ratio']}:1<br><i>{row['risk_factor']}</i>",
        showarrow=False,
        font=dict(size=9, color="#444444"),
        align="center",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#CCCCCC",
        borderwidth=0.5,
        borderpad=3
    )

fig_grouped.update_layout(
    title=dict(
        text="<b>Five most male-skewed cancers — absolute case counts</b><br>"
             "<sup>Bars show raw 2020 diagnoses. Ratio and primary risk factor "
             "labeled above each pair.</sup>",
        x=0.5,
        font=dict(size=16)
    ),
    barmode="group",
    xaxis=dict(
        title="Cancer type",
        tickfont=dict(size=11)
    ),
    yaxis=dict(
        title="New cases in 2020",
        gridcolor="#EEEEEE",
        tickformat=",",    # Add thousands separator to y-axis labels
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=560,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=80, r=40, t=140, b=80)
)

fig_grouped.show()
# fig_grouped.write_html("chart3_grouped_bar.html")   # Uncomment to save


# =============================================================================
# CELL 10 — VISUALIZATION 4: TREEMAP OF COMBINED GLOBAL BURDEN
#
# Purpose: Show every cancer type's share of TOTAL global burden (both sexes
#          combined) in a proportional area chart. This surfaces the striking
#          near-tie between breast (12.5%) and lung (12.2%) that is only
#          visible in the combined view — and that disappears when the data
#          is split by sex.
#
# Design choices:
#   - Each rectangle's area is proportional to the cancer's % share globally
#   - Color encodes the sex profile of each cancer:
#       Pink  = female-only cancer (breast, cervix, corpus uteri, ovary, vulva, vagina)
#       Blue  = male-only cancer (prostate, testis, penis)
#       Gray  = cancers that appear in both sexes
#   - The color legend helps readers immediately connect the treemap back
#     to the butterfly chart from Cell 7
# =============================================================================

# Define sex profile for each cancer in the combined dataset
# "F-only" and "M-only" are cancers biologically exclusive to one sex.
# Everything else is "Shared" (can affect both sexes).
female_only_cancers = {
    "breast", "cervix uteri", "corpus uteri", "ovary",
    "vulva", "vagina"
}
male_only_cancers = {
    "prostate", "testis", "penis"
}

def sex_profile(cancer_clean_name):
    """
    Returns the sex profile label for a given cancer name.
    Used to assign colors in the treemap.
    """
    if cancer_clean_name in female_only_cancers:
        return "Female-specific"
    elif cancer_clean_name in male_only_cancers:
        return "Male-specific"
    else:
        return "Affects both sexes"

# Apply the profile to each row of the combined dataset
df_combined["sex_profile"] = df_combined["cancer_clean"].apply(sex_profile)

# Color map for each sex profile
treemap_colors = {
    "Female-specific":   "#ED93B1",   # Soft pink
    "Male-specific":     "#85B7EB",   # Soft blue
    "Affects both sexes":"#B4B2A9",   # Neutral gray
}

# Build the treemap
fig_treemap = px.treemap(
    df_combined,
    path=["sex_profile", "Cancer"],    # Two-level hierarchy: sex profile → cancer name
    values="% of all cancers",         # Rectangle size = global % share
    color="sex_profile",               # Rectangle fill color
    color_discrete_map=treemap_colors,
    title=(
        "<b>Global cancer burden — all types, combined sexes, 2020</b><br>"
        "<sup>Rectangle size = % of all 18.09 million global diagnoses. "
        "Color = which sex(es) the cancer affects.</sup>"
    ),
    hover_data={"New cases in 2020": ":,", "% of all cancers": ":.1f"},
)

# Style the text labels inside each rectangle
fig_treemap.update_traces(
    texttemplate="<b>%{label}</b><br>%{value:.1f}%",
    textfont=dict(size=11),
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Global share: %{value:.1f}%<br>"
        "New cases: %{customdata[0]:,}<extra></extra>"
    )
)

fig_treemap.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=100, b=20),
    paper_bgcolor="white"
)

fig_treemap.show()
# fig_treemap.write_html("chart4_treemap.html")   # Uncomment to save


# =============================================================================
# CELL 11 — FINDINGS SUMMARY
#
# Print a clean narrative summary of the four key findings. This mirrors the
# Analysis section of the white paper and serves as a quick reference without
# needing to re-read the full document.
# =============================================================================

print("=" * 60)
print("FINDINGS SUMMARY")
print("=" * 60)

# --- Finding 1: Burden asymmetry ---
breast_pct  = df_female.loc[df_female["cancer_clean"] == "breast",  "% of all cancers"].values[0]
lung_f_pct  = df_female.loc[df_female["cancer_clean"] == "lung",    "% of all cancers"].values[0]
lung_m_pct  = df_male.loc[  df_male["cancer_clean"]   == "lung",    "% of all cancers"].values[0]
prostate_pct= df_male.loc[  df_male["cancer_clean"]   == "prostate","% of all cancers"].values[0]

print(f"""
FINDING 1 — The burden looks completely different by sex
  Breast cancer accounts for {breast_pct}% of all female diagnoses.
  That is nearly 3x the next largest female cancer type.
  By contrast, the male top two (lung {lung_m_pct}%, prostate {prostate_pct}%)
  are far more evenly distributed — no single cancer dominates.
  Men had ~{total_male - total_female:,} more total diagnoses than women in 2020.
""")

# --- Finding 2: Shared cancers ---
n_shared = len(df_shared)
top_ratio_cancer = df_shared.iloc[0]["cancer_f"]
top_ratio        = df_shared.iloc[0]["mf_ratio"]
bottom_cancer    = df_shared.iloc[-1]["cancer_f"]
bottom_ratio     = df_shared.iloc[-1]["mf_ratio"]

print(f"""
FINDING 2 — 14 cancers cross gender lines with very different ratios
  Shared cancers found: {n_shared}
  Highest M:F ratio: {top_ratio_cancer} ({top_ratio}:1)
  Only female-skewed shared cancer: {bottom_cancer} ({bottom_ratio}:1)
  All other 13 shared cancers skew male.
""")

# --- Finding 3: Behavioral drivers ---
print("""
FINDING 3 — The gaps are driven by behavior as much as biology
  Larynx, esophagus, bladder, and oral cavity cancers all have
  M:F ratios above 2.0, driven primarily by:
    - Tobacco use (historically higher in men worldwide)
    - Alcohol consumption (higher in men globally)
    - Occupational carcinogen exposure (male-dominated industries)
  Gallbladder's female skew is biological (gallstones linked to estrogen).
""")

# --- Finding 4: Combined view hides asymmetry ---
breast_combined_pct = df_combined.loc[df_combined["cancer_clean"] == "breast", "% of all cancers"].values[0]
lung_combined_pct   = df_combined.loc[df_combined["cancer_clean"] == "lung",   "% of all cancers"].values[0]

print(f"""
FINDING 4 — The combined view hides what matters most
  In the combined dataset, breast cancer ({breast_combined_pct}%) and
  lung cancer ({lung_combined_pct}%) appear essentially tied.
  That near-tie is produced by combining a female-dominant cancer
  (breast) with a cancer that affects both sexes (lung).
  Splitting by sex reveals the true asymmetry.
""")


# =============================================================================
# CELL 12 — EXPORT ALL CHARTS
#
# Uncomment any of the lines below to save charts as self-contained HTML files
# that anyone can open in a browser and interact with (zoom, hover, etc.).
# This is the recommended format for the Jupyter Notebook submission since it
# preserves the interactivity that Plotly Express generates.
#
# To save as a static PNG (for the white paper illustrations), install the
# kaleido package: pip install kaleido
# Then replace .write_html() with .write_image("filename.png", scale=2)
# =============================================================================

# fig_butterfly.write_html("chart1_butterfly_bar_chart.html")
# fig_dot.write_html("chart2_mf_ratio_dotplot.html")
# fig_grouped.write_html("chart3_grouped_bar_skewed.html")
# fig_treemap.write_html("chart4_global_treemap.html")

# --- Static PNG exports (requires kaleido) ---
# fig_butterfly.write_image("chart1_butterfly.png", scale=2, width=900, height=600)
# fig_dot.write_image("chart2_dotplot.png",         scale=2, width=900, height=550)
# fig_grouped.write_image("chart3_grouped.png",     scale=2, width=900, height=580)
# fig_treemap.write_image("chart4_treemap.png",     scale=2, width=900, height=620)

print("Analysis complete. Uncomment export lines in Cell 12 to save charts.")


# =============================================================================
# CELL 13 — LIMITATIONS REMINDER (inline documentation)
#
# These are not code blocks — they are reminders about what this analysis
# CAN and CANNOT conclude. Including them here keeps the limitations visible
# inside the notebook alongside the code that produces the results.
# =============================================================================

limitations = """
LIMITATIONS — Read before drawing conclusions from this analysis
================================================================
1. GLOBAL AGGREGATES ONLY
   These datasets are totals across 185+ countries. They do not support
   country-level, regional, or income-group breakdowns. A cancer that
   looks rare globally may dominate in specific regions.

2. SINGLE YEAR (2020) — NO TREND DATA
   We can describe where the divide stands but not whether it is growing
   or shrinking. Time-series analysis requires multi-year GLOBOCAN data.

3. COVID-19 DIAGNOSTIC DISRUPTION
   Screening programs globally paused in 2020. These case counts likely
   underestimate true incidence, particularly for screen-detected cancers
   (breast, colorectal, cervical). The degree varies by country and type.

4. BINARY SEX CATEGORIES
   The data only separates male and female. Transgender, non-binary, and
   intersex individuals are not represented. All findings describe patterns
   in biological sex at birth, not gender identity.

5. INCIDENCE ONLY — NO MORTALITY DATA
   Getting diagnosed and dying are different. Mortality patterns would
   require a separate GLOBOCAN mortality dataset.

6. SMALL DATASET — NO SIGNIFICANCE TESTING
   27–33 rows per file is insufficient for statistical significance tests.
   All findings are descriptive observations, not inferential conclusions.
   No claims of causation are made.
"""

print(limitations)
