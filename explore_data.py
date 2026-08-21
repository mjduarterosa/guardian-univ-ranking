# Import packages
import pandas as pd
import mplcursors
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load pickle
subject_dataframes = pd.read_pickle("data/data.pkl")

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
dfs = subject_dataframes

TARGET = "guardianScore"
HIGHLIGHT = "Anglia Ruskin"

# assumes all dataframes share the same metric columns from index 7 onward
metrics = list(dfs.values())[0].columns[6:14].to_list()

metrics_labels = ["Satisfied with teaching",
                  "Spend per student/10",
                  "Student to staff ratio",
                  "Career after 15 months",
                  "Value added score/10",
                  "Average entry tariff", 
                  "Satisfied with feedback",                 
                  "Continuation"]

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 6), constrained_layout=True)

for ax, col, label in zip(axes.flat, metrics, metrics_labels):

    # Concatenate this metric's data from every dataframe into one pooled sample
    combined = pd.concat(
        [df[[col, TARGET, "name", "subject"]].assign(source=df_key) for df_key, df in dfs.items()],
        ignore_index=True
    )
     
    # Force numeric dtypes — guards against stray strings/junk in the CSV
    combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined[TARGET] = pd.to_numeric(combined[TARGET], errors="coerce")
    combined = combined.dropna(subset=[col, TARGET])

    x, y = combined[col].values, combined[TARGET].values

    # Fit ONE regression line to the pooled data
    slope, intercept, r_pearson, p_val, se = stats.linregress(x, y)
    r2 = r_pearson ** 2
    r_spearman, _ = stats.spearmanr(x, y)

    # Split highlighted institution from the rest (still pooled across sources)
    is_highlight = combined["name"] == HIGHLIGHT
    rest, hl = combined[~is_highlight], combined[is_highlight]

    # Scatter: rest colored by source dataframe
    sc_rest = ax.scatter(rest[col], rest[TARGET], alpha=0.6, edgecolor="white", s=35,
               color="tab:blue", label="Other Institutions")

    # Highlighted institution always in green, on top
    sc_hl = ax.scatter(hl[col], hl[TARGET], color="green", edgecolor="black",
               s=60, zorder=5, label=HIGHLIGHT)

    cursor_hl = mplcursors.cursor(sc_hl, hover=True)

    def make_hl_annot(sel, hl=hl, col=col):
        row = hl.iloc[sel.index]
        sel.annotation.set_text(
            f"{HIGHLIGHT}\nSubject: {row['subject']}\n{col}: {row[col]:.2f}\n{TARGET}: {row[TARGET]:.2f}"
    )

    cursor_hl.connect("add", make_hl_annot)

    # Regression line fit to ALL combined data
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#b2182b", linewidth=1.5)

    # Annotate stats (computed on pooled data)
    ax.set_title(
        f"{label}\nR²={r2:.2f}  Pearson={r_pearson:.2f}  Spearman={r_spearman:.2f}",
        fontsize=9
    )
    ax.set_xlabel(label, fontsize=8)
    ax.set_ylabel(TARGET, fontsize=8)
    ax.tick_params(labelsize=7)

# Single shared legend for the highlighted institution and each source dataframe
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=10)

plt.tight_layout()
plt.show()

