import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("MoveData")
sys.path.append("autoSfM")
sys.path.append("SemiF-AnnotationPipeline")
sys.path.append("SemiF-SyntheticPipeline")

from pathlib import Path

# cutouts = Path("SemiF-AnnotationPipeline/data/semifield-cutouts").rglob(
# "*.csv")

# dfs = [pd.read_csv(x, low_memory=False) for x in cutouts]
# df = pd.concat(dfs, ignore_index=True)
df = pd.read_csv("weed_cutouts.csv", low_memory=False)

pdf = df.copy()
pdf = pdf[pdf["is_primary"] == True]
pdf["state_id"] = pdf["batch_id"].str.split("_", expand=True).iloc[:, 0]

###
dfgb = pdf.groupby([
    'common_name', 'state_id'
])['cutout_id'].count().reset_index(name='Count').sort_values(['Count'],
                                                              ascending=False)
plt.style.use('seaborn')
g = sns.catplot(data=dfgb,
                y="common_name",
                x="Count",
                kind="bar",
                col="state_id",
                sharey=False,
                margin_titles=True)
g.fig.set_size_inches(15, 8)
g.fig.subplots_adjust(top=0.9)

g.fig.suptitle('Unique cutout count')
g.set_titles("{row_name}")

# iterate through axes
for ax in g.axes.ravel():
    title = ax.get_title().split("=")[1]
    ax.set_title(title)
    # add annotations
    for c in ax.containers:
        labels = [f'{int(v.get_width())}' for v in c]
        ax.bar_label(
            c,
            labels=labels,
            label_type="edge",
        )
g.set_axis_labels("", "")
plt.tight_layout()

plt.savefig("primary_cutout_count.png", dpi=100)
plt.show()