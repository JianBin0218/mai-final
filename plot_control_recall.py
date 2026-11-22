"""Render the A-track comparison from the verified control metrics."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / "phase2/fresh-controls"
with (root / "metrics.csv").open() as f:
    data = {(r["group"],r["method"]):r for r in csv.DictReader(f)}
fig, axes = plt.subplots(1,2,figsize=(9.2,3.7),layout="constrained")
methods = ["text","unrelated","K4","K16","full"]
labels = ["Text only","Unrelated audio","4 slots","16 slots","Full audio"]
for ax,group,title in zip(axes,["all","time_pattern"],["All questions (340)","Temporal pattern (39)"]):
    for metric,marker,color,label in [("yes_recall","o","#22789a","Yes recall"),("no_recall","s","#a4491f","No recall")]:
        vals=[float(data[group,m][metric]) for m in methods]
        ax.scatter(vals,range(len(methods)),label=label,marker=marker,color=color,zorder=3)
    ax.set(yticks=range(len(methods)),yticklabels=labels,xlim=(-2,102),xlabel="Correct within answer class (%)",title=title)
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.25)
axes[0].legend(loc="upper center",bbox_to_anchor=(1.08,-.17),ncol=2,frameon=False)
fig.savefig(root/"answer-class-recall.png",dpi=170)
