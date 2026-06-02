import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = [
    "AASIST\nBaseline",
    "ResNet\nBaseline",
    "MLP\nBaseline",
    "CA\nBaseline",
    "AASIST\n+UR-FFL",
    "ResNet\n+UR-FFL",
    "MLP\n+UR-FFL",
    "CA\n(ResNet\nSensor)",
    "CA+UR-FFL\n(Proposed)"
]

la_eer  = [15.17, 8.54, 6.67, 5.57, 13.26, 5.34, 7.46, 5.29, 3.97]
df_eer  = [33.74, 26.49, 29.99, 25.86, 31.10, 19.96, 20.17, 25.14, 14.99]
gap     = [df - la for la, df in zip(la_eer, df_eer)]

x = np.arange(len(models))
width = 0.28

fig, ax = plt.subplots(figsize=(14, 6))

bars1 = ax.bar(x - width, la_eer, width, label="LA EER (%)",
               color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x,          df_eer, width, label="DF EER (%)",
               color="#DD8452", alpha=0.85)
bars3 = ax.bar(x + width,  gap,    width, label="LA→DF Gap (pp)",
               color="#55A868", alpha=0.85)

# Highlight proposed method
for b in [bars1[-1], bars2[-1], bars3[-1]]:
    b.set_edgecolor("black")
    b.set_linewidth(2.0)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("EER (%) / Gap (pp)")
ax.set_title("Generalization Gap Summary: LA EER, DF EER, and LA→DF Gap")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("figure_109_generalization_gap.png", dpi=300)
plt.show()