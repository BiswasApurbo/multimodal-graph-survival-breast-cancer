# MANUSCRIPT FIGURES 3-8 (revised) -> Revision_R1/manuscript_figures/ + manuscript_figures.zip
import os, shutil, numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

BASE = "/content/drive/MyDrive/Thesis"
RUN_OS, RUN_DSS = f"{BASE}/Revision_R1", f"{BASE}/Revision_R1_DSS"
METABRIC_CSV = f"{BASE}/Dataset/METABRIC_RNA_Mutation.csv"
OUT = f"{RUN_OS}/manuscript_figures"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = "#1f77b4", "#ff7f0e"            # colours of the original Figs 4-5
DARK_BLUE, LIGHT_BLUE = "#1565c0", "#9ec5ea"    # colours of the original Fig 6
TAB = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.dpi": 600, "savefig.bbox": "tight"})

def save(name):
    plt.savefig(f"{OUT}/{name}.pdf"); plt.savefig(f"{OUT}/{name}.png"); plt.show(); plt.close()
    print("saved", name)

SUB = ["clin", "mut", "rna", "clin+rna", "clin+mut", "rna+mut", "clin+rna+mut"]
SUBL = {"clin": "Clinical\n(GraphSAGE)", "mut": "Mutation\n(GraphSAGE)", "rna": "RNA\n(GraphSAGE)",
        "clin+rna": "Clinical + RNA\n(DualGraph)", "clin+mut": "Clinical + Mutation\n(DualGraph)",
        "rna+mut": "RNA + Mutation\n(DualGraph)", "clin+rna+mut": "Clinical + RNA + Mutation\n(TripleGraph)"}

# =====================================================================================
# Fig. 3  Modality-specific patient similarity graphs (training patients, split 42, k = 10)
#         nodes coloured by the held-out (out-of-fold) TripleGraph risk score
# =====================================================================================
def fig3():
    met = pd.read_csv(METABRIC_CSV, low_memory=False)
    meta31 = ["patient_id", "age_at_diagnosis", "type_of_breast_surgery", "cancer_type", "cancer_type_detailed",
              "cellularity", "chemotherapy", "pam50_+_claudin-low_subtype", "cohort", "er_status_measured_by_ihc",
              "er_status", "neoplasm_histologic_grade", "her2_status_measured_by_snp6", "her2_status",
              "tumor_other_histologic_subtype", "hormone_therapy", "inferred_menopausal_state", "integrative_cluster",
              "primary_tumor_laterality", "lymph_nodes_examined_positive", "mutation_count",
              "nottingham_prognostic_index", "oncotree_code", "overall_survival_months", "overall_survival",
              "pr_status", "radio_therapy", "3-gene_classifier_subtype", "tumor_size", "tumor_stage", "death_from_cancer"]
    mutc = [c for c in met.columns if c.endswith("_mut")]
    rnac = [c for c in met.columns if c not in meta31 and c not in mutc]
    lm = lambda s, m: s.astype(str).str.strip().str.lower().map(m)
    clin = pd.DataFrame({
        "age": met.age_at_diagnosis,
        "surg": lm(met.type_of_breast_surgery, {"mastectomy": 1, "breast conserving": 0}),
        "cell": lm(met.cellularity, {"low": 0, "moderate": 1, "high": 2}),
        "chemo": met.chemotherapy, "hormone": met.hormone_therapy, "radio": met.radio_therapy,
        "er": lm(met.er_status, {"positive": 1, "negative": 0}), "pr": lm(met.pr_status, {"positive": 1, "negative": 0}),
        "her2": lm(met.her2_status, {"positive": 1, "negative": 0}),
        "meno": lm(met.inferred_menopausal_state, {"post": 1, "pre": 0}),
        "grade": met.neoplasm_histologic_grade, "nodes": met.lymph_nodes_examined_positive,
        "mutcount": met.mutation_count, "npi": met.nottingham_prognostic_index,
        "size": met.tumor_size, "stage": met.tumor_stage}).astype(float)
    rna = met[rnac].astype(float)
    mut = pd.DataFrame({c: ((~met[c].isna()) & (met[c].astype(str).str.strip() != "0")).astype(float) for c in mutc})
    t = met.overall_survival_months.values.astype(float)
    e = (met.overall_survival == 0).astype(int).values            # 1 = died

    idx = np.arange(len(e))                                       # identical to make_split(eM, 42)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=e)
    tr = np.sort(tr)

    def score(X, tt, ee):                                         # univariate Cox score test (training rows only)
        o = np.argsort(tt, kind="mergesort"); tt, ee, X = tt[o], ee[o], X[o]
        n = len(tt); S0 = np.arange(n, 0, -1, dtype=float)
        S1 = np.cumsum(X[::-1], 0)[::-1]; S2 = np.cumsum((X ** 2)[::-1], 0)[::-1]
        f = np.searchsorted(tt, tt, "left"); S0, S1, S2 = S0[f], S1[f], S2[f]; ev = ee == 1
        m = S1[ev] / S0[ev][:, None]; U = (X[ev] - m).sum(0); I = (S2[ev] / S0[ev][:, None] - m ** 2).sum(0)
        return np.where(I > 1e-12, U ** 2 / np.where(I > 1e-12, I, 1), 0)

    def z(D):
        D = D.iloc[tr].fillna(D.iloc[tr].median())
        return ((D - D.mean()) / D.std(ddof=0).replace(0, 1)).values
    Xc = z(clin)
    Xr_all = z(rna); Xr = Xr_all[:, np.argsort(-score(Xr_all, t[tr], e[tr]))[:100]]
    Mt = mut.iloc[tr]; cand = [c for c in mutc if Mt[c].sum() >= 10]
    Xm = Mt[cand].values[:, np.argsort(-score(Mt[cand].values, t[tr], e[tr]))[:40]]

    oof = pd.read_csv(f"{RUN_OS}/e6_oof_predictions.csv")["risk_TripleGraph"].values[tr]
    er = met.er_status.values[tr]; pam = met["pam50_+_claudin-low_subtype"].replace({"NC": np.nan}).values[tr]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, X, metric, title in zip(axs, (Xc, Xr, Xm), ("euclidean", "cosine", "hamming"),
                                    ("Clinical Patient Similarity Graph", "RNA Expression Similarity Graph",
                                     "Mutation-Based Patient Graph")):
        _, ind = NearestNeighbors(n_neighbors=11, metric=metric, algorithm="brute").fit(X).kneighbors(X)
        edges = {tuple(sorted((a, b))) for a in range(len(X)) for b in [x for x in ind[a] if x != a][:10]}
        same_er = np.mean([er[a] == er[b] for a, b in edges])
        p_er = pd.Series(er).value_counts(normalize=True).values
        ok = [(a, b) for a, b in edges if pd.notna(pam[a]) and pd.notna(pam[b])]
        same_pam = np.mean([pam[a] == pam[b] for a, b in ok])
        p_pam = pd.Series(pam).dropna().value_counts(normalize=True).values
        G = nx.Graph(); G.add_nodes_from(range(len(X))); G.add_edges_from(edges)
        pos = nx.spring_layout(G, k=0.06, iterations=60, seed=0)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.15, alpha=0.10, edge_color="#7f7f7f")
        sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=7, node_color=oof, cmap="coolwarm",
                                    vmin=-2, vmax=2, linewidths=0)
        ax.set_title(f"{title}\nER homophily {same_er:.2f} (random {np.sum(p_er**2):.2f}) · "
                     f"PAM50 {same_pam:.2f} (random {np.sum(p_pam**2):.2f})", fontsize=10)
        ax.axis("off")
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01); cb.set_label("Normalized Risk Score", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    plt.tight_layout(); save("fig3_patient_graphs")

# =====================================================================================
# Fig. 4  Cross-cohort generalisation (same harmonised models: METABRIC test vs TCGA external)
# =====================================================================================
def fig4():
    s = pd.read_csv(f"{RUN_OS}/e7_cross_cohort_summary.csv").query("trained_on == 'METABRIC'")
    order = ["XGBoost-Cox", "CoxPH", "DeepSurv", "TripleGraph-noGraph", "TripleGraph", "DualGraph",
             "UnifiedGraph", "GraphSAGE-RNA"]
    lab = {"XGBoost-Cox": "XGBoost (Cox)", "TripleGraph-noGraph": "TripleGraphSurvival\n(no graph)",
           "TripleGraph": "TripleGraphSurvival", "DualGraph": "DualGraphSurvival", "UnifiedGraph": "Unified graph",
           "GraphSAGE-RNA": "GraphSAGE (RNA)"}
    a = s[s.evaluated_on == "METABRIC test"].set_index("model").reindex(order)
    b = s[(s.evaluated_on == "TCGA all") & (s.graph_mode == "within")].set_index("model").reindex(order)
    y = np.arange(len(order))[::-1]; h = 0.38
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for d, off, col, name in ((a, h / 2, BLUE, "METABRIC (internal test)"), (b, -h / 2, ORANGE, "TCGA (external)")):
        ax.barh(y + off, d.harrell_c_mean - 0.5, left=0.5, height=h, color=col, label=name,
                xerr=d.harrell_c_std, error_kw=dict(ecolor="#404040", elinewidth=0.9, capsize=2))
        for yy, m, sd in zip(y + off, d.harrell_c_mean, d.harrell_c_std):
            ax.text(m + sd + 0.004, yy, f"{m:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([lab.get(m, m) for m in order], fontsize=9)
    ax.set_xlim(0.5, 0.82); ax.set_xlabel("Concordance Index (C-index), mean ± SD over 10 splits")
    ax.set_title("Cross-Cohort Survival Model Generalization", fontsize=11); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    plt.tight_layout(); save("fig4_cross_cohort")

# =====================================================================================
# Fig. 5  Kaplan-Meier (held-out METABRIC test + TCGA), produced by the experiment notebook
# =====================================================================================
def fig5():
    for ext in ("pdf", "png"):
        shutil.copy(f"{RUN_OS}/figures/fig_km.{ext}", f"{OUT}/fig5_km.{ext}")
    print("saved fig5_km (copied from figures/fig_km)")

# =====================================================================================
# Fig. 6  Ablation: every modality subset, with graph vs the identical network without edges
# =====================================================================================
def fig6():
    s = pd.read_csv(f"{RUN_OS}/e2_ablation_summary.csv")
    y = np.arange(len(SUB))[::-1]; h = 0.38
    fig, ax = plt.subplots(figsize=(10, 7))
    for g, off, col, name in (("knn", h / 2, DARK_BLUE, "With patient similarity graph"),
                              ("none", -h / 2, LIGHT_BLUE, "Same network without graph edges")):
        d = s[s.graph == g].set_index("modalities").reindex(SUB)
        ax.barh(y + off, d["mean"] - 0.5, left=0.5, height=h, color=col, label=name,
                xerr=d["std"], error_kw=dict(ecolor="#404040", elinewidth=0.9, capsize=2))
        for yy, m, sd in zip(y + off, d["mean"], d["std"]):
            ax.text(m + sd + 0.003, yy, f"{m:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([SUBL[m] for m in SUB], fontsize=10, fontweight="bold")
    ax.set_xlim(0.50, 0.75); ax.set_xlabel("Concordance Index (C-index), mean ± SD over 10 splits", fontweight="bold")
    ax.set_title("Ablation Study: Modality Contribution", fontsize=14, fontweight="bold"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    plt.tight_layout(); save("fig6_ablation")

# =====================================================================================
# Fig. 7  5-year calibration and decision curve (out-of-fold, METABRIC)
# =====================================================================================
def fig7():
    c = pd.read_csv(f"{RUN_OS}/e6_calibration_5y.csv"); d = pd.read_csv(f"{RUN_OS}/e6_decision_curve_5y.csv")
    models = ["TripleGraph", "TripleGraph-noGraph", "CoxPH", "DeepSurv", "NPI-Cox"]
    lab = {"TripleGraph": "TripleGraphSurvival", "TripleGraph-noGraph": "TripleGraphSurvival (no graph)", "NPI-Cox": "NPI"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for m, col in zip(models, TAB):
        cc = c[c.model == m]; axs[0].plot(cc.predicted_S60, cc.observed_KM_S60, "o-", color=col, ms=5, lw=1.6,
                                          label=lab.get(m, m))
        dd = d[d.model == m]; axs[1].plot(dd.threshold, dd.net_benefit_model, color=col, lw=1.8, label=lab.get(m, m))
    lo = min(c.predicted_S60.min(), c.observed_KM_S60.min()) - 0.03
    axs[0].plot([lo, 1], [lo, 1], "k--", lw=0.9, label="Perfect calibration")
    axs[0].set_xlim(lo, 1); axs[0].set_ylim(lo, 1)
    axs[0].set_xlabel("Predicted 5-year survival"); axs[0].set_ylabel("Observed 5-year survival (Kaplan–Meier)")
    axs[0].set_title("(a) Calibration by risk quintile"); axs[0].legend(fontsize=8, loc="upper left")
    dd = d[d.model == models[0]]
    axs[1].plot(dd.threshold, dd.net_benefit_treat_all, color="#7f7f7f", ls=":", lw=1.6, label="Treat all")
    axs[1].axhline(0, color="k", lw=0.9, label="Treat none")
    axs[1].set_ylim(-0.02, max(d.net_benefit_model.max(), dd.net_benefit_treat_all.max()) + 0.02)
    axs[1].set_xlabel("Threshold probability (5-year death)"); axs[1].set_ylabel("Net benefit")
    axs[1].set_title("(b) Decision curve at 5 years"); axs[1].legend(fontsize=8, loc="upper right")
    plt.tight_layout(); save("fig7_calibration_dca")

# =====================================================================================
# Fig. 8  Paired difference (graph - no graph), 95% CI by corrected resampled t-test, OS and DSS
# =====================================================================================
def fig8():
    n_tr, n_te, J = 1332, 286, 10; tq = stats.t.ppf(0.975, J - 1)
    y = np.arange(len(SUB))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for run, col, off, name in ((RUN_OS, BLUE, 0.14, "Overall survival"), (RUN_DSS, ORANGE, -0.14, "Disease-specific survival")):
        p = pd.read_csv(f"{run}/e2_ablation_paired_tests.csv")
        p = p[p.comparison == "graph - no graph"].set_index("modalities").reindex(SUB)
        ci = tq * np.sqrt(1 / J + n_te / n_tr) * p.sd_diff
        ax.errorbar(p.mean_diff, y + off, xerr=ci, fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.6,
                    capsize=3, label=name)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([SUBL[m].replace("\n", " ") for m in SUB], fontsize=9)
    ax.set_xlabel("Δ C-index (graph − no graph), 95% CI over 10 splits")
    ax.set_title("Effect of the Patient Similarity Graph", fontsize=11)
    ax.text(0.02, 0.02, "← graph worse", transform=ax.transAxes, color="#555555", fontsize=9)
    ax.text(0.98, 0.02, "graph better →", transform=ax.transAxes, color="#555555", fontsize=9, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False); ax.grid(axis="y", visible=False)
    plt.tight_layout(); save("fig8_graph_vs_nograph")

for f in (fig3, fig4, fig5, fig6, fig7, fig8):
    try:
        f()
    except Exception as ex:
        print(f"!! {f.__name__} failed: {type(ex).__name__}: {ex}")

shutil.make_archive(OUT, "zip", OUT)
print("\nFiles:", sorted(os.listdir(OUT)))
print("Zip:", OUT + ".zip")
