"""Human-readable rendering of a report dict. No numbers are invented here."""
from __future__ import annotations


def _ci(pair):
    return f"[{pair[0]:.4f}, {pair[1]:.4f}]"


def render_markdown(r: dict) -> str:
    L = []
    A = L.append
    A(f"# contamcheck report — `{r['run']['input']}`")
    A("")
    A(f"- tool version: {r['run']['contamcheck_version']}")
    A(f"- master seed: **{r['run']['seed']}** "
      f"(cv_seed={r['run']['cv_seed']}, bootstrap_seed={r['run']['bootstrap_seed']}, "
      f"nn_seed={r['run']['nn_seed']})")
    A(f"- label column `{r['run']['label_col']}` mapped {r['run']['label_mapping']}; "
      f"group column `{r['run']['group_col']}`")
    A(f"- rows {r['group_structure']['n_rows']}, "
      f"positives {r['run']['n_pos']}, negatives {r['run']['n_neg']}")
    A(f"- feature columns ({r['run']['feature_source']}): "
      f"{', '.join(r['run']['feature_cols']) if r['run']['feature_cols'] else 'none'}")
    A("")
    v = r["verdict"]
    A(f"## Verdict: **{v['level']}**")
    A("")
    A(v["reading"])
    A("")
    if v["flags_triggered"]:
        A("Flags triggered:")
        for f in v["flags_triggered"]:
            A(f"- {f}")
    else:
        A("No flags triggered.")
    A("")
    A(f"_{v['thresholds_are_conventions']}_")
    A("")

    g = r["group_identity"]
    A("## (a) Can the group column alone predict the label?")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| leave-one-out target-encoded AUC of group identity | "
      f"**{g['auc_group_identity_LOO_target_encoding']:.4f}** {_ci(g['ci95'])} |")
    A(f"| bootstrap draws | {g['n_bootstrap_draws']} (seed {g['bootstrap_seed']}) |")
    A(f"| rows falling back to the global prior (singleton groups) | "
      f"{g['n_rows_scored_with_prior_fallback']} |")
    A("")

    s = r["group_structure"]
    gs = s["group_size"]
    A("## (b) Group purity and size distribution")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| groups | {s['n_groups']} |")
    A(f"| fraction of groups that are single-label | **{s['frac_groups_label_pure']:.4f}** |")
    A(f"| fraction of rows in a label-pure group | {s['frac_rows_in_a_label_pure_group']:.4f} |")
    A(f"| fraction of rows in a mixed-label group | {s['frac_rows_in_a_mixed_label_group']:.4f} |")
    A(f"| group size min / median / mean / max | {gs['min']} / {gs['median']:.1f} / "
      f"{gs['mean']:.4f} / {gs['max']} |")
    A(f"| group size p25 / p75 / p95 | {gs['p25']:.1f} / {gs['p75']:.1f} / {gs['p95']:.1f} |")
    A(f"| singleton groups | {gs['n_singleton_groups']} |")
    A(f"| fraction of rows in the largest group | {gs['frac_rows_in_largest_group']:.4f} |")
    A("")

    sc = r.get("split_comparison")
    A("## (c) Random split vs group split, same baseline classifier")
    A("")
    if sc is None:
        A(f"Not computed: {r['run'].get('split_comparison_skipped_because')}")
    else:
        A(f"Classifier: {sc['classifier']}; {sc['n_folds']}-fold, "
          f"{sc['n_repeats_random_split']} repeats on the random split, cv seed {sc['cv_seed']}.")
        A("")
        A("| split | AUC | 95% CI |")
        A("|---|---|---|")
        A(f"| random (stratified k-fold) | {sc['random_split']['mean_auc_over_repeats']:.4f} "
          f"(mean of repeats); {sc['random_split']['auc_last_repeat']:.4f} (last repeat) | "
          f"{_ci(sc['random_split']['ci95_last_repeat'])} (last repeat) |")
        A(f"| grouped (GroupKFold) | {sc['group_split']['auc']:.4f} | "
          f"{_ci(sc['group_split']['ci95'])} |")
        A("")
        d = sc["delta_group_minus_random"]
        A(f"**delta (grouped - random), paired bootstrap: {d['point']:+.4f} "
          f"{_ci(d['ci95'])}**; {d['frac_bootstrap_draws_le_0']*100:.2f}% of draws <= 0. "
          f"Drop from the mean random-split AUC: {sc['drop_mean_random_to_group']:+.4f}.")
    A("")

    ac = r.get("arm_comparison")
    if ac is not None:
        A("## (c2) Model arm vs comparison arm, under both splits")
        A("")
        A("| split | "
          f"{ac['arm_a']} | {ac['arm_b']} | delta (A - B) | 95% CI |")
        A("|---|---|---|---|---|")
        rs, gsp = ac["random_split"], ac["group_split"]
        A(f"| random | {rs['mean_auc_'+ac['arm_a']]:.4f} | {rs['mean_auc_'+ac['arm_b']]:.4f} | "
          f"{rs['delta_a_minus_b_last_repeat']:+.4f} | {_ci(rs['ci95'])} |")
        A(f"| grouped | {gsp['auc_'+ac['arm_a']]:.4f} | {gsp['auc_'+ac['arm_b']]:.4f} | "
          f"{gsp['delta_a_minus_b']:+.4f} | {_ci(gsp['ci95'])} |")
        A("")
        A(f"Shift in A's advantage, random -> grouped: "
          f"{ac['shift_in_advantage_random_to_group']:+.4f}.")
        A("")

    st = r.get("structure")
    A("## (d) Structure diagnostics")
    A("")
    if st is None:
        A(f"Not computed: {r['run'].get('structure_skipped_because')}")
    else:
        A("| quantity | value |")
        A("|---|---|")
        A(f"| molecules parsed | {st['n_parsed']} of {st['n_rows']} |")
        A(f"| nearest-neighbour Tanimoto, same class (mean over "
          f"{st['nn_sample_size']} sampled queries) | {st['nn_tanimoto_same_class_mean']:.4f} |")
        A(f"| nearest-neighbour Tanimoto, cross class | "
          f"{st['nn_tanimoto_cross_class_mean']:.4f} |")
        A(f"| Murcko scaffolds, positive / negative / shared | "
          f"{st['n_scaffolds_pos']} / {st['n_scaffolds_neg']} / {st['n_scaffolds_shared']} |")
        jac = st.get("scaffold_jaccard_pos_vs_neg")
        A(f"| between-class scaffold Jaccard | {jac:.4f} |" if jac is not None
          else "| between-class scaffold Jaccard | n/a |")
        for key, lbl in (("scaffold_leakage_random_split", "random split"),
                         ("scaffold_leakage_group_split", "group split")):
            val = st.get(key)
            if val is not None:
                A(f"| mean fraction of test scaffolds also seen in train, {lbl} | {val:.4f} |")
    A("")
    A("---")
    A("")
    A("Every number above is also in `report.json` in this directory. "
      "Diagnostics are descriptive: they show what the group column encodes on "
      "this table, not whether your model is right.")
    A("")
    return "\n".join(L)
