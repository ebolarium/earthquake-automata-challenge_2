# Can a Causal Spatial Correction Improve ETAS Across Tectonic Regimes? Retrospective Evidence and a Frozen Prospective Test

**Saban Baris Boga**

Independent Researcher, Adana, Turkiye
ORCID: https://orcid.org/0009-0000-9076-946X
Correspondence: hello@bboga.com
Research archive DOI: https://doi.org/10.5281/zenodo.22832305

**Manuscript status:** Version 1.0, 18 September 2026. Research article and prospective protocol. The retrospective analyses reported here used opened data; the confirmatory 365-day experiment was frozen before its first eligible four-region forecast and had not produced a prospective result at the time of writing.

## Abstract

Epidemic-type aftershock sequence (ETAS) models are strong short-term earthquake-forecasting baselines, but their prescribed spatial kernel may leave repeatable, catalog-dependent structure unexplained. We ask two linked questions: can a causal neural correction improve the spatial log score of frozen ETAS across tectonically distinct regions, and can an evidence gate limit negative transfer when that correction is deployed? We constructed a translation-invariant fast-minus-slow event-history network. For each target region, a three-member ensemble was trained only on the other regions. The network reallocates, but does not change, the daily ETAS expected count. Its correction is restricted to cells in the upper 1% of the ETAS rate field and is mixed at 0.5 with a causal regional safe forecast. A Bayes-factor hurdle of 20 activates the neural expert using only scores from completed prior days. In retrospective daily-grid replays comprising 8,152 earthquakes, the gated forecast achieved positive information gain per earthquake (IGPE) against ETAS in California (0.00944; 90-day stationary-block 95% interval lower bound 0.00586), New Zealand (0.03046; 0.01965), Chile (0.01073; 0.00694), and Japan C (0.00150; 0.00100). The pooled descriptive IGPE was 0.01525. However, against the regional safe forecast, the gate was exactly neutral in New Zealand and Japan C, positive in California, and slightly negative in Chile; the external-region event-weighted difference was -0.0000019 IGPE. Thus, the retrospective answer is qualified: the frozen hybrid improved ETAS in all four replays, but independent transport and superiority over a strong safe correction are not established. We therefore froze a 365-day, four-region prospective experiment with daily forecasts, a minimum of 500 pooled target earthquakes, paired IGPE as the primary endpoint, and CSEP N-, L-, and R-tests as secondary diagnostics. The protocol forbids backfill, post-activation refitting, and post-hoc region exclusion.

**Keywords:** earthquake forecasting; ETAS; neural point process; information gain per earthquake; CSEP; prospective evaluation; causal prediction; negative transfer

## 1. Introduction

Earthquake forecasting is a probabilistic point-process problem: a useful model must assign rate to future events before they occur and must be judged with data unavailable at model construction. ETAS represents seismicity as a background process plus self-exciting offspring from previous earthquakes [1,2]. It remains a demanding benchmark because its temporal productivity and spatial decay encode empirically persistent features of clustered seismicity.

Neural point-process models can learn dependencies that are not captured by a fixed parametric kernel. Recent studies have shown that neural encoders can match or exceed ETAS in selected catalogs [3--5]. Flexibility, however, creates three hazards. First, absolute coordinates can let a network memorize a training geography. Second, standard random splits leak later seismic regimes into training. Third, an apparently successful correction can degrade a well-calibrated baseline when transported to a different tectonic regime. These hazards are especially important when the intended claim spans transform, subduction, and complex island-arc settings.

We address those hazards by narrowing the learning problem. The neural component does not predict the total number of earthquakes. Instead, it learns a relative spatial density ratio from causal event history; ETAS continues to supply the daily expected count. The network uses coordinates expressed relative to each candidate location and contrasts recent activity with a compressed slower history. Every target-region model is leave-one-region-out, so the target geography contributes no training example. The learned redistribution is further confined to high-rate ETAS support and is admitted only after completed-day evidence clears a fixed hurdle.

The present study asks: **Does this constrained, evidence-gated spatial correction improve frozen ETAS across tectonically distinct regions without sacrificing calibration?** We answer the retrospective portion using four daily-grid replays, report the failed as well as passed gates, and then specify the independent prospective test that can adjudicate generalization. This distinction matters: all retrospective target outcomes are now known and are development evidence, irrespective of whether individual model fits excluded a target region.

Our contributions are:

- a translation-invariant fast-minus-slow neural residual that begins at the exact ETAS control and conserves total daily rate;
- leave-one-region-out training and causal permutation interventions that test whether event history, rather than geography alone, carries the signal;
- a support restriction and anytime evidence hurdle designed to reduce negative transfer;
- a complete four-region retrospective accounting, including the near-zero external shortfall against the safe incumbent; and
- a frozen, public, 365-day protocol that reports IGPE and CSEP N-, L-, and R-tests for every region and for the pooled experiment.

![Figure 1. Forecast construction. ETAS fixes the daily expected count. A regional safe redistribution and a leave-one-region-out neural spatial expert are combined only inside the upper 1% of the ETAS rate field. A prior-day evidence hurdle determines whether the neural expert receives positive weight.](figures/causal-spatial-method.png)

## 2. Models

### 2.1 Frozen ETAS baseline

For location x and time t, the ETAS conditional intensity is written schematically as

`lambda_ETAS(t,x) = mu(x) + sum_{t_i<t} K(m_i) g(t-t_i) f(x-x_i | m_i)`.

The sum includes only earthquakes observed strictly before forecast issue time. The productivity term K increases with parent magnitude; g is a tapered Omori-type temporal kernel; and f is a magnitude-scaled spatial power-law kernel. Each regional ETAS calibration is frozen before prospective activation. Daily integration over the regional grid gives cell rates `lambda_E,j` and total expected count `Lambda_E = sum_j lambda_E,j`.

The experiment evaluates spatial allocation conditional on the daily count supplied by ETAS. Every challenger stage is constrained so that `sum_j lambda_C,j = Lambda_E`. Consequently, the event-wise log-rate ratio reduces to a spatial probability ratio, while the CSEP count diagnostics remain identical unless an implementation error violates the invariant.

### 2.2 Regional safe forecast

The safe expert is a causal, non-neural redistribution of ETAS background intensity. California uses a supported-neighbor renewal residual based on background-attributed activity in fixed 7- and 30-day windows. Its three frozen parameters are an acceleration support power of 0.7148116, additive acceleration mix of 0.0797456, and residual scale of 0.7423419. New Zealand, Chile, and Japan C use the same exposure-normalized renewal construction with regional exposure scales fixed before the prospective run. The triggered ETAS component passes through unchanged, and the adjusted background is renormalized to preserve the ETAS daily count.

This expert is called “safe” only in a relative engineering sense: it is the fallback selected before the prospective experiment, not a claim that it is the true data-generating process. The evidence hurdle therefore provides a deployment rule, not a formal universal error guarantee.

### 2.3 Translation-invariant fast-minus-slow network

The neural expert receives the 256 most recent catalog events available before issue time. Every event has five normalized features: log inter-event gap, along-strike coordinate, cross-strike coordinate, normalized depth, and magnitude above the regional completeness threshold. The final 32 events form the fast branch. The preceding 224 events are grouped chronologically in blocks of eight, yielding 28 slow tokens with mean features and maximum magnitude.

For a candidate cell, the network forms event-candidate pairs containing two relative coordinates, log radial distance, normalized gap, normalized depth, mean magnitude, and maximum magnitude. A two-layer multilayer perceptron maps seven pair features through 32 and 48 hidden units with GELU activations. Pair embeddings are averaged separately over the fast and slow histories; their difference is passed through layer normalization, a 48-unit hidden layer, dropout 0.1, and a scalar output. Candidate logits are centered so that only relative allocation is represented. The output layer is initialized to zero, making epoch zero the exact ETAS control.

Training uses contrastive spatial windows. A positive observed target location competes against ETAS-drawn counterfactual locations. Cross-entropy is supplemented by penalties that push the correction toward neutrality when event contexts are mismatched across examples or shuffled in time. Chronological validation selects checkpoints. Learning rate is 0.0003, weight decay 0.001, batch size 64, maximum 35 epochs, early-stopping patience six, and gradient norm clip 1.0.

For each held-out target region, training uses only the other regions. Three fixed seeds (14002, 14003, and 14004) are averaged in logit space. This is equivalent to a geometric mean of learned density ratios before candidate-set normalization. No seed is selected by target outcome.

### 2.4 Support-restricted transport

Unrestricted neural renormalization moved rate into cells where ETAS assigned little support and produced a negative California ablation. We therefore define active support A as grid cells at or above the 0.99 quantile of the daily ETAS rates. The neural allocation is renormalized to the safe expert's mass within A; cells outside A remain exactly equal to the safe expert. The fixed expert is

`lambda_F = lambda_S + 0.5 (lambda_N,A - lambda_S,A)` within A,

and `lambda_F = lambda_S` outside A. Here `lambda_N,A` is the neural allocation rescaled to the safe mass on A. The mixture fraction 0.5 and quantile 0.99 were selected on opened California development outcomes and are not independent evidence.

### 2.5 Prior-day evidence hurdle

Let `S_d` be the cumulative event log-rate ratio of the fixed expert to the safe expert through completed issue day d. No within-day outcome changes a forecast already issued. For the next day,

`w_{d+1} = max(0, tanh((S_d - log 20)/2))`,

`lambda_C,d+1 = lambda_S,d+1 + w_{d+1}(lambda_F,d+1 - lambda_S,d+1)`.

The initial state is `S_0 = 0`, so the first forecast is exactly safe. The hurdle corresponds to a fixed Bayes-factor threshold of 20. Historical retrospective evidence is not carried into prospective activation. A region that never qualifies remains exactly at the safe expert; a qualified region can deactivate if cumulative evidence falls back below the hurdle.

## 3. Data and experimental design

### 3.1 Regions and catalogs

The regions span distinct tectonic settings and catalog regimes. California RELM is dominated by transform faulting with a dense low-magnitude catalog. The New Zealand CSEP region combines subduction and crustal deformation. Chile covers a long subduction corridor. Japan C samples the Japan and Kuril trench environment. Table 1 lists the frozen prospective adapters; retrospective windows differ where catalog provenance required it.

| Region | Grid | Target threshold | Depth | Retrospective scoring interval | Events |
|---|---:|---:|---:|---|---:|
| California RELM | 7,682 cells, 0.1 deg | M >= 2.5 | all protocol depths | 2014-01-07 to 2018-12-31 | 3,619 |
| New Zealand CSEP | 6,343 cells, 0.1 deg | M >= 4.0 | 0--40 km | 2008-01-01 to 2025-12-31 | 2,270 |
| Chile subduction | 1,560 cells, 0.5 deg | M >= 4.5 | 0--100 km | 2015-01-01 to 2025-12-31 | 1,909 |
| Japan C | 572 cells, 0.5 deg | M >= 5.0 | 0--100 km | 2004-01-01 to pre-Tohoku 2011-03-11 | 354 |

California, Chile, and the prospective Japan stream use USGS ANSS ComCat FDSN data. New Zealand uses the GeoNet FDSN service. The retrospective Japan experiment uses the historical catalog associated with the FERN benchmark; prospective Japan targets come from ComCat and are therefore reported with a catalog-continuity warning. Raw responses, normalized catalogs, and SHA-256 manifests are retained.

### 3.2 Evidence chronology and leakage controls

The development path was sequential. Earlier models motivated the fast-minus-slow architecture; target outcomes were opened during model development. We therefore label all four daily-grid results retrospective development evidence. Leave-one-region-out training prevents direct target-region fitting but does not restore prospective blindness after a target catalog has been inspected.

Three controls target shortcut learning. First, regional coordinates are centered, rotated into a principal along/cross frame, and scaled, while the network acts on candidate-relative displacement. Second, the target region is absent from training for its ensemble. Third, mismatched-context and time-shuffled interventions are scored. In the final three-seed contrastive ensemble, mean log gain was positive in all four held-out regions and the 90-day stationary-block lower bound exceeded zero in each. The context-permutation and time-shuffle drops were also positive in every region, indicating that the model used causal history rather than only static candidate position. These interventions establish dependence within the constructed benchmark; they do not prove a physical causal mechanism.

### 3.3 Scoring

For N target earthquakes and forecasts A and B with equal total expected count, information gain per earthquake is

`IGPE(A,B) = (1/N) sum_i log(lambda_A,i / lambda_B,i)`.

Positive IGPE favors A. `exp(IGPE)` is the geometric mean probability factor. Because daily scores are serially dependent, uncertainty is estimated with a stationary bootstrap over complete issue days [6]. We report 10,000 replicates with mean block lengths of 30 and 90 days and percentile 95% intervals. These intervals quantify replay variability under the chosen dependence resampling; they do not convert development results into confirmatory evidence.

The frozen prospective test also reports CSEP diagnostics [7--10]. The N-test compares observed and expected counts. The L-test evaluates the absolute likelihood consistency of each conditional Poisson spatial forecast using 2,000 simulations per region-day. The R-test compares the challenger with ETAS. Results are displayed daily and cumulatively, per region and pooled. Calibration tests are secondary to the preregistered IGPE endpoint.

## 4. Retrospective results

### 4.1 Held-out contrastive mechanism test

The three-seed leave-one-region-out ensemble produced positive contrastive log gain in every target region: 0.00534 in California, 0.02815 in New Zealand, 0.01913 in Chile, and 0.00681 in Japan C. The corresponding 90-day stationary-block lower bounds were 0.00279, 0.02012, 0.01215, and 0.00359. Permuting context reduced mean gain by 0.00506, 0.00062, 0.00409, and 0.00285; shuffling history in time reduced it by 0.00260, 0.00157, 0.00120, and 0.00335, respectively. The smaller New Zealand intervention drop cautions that much of its contrastive advantage may be relatively stable spatial structure.

### 4.2 Support restriction

On the opened 2014--2018 California daily grid, the support-restricted fixed expert gained 0.01070 IGPE over ETAS (3,619 events), with lower bounds of 0.00659 and 0.00629 for 30- and 90-day blocks. Relative to the safe forecast, it gained 0.00491 IGPE, with lower bounds 0.00152 and 0.00084. Every annual difference from the safe forecast was non-negative, although the 2018 value was only 0.00009. By contrast, global renormalization without the support restriction lost 0.00076 IGPE relative to the safe model. The support restriction therefore corrected an observed failure mode, but the rule was selected after seeing California outcomes.

Zero-refit transfer of the same support rule yielded positive mean IGPE against ETAS in New Zealand (0.02752), Chile (0.01221), and pre-Tohoku Japan C (0.00288), with positive 30- and 90-day lower bounds. It did not uniformly improve the regional safe forecast: mean differences were -0.00294, +0.00148, and +0.00137, and the uncertainty bounds for the latter two included zero. This motivated the evidence hurdle.

### 4.3 Evidence-gated daily-grid replay

Table 2 gives the final retrospective replay used to freeze the prospective policy. The gate begins at the safe forecast independently in every region. It strongly activates in California, never activates in New Zealand or Japan C, and briefly activates in Chile. Against ETAS, all four regional means and both reported lower bounds are positive.

| Region | N | IGPE vs ETAS | 30-day lower | 90-day lower | IGPE vs safe | Ever qualified |
|---|---:|---:|---:|---:|---:|---|
| California | 3,619 | 0.009437 | 0.005884 | 0.005859 | 0.003655 | Yes |
| New Zealand | 2,270 | 0.030459 | 0.019435 | 0.019649 | 0.000000 | No |
| Chile | 1,909 | 0.010728 | 0.007602 | 0.006935 | -0.0000045 | Yes |
| Japan C | 354 | 0.001503 | 0.001025 | 0.001001 | 0.000000 | No |

Across 8,152 events, summed information gain against ETAS was 124.306 nats, corresponding to a descriptive event-weighted IGPE of 0.015249 and probability factor 1.01537. This pooled number has no preregistered retrospective interval and should not be interpreted as confirmatory.

The failure is also informative. For the three external regions, the event-weighted gated difference from the safe forecast was -0.0000019 IGPE. The fixed neural expert had lost materially to safe in New Zealand, and the hurdle successfully prevented that transfer. In Chile, however, short activation around the threshold incurred a total loss of 0.0087 nat, or -0.0000045 per event. Therefore the preregistered development admission rule requiring non-negative external event-weighted and macro differences from safe did not pass. The gate is a high-conservatism transport device, not demonstrated dominance over the safe forecast.

![Figure 2. Retrospective IGPE of the evidence-gated challenger relative to frozen ETAS. Points show regional means; thick and thin intervals are 30- and 90-day stationary-bootstrap 95% intervals. Every regional lower bound is positive, but all target data were opened during development.](figures/retrospective-igpe.png)

## 5. Frozen prospective test

### 5.1 Research question and activation

The confirmatory question is: **Across tectonically distinct regions, does the frozen causal spatial correction achieve positive prospective information gain over frozen ETAS without degrading CSEP count and likelihood calibration?** The experiment activates on the first atomic forecast that publishes all four regions for the same next-UTC-day target. No dry run is required. No forecast created after its target window begins can be backfilled or scored as prospective.

The planned duration is 365 consecutive target days with no calendar extension. At least 500 pooled target earthquakes are required; otherwise the primary conclusion is “inconclusive.” All regions remain in the analysis regardless of performance or downtime. A model, feature, geometry, threshold, catalog, or parameter change requires a new protocol.

### 5.2 Primary and secondary endpoints

The primary endpoint is pooled paired IGPE of the challenger against frozen ETAS using catalog-settled final scores. Success requires all three conditions: mean IGPE greater than zero, the 95% stationary-bootstrap lower bound greater than zero with a 30-day mean block, and the analogous lower bound greater than zero with a 90-day mean block. Ten thousand replicates are fixed. Regional results are reported without post-hoc exclusion but are not separate pass/fail gates.

Secondary endpoints are per-region and pooled CSEP N-, L-, and R-tests, daily and cumulative. The challenger is count-conserving by design, so the N-test should match ETAS; a difference flags a pipeline defect. The L-test examines absolute consistency, and the R-test compares relative fit. Secondary failures qualify interpretation even when the IGPE endpoint passes.

Catalog snapshots are immutable. Initial scores are provisional; after a seven-day settlement delay they are recomputed and labeled final while forecasting continues daily. This delay is for catalog revision, not a waiting period. Every forecast and catalog object has a manifest and SHA-256 digest.

### 5.3 Frozen parameters

The challenger-wide parameters are: 0.99 active ETAS quantile, 0.5 mixture fraction, Bayes-factor hurdle 20, initial log evidence zero, 256 context events, 32 recent events, and three ensemble members. Table 3 reports the observed range of regional ETAS calibrations. These are fitted regional parameters, not transferable universal constants.

| Parameter | Minimum | Maximum | Role |
|---|---:|---:|---|
| magnitude reference | 2.500 | 5.000 | catalog threshold |
| beta | 2.147 | 2.366 | magnitude density |
| log10 mu | -8.546 | -6.333 | background rate |
| log10 K0 | -2.672 | 0.228 | productivity |
| a | 1.556 | 2.657 | magnitude productivity |
| log10 c | -2.798 | -2.156 | short-time offset |
| omega | -0.0619 | 0.0404 | temporal decay adjustment |
| log10 tau | 3.208 | 3.838 | temporal taper |
| log10 d | -0.773 | 2.317 | spatial scale |
| gamma | 0.581 | 1.011 | magnitude-spatial scaling |
| rho | 0.557 | 1.014 | spatial tail exponent |

![Figure 3. Prospective evaluation logic. A single atomic issue opens four regional target windows. Forecasts are immutable before the target day; provisional scores continue daily, final scores follow catalog settlement, and the primary decision is made after 365 days only if at least 500 pooled events are observed.](figures/prospective-protocol.png)

## 6. Discussion

The retrospective evidence answers the first question narrowly: a constrained hybrid can improve the log score of ETAS across four heterogeneous replays. The network's role is not to replace ETAS. It proposes where, within already elevated ETAS support, some rate should move. Count conservation makes the comparison interpretable and prevents the neural component from winning through a separate rate forecast.

The results also show why unrestricted claims would be premature. The support rule was chosen after California outcomes were visible. New Zealand's safe model outperformed the fixed neural expert, and the final hurdle avoided that loss only by remaining inactive. Chile crossed the hurdle briefly and produced a tiny loss relative to safe. Moreover, catalogs, magnitude thresholds, spatial grids, and retrospective periods differ by region. Positive IGPE against ETAS across these replays is therefore evidence of a reusable mechanism, not proof of universal tectonic transfer.

The design combines ideas from seismology, causal prediction, and online decision theory. ETAS contributes a scientifically interpretable intensity and high-rate support. Relative coordinates and leave-one-region-out training reduce geographic memorization. History interventions ask whether predictions depend on the correct temporal context. The evidence hurdle treats the learned correction as a risky expert that must earn deployment using only past scores. None of these devices alone guarantees validity; together they make failure modes observable and restrict their consequences.

Tectonic labels are deliberately absent from the final gate. Earlier attribution experiments indicated that regime information could explain model behavior, but a fitted tectonic mixture did not establish a reliable improvement. The frozen experiment instead asks whether one invariant formulation transfers across regimes. Region-specific coordinate frames and ETAS calibrations are permitted because they are estimated before scoring and are part of each catalog adapter; the neural architecture, support rule, and gate are common.

## 7. Limitations

First, all reported outcomes are retrospective and were available during the research program. Leave-one-region-out fitting is weaker than a genuinely untouched stream. Second, the four experiments do not share identical catalog provenance or completeness. In particular, historical and prospective Japan catalogs differ. Third, the conditional Poisson CSEP tests simplify the overdispersion present in clustered seismicity; their interpretation must accompany, not replace, paired score comparisons [10,11]. Fourth, low target counts, especially in Japan C, can produce low-power regional tests. Fifth, a positive log-score difference is not an earthquake-prediction capability and provides no deterministic warning. Finally, the Bayes-factor terminology describes a likelihood-ratio evidence policy; without assuming the safe forecast is the true conditional model, the threshold is not a guaranteed type-I error bound.

## 8. Conclusions

A causal, support-restricted, evidence-gated spatial correction achieved positive retrospective IGPE against frozen ETAS in California, New Zealand, Chile, and Japan C. The pooled descriptive gain was 0.01525 nat per earthquake. The result is nonetheless qualified: the correction did not demonstrate uniform superiority over the regional safe forecast, and the external aggregate missed that development criterion by approximately two millionths of a nat per earthquake. The correct conclusion is therefore not that ETAS has been superseded, but that a tightly constrained residual merits independent prospective evaluation.

The model, regional adapters, decision rule, and failure handling are now frozen in a 365-day four-region experiment. Its answer will be based on forecasts issued before observation, a minimum of 500 pooled events, paired IGPE with two dependence scales, and CSEP calibration diagnostics. Until that experiment matures, the prospective question remains open.

## 9. Reproducibility and data availability

Source code, frozen configurations, model weights, protocol hashes, and the public scorecard are available at https://github.com/ebolarium/earthquake-automata-challenge_2 and https://etas2.bboga.com. The archived research release is permanently identified by https://doi.org/10.5281/zenodo.22832305. The repository records the exact protocol, regional ETAS parameter files, neural ensemble digests, and forecast runtime digests. Retrospective result artifacts originated in the preceding development repository and are reported with their claim-boundary metadata. Public catalogs are retrieved from USGS ANSS ComCat and GeoNet FDSN endpoints. Raw responses are archived by the operational pipeline subject to provider terms. Reproductions should cite the Zenodo DOI and the corresponding repository release or commit hash.

## 10. Declarations

**Author contributions.** Saban Baris Boga conceived the study, defined the research question and evaluation constraints, directed the software-assisted implementation, reviewed the experiments, interpreted the results, and prepared the manuscript.

**Competing interests.** The author declares no competing interests.

**Funding.** This independent research received no external funding.

**Generative AI disclosure.** Generative AI systems were used as research-assistance tools for software implementation, documentation synthesis, language editing, and preparation of the manuscript draft under the author's direction. They are not authors. The author retains responsibility for verification, interpretation, and the submitted text.

**Ethics and safety.** The study uses public earthquake catalogs and does not involve human or animal subjects. Forecasts are research outputs, not earthquake warnings, operational hazard products, or substitutes for official guidance.

## References

1. Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83, 9--27. https://doi.org/10.1080/01621459.1988.10478560
2. Ogata, Y. (1998). Space-time point-process models for earthquake occurrences. *Annals of the Institute of Statistical Mathematics*, 50, 379--402. https://doi.org/10.1023/A:1003403601725
3. Zlydenko, O., et al. (2023). A neural encoder for earthquake rate forecasting. *Scientific Reports*, 13, 12350. https://doi.org/10.1038/s41598-023-38033-9
4. Dascher-Cousineau, K., et al. (2023). Using deep learning for flexible and scalable earthquake forecasting. *Geophysical Research Letters*, 50, e2023GL103909. https://doi.org/10.1029/2023GL103909
5. Stockman, S., Lawson, D. J., & Werner, M. J. (2023). Forecasting the 2016--2017 Central Apennines earthquake sequence with a neural point process. *Earth's Future*, 11, e2023EF003777. https://doi.org/10.1029/2023EF003777
6. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89, 1303--1313. https://doi.org/10.1080/01621459.1994.10476870
7. Schorlemmer, D., Gerstenberger, M. C., Wiemer, S., Jackson, D. D., & Rhoades, D. A. (2007). Earthquake likelihood model testing. *Seismological Research Letters*, 78, 17--29. https://doi.org/10.1785/gssrl.78.1.17
8. Zechar, J. D., Gerstenberger, M. C., & Rhoades, D. A. (2010). Likelihood-based tests for evaluating space-rate-magnitude earthquake forecasts. *Bulletin of the Seismological Society of America*, 100, 1184--1195. https://doi.org/10.1785/0120090192
9. Rhoades, D. A., Schorlemmer, D., Gerstenberger, M. C., Christophersen, A., Zechar, J. D., & Imoto, M. (2011). Efficient testing of earthquake forecasting models. *Acta Geophysica*, 59, 728--747. https://doi.org/10.2478/s11600-011-0013-5
10. Rhoades, D. A., et al. (2010). Establishing a New Zealand earthquake forecast testing centre. *Pure and Applied Geophysics*, 167, 877--892. https://doi.org/10.1007/s00024-010-0082-4
11. Mizrahi, L., Nandan, S., & Wiemer, S. (2024). Developing, testing, and communicating earthquake forecasts: Current practices and future directions. *Reviews of Geophysics*, 62, e2023RG000823. https://doi.org/10.1029/2023RG000823
12. Mizrahi, L., Nandan, S., & Wiemer, S. (2021). Embracing data incompleteness for better earthquake forecasting. *Journal of Geophysical Research: Solid Earth*, 126, e2021JB022379. https://doi.org/10.1029/2021JB022379
13. Field, E. H., et al. (2013). Uniform California Earthquake Rupture Forecast, Version 3 (UCERF3): The time-independent model. U.S. Geological Survey Open-File Report 2013-1165. https://doi.org/10.3133/ofr20131165
14. Mizrahi, L., Nandan, S., & Wiemer, S. (2022). ETAS: Python package for fitting and simulating epidemic-type aftershock sequence models. Zenodo. https://doi.org/10.5281/zenodo.6583992
