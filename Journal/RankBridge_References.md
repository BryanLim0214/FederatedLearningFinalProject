# RankBridge — Corrected BibTeX Reference File

**Paper:** *"RankBridge: Mitigating Heterogeneity in Federated Phishing Detection via Privacy-Preserving Rank-Based Explanation Clustering"*

All 15 entries verified against official publisher records. Errors from the original user-provided citations are noted inline with `% FIX:` comments. Copy the block below directly into your `.bib` file.

---

## Summary

| Stat | Count |
|------|-------|
| Total entries | 15 |
| Verified | 15 / 15 |
| Entries with corrections | 8 (sources 1, 2, 3, 6, 8, 10, 12, 13) |
| arXiv-preprint-only entries | 2 (sources 5, 11) — flagged below |

---

## BibTeX Entries

```bibtex
% ============================================================
%  RankBridge — Corrected BibTeX Reference File
%  Paper: "RankBridge: Mitigating Heterogeneity in Federated
%  Phishing Detection via Privacy-Preserving Rank-Based
%  Explanation Clustering"
%
%  All 15 entries verified against official publisher records.
%  Errors from original user-provided citations are noted
%  inline with % FIX: comments.
% ============================================================


% ────────────────────────────────────────────────────────────
%  CATEGORY 1: Federated Learning Foundations
% ────────────────────────────────────────────────────────────

% [1] McMahan et al. 2017 — FedAvg
% FIX: year corrected from 2023 (arXiv export date) → 2017 (AISTATS publication year)
% FIX: entry type changed from @misc → @inproceedings with proper AISTATS booktitle
@inproceedings{mcmahan2017communication,
  title         = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author        = {McMahan, H. Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and {Ag{\"u}era y Arcas}, Blaise},
  booktitle     = {Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series        = {Proceedings of Machine Learning Research},
  volume        = {54},
  pages         = {1273--1282},
  year          = {2017},
  publisher     = {PMLR},
  url           = {https://arxiv.org/abs/1602.05629},
  eprint        = {1602.05629},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% [2] Li et al. 2020 — FedProx
% FIX: journal field "arXiv: Learning" is not a real journal — corrected to MLSys 2020 @inproceedings
% FIX: arXiv ID added (1812.06127); Semantic Scholar URL replaced with proper arXiv URL
% FIX: author order corrected — Tian Li is first author per arXiv and MLSys proceedings
% FIX: year corrected from 2018 (arXiv submission) → 2020 (MLSys publication year)
@inproceedings{li2020fedprox,
  title         = {Federated Optimization in Heterogeneous Networks},
  author        = {Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  booktitle     = {Proceedings of Machine Learning and Systems (MLSys)},
  pages         = {429--450},
  year          = {2020},
  url           = {https://arxiv.org/abs/1812.06127},
  eprint        = {1812.06127},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}


% ────────────────────────────────────────────────────────────
%  CATEGORY 2: Non-IID / Statistical Heterogeneity in FL
% ────────────────────────────────────────────────────────────

% [3] Li et al. 2020 — IEEE Signal Processing Magazine survey
% FIX: author "Anit Kumar Sahu" was missing from the BibTeX author field
% VERIFIED: DOI 10.1109/MSP.2020.2975749, vol. 37, no. 3, pp. 50–60, May 2020
@article{li2020challenges,
  title   = {Federated Learning: Challenges, Methods, and Future Directions},
  author  = {Li, Tian and Sahu, Anit Kumar and Talwalkar, Ameet and Smith, Virginia},
  journal = {IEEE Signal Processing Magazine},
  volume  = {37},
  number  = {3},
  pages   = {50--60},
  year    = {2020},
  month   = {May},
  doi     = {10.1109/MSP.2020.2975749},
  url     = {https://arxiv.org/abs/1908.07873}
}

% [4] Kairouz et al. 2021 — Advances and Open Problems in FL
% NO ERRORS FOUND. Verified: Foundations and Trends in ML, vol. 14, no. 1–2, pp. 1–210
@article{kairouz2021advances,
  title   = {Advances and Open Problems in Federated Learning},
  author  = {Kairouz, Peter and McMahan, H. Brendan and Avent, Brendan and Bellet, Aur{\'e}lien and Bennis, Mehdi and Bhagoji, Arjun Nitin and Bonawitz, Kallista and Charles, Zachary and Cormode, Graham and Cummings, Rachel and others},
  journal = {Foundations and Trends in Machine Learning},
  volume  = {14},
  number  = {1--2},
  pages   = {1--210},
  year    = {2021},
  doi     = {10.1561/2200000083},
  url     = {https://arxiv.org/abs/1912.04977}
}

% [5] Hsu et al. 2019 — Dirichlet non-IID partitioning
% NO CITATION ERRORS FOUND.
% STATUS NOTE: arXiv preprint only — never formally peer-reviewed or
%   published in a conference/journal. Widely cited (1,395+ citations).
%   Flag if your venue requires all sources to be peer-reviewed.
@article{hsu2019measuring,
  title         = {Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification},
  author        = {Hsu, Tzu-Ming Harry and Qi, Hang and Brown, Matthew},
  journal       = {CoRR},
  volume        = {abs/1909.06335},
  year          = {2019},
  url           = {https://arxiv.org/abs/1909.06335},
  eprint        = {1909.06335},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}


% ────────────────────────────────────────────────────────────
%  CATEGORY 3: Clustered Federated Learning
% ────────────────────────────────────────────────────────────

% [6] Ghosh et al. 2020 — IFCA (NeurIPS 2020)
% FIX: year corrected from 2021 (arXiv v2 revision date) → 2020 (NeurIPS 2020 publication year)
% FIX: entry type changed from @misc → @inproceedings with NeurIPS 2020 booktitle
@inproceedings{ghosh2020ifca,
  title         = {An Efficient Framework for Clustered Federated Learning},
  author        = {Ghosh, Avishek and Chung, Jichan and Yin, Dong and Ramchandran, Kannan},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume        = {33},
  pages         = {19586--19597},
  year          = {2020},
  url           = {https://arxiv.org/abs/2006.04088},
  eprint        = {2006.04088},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}

% [7] Sattler et al. 2021 — CFL (IEEE TNNLS)
% NO ERRORS FOUND. Verified: IEEE TNNLS vol. 32, no. 8, pp. 3710–3722, Aug. 2021
%   PubMed PMID 32833654 confirms DOI 10.1109/TNNLS.2020.3015958
@article{sattler2021cfl,
  title   = {Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints},
  author  = {Sattler, Felix and M{\"u}ller, Klaus-Robert and Samek, Wojciech},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {32},
  number  = {8},
  pages   = {3710--3722},
  year    = {2021},
  month   = {Aug},
  doi     = {10.1109/TNNLS.2020.3015958},
  url     = {https://arxiv.org/abs/1910.01991}
}


% ────────────────────────────────────────────────────────────
%  CATEGORY 4: Explainable AI / SHAP
% ────────────────────────────────────────────────────────────

% [8] Lundberg & Lee 2017 — SHAP (NeurIPS 2017)
% FIX (minor): user-provided pages 4766–4777; corrected to 4765–4774 per
%   NeurIPS 2017 proceedings (ACM DL record). All other fields correct.
@inproceedings{lundberg2017shap,
  title         = {A Unified Approach to Interpreting Model Predictions},
  author        = {Lundberg, Scott M. and Lee, Su-In},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume        = {30},
  pages         = {4765--4774},
  year          = {2017},
  url           = {https://papers.nips.cc/paper/7062},
  eprint        = {1705.07874},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% [9] Lundberg et al. 2020 — TreeSHAP (Nature Machine Intelligence)
% NO ERRORS FOUND. Verified: Nature Machine Intelligence vol. 2, no. 1, pp. 56–67, Jan. 2020
%   DOI 10.1038/s42256-019-0138-9 confirmed on Nature.com
@article{lundberg2020treeshap,
  title   = {From Local Explanations to Global Understanding with Explainable {AI} for Trees},
  author  = {Lundberg, Scott M. and Erion, Gabriel and Chen, Hugh and DeGrave, Alex and Prutkin, Jordan M. and Nair, Bala and Katz, Ronit and Himmelfarb, Jonathan and Bansal, Nisha and Lee, Su-In},
  journal = {Nature Machine Intelligence},
  volume  = {2},
  number  = {1},
  pages   = {56--67},
  year    = {2020},
  month   = {Jan},
  doi     = {10.1038/s42256-019-0138-9},
  url     = {https://www.nature.com/articles/s42256-019-0138-9}
}


% ────────────────────────────────────────────────────────────
%  CATEGORY 5: Privacy in Federated Learning
% ────────────────────────────────────────────────────────────

% [10] Zhu et al. 2019 — Deep Leakage from Gradients (NeurIPS 2019)
% FIX: entry type changed from @misc → @inproceedings (NeurIPS 2019, Vancouver)
%   The paper was published at NeurIPS 2019, not just posted to arXiv.
@inproceedings{zhu2019deepleakage,
  title         = {Deep Leakage from Gradients},
  author        = {Zhu, Ligeng and Liu, Zhijian and Han, Song},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume        = {32},
  pages         = {14774--14784},
  year          = {2019},
  url           = {https://arxiv.org/abs/1906.08935},
  eprint        = {1906.08935},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% [11] Lyu et al. 2020 — Threats to Federated Learning (arXiv survey)
% NO CITATION ERRORS FOUND.
% STATUS NOTE: arXiv preprint only — no peer-reviewed journal publication
%   confirmed under this title. Widely cited (1,000+). Flag if your venue
%   requires all sources to be peer-reviewed.
@misc{lyu2020threats,
  title         = {Threats to Federated Learning: A Survey},
  author        = {Lyu, Lingjuan and Yu, Han and Yang, Qiang},
  year          = {2020},
  eprint        = {2003.02133},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2003.02133}
}


% ────────────────────────────────────────────────────────────
%  CATEGORY 6: Phishing Detection / Cybersecurity ML
% ────────────────────────────────────────────────────────────

% [12] Mamun et al. 2016 — ISCX-URL2016 dataset paper
% FIX: first author name "Mamun, Mohammad" corrected to full name
%   "Mamun, Mohammad Saiful Islam"
% FIX: second author "Rathore, Muhammad" corrected to "Rathore, Mohammad Ahmad"
% FIX: last author "Ghorbani, Ali" corrected to "Ghorbani, Ali A."
% FIX: Added series = {LNCS} and proper booktitle for NSS 2016
@inproceedings{mamun2016detecting,
  title     = {Detecting Malicious {URLs} Using Lexical Analysis},
  author    = {Mamun, Mohammad Saiful Islam and Rathore, Mohammad Ahmad and Lashkari, Arash Habibi and Stakhanova, Natalia and Ghorbani, Ali A.},
  booktitle = {Proceedings of the 10th International Conference on Network and System Security (NSS 2016)},
  series    = {Lecture Notes in Computer Science},
  volume    = {9955},
  pages     = {467--482},
  year      = {2016},
  month     = {Sep},
  publisher = {Springer},
  doi       = {10.1007/978-3-319-46298-1_30},
  isbn      = {978-3-319-46297-4}
}

% [13] Salloum et al. 2022 — Phishing email detection NLP survey (IEEE Access)
% FIX: pages field was empty ({}) — corrected to 65703--65727
% VERIFIED: DOI 10.1109/ACCESS.2022.3183083 confirmed on IEEE Xplore,
%   published June 14, 2022, vol. 10, pp. 65703–65727
@article{salloum2022phishing,
  title   = {A Systematic Literature Review on Phishing Email Detection Using Natural Language Processing Techniques},
  author  = {Salloum, Said and Gaber, Tarek and Vadera, Sunil and Shaalan, Khaled},
  journal = {IEEE Access},
  volume  = {10},
  pages   = {65703--65727},
  year    = {2022},
  month   = {Jun},
  doi     = {10.1109/ACCESS.2022.3183083},
  url     = {https://ieeexplore.ieee.org/document/9795286}
}


% ────────────────────────────────────────────────────────────
%  CATEGORY 7: Rank Statistics & Hierarchical Clustering
% ────────────────────────────────────────────────────────────

% [14] Ward 1963 — Ward's hierarchical clustering (JASA)
% NO ERRORS FOUND. DOI 10.2307/2282967 is the correct JSTOR DOI
%   for this paper. Verified against Taylor & Francis
%   (publisher DOI: 10.1080/01621459.1963.10500845), Semantic Scholar
%   (13,418 CrossRef citations), and ScienceOpen.
@article{ward1963hierarchical,
  title   = {Hierarchical Grouping to Optimize an Objective Function},
  author  = {Ward, Joe H.},
  journal = {Journal of the American Statistical Association},
  volume  = {58},
  number  = {301},
  pages   = {236--244},
  year    = {1963},
  month   = {Mar},
  doi     = {10.2307/2282967},
  url     = {https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500845}
}

% [15] Spearman 1904 — Spearman rank correlation (AJP)
% NO ERRORS FOUND. DOI 10.2307/1412159 is the correct JSTOR DOI
%   for Spearman (1904). Verified against JSTOR stable record and
%   PubMed commentary (PMID 21051367).
@article{spearman1904proof,
  title   = {The Proof and Measurement of Association between Two Things},
  author  = {Spearman, C.},
  journal = {The American Journal of Psychology},
  volume  = {15},
  number  = {1},
  pages   = {72--101},
  year    = {1904},
  month   = {Jan},
  doi     = {10.2307/1412159},
  url     = {https://www.jstor.org/stable/1412159}
}

% ============================================================
%  END OF FILE
%  Total entries: 15
%  Verified: 15/15
%  Entries with corrections: 8 (sources 1, 2, 3, 6, 8, 10, 12, 13)
%  arXiv-preprint-only entries: 2 (sources 5, 11) — flagged above
% ============================================================
```
