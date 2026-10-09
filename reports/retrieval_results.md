Questions: 140 (excluding out-of-scope) | chunks: 634 | embedder: all-MiniLM-L6-v2-int8

| System | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | R@5 lay | R@5 legal | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.443 | 0.564 | 0.636 | 0.786 | 0.536 | 0.593 | 0.536 | 1.000 | 0.88 |
| dense | 0.621 | 0.807 | 0.857 | 0.914 | 0.723 | 0.762 | 0.818 | 1.000 | 4.48 |
| hybrid_rrf | 0.593 | 0.786 | 0.843 | 0.921 | 0.702 | 0.750 | 0.800 | 1.000 | 5.01 |
| hybrid_weighted_cv | 0.657 | 0.814 | 0.857 | 0.936 | 0.751 | 0.788 | 0.818 | 1.000 | 5.59 |
| hybrid+ltr_rerank_cv | 0.600 | 0.750 | 0.814 | 0.921 | 0.695 | 0.743 | 0.764 | 1.000 | 10.79 |
| bm25+expansion | 0.493 | 0.621 | 0.700 | 0.836 | 0.590 | 0.644 | 0.618 | 1.000 | 1.09 |
| dense+expansion | 0.650 | 0.829 | 0.893 | 0.950 | 0.759 | 0.797 | 0.864 | 1.000 | 4.18 |
| hybrid_weighted_cv+expansion | 0.686 | 0.879 | 0.921 | 0.957 | 0.787 | 0.822 | 0.900 | 1.000 | 5.91 |

Held-out set (38 questions written after the glossary was frozen):

| System | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | 0.526 | 0.684 | 0.816 | 0.842 | 0.626 |
| dense | 0.553 | 0.816 | 0.895 | 0.974 | 0.705 |
| hybrid_weighted_a0.8 | 0.632 | 0.842 | 0.947 | 0.974 | 0.756 |
| bm25+expansion | 0.632 | 0.895 | 0.921 | 0.947 | 0.749 |
| dense+expansion | 0.684 | 0.895 | 0.974 | 1.000 | 0.803 |
| hybrid_weighted_a0.8+expansion | 0.816 | 0.947 | 0.974 | 0.974 | 0.879 |

Best by MRR@10: **hybrid_weighted_cv+expansion** - 17 questions with the gold article outside the top 3.
