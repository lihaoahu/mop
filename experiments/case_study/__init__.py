"""Application to platform discourse on vehicle energy types (Section 3.2).

``estimator``  load the corpus, fit, select the count
``fits``       Tables 2 and 4, and the yearly table
``dim_curves`` Table 3, and the BIC curve behind each dimension
``bic_curve``  Figure 3
``bootstrap``  article-level bootstrap of the selected count
``holdout``    the cross-validation tables and the retained-gain column
``specification``  free versus pinned dispersion
``curvature``  the second derivative that identifies the dispersion
``claims``     the refit on the six named dimensions
``audit``      what the low-weight component is made of
``tables_check``  every printed number against its artifact
"""

