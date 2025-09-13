# Analysis methods

`fev` provides 3 main methods for aggregating the evaluation summaries produced by [`Task.evaluation_summary()`][fev.Task.evaluation_summary]:

- [`pivot_table()`][fev.pivot_table] - A table of model scores with tasks as index and model names as columns.
- [`leaderboard()`][fev.leaderboard] - Aggregate performance for each individual model.
- [`pairwise_comparison()`][fev.pairwise_comparison] - Aggregate performance for each pair of models.

On this page `SummaryType` is an alias for one of the following types:
::: fev.analysis.SummaryType
    options:
        heading_level: 4

## Functions
::: fev.leaderboard
    options:
        heading_level: 4

::: fev.pairwise_comparison
    options:
        heading_level: 4

::: fev.pivot_table
    options:
        heading_level: 4
