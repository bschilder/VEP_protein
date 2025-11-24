# Epistasis Metrics Explanation

## Available Metrics for Ranking Epistasis Strength

### 1. `deviation_from_additive` ⭐ **RECOMMENDED FOR MAGNITUDE**
- **Definition**: `joint_effect - expected_additive_effect`
- **Units**: Same as VEP scores (directly interpretable)
- **Interpretation**: The raw difference between what we observe and what we'd expect if effects were purely additive
- **Use case**: Best for ranking the **magnitude** of epistatic effects
- **Example**: If deviation = -0.5, the joint effect is 0.5 units more pathogenic than expected from additivity

### 2. `delta_r2` ⭐ **RECOMMENDED FOR MODEL IMPROVEMENT**
- **Definition**: `R²_interaction - R²_additive`
- **Units**: Unitless (0 to 1 scale)
- **Interpretation**: How much the model fit improves when we add the epistasis term
- **Use case**: Best for ranking epistasis by **statistical/model improvement**
- **Limitation**: Depends on sample size and variance - small improvements can still be significant
- **Example**: delta_r2 = 0.1 means the epistasis term explains 10% additional variance

### 3. `epistasis_coefficient` (β₂) - **FIXED BUG**
- **Definition**: Coefficient of the epistasis term in the interaction model
- **Units**: Scaled by the deviation_from_baseline term
- **Interpretation**: The strength of the epistasis interaction term in the model
- **Use case**: Model parameter, but less directly interpretable than deviation_from_additive
- **Note**: This was previously buggy (accessing coef_[3] instead of coef_[2]) - now fixed

### 4. `epistasis_fstat` ⭐ **RECOMMENDED FOR STATISTICAL STRENGTH**
- **Definition**: F-statistic from comparing additive vs interaction models
- **Units**: Unitless (higher = more significant)
- **Interpretation**: Statistical strength of the epistasis signal
- **Use case**: Best for ranking by **statistical significance/strength**
- **Example**: F-stat = 10 means the interaction model is significantly better

### 5. `epistasis_pvalue` - **FOR FILTERING**
- **Definition**: P-value from F-test
- **Units**: Probability (0 to 1)
- **Interpretation**: Statistical significance of epistasis
- **Use case**: Filtering (p < 0.05) rather than ranking
- **Note**: Lower is better, so use `-epistasis_pvalue` for ranking

### 6. `interaction_strength` - **ORIGINAL METRIC**
- **Definition**: Absolute value of the joint effect coefficient from main model
- **Units**: Same as VEP scores
- **Interpretation**: Overall interaction strength (includes both additive and epistatic components)
- **Use case**: General interaction strength, but doesn't distinguish additive vs epistatic

## Recommended Ranking Strategies

### For Biological Magnitude:
```python
sort_interactions = "deviation_from_additive"  # or abs(deviation_from_additive)
```
- Directly interpretable in VEP units
- Shows how much more/less pathogenic/benign than expected

### For Statistical Strength:
```python
sort_interactions = "epistasis_fstat"  # or "-epistasis_pvalue"
```
- Ranks by how statistically significant the epistasis is
- Good for finding robust epistatic signals

### For Model Improvement:
```python
sort_interactions = "delta_r2"
```
- Ranks by how much the epistasis term improves model fit
- Good for finding epistasis that explains substantial variance

### Combined Approach (Recommended):
```python
# Filter for significant epistasis first, then rank by magnitude
filter_interactions = lambda df: df["is_epistatic"] == True
sort_interactions = "deviation_from_additive"  # or abs(deviation_from_additive)
```

## Key Differences

| Metric | Measures | Units | Best For |
|--------|----------|-------|----------|
| `deviation_from_additive` | Magnitude of epistatic effect | VEP units | **Biological interpretation** |
| `delta_r2` | Model improvement | Unitless (0-1) | Statistical/model fit |
| `epistasis_fstat` | Statistical strength | Unitless | **Statistical significance** |
| `epistasis_pvalue` | Statistical significance | Probability | Filtering |
| `epistasis_coefficient` | Model parameter | Scaled | Model interpretation |

## Recommendation

For visualizing epistatic interactions on structure, use:
- **Filter**: `filter_interactions = lambda df: df["is_epistatic"] == True`
- **Sort**: `sort_interactions = "deviation_from_additive"` (or `abs(deviation_from_additive)` for magnitude regardless of direction)

This will show the strongest epistatic effects (by magnitude) that are statistically significant.

