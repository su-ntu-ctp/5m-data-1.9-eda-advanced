import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Example 1: Same relationship, different scales
print("=" * 70)
print("EXAMPLE 1: Same Relationship, Different Scales")
print("=" * 70)

# Dataset 1: Height (cm) vs Weight (kg)
height_cm = np.random.normal(170, 10, 100)
weight_kg = height_cm * 0.5 + np.random.normal(0, 5, 100)

# Dataset 2: Height (inches) vs Weight (pounds)
height_inches = height_cm / 2.54
weight_pounds = weight_kg * 2.20462

# Calculate covariance
cov_metric1 = np.cov(height_cm, weight_kg)[0, 1]
cov_metric2 = np.cov(height_inches, weight_pounds)[0, 1]

# Calculate correlation
corr_metric1 = np.corrcoef(height_cm, weight_kg)[0, 1]
corr_metric2 = np.corrcoef(height_inches, weight_pounds)[0, 1]

print(f"\nDataset 1: Height (cm) vs Weight (kg)")
print(f"  Covariance: {cov_metric1:.2f}")
print(f"  Correlation: {corr_metric1:.4f}")

print(f"\nDataset 2: Height (inches) vs Weight (pounds)")
print(f"  Covariance: {cov_metric2:.2f}")
print(f"  Correlation: {corr_metric2:.4f}")

print(f"\n⚠️  OBSERVATION:")
print(f"  - Covariances are VERY DIFFERENT ({cov_metric1:.2f} vs {cov_metric2:.2f})")
print(f"  - But they represent the SAME relationship!")
print(f"  - Correlations are IDENTICAL ({corr_metric1:.4f} vs {corr_metric2:.4f})")
print(f"  - This is because correlation is scale-independent")

# Example 2: Different strength of relationships
print("\n" + "=" * 70)
print("EXAMPLE 2: Different Strength of Relationships")
print("=" * 70)

# Weak relationship
X_weak = np.random.normal(100, 15, 100)
Y_weak = X_weak * 0.2 + np.random.normal(0, 20, 100)

# Strong relationship
X_strong = np.random.normal(100, 15, 100)
Y_strong = X_strong * 0.9 + np.random.normal(0, 5, 100)

# Calculate metrics
cov_weak = np.cov(X_weak, Y_weak)[0, 1]
corr_weak = np.corrcoef(X_weak, Y_weak)[0, 1]

cov_strong = np.cov(X_strong, Y_strong)[0, 1]
corr_strong = np.corrcoef(X_strong, Y_strong)[0, 1]

print(f"\nWeak Relationship:")
print(f"  Covariance: {cov_weak:.2f}")
print(f"  Correlation: {corr_weak:.4f}")

print(f"\nStrong Relationship:")
print(f"  Covariance: {cov_strong:.2f}")
print(f"  Correlation: {corr_strong:.4f}")

print(f"\n⚠️  OBSERVATION:")
print(f"  - Covariances are similar in magnitude ({cov_weak:.2f} vs {cov_strong:.2f})")
print(f"  - But correlations clearly show the difference!")
print(f"  - Weak correlation: {corr_weak:.4f} (closer to 0)")
print(f"  - Strong correlation: {corr_strong:.4f} (closer to 1)")

# Example 3: Opposite relationships
print("\n" + "=" * 70)
print("EXAMPLE 3: Opposite Relationships")
print("=" * 70)

X = np.linspace(0, 10, 100)

# Positive relationship
Y_positive = X + np.random.normal(0, 1, 100)

# Negative relationship
Y_negative = -X + np.random.normal(0, 1, 100)

cov_pos = np.cov(X, Y_positive)[0, 1]
corr_pos = np.corrcoef(X, Y_positive)[0, 1]

cov_neg = np.cov(X, Y_negative)[0, 1]
corr_neg = np.corrcoef(X, Y_negative)[0, 1]

print(f"\nPositive Relationship (X increases, Y increases):")
print(f"  Covariance: {cov_pos:.2f}")
print(f"  Correlation: {corr_pos:.4f}")

print(f"\nNegative Relationship (X increases, Y decreases):")
print(f"  Covariance: {cov_neg:.2f}")
print(f"  Correlation: {corr_neg:.4f}")

print(f"\n⚠️  OBSERVATION:")
print(f"  - Both signs are clear in covariance AND correlation")
print(f"  - But correlation magnitude is bounded: -1 to 1")
print(f"  - Covariance magnitude is unbounded")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Correlation vs Covariance: Visual Examples', fontsize=16, fontweight='bold')

# Example 1 plots
axes[0, 0].scatter(height_cm, weight_kg, alpha=0.6, color='blue')
axes[0, 0].set_xlabel('Height (cm)')
axes[0, 0].set_ylabel('Weight (kg)')
axes[0, 0].set_title(f'Height vs Weight (cm, kg)\nCov: {cov_metric1:.2f}\nCorr: {corr_metric1:.4f}')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(height_inches, weight_pounds, alpha=0.6, color='green')
axes[0, 1].set_xlabel('Height (inches)')
axes[0, 1].set_ylabel('Weight (pounds)')
axes[0, 1].set_title(f'Height vs Weight (in, lbs)\nCov: {cov_metric2:.2f}\nCorr: {corr_metric2:.4f}')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].text(0.5, 0.5, 'SAME RELATIONSHIP\n\nDifferent scales cause\ndifferent covariances\n\nCorrelation remains\nidentical!',
                ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
axes[0, 2].axis('off')

# Example 2 plots
axes[1, 0].scatter(X_weak, Y_weak, alpha=0.6, color='orange')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')
axes[1, 0].set_title(f'Weak Relationship\nCov: {cov_weak:.2f}\nCorr: {corr_weak:.4f}')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(X_strong, Y_strong, alpha=0.6, color='purple')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
axes[1, 1].set_title(f'Strong Relationship\nCov: {cov_strong:.2f}\nCorr: {corr_strong:.4f}')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].text(0.5, 0.5, 'DIFFERENT STRENGTHS\n\nCovariances are\nsimilar in magnitude\n\nBut correlation clearly\nshows the difference!',
                ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('/Users/jawad/Documents/work/dsai/5m-data-1.9-eda-advanced/correlation_vs_covariance.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 70)
print("Visualization saved to: correlation_vs_covariance.png")
print("=" * 70)

# Summary Table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

summary_data = {
    'Aspect': ['Range', 'Units', 'Interpretability', 'Scale Dependent', 'Use Case'],
    'Covariance': ['-∞ to +∞', 'X units × Y units', 'Hard', 'Yes', 'Theoretical calculations'],
    'Correlation': ['-1 to +1', 'None (unitless)', 'Easy', 'No', 'Practical analysis']
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. SCALE DEPENDENCY
   - Covariance changes when you change units of measurement
   - Correlation stays the same regardless of units

2. INTERPRETABILITY
   - Correlation ranges from -1 to 1 (easy to understand)
   - Covariance is unbounded (hard to interpret magnitude)

3. WHEN TO USE
   - Use CORRELATION for: comparing relationships across datasets,
     understanding strength of relationship
   - Use COVARIANCE for: mathematical derivations, matrix calculations,
     variance-covariance matrices

4. RELATIONSHIP
   - Correlation is normalized covariance
   - Formula: Correlation = Covariance / (σₓ × σᵧ)
   - This normalization removes the scale dependency
""")
