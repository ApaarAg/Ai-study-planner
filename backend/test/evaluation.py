import numpy as np
import os, sys
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.model_loader import predict_with_classification
from app.scheduler import generate_plan


# -------------------------
# 1. Test Topics
# -------------------------

topics = [
    {
        "topic_name": "Calculus",
        "difficulty": 5,
        "past_score": 40,
        "hours_spent": 3,
        "revision_count": 1,
        "days_to_exam": 3,
        "confidence": 2
    },
    {
        "topic_name": "Linear Algebra",
        "difficulty": 3,
        "past_score": 65,
        "hours_spent": 6,
        "revision_count": 3,
        "days_to_exam": 7,
        "confidence": 3
    },
    {
        "topic_name": "Machine Learning",
        "difficulty": 4,
        "past_score": 75,
        "hours_spent": 8,
        "revision_count": 4,
        "days_to_exam": 10,
        "confidence": 4
    }
]


# -------------------------
# 2. Predictions
# -------------------------

predictions = predict_with_classification(topics)

print("\nPredictions:")
for p in predictions:
    print(
        f"{p['topic_name']}: "
        f"Mean={p['predicted_gain']:.2f}, "
        f"Std={p['prediction_std']:.2f}"
    )


# -------------------------
# 3. Budget Stability Test
# -------------------------

budgets = [6, 8, 10]
allocations = []
rankings = []

print("\nBudget Allocations:")

for H in budgets:
    plan = generate_plan(predictions.copy(), H)
    alloc = [t["allocated_minutes"] for t in plan["study_plan"]]

    allocations.append(alloc)
    rankings.append(np.argsort(alloc))

    print(f"Budget {H}h → {alloc}")


# -------------------------
# 4. Ranking Stability
# -------------------------

corr1, _ = spearmanr(rankings[0], rankings[1])
corr2, _ = spearmanr(rankings[1], rankings[2])

print("\nRanking Stability:")
print(f"6h vs 8h  → {corr1:.3f}")
print(f"8h vs 10h → {corr2:.3f}")