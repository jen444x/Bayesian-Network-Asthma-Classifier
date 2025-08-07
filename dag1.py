import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, auc

# Load dataset (semicolon-separated)
df = pd.read_csv("asthma_data.csv", sep=",")

# Preprocess dataset according to multinomial states
def preprocess_data(df):
    processed = pd.DataFrame()
    processed['A'] = df['age'].map({'young': 'Y', 'adult': 'Ad', 'old': 'O'})
    processed['S'] = df['sex'].map({'male': 'M', 'female': 'F'})
    processed['E'] = df['education'].map({'high': 'H'}).fillna('L')
    processed['Sm'] = df['smoke'].map({'yes': 'T', 'no': 'F'})
    processed['U'] = df['urbanization'].map({'high': 'H', 'medium': 'M'}).fillna('L')
    processed['AS'] = df['asthma'].map({'yes': 'T', 'no': 'F'})
    return processed

df = preprocess_data(df)

# Small epsilon to prevent zero probabilities
epsilon = 1e-6

# Helper to build keys
def key(*args):
    return '_'.join(args)

# Count tables
counts = {
    'A': defaultdict(lambda: epsilon),
    'S': defaultdict(lambda: epsilon),
    'E': defaultdict(lambda: epsilon),
    'Sm': defaultdict(lambda: epsilon),  # P(Sm | A, S)
    'U': defaultdict(lambda: epsilon),   # P(U | E)
    'AS': defaultdict(lambda: epsilon)   # P(AS | Sm, U)
}

accuracies = []
correct = 0
total = 0

# So: var = 'Sm', val = 'T', parents = ['Y', 'M']
# key('Y', 'M', 'T') → "Y_M_T"
# table = counts['Sm']
# It sums counts of all values like "Y_M_T" and "Y_M_F", then divides the count of "Y_M_T" by the total.
def get_prob(var, val, parents):
    table = counts[var]
    prefix = key(*parents)
    
    # Find all matching keys for this parent combination
    matching_keys = [k for k in table if k.startswith(prefix + '_') or (prefix == '' and '_' not in k)]
    value_keys = set(k.split('_')[-1] for k in matching_keys)

    total_count = sum(table[key(prefix, v)] for v in value_keys)

    if total_count > 0:
        return table[key(prefix, val)] / total_count
    else:
        return 1.0 / max(len(value_keys), 2)  # Avoid divide-by-zero, assume at least 2 values

# Initialize performance values
precisions = []
recalls = []
f1s = []
# Store probabilities and true labels for ROC
probs_asthma = []  # predicted probability asthma = T
true_labels = []   # 1 if actual == 'T', else 0

TP = 0
FP = 0
FN = 0

asthma_probabilities_over_time = []

for _, row in df.iterrows():
    # Inference
    prob_T = (
        get_prob('A', row['A'], []) *
        get_prob('S', row['S'], []) *
        get_prob('E', row['E'], []) *
        # get_prob('Sm', 'T', [row['A'], row['S']]) *   # always assumes the person smokes
        get_prob('Sm', row['Sm'], [row['A'], row['S']]) * # use the actual smoking status
        get_prob('U', row['U'], [row['E']]) *
        # get_prob('AS', 'T', ['T', row['U']]) # always uses smoker=True in asthma 
        get_prob('AS', 'T', [row['Sm'], row['U']])  # uses actual smoking status
    )
    prob_F = (
        get_prob('A', row['A'], []) *
        get_prob('S', row['S'], []) *
        get_prob('E', row['E'], []) *
        # get_prob('Sm', 'T', [row['A'], row['S']]) *
        get_prob('Sm', row['Sm'], [row['A'], row['S']]) *
        get_prob('U', row['U'], [row['E']]) *
        # get_prob('AS', 'F', ['T', row['U']])
        get_prob('AS', 'F', [row['Sm'], row['U']])
    )

    # predicted = 'T' if prob_T >= prob_F else 'F'    # not normalized
    prob_norm = prob_T / (prob_T + prob_F)      # normalized
    
    predicted = 'T' if prob_norm >= 0.5 else 'F'
    actual = row['AS']
    
    
    asthma_probabilities_over_time.append({
    'Iteration': total + 1,
    'P(asthma=T)': prob_norm,
    'Actual_Label': actual
    })

    probs_asthma.append(prob_norm)
    true_labels.append(1 if actual == 'T' else 0)

    if predicted == actual:
        correct += 1
    total += 1
    accuracies.append(correct / total)

    # Track TP, FP, FN for F1
    if predicted == 'T':
        if actual == 'T':
            TP += 1
        else:
            FP += 1
    elif actual == 'T':
        FN += 1

    # Calculate Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    # Update counts
    counts['A'][row['A']] += 1
    counts['S'][row['S']] += 1
    counts['E'][row['E']] += 1
    counts['Sm'][key(row['A'], row['S'], row['Sm'])] += 1
    counts['U'][key(row['E'], row['U'])] += 1
    counts['AS'][key(row['Sm'], row['U'], row['AS'])] += 1



asthma_prob_df = pd.DataFrame(asthma_probabilities_over_time)
asthma_prob_df.to_csv('asthma_probabilities.csv', index=False)

print(asthma_prob_df)  # show first few rows

plt.figure(figsize=(10, 6))
plt.scatter(
    asthma_prob_df['Iteration'], 
    asthma_prob_df['P(asthma=T)'], 
    label='P(asthma=T)', 
    s=10,  # size of the dots
    alpha=0.7  # transparency to help see overlapping points
)
plt.xlabel("Iteration")
plt.ylabel("Probability")
plt.title("Predicted Probability of Asthma Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("asthma_probability_over_time.png")


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracies)+1), accuracies, label="Accuracy")
plt.plot(range(1, len(precisions)+1), precisions, label="Precision")
plt.plot(range(1, len(recalls)+1), recalls, label="Recall")
plt.plot(range(1, len(f1s)+1), f1s, label="F1 Score")
plt.xlabel("Cases")
plt.ylabel("Performance Metric")
plt.title("Accuracy, Precision, Recall, and F1 Score Over Time")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("performance1.png")

# asthma_counts = df['AS'].value_counts()
# print("Actual counts:")
# print("Asthma (T):", asthma_counts.get('T', 0))
# print("No Asthma (F):", asthma_counts.get('F', 0))

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, probs_asthma)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve - Bayesian Asthma Classifier')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve1.png")
 