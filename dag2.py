import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset (semicolon-separated)
df = pd.read_csv("asthma_data.csv", sep=",")

# Preprocess dataset according to multinomial states
def preprocess_data(df):
    processed = pd.DataFrame()
    processed['S'] = df['sex'].map({'male': 'M', 'female': 'F'})
    processed['Sm'] = df['smoke'].map({'yes': 'T', 'no': 'F'})
    processed['E'] = df['education'].map({'high': 'H'}).fillna('L')
    processed['Sed'] = df['sedentary'].map({'yes': 'T', 'no': 'F'})
    processed['A'] = df['allergy'].map({'yes': 'T', 'no': 'F'})
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
    'S': defaultdict(lambda: epsilon),
    'Sm': defaultdict(lambda: epsilon),
    'E': defaultdict(lambda: epsilon),
    'Sed': defaultdict(lambda: epsilon),   # P(Sed | Sm, E)
    'A': defaultdict(lambda: epsilon),     # P(Allergy | S)
    'AS': defaultdict(lambda: epsilon)     # P(Asthma | A, Sed)
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
        get_prob('S', row['S'], []) *
        get_prob('Sm', row['Sm'], []) *
        get_prob('E', row['E'], []) *
        get_prob('Sed', row['Sed'], [row['Sm'], row['E']]) *
        get_prob('A', row['A'], [row['S']]) *
        get_prob('AS', 'T', [row['A'], row['Sed']])
    )

    prob_F = (
        get_prob('S', row['S'], []) *
        get_prob('Sm', row['Sm'], []) *
        get_prob('E', row['E'], []) *
        get_prob('Sed', row['Sed'], [row['Sm'], row['E']]) *
        get_prob('A', row['A'], [row['S']]) *
        get_prob('AS', 'F', [row['A'], row['Sed']])
    )

    # not normalized
    # predicted = 'T' if prob_T >= prob_F else 'F'
    # normalized
    prob_norm = prob_T / (prob_T + prob_F)
    # predicted = 'T' if prob_norm >= 0.5 else 'F' # using original
    predicted = 'T' if prob_norm >= 0.5 else 'F'    # using best threshold

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
    counts['S'][row['S']] += 1
    counts['Sm'][row['Sm']] += 1
    counts['E'][row['E']] += 1
    counts['Sed'][key(row['Sm'], row['E'], row['Sed'])] += 1
    counts['A'][key(row['S'], row['A'])] += 1
    counts['AS'][key(row['A'], row['Sed'], row['AS'])] += 1
    

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
plt.savefig("performance2.png")

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, probs_asthma)
roc_auc = auc(fpr, tpr)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
print(best_threshold)

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
plt.savefig("roc_curve2.png")

##### asthma count #####
# asthma_counts = df['AS'].value_counts()
# print("Actual counts:")
# print("Asthma (T):", asthma_counts.get('T', 0))
# print("No Asthma (F):", asthma_counts.get('F', 0))

#### tune threshold ######
# Sweep thresholds from 0.0 to 1.0
thresholds_to_test = np.linspace(0.0, 1.0, 101)
best_f1 = 0
best_threshold = 0
f1_scores = []

for threshold in thresholds_to_test:
    predicted_labels = ['T' if p >= threshold else 'F' for p in probs_asthma]
    
    # Convert to binary
    predicted_binary = [1 if label == 'T' else 0 for label in predicted_labels]

    # Compute F1
    f1 = f1_score(true_labels, predicted_binary)
    f1_scores.append(f1)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Plot F1 vs Threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds_to_test, f1_scores, label="F1 Score")
plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("f1_vs_threshold.png")