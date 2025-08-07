# Asthma Prediction Expert System

A Bayesian Network-based expert system for medical diagnosis that predicts asthma risk using patient health data. The system implements iterative learning, starting with no prior training data and improving predictions as it processes each patient case.

## 🎯 Project Overview

This project explores the construction of an expert system for asthma prediction using Bayesian Networks (BN). The system models a real-world scenario where medical expert systems learn from experience and improve diagnostic accuracy over time.

**Key Features:**

- **Iterative Learning**: System starts with no training data and learns from each patient case
- **Two DAG Architectures**: Compared different causal relationship structures
- **Performance Optimization**: Achieved 90% recall through decision threshold tuning
- **Comprehensive Evaluation**: ROC curves, precision-recall analysis, and convergence tracking

## 📊 Results Summary

| Model                 | Accuracy | Precision | Recall   | F1-Score | AUC      |
| --------------------- | -------- | --------- | -------- | -------- | -------- |
| DAG 1                 | 55%      | 0.45      | 0.08     | 0.14     | 0.48     |
| DAG 2                 | 70%      | 0.75      | 0.48     | 0.58     | 0.68     |
| **DAG 2 (Optimized)** | **65%**  | **0.44**  | **0.90** | **0.61** | **0.68** |

## 🏗️ Architecture

### DAG Structure 1: Traditional Risk Factors

```
Age + Sex → Smoking → Asthma
Education → Urbanization → Asthma
```

### DAG Structure 2: Medical Risk Pathway (Better Performance)

```
Sex → Allergy → Asthma
Smoking + Education → Sedentary Lifestyle → Asthma
```

## 🔬 Dataset

- **Size**: 2,755 patient records
- **Features**: sex, age, urbanization, education, geographic_area, allergy, smoke, sedentary
- **Target**: asthma (binary classification)
- **Challenge**: Imbalanced dataset (43% positive cases)

## 🚀 Key Technical Achievements

### 1. Iterative Learning System

- Implemented incremental probability updates
- Started from uniform priors with epsilon smoothing
- Demonstrated convergence from random to stable predictions

### 2. Decision Threshold Optimization

- Systematic threshold tuning from 0.0 to 1.0
- Optimized for F1-score to balance precision and recall
- **Achieved 90% recall** (up from 48%) while maintaining reasonable precision

### 3. Comprehensive Model Comparison

- Designed and tested two different causal structures
- Analyzed trade-offs between conservative vs. adaptive learning
- Demonstrated importance of domain knowledge in DAG design

## 📈 Performance Analysis

### Learning Convergence

- **Early Phase** (0-500 iterations): High variance, rapid learning
- **Stabilization** (500+ iterations): Consistent predictions around learned probabilities
- **Plateau Effect**: Model reaches stable performance by iteration 1000

### Model Behavior Insights

- **DAG 1**: Overcautious, high accuracy but poor recall (missed 92% of asthma cases)
- **DAG 2**: Better feature relationships, improved recall and F1-score
- **Optimized**: Threshold tuning achieved clinical-grade recall for screening applications

## 🛠️ Implementation Details

### Technologies Used

- **Python**: Core implementation
- **Pandas/NumPy**: Data preprocessing and numerical computations
- **Matplotlib**: Performance visualization and analysis
- **Scikit-learn**: ROC curve analysis and evaluation metrics

### Key Algorithms

- **Bayesian Inference**: Joint probability calculations with conditional independence
- **Incremental Learning**: Real-time probability table updates
- **Threshold Optimization**: Grid search for optimal decision boundary

## 📋 Files Structure

```
├── dag.py                     # DAG Structure 1 implementation
├── dag2.py                    # DAG Structure 2 implementation
├── asthma_data.csv           # Patient dataset (2,755 records)
├── requirements.txt          # Python dependencies
├── results/                  # Generated visualizations
│   ├── performance1.png      # DAG 1 metrics over time
│   ├── performance2.png      # DAG 2 metrics over time
│   ├── roc_curve1.png        # DAG 1 ROC analysis
│   ├── roc_curve2.png        # DAG 2 ROC analysis
│   └── f1_vs_threshold.png   # Threshold optimization results
└── README.md
```

## 🔧 Installation & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Models

```bash
# Run DAG Structure 1
python dag.py

# Run DAG Structure 2 with optimization
python dag2.py
```

### Expected Output

- Performance metrics visualizations
- ROC curve analysis
- Probability predictions over time
- Optimal threshold recommendations

## 🎯 Real-World Applications

This expert system demonstrates practical applications in:

- **Medical Screening**: High recall configuration suitable for initial asthma screening
- **Risk Assessment**: Probability-based risk stratification for patient populations
- **Clinical Decision Support**: Integration with electronic health records
- **Adaptive Learning**: Template for other iterative medical diagnostic systems

## 🔍 Key Insights

### Medical Domain Knowledge Matters

The superior performance of DAG 2 highlights how proper feature relationships (sex→allergy→asthma, lifestyle→sedentary→asthma) outperform generic structures.

### Threshold Optimization Critical

Moving from 0.5 to 0.3 threshold increased recall from 48% to 90%, demonstrating the importance of application-specific optimization for medical screening.

### Trade-off Management

Successfully balanced the precision-recall trade-off, achieving clinically useful sensitivity while maintaining acceptable specificity.

## 🚀 Future Enhancements

- **Feature Engineering**: Additional patient risk factors (family history, environmental data)
- **Deep Learning Integration**: Neural network components for complex feature interactions
- **Temporal Modeling**: Longitudinal patient data for risk progression analysis
- **Uncertainty Quantification**: Confidence intervals for probability predictions

---

_This project demonstrates advanced machine learning techniques applied to healthcare, showcasing iterative learning, model comparison, and performance optimization in medical diagnostic systems._
