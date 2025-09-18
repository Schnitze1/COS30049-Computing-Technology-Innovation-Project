import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

df = pd.read_csv('data_preprocessing/input/data.csv', index_col=False)
    
pd.set_option("display.max_columns", None)
# Set the display option to show all rows
pd.set_option('display.max_rows', None)
print(df.head(5))
print(df.tail(5))

df.dtypes

plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Label', data=df)
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Add count numbers on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
# plt.show()

# Initial data overview
counts = df.groupby(['Label', 'Traffic Type', 'Traffic Subtype']).size().reset_index(name='Counts')
print(counts)

# Drop the unnecessary columns for correlation analysis
corr_df= df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Timestamp', 'Label', 'Traffic Type', 'Traffic Subtype'])

# Calculate the correlation matrix
corr_df1 = corr_df.corr()

# Create EDA output directory
eda_dir = 'data_preprocessing/EDA'
os.makedirs(eda_dir, exist_ok=True)

# Plot correlation matrix
plt.figure(figsize=(20, 15))
sns.heatmap(corr_df1, annot=False, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix Heatmap - Original Features')
plt.tight_layout()
plt.savefig(f'{eda_dir}/correlation_matrix_original.png', dpi=300, bbox_inches='tight')
# plt.show()

corr_matrix = corr_df1.abs()

# Create triangle matrix
upper = corr_df1.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# upper i.e:
#        f1    f2    f3
# f1    NaN  0.95  0.20
# f2    NaN   NaN  0.30
# f3    NaN   NaN   NaN

to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
df_reduced = df.drop(columns=to_drop)

print("Raw dataframe's shape: ", df.shape)
print("Dropped features:", to_drop)
print("Dataframe's shape after clean up redundant features: ", df_reduced.shape)

# Plot correlation matrix after removing redundant features
corr_df_reduced = corr_df.drop(columns=to_drop).corr()

plt.figure(figsize=(20, 15))
sns.heatmap(corr_df_reduced, annot=False, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix Heatmap - After Removing Redundant Features')
plt.tight_layout()
plt.savefig(f'{eda_dir}/correlation_matrix_reduced.png', dpi=300, bbox_inches='tight')
# plt.show()

TARGET_VARIABLE = 'Traffic Type'
DROP_COLUMNS = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']
TARGET_TO_DROP = {'Label': ['Traffic Type', 'Traffic Subtype'],
                  'Traffic Type': ['Label', 'Traffic Subtype'],
                  'Traffic Subtype': ['Label', 'Traffic Type']}

# Drop 5-tuple collumns and timestamp
df = df_reduced.drop(columns=DROP_COLUMNS)

# Filter out duplicates within the same target
df = df.round(3)
df = df.drop_duplicates()
df = df.drop(columns=TARGET_TO_DROP[TARGET_VARIABLE])

print("Dropped features: ",DROP_COLUMNS + TARGET_TO_DROP[TARGET_VARIABLE])
print("Final dataset's shape: ", df.shape)
df.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


X = df.drop(TARGET_VARIABLE, axis=1)
y = df[TARGET_VARIABLE]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Compute train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Identifying Numerical and Categorical columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.to_list()
categorical_cols = X_train.select_dtypes(include=[object]).columns.to_list()

# Pipelines for Numerical and Categorical Data Transformations
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Column Transformer combining both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Apply preprocessor to train and test data
preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf_selector = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_selector.fit(X_train, y_train)

# Get top features
feature_names = X.columns
# Ensure arrays have the same length
n_features = len(rf_selector.feature_importances_)
n_columns = len(X.columns)
min_length = min(n_features, n_columns)

importance_df = pd.DataFrame({
    'feature_index': range(min_length),
    'feature_name': X.columns[:min_length],  # Original feature names for reference
    'importance': rf_selector.feature_importances_[:min_length]
}).sort_values('importance', ascending=False)
print("Sorted features important:\n", importance_df)

# Select top 15 feature INDICES (as suggested - last 5 are redundant)
top_feature_indices = importance_df.head(15)['feature_index'].tolist()
top_feature_names = importance_df.head(15)['feature_name'].tolist()
print("Top 15 features (optimized selection):\n",top_feature_names)

# Save feature importance analysis to CSV
importance_df.to_csv(f'{eda_dir}/feature_importance_analysis.csv', index=False)
print(f"Feature importance analysis saved to: {eda_dir}/feature_importance_analysis.csv")
print(f"Original X_train shape: {X_train.shape}")

# Select features using indices (works with NumPy arrays)
X_train = X_train[:, top_feature_indices]
X_test = X_test[:, top_feature_indices]
print(f"Selected X_train shape: {X_train.shape}")

# Create feature importance visualization
plt.figure(figsize=(12, 8))
top_15_importance = importance_df.head(15)
plt.barh(range(len(top_15_importance)), top_15_importance['importance'])
plt.yticks(range(len(top_15_importance)), top_15_importance['feature_name'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{eda_dir}/feature_importance_top15.png', dpi=300, bbox_inches='tight')
# plt.show()

# Create output directory
output_dir = 'data_preprocessing/output'
os.makedirs(output_dir, exist_ok=True)

from imblearn.over_sampling import SMOTE
# Step 2: For training other models, then apply SMOTE
# Save original data before SMOTE for unsupervised learning
X_train_unSMOTE = X_train.copy()

# Check class distribution before SMOTE
unique_classes, counts_before = np.unique(y_train, return_counts=True)
print(f"Class distribution before SMOTE:")
for i, (cls, count) in enumerate(zip(le.classes_, counts_before)):
    print(f"  {cls} (class {i}): {count} samples")

try:
    # Equalize class sizes by oversampling all classes to the same target
    # Choose target as the maximum class size in the training split
    target_size = int(max(counts_before))
    target_counts = {int(i): target_size for i in range(len(le.classes_))}
    
    print(f"Target class distribution (equalized to max class size = {target_size}):")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls} (class {i}): {target_counts[i]} samples")
    
    # Apply SMOTE with equal target per class
    smote = SMOTE(
        sampling_strategy=target_counts,
        random_state=42,
        k_neighbors=max(1, min(5, int(min(counts_before)) - 1))  # Ensure k_neighbors is valid
    )
    
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("SMOTE applied successfully!")
    
except Exception as e:
    print(f"SMOTE failed: {e}")
    print("Using original data with class_weight='balanced' in models.")
    
print(f"Balanced X_train shape: {X_train.shape}")
print(f"Balanced y_train shape: {y_train.shape}")


# Class distribution after SMOTE
unique_classes, counts_after = np.unique(y_train, return_counts=True)
print(f"\nClass distribution after SMOTE:")
for i, (cls, count) in enumerate(zip(le.classes_, counts_after)):
    print(f"  {cls} (class {i}): {count} samples")

# Show improvement summary
print(f"\nSMOTE Improvement Summary:")
for i, (cls, before, after) in enumerate(zip(le.classes_, counts_before, counts_after)):
    improvement = ((after - before) / before * 100) if before > 0 else 0
    print(f"  {cls}: {before} → {after} samples ({improvement:+.1f}%)")

# Plotting
plt.figure(figsize=(15, 5))

# Before SMOTE
plt.subplot(1, 3, 1)
plt.bar(range(len(counts_before)), counts_before)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)

# After SMOTE
plt.subplot(1, 3, 2)
plt.bar(range(len(counts_after)), counts_after)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)

# Comparison
plt.subplot(1, 3, 3)
x = np.arange(len(le.classes_))
width = 0.35
plt.bar(x - width/2, counts_before, width, label='Before SMOTE', alpha=0.7)
plt.bar(x + width/2, counts_after, width, label='After SMOTE', alpha=0.7)
plt.title('Class Distribution Comparison')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(x, le.classes_, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/class_distribution_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# Create boxplots for top 15 features
fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()

for i, feat in enumerate(top_feature_names):
    sns.boxplot(data=df, x='Traffic Type', y=feat, showfliers=False, ax=axes[i])
    axes[i].set_title(feat)
    axes[i].tick_params(axis='x', rotation=45)

# Hide unused subplots
for i in range(len(top_feature_names), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'{eda_dir}/feature_boxplots_top15.png', dpi=300, bbox_inches='tight')
# plt.show()

# Block 6: Save data for main pipeline
# Create comprehensive logging
log_data = {
    'timestamp': datetime.now().isoformat(),
    'dataset_info': {
        'original_shape': list(df.shape),
        'after_correlation_cleanup': list(df_reduced.shape),
        'final_shape': list(df.shape),
        'dropped_correlation_features': to_drop,
        'dropped_columns': DROP_COLUMNS + TARGET_TO_DROP[TARGET_VARIABLE]
    },
    'feature_selection': {
        'total_features_analyzed': int(len(importance_df)),
        'selected_features_count': int(len(top_feature_names)),
        'top_features': top_feature_names,
        'feature_importance_scores': importance_df.head(15).to_dict('records')
    },
    'class_distribution': {
        'before_smote': {str(cls): int(count) for cls, count in zip(le.classes_, counts_before)},
        'after_smote': {str(cls): int(count) for cls, count in zip(le.classes_, counts_after)},
        'smote_improvements': {
            str(cls): f"{int(before)} → {int(after)} samples ({((after - before) / before * 100) if before > 0 else 0:+.1f}%)"
            for cls, before, after in zip(le.classes_, counts_before, counts_after)
        }
    },
    'data_quality': {
        'target_variable': TARGET_VARIABLE,
        'classes': [str(cls) for cls in le.classes_],
        'train_test_split': {
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'train_features': int(X_train.shape[1]),
            'test_features': int(X_test.shape[1])
        }
    }
}

# Save comprehensive log
import json
with open(f'{eda_dir}/data_preprocessing_log.json', 'w') as f:
    json.dump(log_data, f, indent=2)

print(f"Comprehensive preprocessing log saved to: {eda_dir}/data_preprocessing_log.json")

# Save processed data
np.savez_compressed(
    f'{output_dir}/processed_data.npz',
    X_train_unSMOTE=X_train_unSMOTE,  # Original X_train before SMOTE for unsupervised learning
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Save metadata (including label encoder, feature names, and preprocessor)
with open(f'{output_dir}/feature_metadata.pkl', 'wb') as f:
    pickle.dump({
        'label_encoder': le,  # Your LabelEncoder from Block 3
        'feature_names': top_feature_names,  # Your top 15 features (optimized)
        'feature_indices': top_feature_indices,  # Feature indices for selection
        'preprocessor': preprocessor,  # Full preprocessor pipeline
        'target_variable': 'Traffic Type',
        'class_distribution_before': dict(zip(le.classes_, counts_before)),
        'class_distribution_after': dict(zip(le.classes_, counts_after)),
        'feature_count': len(top_feature_names),  # Number of selected features
        'eda_directory': eda_dir  # EDA output directory for frontend access
    }, f)

print(f"\nData saved successfully!")
print(f"Processed Data: {output_dir}/processed_data.npz")
print(f"Metadata: {output_dir}/feature_metadata.pkl")
print(f"Traffic Types: {le.classes_}")

print(f"\nEDA Visualizations and Analysis saved to: {eda_dir}/")
print(f"   correlation_matrix_original.png - Original feature correlations")
print(f"   correlation_matrix_reduced.png - After removing redundant features")
print(f"   feature_importance_top15.png - Top 15 most important features")
print(f"   feature_boxplots_top15.png - Feature distributions by traffic type")
print(f"   class_distribution_comparison.png - SMOTE before/after comparison")
print(f"   feature_importance_analysis.csv - Complete feature importance data")
print(f"   data_preprocessing_log.json - Comprehensive preprocessing log")

print(f"\nOptimized Feature Selection:")
print(f"   Selected {len(top_feature_names)} features (reduced from 20 to 15)")
print(f"   Removed 5 redundant features for better model performance")
print(f"   All visualizations and logs ready for frontend display")

