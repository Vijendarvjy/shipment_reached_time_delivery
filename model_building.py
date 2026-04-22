#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from scipy.stats import uniform, randint
from collections import Counter
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

data1 = pd.read_csv("/content/Train (1).csv")
data1

data1.info()

data1.shape

data1.head()

data1.describe(include='all')

data1.isnull().sum()

data1.columns.tolist()

data1.nunique()

# Check for duplicate rows
duplicates = data1.duplicated().sum()
print("Number of duplicate rows:", duplicates)

# If duplicates exist, remove them
data1 = data1.drop_duplicates()

counts = data1['Reached.on.Time_Y.N'].value_counts()
print(counts)

"""# Univariate analysis for categorical columns"""

#Target Distribution
sns.countplot(x='Reached.on.Time_Y.N',data=data1)
plt.title('Target Distribution')
plt.xlabel('Reached on time?')
plt.show()

sns.histplot(data1["Customer_care_calls"])
plt.show()

sns.boxplot(x=data1["Cost_of_the_Product"])
plt.show()

sns.histplot(data1["Weight_in_gms"])
plt.show()

sns.countplot(x='Warehouse_block', data=data1)
plt.title('Warehouse Block Distribution')
plt.xlabel('Warehouse Block')
plt.show()

sns.countplot(x='Mode_of_Shipment', data=data1)
plt.title('Mode_of_Shipment Distribution')
plt.xlabel('Mode_of_Shipment')
plt.show()

sns.countplot(x='Product_importance', data=data1)
plt.title('Product_importance Distribution')
plt.xlabel('Product_importance')
plt.show()

sns.countplot(x='Gender', data=data1)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.show()

"""# Univariate analysis for numerical columns"""

cols = ['Customer_care_calls','Customer_rating','Cost_of_the_Product','Prior_purchases','Discount_offered','Weight_in_gms']
for z in cols:
    sns.histplot(data1[z])
    plt.title(z)
    plt.show()

"""# Bivariate Analysis: Relationship between two columns"""

s = pd.crosstab(data1['Mode_of_Shipment'],data1['Reached.on.Time_Y.N'])
s.plot(kind='barh')
plt.xlabel('Proportion on-time')
plt.show()

t = pd.crosstab(data1['Product_importance'],data1['Reached.on.Time_Y.N'])
t.plot(kind='bar')
plt.show()

r = pd.crosstab(data1['Warehouse_block'],data1['Reached.on.Time_Y.N'])
r.plot(kind='bar')
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10,6))
corr = data1[cols + ['Reached.on.Time_Y.N']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# Pairplot: shows scatter plots + histograms together
sns.pairplot(data1[['Customer_care_calls','Customer_rating',
                    'Cost_of_the_Product','Discount_offered',
                    'Weight_in_gms','Reached.on.Time_Y.N']],
             hue='Reached.on.Time_Y.N')
plt.show()

# Outlier detection using boxplots
for z in cols:
    sns.boxplot(x=data1[z])
    plt.title(f'Outlier Check: {z}')
    plt.show()

sns.countplot(x='Reached.on.Time_Y.N', data=data1)
plt.title("Class Distribution: Reached on Time (Yes=1, No=0)")
plt.xlabel("Reached on Time (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

"""# Label encoding and one hot encoding


"""

# Copy dataset
encode_data = data1.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Label Encode
encoders = {}  # dictionary to store encoders

for col in ['Product_importance', 'Gender']:
    le = LabelEncoder()
    encode_data[col] = le.fit_transform(encode_data[col])
    encoders[col] = le  # save encoder for later use

# One-Hot Encode nominal categorical columns
encode_data = pd.get_dummies(encode_data,
                             columns=['Warehouse_block', 'Mode_of_Shipment'],
                             drop_first=True)

# Check
encode_data.head()

plt.figure(figsize=(10,8))
sns.heatmap(encode_data.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap After Encoding")
plt.show()

# Check only the encoded columns
encode_data[['Product_importance', 'Gender'] +
            [col for col in encode_data.columns if 'Warehouse_block' in col or 'Mode_of_Shipment' in col]].head()
# Convert all boolean columns to 0/1
encode_data = encode_data.astype(int)
encode_data.head()

# Initialize and apply scaler
std_scaler = StandardScaler()

# Columns to normalize
num_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
            'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
# Apply to numerical columns
std_scaled_data = encode_data.copy()
std_scaled_data[num_cols] = std_scaler.fit_transform(encode_data[num_cols])

# Check results
print(std_scaled_data[num_cols].describe())

"""# Feature Engineering of Cost of the product & weight in grams
1.   handling the infinity and nan values
2.   filling nan values with the median values
"""

encode_data['Cost_to_Weight_ratio'] = encode_data['Cost_of_the_Product'] / encode_data['Weight_in_gms']

#Handling infinity and NaN values
encode_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values  with the median of valid ratios
encode_data['Cost_to_Weight_ratio'].fillna(encode_data['Cost_to_Weight_ratio'].median(), inplace=True)

encode_data['Cost*Weight'] = encode_data['Cost_of_the_Product'] * encode_data['Weight_in_gms']

# Feature 2: Discount Ratio (Discount / Cost)
encode_data['Discount_Ratio'] = encode_data['Discount_offered'] / encode_data['Cost_of_the_Product']

# Handle infinite or NaN values after new features
encode_data.replace([np.inf, -np.inf], np.nan, inplace=True)
encode_data.fillna(0, inplace=True)

#  Additional Interaction Features

#  Ratio of Customer Care Calls to Prior Purchases
encode_data['CareCalls_to_Purchases'] = encode_data['Customer_care_calls'] / (encode_data['Prior_purchases'] + 1)
# Add +1 to avoid division by zero

#  Interaction between Cost-to-Weight Ratio and Discount
encode_data['CostWeight_Discount_Interaction'] = encode_data['Cost_to_Weight_ratio'] * (encode_data['Discount_offered'] + 1)
# Adding +1 to keep values non-zero and avoid scaling issues

# Handle infinity and NaN values (if any)
encode_data.replace([np.inf, -np.inf], np.nan, inplace=True)
encode_data.fillna(encode_data.median(numeric_only=True), inplace=True)

#  Quick verification
print("\nNew feature columns added:")
print(['CareCalls_to_Purchases', 'CostWeight_Discount_Interaction'])
print("\nFeature summaries:")
print(encode_data[['CareCalls_to_Purchases', 'CostWeight_Discount_Interaction']].describe())

#  Visualize the new features (optional)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(encode_data['CareCalls_to_Purchases'], bins=30, edgecolor='black')
plt.title('Care Calls / Purchases Distribution')
plt.xlabel('Ratio'); plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.hist(encode_data['CostWeight_Discount_Interaction'], bins=30, edgecolor='black')
plt.title('CostWeight *Discount Interaction')
plt.xlabel('Interaction Value'); plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Confirm all columns are numeric
print("\nAll numeric columns?", encode_data.dtypes.apply(lambda x: x != 'object').all())
print("\nCost_to_Weight_ratio summary:\n", encode_data['Cost_to_Weight_ratio'].describe())

# Preview first few rows
encode_data.head()

plt.figure(figsize=(8,5))
plt.hist(encode_data['Cost_to_Weight_ratio'], bins=30, edgecolor='black')
plt.title('Distribution of Cost-to-Weight Ratio', fontsize=14)
plt.xlabel('Cost-to-Weight Ratio')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

"""# Value Count"""

# For target column
print("Target Variable (Reached.on.Time_Y.N):")
print(data1['Reached.on.Time_Y.N'].value_counts())
print("\nNormalized Value Counts (Proportion):")
print(data1['Reached.on.Time_Y.N'].value_counts(normalize=True))

# For all categorical columns
categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']

for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(data1[col].value_counts())
    print("\nNormalized:")
    print(data1[col].value_counts(normalize=True))

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=data1)
    plt.title(f'{col} Distribution')
    plt.show()

encode_data.select_dtypes(include=['object']).columns
encode_data.dtypes

sns.countplot(x='Reached.on.Time_Y.N', data=data1)
plt.title('Target Distribution (Normalized)')
plt.xlabel('Reached on Time (0=No, 1=Yes)')
plt.ylabel('Proportion')
plt.show()

"""# Spliting the dataset into training and testing sets

"""

# Spliting dataset into features (X) and target (y)
X = encode_data.drop('Reached.on.Time_Y.N', axis=1)
y = encode_data['Reached.on.Time_Y.N']

# Spliting the data into train(80%) and test(20%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# shape of the Training, testing dataset after splitting
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
from sklearn.model_selection import train_test_split

# Further split training data into train + validation
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Training Set Class Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=y_test)
plt.title("Testing Set Class Distribution")
plt.show()

"""# Save training and testing data into separate CSV files

"""

# Combine features and target for saving
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save to CSV files
train_data.to_csv("ShipmentSure_Train_Data.csv", index=False)
test_data.to_csv("ShipmentSure_Test_Data.csv", index=False)

print("Training and Testing datasets saved successfully!")

"""# Handle Class Imbalance using SMOTE

"""

# Check original class distribution
print("Before SMOTE class distribution:", Counter(y_train))

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply to training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
print("After SMOTE class distribution:", Counter(y_train_resampled))

# Visualization before & after
fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.countplot(x=y_train, ax=ax[0])
ax[0].set_title("Before SMOTE (Training Data)")
sns.countplot(x=y_train_resampled, ax=ax[1])
ax[1].set_title("After SMOTE (Balanced Training Data)")
plt.show()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),  # L2 regularization (smaller C = stronger)

    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=42),

    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1),

    "Naive Bayes": GaussianNB(),

    "KNN": KNeighborsClassifier(n_neighbors=9),  # more neighbors → smoother, less overfitting

    "SVM": SVC(C=0.5, kernel='rbf', gamma='scale', random_state=42),

    "XGBoost": XGBClassifier(
        learning_rate=0.05, n_estimators=500, max_depth=6,subsample=0.8, colsample_bytree=0.8, reg_lambda=0.5, use_label_encoder=False, eval_metric='logloss', random_state=42),

    "LightGBM": LGBMClassifier(
        learning_rate=0.05, n_estimators=500, max_depth=6,num_leaves=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=0.5, random_state=42),

    "CatBoost": CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=5, verbose=False, random_state=42)
}

results = []

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])

    # Cross-validation accuracy (10-fold)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

    results.append({
        'Model': name,
        'Mean CV Accuracy': round(np.mean(cv_scores), 4),
        'Std Dev': round(np.std(cv_scores), 4)
    })

results_df = pd.DataFrame(results).sort_values(by='Mean CV Accuracy', ascending=False)
print("\n Cross-validation results (Overfitting check):")
print(results_df)

# Fix random seed for consistency
RND = 42

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RND)

# Define models with anti-overfitting hyperparameters
models = {
    "Logistic Regression": LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=RND),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=4, random_state=RND),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                            min_samples_leaf=3, max_features='sqrt', random_state=RND),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(C=0.5, kernel='rbf', gamma='scale', random_state=RND),
    "XGBoost": XGBClassifier(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1, n_estimators=200, random_state=RND, eval_metric='logloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(num_leaves=20, max_depth=6, subsample=0.8, reg_lambda=0.5,
                               learning_rate=0.05, n_estimators=200, random_state=RND),
    "CatBoost": CatBoostClassifier(depth=6, learning_rate=0.05, l2_leaf_reg=3, n_estimators=200,
                                   verbose=0, random_state=RND)
}

# Run cross-validation to check improvement
cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    cv_results.append({
        "Model": name,
        "Mean CV Accuracy": np.mean(scores),
        "Std Dev": np.std(scores)
    })

# Display results
cv_df = pd.DataFrame(cv_results).sort_values(by="Mean CV Accuracy", ascending=False)
print("\nCross-validation results (After Regularization to Fix Overfitting):")
print(cv_df.to_string(index=False))

models = {
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "SVM": SVC(C=0.5, kernel='rbf', gamma='scale', probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        learning_rate=0.05, n_estimators=500, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=0.5,
        use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(
        learning_rate=0.05, n_estimators=500, max_depth=6, num_leaves=20,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=0.5, random_state=42),
    "CatBoost": CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=5, verbose=False, random_state=42)
}

# Store results
results = {}
comparison = []

plt.figure(figsize=(10, 7))
for name, model in models.items():
    print(f"\n Training and evaluating {name}...")

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Metrics
    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_prob)

    print(f"\nClassification Report for {name}:")
    print(classification_report(y_val, y_val_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    comparison.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    })

# ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC-AUC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Final Comparison Table
comparison_df = pd.DataFrame(comparison).sort_values(by='ROC-AUC', ascending=False)
print("\nFinal Model Comparison Table:")
print(comparison_df)

# Optional: save to CSV for report
comparison_df.to_csv("model_comparison_results.csv", index=False)

models = {
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "SVM": SVC(C=0.5, kernel='rbf', gamma='scale', probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        learning_rate=0.05, n_estimators=500, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=0.5,
        use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(
        learning_rate=0.05, n_estimators=500, max_depth=6, num_leaves=20,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=0.5, random_state=42),
    "CatBoost": CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=5, verbose=False, random_state=42)
}

comparison = []

for name, model in models.items():
    print(f"\nTraining and Evaluating: {name}")


    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)


    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))


    plt.figure(figsize=(5,4))
    sns.heatmap(pd.crosstab(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Store Results
    comparison.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    })

comparison_df = pd.DataFrame(comparison).sort_values(by="ROC-AUC", ascending=False)
print("\nFinal Model Comparison Table:")
print(comparison_df)

# Step 1: Find the best model based on ROC-AUC
best_model_name = comparison_df.iloc[0]['Model']
print(f"\nBest model based on ROC-AUC: {best_model_name}")

#  Step 2: Save that model
best_model = models[best_model_name]
joblib.dump(best_model, f"{best_model_name}_best_model.pkl")
print(f"Model saved successfully as '{best_model_name}_best_model.pkl'")

feature_names = X_train.columns

def plot_feature_importance(model, model_name):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_[0]
    else:
        print(f"{model_name} does not support feature importance.")
        return

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(7,4))
    plt.barh(imp_df["Feature"], imp_df["Importance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# Loop directly over your existing models
for name, model in models.items():
    plot_feature_importance(model, name)

# Only include models that support feature importance or coefficients
supported_models = {name: model for name, model in models.items()
                    if hasattr(model, "feature_importances_") or hasattr(model, "coef_")}

feature_names = X_train.columns

# Store importance values for each model
importance_df = pd.DataFrame(index=feature_names)

for name, model in supported_models.items():
    if hasattr(model, "feature_importances_"):
        importance_df[name] = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_df[name] = np.abs(model.coef_[0])  # use absolute values for coefficients

# Compute average importance across models
importance_df["Average_Importance"] = importance_df.mean(axis=1)

# Sort by average importance
top_features = importance_df.sort_values(by="Average_Importance", ascending=False).head(10)

# Display table
print("\nTop 10 Most Important Features Across All Models:\n")
print(top_features)

# Plot
plt.figure(figsize=(8,5))
plt.barh(top_features.index, top_features["Average_Importance"], color="skyblue")
plt.gca().invert_yaxis()
plt.title("Overall Top 10 Features (Average Importance Across Models)")
plt.xlabel("Average Importance Score")
plt.tight_layout()
plt.show()

# Check class balance in test set
print("Class distribution in Test Data:")
print(y_test.value_counts(normalize=True))

# Optional: Visualize
plt.figure(figsize=(5,4))
sns.countplot(x=y_test)
plt.title("Class Distribution in Test Data")
plt.xlabel("Reached on Time (Y/N)")
plt.ylabel("Count")
plt.show()

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4)
    })

#  Create and display the comparison table
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("Model Evaluation on Test Data:")
print(results_df)

#  Redefine all models with regularization and simpler configs

models = {
    "Logistic Regression": LogisticRegression(C=0.3, penalty='l2', solver='liblinear', max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance'),
    "SVM": LinearSVC(C=0.5, max_iter=2000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        learning_rate=0.05, n_estimators=200, max_depth=4, subsample=0.7,
        colsample_bytree=0.7, reg_lambda=2, reg_alpha=0.5,
        use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(
        learning_rate=0.05, n_estimators=200, max_depth=4, num_leaves=15,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=2, reg_alpha=0.5, random_state=42),
    "CatBoost": CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=4, l2_leaf_reg=6, verbose=False, random_state=42)
}

# Retrain all models on balanced (SMOTE) training data
print("Training Models with Regularized Parameters...\n")
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
print("All models retrained successfully!")

#   Check Cross-Validation Accuracy (to ensure models generalize well)
from sklearn.model_selection import cross_val_score

for name, model in models.items():
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"{name}: Mean CV Accuracy = {scores.mean():.4f}")

# Evaluate retrained models on Test Data
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4)
    })

results_df_new = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("\nModel Evaluation on Test Data (After Regularization):")
print(results_df_new)

# Save model
pickle.dump(best_model, open("XGBoost_best_model.pkl", "wb"))

# Save label encoders (dictionary)
pickle.dump(encoders, open("encoder.pkl", "wb"))

# Save standard scaler
pickle.dump(std_scaler, open("scaler.pkl", "wb"))

print(" Model, encoders, and scaler saved successfully!")

#
