import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor

def run_training_and_evaluation(data_path="data/transformed_train_data.csv", output_dir="predict_model"):
    """
    KDD Phase 3 & 4: Model Implementation, GroupKFold Cross-Validation, Checkpointing,
    and Comprehensive Evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Transformed data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Define features, target, and groups
    ignore_cols = ['case_id', 'input_DAT', 'DAT', 'predicted_weight_g']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols]
    y = df['predicted_weight_g']
    groups = df['case_id']
    
    print(f"Loaded dataset for modeling. Shape: {X.shape}, Cases: {groups.nunique()}")
    print(f"Features count: {len(feature_cols)}")
    
    # Define 5 distinct regression models
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Random Forest': RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        ),
        'Support Vector Regression (SVR)': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(kernel='rbf', C=50.0, epsilon=0.1))
        ]),
        'Gradient Boosting': LGBMRegressor(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            random_state=42, verbose=-1, n_jobs=-1
        ),
        'Multi-Layer Perceptron (MLP)': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=1000,
                learning_rate_init=0.005, random_state=42, early_stopping=True
            ))
        ])
    }
    
    # Setup GroupKFold CV grouping by 'case_id'
    gkf = GroupKFold(n_splits=5)
    
    evaluation_results = []
    oof_predictions = {}
    fitted_models = {}
    
    print("\n--- Starting GroupKFold Cross-Validation ---")
    for model_name, model in models.items():
        print(f"\nEvaluating Model: {model_name}")
        oof_preds = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Fit model on training fold
            model.fit(X_train, y_train)
            
            # Predict on validation fold
            oof_preds[val_idx] = model.predict(X_val)
            
        oof_predictions[model_name] = oof_preds
        
        # Calculate OOF metrics
        rmse = np.sqrt(mean_squared_error(y, oof_preds))
        mae = mean_absolute_error(y, oof_preds)
        r2 = r2_score(y, oof_preds)
        
        evaluation_results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R-Squared': r2
        })
        
        print(f"[{model_name}] OOF RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        
        # Retrain full model on 100% data and save checkpoint
        model.fit(X, y)
        fitted_models[model_name] = model
        
        filename_safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        checkpoint_path = os.path.join(output_dir, f"{filename_safe}.pkl")
        joblib.dump(model, checkpoint_path)
        print(f"Saved model checkpoint to: {checkpoint_path}")
        
    # Phase 4: Interpretation & Evaluation Summary
    results_df = pd.DataFrame(evaluation_results)
    results_df = results_df.sort_values(by='RMSE', ascending=True).reset_index(drop=True)
    
    # Save evaluation summary CSV
    summary_path = "evaluation_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print("\n================ EVALUATION SUMMARY ================")
    print(results_df.to_string(index=False))
    print(f"\nSaved evaluation summary table to: {summary_path}")
    
    # Generate Visualizations
    best_model_name = results_df.iloc[0]['Model']
    print(f"\nBest Performing Model: {best_model_name}")
    
    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    metrics_melted = results_df.melt(id_vars=['Model'], value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value')
    
    ax = sns.barplot(data=metrics_melted, x='Model', y='Value', hue='Metric', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Model Performance Comparison (GroupKFold CV)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Data Mining Regression Model', fontsize=12, labelpad=10)
    plt.ylabel('Error Value (Grams)', fontsize=12)
    plt.xticks(rotation=25, ha='right', fontsize=10)
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plot1_path = "model_comparison.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to: {plot1_path}")
    
    # 2. Feature Importance Plot (using Gradient Boosting / Random Forest)
    imp_model = fitted_models['Gradient Boosting']
    if hasattr(imp_model, 'feature_importances_'):
        importances = imp_model.feature_importances_
    elif hasattr(imp_model, 'named_steps') and hasattr(imp_model.named_steps['regressor'], 'feature_importances_'):
        importances = imp_model.named_steps['regressor'].feature_importances_
    else:
        imp_model = fitted_models['Random Forest']
        importances = imp_model.feature_importances_
        
    feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 7))
    sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='crest')
    plt.title('Top 15 Environmental & Telemetry Feature Importances', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Feature Importance (Biomass Drivers)', fontsize=12)
    plt.ylabel('Engineered Feature', fontsize=12)
    plt.tight_layout()
    
    plot2_path = "feature_importance.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {plot2_path}")
    
    # 3. Scatter Plot: Predicted vs Actual Weights for Best Model
    best_oof_preds = oof_predictions[best_model_name]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y, best_oof_preds, alpha=0.6, color='#2ca02c', edgecolors='k', linewidth=0.5)
    
    # Perfect fit identity line y = x
    max_val = max(y.max(), best_oof_preds.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Perfect Fit (y = x)')
    
    plt.title(f'Predicted vs Actual Lettuce Weight ({best_model_name})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Actual Leaf Weight (predicted_weight_g)', fontsize=12)
    plt.ylabel('Model Out-of-Fold Predicted Weight (g)', fontsize=12)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot3_path = "predicted_vs_actual.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved predicted vs actual plot to: {plot3_path}")

if __name__ == "__main__":
    run_training_and_evaluation()
