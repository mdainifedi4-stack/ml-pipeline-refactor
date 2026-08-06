# Standard library imports
import csv
import os
import time
import chardet

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns

# Flask imports
from flask import Flask, render_template, request, redirect, send_from_directory,json

# Matplotlib imports
import matplotlib

import openai
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import plot_tree, DecisionTreeClassifier






#******************************  Begin  ************************************

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'
data_path=""
processed_data_path=""
encoded_data_path=""

previous_selected_columns=[]
previous_target_variable=""
previous_test_size=0.0
previous_results={}


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])




@app.route('/')
def home():
    show=True
    return render_template('index.html',show=show)

@app.route('/upload', methods=['POST'])
def upload_file():

    global data_path,processed_data_path,encoded_data_path

    if 'dataset' not in request.files:
        return redirect('/')

    file=request.files['dataset']
    if file.filename == '' :
        return redirect('/')

    if file:
        data_path= os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(data_path)

    df = pd.read_csv(data_path) if file.filename.endswith('.csv') else pd.read_excel(data_path)



    # Read dataset
    if file.filename.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)


    descriptions = {}

    head=df.head()
    shape=df.shape
    description=df.describe()
    columns=df.columns



    processed_df,processed_df_encoded , raport = preprocess_data(df)
    encoded_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_encoded' + file.filename)
    processed_df_encoded.to_csv(encoded_data_path, index=False)


    show=False

    processed_data_path = os.path.join(app.config['UPLOAD_FOLDER'],'processed'+file.filename)

    processed_df.to_csv(processed_data_path, index=False)

    results={
        'head' : head,
        'shape' : shape,
        'description' : description,
        'columns' : columns
    }

    return render_template('index.html',
                           message=f"Preprocessing complete! {len(df)} rows processed.",
                           filename=file.filename,
                           dataset_name=file.filename,
                           results=results,
                           descriptions=descriptions,
                           show=show)


def preprocess_data(df):
    processed_df = df.copy()

    report= {
        'missing_values' : {},
        'categorical_encoded' : [],
        'numerical_scaled' : []
    }

    for col in processed_df.columns:
        missing = processed_df[col].isnull().sum()
        report['missing_values'][col]=missing

        if missing > 0:
            # Handle numerical columns
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                median_v = processed_df[col].median()
                processed_df[col].fillna(median_v , inplace=True)
                report['missing_values'][col] = f"Filled with median: {median_v:.2f}"
            # Handle datetime columns
            elif pd.api.types.is_datetime64_any_dtype(processed_df[col]):
                mode_v = processed_df[col].mode()[0]
                processed_df[col].fillna(mode_v, inplace=True)
                report['missing_values'][col] = f"Filled with mode: {mode_v}"
            # Handle categorical columns
            else:
                mode_v = processed_df[col].mode()[0]
                processed_df[col].fillna(mode_v, inplace=True)
                report['missing_values'][col] = f"Filled with mode: {mode_v}"


    for col in processed_df.select_dtypes(include=['object','category']).columns:
        le=LabelEncoder()
        processed_df[col]= le.fit_transform(processed_df[col])
        report['categorical_encoded'].append({
            'column': col,
            'n_categories': len(le.classes_),
            'mapping': dict(zip(le.classes_, range(len(le.classes_))))
        })

    processed_df_encoded = processed_df

    return processed_df,processed_df_encoded, report



#******************************  KNN  ************************************

@app.route('/model/knn')
def KNN():
    processed_path = processed_data_path

    df= pd.read_csv(processed_path, nrows=0)
    columns=df.columns.tolist()

    return render_template('KNN.html',columns=columns)

@app.route('/manual_configure', methods=['POST'])
def KNN_manual_configuration():

    global previous_selected_columns, previous_target_variable, previous_test_size,previous_results


    selected_columns = request.form.getlist('selected_columns')
    target_variable = request.form['target_variable']
    test_size = float(request.form['test_size'])
    n_neighbors = int(request.form['n_neighbors'])

    previous_selected_columns=selected_columns
    previous_target_variable=target_variable
    previous_test_size=test_size

    df = pd.read_csv(processed_data_path)

    X = df[selected_columns].copy()
    y = df[target_variable].copy()
    z = pd.concat([X, y], axis=1)

    # Generate unique timestamp for filenames
    timestamp = str(int(time.time()))

    # Create directory for plots if not exists
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    corr_filename=f'correlation_{timestamp}.png'
    conf_filename=f'confusion_{timestamp}.png'


    # Correlation heatmap
    cor = z.corr()
    full_corr_path = os.path.join(plot_dir, corr_filename)
    full_conf_path = os.path.join(plot_dir, conf_filename)

    corr_plot_url=f'plots/{corr_filename}'
    conf_plot_url=f'plots/{conf_filename}'

    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    clf_report = classification_report(y_test, y_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_plot_path = os.path.join('plots', f'confusion_{timestamp}.png')
    full_conf_path = os.path.join(plot_dir, f'confusion_{timestamp}.png')

    # Get unique class labels
    classes = np.unique(np.concatenate((y_test, y_pred)))

    plt.figure(figsize=(7, 5), dpi=100)
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(full_conf_path, bbox_inches='tight')
    plt.close()

    # Prepare results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': clf_report,
        'corr_plot': corr_plot_url,
        'conf_matrix_plot': conf_plot_url,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size,
        'n_neighbors': n_neighbors
    }

    previous_results=results

    # Get columns for the form
    df = pd.read_csv(processed_data_path, nrows=0)
    columns = df.columns.tolist()

    return render_template('KNN.html', results=results, columns=columns)

@app.route('/automatic_configuration', methods=['POST'])
def KNN_automatic_configuration():
    selected_columns = request.form.getlist('selected_columns')
    target_variable = request.form['target_variable']
    test_size = float(request.form['test_size'])

    df = pd.read_csv(processed_data_path)
    columns = df.columns.tolist()

    X = df[selected_columns].copy()
    y = df[target_variable].copy()

    timestamp = str(int(time.time()))

    # Create plots directory
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation heatmap
    cor = df[selected_columns + [target_variable]].corr()
    corr_filename = f'correlation_{timestamp}.png'
    corr_plot_url = f'plots/{corr_filename}'
    full_corr_path = os.path.join(plot_dir, corr_filename)

    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # GridSearchCV for KNN
    param_grid = {
        'n_neighbors': range(1, 20),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }

    grid_KNN = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_KNN.fit(X_train, y_train)

    best_params = grid_KNN.best_params_

    Final_model = grid_KNN.best_estimator_
    y_pred_knn = Final_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred_knn)
    precision = precision_score(y_test, y_pred_knn, average='weighted')
    recall = recall_score(y_test, y_pred_knn, average='weighted')
    f1 = f1_score(y_test, y_pred_knn, average='weighted')
    clf_report = classification_report(y_test, y_pred_knn)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_knn)
    conf_filename = f'confusion_{timestamp}.png'
    conf_plot_url = f'plots/{conf_filename}'
    full_conf_path = os.path.join(plot_dir, conf_filename)

    classes = np.unique(np.concatenate((y_test, y_pred_knn)))

    plt.figure(figsize=(7, 5), dpi=100)
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(full_conf_path, bbox_inches='tight')
    plt.close()

    # Results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': clf_report,
        'corr_plot': corr_plot_url,
        'conf_matrix_plot': conf_plot_url,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size,
        'best_parms': best_params
    }


    return render_template('KNN.html', results=results, columns=columns)






#******************************  Decision Tree  ************************************

@app.route('/model/decision_tree')
def decision_tree():
    processed_path = processed_data_path

    df = pd.read_csv(processed_path, nrows=0)
    columns = df.columns.tolist()

    return render_template('decision_tree.html',columns=columns)

@app.route('/dt_manual_configure', methods=['POST'])
def dt_manual_configuration():

    global previous_selected_columns,previous_target_variable,previous_test_size,previous_results

    # Get form data
    selected_columns = request.form.getlist('selected_columns')
    target_variable = request.form['target_variable']
    test_size = float(request.form['test_size'])
    depth = int(request.form['max_depth'])

    previous_selected_columns = selected_columns
    previous_target_variable = target_variable
    previous_test_size = test_size

    # Load data
    df = pd.read_csv(processed_data_path)
    X = df[selected_columns].copy()
    y = df[target_variable].copy()
    columns = df.columns.tolist()

    # Create plot directory
    timestamp = str(int(time.time()))
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation plot
    cor = df[selected_columns + [target_variable]].corr()
    corr_filename = f'correlation_{timestamp}.png'
    corr_plot_url = f'plots/{corr_filename}'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train model
    tree = DecisionTreeClassifier(criterion='gini', max_depth=depth)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    clf_report = classification_report(y_test, y_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_filename = f'confusion_matrix_{timestamp}.png'
    conf_plot_url = f'plots/{conf_filename}'
    full_conf_path = os.path.join(plot_dir, conf_filename)
    classes = np.unique(np.concatenate((y_test, y_pred)))
    plt.figure(figsize=(7, 5), dpi=100)
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(full_conf_path, bbox_inches='tight')
    plt.close()

    # Decision tree plot
    dt_filename = f'decisionTree_{timestamp}.png'
    dt_plot_url = f'plots/{dt_filename}'
    full_dt_path = os.path.join(plot_dir, dt_filename)
    class_names = tree.classes_.astype(str)
    plt.figure(figsize=(50, 15), dpi=200)
    plot_tree(tree, filled=True, feature_names=X_train.columns,
              class_names=class_names, rounded=True, fontsize=14)
    plt.savefig(full_dt_path, dpi=200)
    plt.close()

    # Results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': clf_report,
        'corr_plot': corr_plot_url,
        'conf_matrix_plot': conf_plot_url,
        'dt_plot_url': dt_plot_url,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size,
        'depth': depth
    }

    previous_results = results

    return render_template('decision_tree.html',results=results,columns=columns)

@app.route('/dt_automatic_configure', methods=['POST'])
def dt_automatic_configure():
    selected_columns = request.form.getlist('selected_columns')
    target_variable = request.form['target_variable']
    test_size = float(request.form['test_size'])

    df = pd.read_csv(processed_data_path)
    X = df[selected_columns].copy()
    y = df[target_variable].copy()
    columns = df.columns.tolist()

    timestamp = str(int(time.time()))
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)


    cor = df[selected_columns + [target_variable]].corr()
    corr_filename = f'correlation_{timestamp}.png'
    corr_plot_url = f'plots/{corr_filename}'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    param_grid_tree = {'max_depth': range(1, 10)}
    grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid_tree, cv=5)
    grid_tree.fit(X_train, y_train)

    best_params = grid_tree.best_params_
    best_model = grid_tree.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    clf_report = classification_report(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_filename = f'confusion_matrix_{timestamp}.png'
    conf_plot_url = f'plots/{conf_filename}'
    full_conf_path = os.path.join(plot_dir, conf_filename)
    classes = np.unique(np.concatenate((y_test, y_pred)))
    plt.figure(figsize=(7, 5), dpi=100)
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(full_conf_path, bbox_inches='tight')
    plt.close()

    dt_filename = f'decisionTree_{timestamp}.png'
    dt_plot_url = f'plots/{dt_filename}'
    full_dt_path = os.path.join(plot_dir, dt_filename)
    class_names = best_model.classes_.astype(str)
    plt.figure(figsize=(50, 15), dpi=200)
    plot_tree(best_model, filled=True, feature_names=X_train.columns,
              class_names=class_names, rounded=True, fontsize=14)
    plt.savefig(full_dt_path, dpi=200)
    plt.close()

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': clf_report,
        'corr_plot': corr_plot_url,
        'conf_matrix_plot': conf_plot_url,
        'dt_plot_url': dt_plot_url,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size,
        'best_params': best_params
    }

    return render_template('decision_tree.html', results=results, columns=columns)


# ******************************  Random Forest  ************************************
@app.route('/model/random_forest')
def random_forest():
    processed_path = processed_data_path
    df = pd.read_csv(processed_path, nrows=0)
    columns = df.columns.tolist()
    return render_template('random_forest.html', columns=columns)


@app.route('/rf_manual_configure', methods=['POST'])
def rf_manual_configuration():
    global previous_selected_columns, previous_target_variable, previous_test_size, previous_results

    selected_columns = request.form.getlist('selected_columns')
    target_variable = request.form['target_variable']
    test_size = float(request.form['test_size'])
    n_estimators = int(request.form['n_estimators'])
    max_depth = int(request.form['max_depth']) if request.form['max_depth'] else None

    previous_selected_columns = selected_columns
    previous_target_variable = target_variable
    previous_test_size = test_size

    df = pd.read_csv(processed_data_path)
    X = df[selected_columns].copy()
    y = df[target_variable].copy()
    columns = df.columns.tolist()

    # Create plot directory
    timestamp = str(int(time.time()))
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation plot
    cor = df[selected_columns + [target_variable]].corr()
    corr_filename = f'correlation_{timestamp}.png'
    corr_plot_url = f'plots/{corr_filename}'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    clf_report = classification_report(y_test, y_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_filename = f'confusion_matrix_{timestamp}.png'
    conf_plot_url = f'plots/{conf_filename}'
    full_conf_path = os.path.join(plot_dir, conf_filename)
    classes = np.unique(np.concatenate((y_test, y_pred)))
    plt.figure(figsize=(7, 5), dpi=100)
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(full_conf_path, bbox_inches='tight')
    plt.close()

    # Feature importance plot
    feature_importances = rf.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")

    fi_filename = f'feature_importance_{timestamp}.png'
    fi_plot_url = f'plots/{fi_filename}'
    full_fi_path = os.path.join(plot_dir, fi_filename)
    plt.savefig(full_fi_path, bbox_inches='tight')
    plt.close()

    # Results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': clf_report,
        'corr_plot': corr_plot_url,
        'conf_matrix_plot': conf_plot_url,
        'fi_plot_url': fi_plot_url,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'feature_importances': dict(zip(X.columns, feature_importances))
    }

    previous_results = results
    return render_template('random_forest.html', results=results, columns=columns)


@app.route('/rf_automatic_configure', methods=['POST'])
def rf_automatic_configure():
    selected_columns = request.form.getlist('selected_columns')
    target_variable = request.form['target_variable']
    test_size = float(request.form['test_size'])

    df = pd.read_csv(processed_data_path)
    X = df[selected_columns].copy()
    y = df[target_variable].copy()
    columns = df.columns.tolist()

    timestamp = str(int(time.time()))
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation plot
    cor = df[selected_columns + [target_variable]].corr()
    corr_filename = f'correlation_{timestamp}.png'
    corr_plot_url = f'plots/{corr_filename}'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # GridSearch for optimal parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)

    best_params = grid_rf.best_params_
    best_model = grid_rf.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    clf_report = classification_report(y_test, y_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_filename = f'confusion_matrix_{timestamp}.png'
    conf_plot_url = f'plots/{conf_filename}'
    full_conf_path = os.path.join(plot_dir, conf_filename)
    classes = np.unique(np.concatenate((y_test, y_pred)))
    plt.figure(figsize=(7, 5), dpi=100)
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(full_conf_path, bbox_inches='tight')
    plt.close()

    # Feature importance plot
    feature_importances = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")

    fi_filename = f'feature_importance_{timestamp}.png'
    fi_plot_url = f'plots/{fi_filename}'
    full_fi_path = os.path.join(plot_dir, fi_filename)
    plt.savefig(full_fi_path, bbox_inches='tight')
    plt.close()

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': clf_report,
        'corr_plot': corr_plot_url,
        'conf_matrix_plot': conf_plot_url,
        'fi_plot_url': fi_plot_url,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size,
        'best_params': best_params,
        'feature_importances': dict(zip(X.columns, feature_importances))
    }

    return render_template('random_forest.html', results=results, columns=columns)





#******************************  Linear Regression  ************************************

@app.route('/model/linear_regression')
def linear_regression():
    processed_path = processed_data_path

    df = pd.read_csv(processed_path, nrows=0)
    columns = df.columns.tolist()

    return render_template('linear_regression.html', columns=columns)


@app.route('/linear_regression_configure', methods=['POST'])
def linear_regression_configure():
    global previous_selected_columns, previous_target_variable, previous_test_size, previous_results

    selected_columns = request.form.getlist('selected_columns')
    target_variable = request.form['target_variable']
    test_size = float(request.form['test_size'])

    df = pd.read_csv(processed_data_path)
    X = df[selected_columns].copy()
    y = df[target_variable].copy()
    z = pd.concat([X, y], axis=1)

    previous_selected_columns=selected_columns
    previous_target_variable=target_variable
    previous_test_size=test_size


    # Generate unique timestamp for filenames
    timestamp = str(int(time.time()))

    # Create directory for plots if not exists
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation heatmap
    corr_filename = f'correlation_{timestamp}.png'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    corr_plot_url = f'plots/{corr_filename}'

    cor = z.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Model training
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)

    # Make predictions
    y_pred = lr.predict(X_test_sc)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Get coefficients
    coefficients = dict(zip(selected_columns, lr.coef_))
    intercept = lr.intercept_

    # Actual vs Predicted plot
    lr_name = f'lr_plot_{timestamp}.png'
    lr_url = f'plots/{lr_name}'
    lr_path = os.path.join(plot_dir, lr_name)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(lr_path, bbox_inches='tight')
    plt.close()

    # Residual plot
    residuals = y_test - y_pred
    residual_name = f'residuals_{timestamp}.png'
    residual_url = f'plots/{residual_name}'
    residual_path = os.path.join(plot_dir, residual_name)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(residual_path, bbox_inches='tight')
    plt.close()

    # Prepare results
    results = {
        'corr_plot': corr_plot_url,
        'lr_plot': lr_url,
        'residual_plot': residual_url,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'intercept': intercept,
        'coefficients': coefficients,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size
    }

    previous_results=results

    return render_template('linear_regression.html', results=results)




#******************************  Clustering  ************************************
@app.route('/model/clustering')
def clustering():
    processed_path = processed_data_path
    timestamp = str(int(time.time()))

    # Read the full dataset
    df = pd.read_csv(processed_path)
    columns = df.columns.tolist()

    # Prepare plot filenames
    elbow_name = f'elbow_{timestamp}.png'
    silhouette_name = f'silhouette_{timestamp}.png'
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    elbow_full_path = os.path.join(plot_dir, elbow_name)
    silhouette_full_path = os.path.join(plot_dir, silhouette_name)

    # Elbow Method
    inertia = []
    K = range(2, 8)
    for k in K:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(df[columns])
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K, inertia, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(elbow_full_path, dpi=200)
    plt.close()  # Close the figure

    # Silhouette Score
    silhouette_scores = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[columns])
        score = silhouette_score(df[columns], kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 6))
    plt.plot(K, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(silhouette_full_path, dpi=200)
    plt.close()  # Close the figure

    # Prepare URLs for the template
    elbow_url = f'plots/{elbow_name}'
    silhouette_url = f'plots/{silhouette_name}'

    return render_template('clustering.html',
                         columns=columns,
                         elbow_plot=elbow_url,
                         silhouette_plot=silhouette_url)

@app.route('/clustering_configure', methods=['POST'])
def clustering_configure():
    global previous_selected_columns, previous_results

    selected_columns = request.form.getlist('selected_columns')
    n_clusters = int(request.form['n_clusters'])

    previous_selected_columns=selected_columns

    df = pd.read_csv(processed_data_path)
    X = df[selected_columns].copy()

    timestamp = str(int(time.time()))
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation heatmap
    cor = df[selected_columns].corr()
    corr_filename = f'correlation_{timestamp}.png'
    corr_plot_url = f'plots/{corr_filename}'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    km = KMeans(n_clusters=n_clusters, random_state=0)
    km.fit(X_scaled)
    labels = km.labels_

    # Add cluster labels to dataframe
    df['Cluster'] = labels

    # PCA for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X_scaled)
    reduced_centers = pca.transform(km.cluster_centers_)

    # Cluster visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, alpha=0.6, cmap='viridis')
    plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'K-means Clustering (k={n_clusters})')
    plt.legend()
    plt.colorbar(label='Cluster')

    cluster_plot_filename = f'clusters_{timestamp}.png'
    cluster_plot_url = f'plots/{cluster_plot_filename}'
    full_cluster_path = os.path.join(plot_dir, cluster_plot_filename)
    plt.savefig(full_cluster_path, bbox_inches='tight')
    plt.close()

    # Prepare results for template
    results = {
        'corr_plot': corr_plot_url,
        'cluster_plot': cluster_plot_url,
        'n_clusters': n_clusters,
        'selected_columns': selected_columns,
        'cluster_counts': df['Cluster'].value_counts().to_dict(),
        'timestamp': timestamp
    }

    previous_results=results

    return render_template('clustering.html', results=results, columns=df.columns.tolist())




#******************************  Model Comparison  ************************************


@app.route('/model/comparison', methods=['POST'])
def model_comparison_classification():
    selected_columns = previous_selected_columns
    target_variable = previous_target_variable
    test_size = previous_test_size

    df = pd.read_csv(processed_data_path)

    if request.form['model_type']=='knn':

        n_neighbors = int(request.form['new_n_neighbors'])




        X = df[selected_columns].copy()
        y = df[target_variable].copy()
        z = pd.concat([X, y], axis=1)  # Fixed concatenation

        # Generate unique timestamp for filenames
        timestamp = str(int(time.time()))

        # Create directory for plots if not exists
        plot_dir = os.path.join(app.static_folder, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        corr_filename = f'correlation_{timestamp}.png'
        conf_filename = f'confusion_{timestamp}.png'

        # Correlation heatmap
        cor = z.corr()
        full_corr_path = os.path.join(plot_dir, corr_filename)
        full_conf_path = os.path.join(plot_dir, conf_filename)

        corr_plot_url = f'plots/{corr_filename}'
        conf_plot_url = f'plots/{conf_filename}'

        plt.figure(figsize=(8, 4))
        sns.heatmap(cor, annot=True)
        plt.savefig(full_corr_path, bbox_inches='tight')
        plt.close()

        # Train-test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        KNN.fit(X_train, y_train)
        y_pred = KNN.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        clf_report = classification_report(y_test, y_pred)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_plot_path = os.path.join('plots', f'confusion_{timestamp}.png')
        full_conf_path = os.path.join(plot_dir, f'confusion_{timestamp}.png')

        # Get unique class labels
        classes = np.unique(np.concatenate((y_test, y_pred)))

        plt.figure(figsize=(7, 5), dpi=100)
        sns.heatmap(conf_matrix, annot=True, fmt="d",
                    xticklabels=classes,
                    yticklabels=classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig(full_conf_path, bbox_inches='tight')
        plt.close()

        # Prepare results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': clf_report,
            'corr_plot': corr_plot_url,
            'conf_matrix_plot': conf_plot_url,
            'selected_columns': selected_columns,
            'target_variable': target_variable,
            'test_size': test_size,
            'n_neighbors': n_neighbors
        }

        df = pd.read_csv(processed_data_path, nrows=0)
        columns = df.columns.tolist()

    else:

        depth = int(request.form['new_max_depth'])
        # Load data

        X = df[selected_columns].copy()
        y = df[target_variable].copy()
        columns = df.columns.tolist()

        # Create plot directory
        timestamp = str(int(time.time()))
        plot_dir = os.path.join(app.static_folder, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Correlation plot
        cor = df[selected_columns + [target_variable]].corr()
        corr_filename = f'correlation_{timestamp}.png'
        corr_plot_url = f'plots/{corr_filename}'
        full_corr_path = os.path.join(plot_dir, corr_filename)
        plt.figure(figsize=(8, 4))
        sns.heatmap(cor, annot=True)
        plt.savefig(full_corr_path, bbox_inches='tight')
        plt.close()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train model
        tree = DecisionTreeClassifier(criterion='gini', max_depth=depth)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        clf_report = classification_report(y_test, y_pred)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)  # ADDED
        conf_filename = f'confusion_matrix_{timestamp}.png'  # ADDED
        conf_plot_url = f'plots/{conf_filename}'  # ADDED
        full_conf_path = os.path.join(plot_dir, conf_filename)  # ADDED
        classes = np.unique(np.concatenate((y_test, y_pred)))  # Fixed variable
        plt.figure(figsize=(7, 5), dpi=100)
        sns.heatmap(conf_matrix, annot=True, fmt="d",
                    xticklabels=classes,
                    yticklabels=classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig(full_conf_path, bbox_inches='tight')
        plt.close()

        # Decision tree plot
        dt_filename = f'decisionTree_{timestamp}.png'
        dt_plot_url = f'plots/{dt_filename}'  # Relative URL
        full_dt_path = os.path.join(plot_dir, dt_filename)
        class_names = tree.classes_.astype(str)  # Dynamic classes
        plt.figure(figsize=(50, 15), dpi=200)
        plot_tree(tree, filled=True, feature_names=X_train.columns,
                  class_names=class_names, rounded=True, fontsize=14)
        plt.savefig(full_dt_path, dpi=200)  # Removed plt.show()
        plt.close()

        # Results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': clf_report,
            'corr_plot': corr_plot_url,
            'conf_matrix_plot': conf_plot_url,  # Fixed variable
            'dt_plot_url': dt_plot_url,  # Relative path
            'selected_columns': selected_columns,
            'target_variable': target_variable,
            'test_size': test_size,
            'depth': depth  # Comma added
        }
        df = pd.read_csv(processed_data_path, nrows=0)
        columns = df.columns.tolist()


    return render_template('comparison.html', results=results, columns=columns,previous_results=previous_results)



@app.route('/model/comparison_regression', methods=['POST'])
def model_comparison_Regression():
    selected_columns = previous_selected_columns
    target_variable = previous_target_variable
    test_size = float(request.form['test_new_size'])

    df = pd.read_csv(processed_data_path)

    X = df[selected_columns].copy()
    y = df[target_variable].copy()
    z = pd.concat([X, y], axis=1)  # Combined for correlation

    # Generate unique timestamp for filenames
    timestamp = str(int(time.time()))

    # Create directory for plots if not exists
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation heatmap
    corr_filename = f'correlation_{timestamp}.png'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    corr_plot_url = f'plots/{corr_filename}'

    cor = z.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Model training
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)  # Corrected: train on y_train

    # Make predictions
    y_pred = lr.predict(X_test_sc)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Get coefficients
    coefficients = dict(zip(selected_columns, lr.coef_))
    intercept = lr.intercept_

    # Actual vs Predicted plot
    lr_name = f'lr_plot_{timestamp}.png'
    lr_url = f'plots/{lr_name}'
    lr_path = os.path.join(plot_dir, lr_name)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(lr_path, bbox_inches='tight')
    plt.close()

    # Residual plot
    residuals = y_test - y_pred
    residual_name = f'residuals_{timestamp}.png'
    residual_url = f'plots/{residual_name}'
    residual_path = os.path.join(plot_dir, residual_name)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(residual_path, bbox_inches='tight')
    plt.close()

    # Prepare results
    results = {
        'corr_plot': corr_plot_url,
        'lr_plot': lr_url,
        'residual_plot': residual_url,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'intercept': intercept,
        'coefficients': coefficients,
        'selected_columns': selected_columns,
        'target_variable': target_variable,
        'test_size': test_size
    }



    return render_template('comparison_regression.html',results=results,previous_results=previous_results)



@app.route('/model/comparison_clustering', methods=['POST'])
def model_comparison_clustering():
    selected_columns = previous_selected_columns

    n_clusters = int(request.form['new_n_clusters'])

    df = pd.read_csv(processed_data_path)

    X = df[selected_columns].copy()

    timestamp = str(int(time.time()))
    plot_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Correlation heatmap
    cor = df[selected_columns].corr()
    corr_filename = f'correlation_{timestamp}.png'
    corr_plot_url = f'plots/{corr_filename}'
    full_corr_path = os.path.join(plot_dir, corr_filename)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cor, annot=True)
    plt.savefig(full_corr_path, bbox_inches='tight')
    plt.close()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    km = KMeans(n_clusters=n_clusters, random_state=0)
    km.fit(X_scaled)
    labels = km.labels_

    # Add cluster labels to dataframe
    df['Cluster'] = labels

    # PCA for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X_scaled)
    reduced_centers = pca.transform(km.cluster_centers_)

    # Cluster visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, alpha=0.6, cmap='viridis')
    plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'K-means Clustering (k={n_clusters})')
    plt.legend()
    plt.colorbar(label='Cluster')

    cluster_plot_filename = f'clusters_{timestamp}.png'
    cluster_plot_url = f'plots/{cluster_plot_filename}'
    full_cluster_path = os.path.join(plot_dir, cluster_plot_filename)
    plt.savefig(full_cluster_path, bbox_inches='tight')
    plt.close()

    # Prepare results for template
    results = {
        'corr_plot': corr_plot_url,
        'cluster_plot': cluster_plot_url,
        'n_clusters': n_clusters,
        'selected_columns': selected_columns,
        'cluster_counts': df['Cluster'].value_counts().to_dict(),
        'timestamp': timestamp
    }


    return render_template('comparison_clustering.html',results=results,previous_results=previous_results)

#******************************  Error Handling  ************************************


@app.errorhandler(ValueError)
def handle_value_error(error):
    return render_template('error.html',error_message=str(error)),400

@app.errorhandler(Exception)
def handle_generic_error(error):
    return render_template('error.html',error_message="An unexpected error occurred"),500


#******************************  clean messy CSV files  ************************************
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


@app.route('/clean-csv', methods=['POST'])
def clean_csv():
    if 'file' not in request.files:
        return json.jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return json.jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        # Save original file
        orig_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{file.filename}")
        file.save(orig_path)

        # Clean the CSV
        cleaned_path = os.path.join(app.config['UPLOAD_FOLDER'], f"cleaned_{file.filename}")
        clean_messy_csv(orig_path, cleaned_path)

        # Create download link
        return json.jsonify({
            'message': 'File cleaned successfully!',
            'download_link': f'/download/cleaned_{file.filename}',
            'filename': f"cleaned_{file.filename}"
        })

    return json.jsonify({'error': 'Invalid file type. Only CSV files accepted.'}), 400


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


def clean_messy_csv(input_path, output_path):

    # Detect file encoding
    with open(input_path, 'rb') as f:
        raw_data = f.read(10000)
        encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'

    # Read file with detected encoding
    with open(input_path, 'r', encoding=encoding, errors='replace') as f:
        lines = f.readlines()

    # Skip empty lines
    lines = [line.strip() for line in lines if line.strip()]

    if not lines:
        # Create empty cleaned file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")
        return output_path

    # Detect delimiter from first non-empty line
    first_line = lines[0]
    delimiter_candidates = [',', ';', '\t', '|', '/', '-', ' - ', ':']
    delimiter_counts = {delim: first_line.count(delim) for delim in delimiter_candidates}
    detected_delimiter = max(delimiter_counts, key=delimiter_counts.get) if delimiter_counts else ','

    # Process all lines
    cleaned_lines = []
    max_columns = 0

    for line in lines:
        # Handle lines where all data is in one column
        if detected_delimiter not in line and any(c in line for c in delimiter_candidates):
            # Find which delimiter actually exists in the line
            for delim in delimiter_candidates:
                if delim in line:
                    parts = line.split(delim)
                    cleaned_line = [part.strip() for part in parts]
                    cleaned_lines.append(cleaned_line)
                    max_columns = max(max_columns, len(cleaned_line))
                    break
        else:
            # Split using detected delimiter
            parts = line.split(detected_delimiter)
            cleaned_line = [part.strip() for part in parts]
            cleaned_lines.append(cleaned_line)
            max_columns = max(max_columns, len(cleaned_line))

    # Normalize all rows to have the same number of columns
    normalized_lines = []
    for line in cleaned_lines:
        if len(line) < max_columns:
            # Pad with empty values
            normalized_line = line + [''] * (max_columns - len(line))
        else:
            # Truncate extra columns
            normalized_line = line[:max_columns]
        normalized_lines.append(normalized_line)

    # Write cleaned CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(normalized_lines)

    return output_path




if __name__ == '__main__' :
    app.run(debug=True)
