import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y, iris.target_names

def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def find_best_k(X_train, y_train, X_val, y_val, k_range=range(1,21)):
    results = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5).mean()
        results.append((k, acc, cv))
    df = pd.DataFrame(results, columns=["k","val_accuracy","cv_mean"])
    best_row = df.sort_values(["val_accuracy","cv_mean"], ascending=False).iloc[0]
    return df, int(best_row["k"])

def plot_accuracy_vs_k(df):
    plt.figure(figsize=(8,4))
    plt.plot(df['k'], df['val_accuracy'], marker='o', label='val accuracy')
    plt.plot(df['k'], df['cv_mean'], marker='s', label='cv mean (train)')
    plt.xlabel('k (neighbors)')
    plt.ylabel('Accuracy')
    plt.title('K vs Accuracy')
    plt.xticks(df['k'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=target_names))
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_decision_boundaries(X, y, model, feature_names, h=0.02):
    
    if X.shape[1] < 2:
        print("Need at least two features for boundary plot.")
        return

    X2 = X[:, :2]
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), 
                             
                             np.zeros((xx.ravel().shape[0], max(0, X.shape[1]-2)))])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor='k', s=50)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision boundaries (first two features)')
    plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(y).astype(str))
    plt.tight_layout()
    plt.show()

def main():
    
    X_df, y_series, target_names = load_data()
    X, scaler = preprocess(X_df)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y_series, test_size=0.25, random_state=42, stratify=y_series)

   
    df_k, best_k = find_best_k(X_train, y_train, X_test, y_test, k_range=range(1,21))
    print("K search results:\n", df_k)
    print(f"\nBest k selected: {best_k}")

    plot_accuracy_vs_k(df_k)

 
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    y_pred = knn_final.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy with k={best_k}: {test_acc:.4f}")

    
    plot_confusion(y_test, y_pred, target_names)

  
    X2 = X[:, :2]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y_series, test_size=0.25, random_state=42, stratify=y_series)
    knn_2d = KNeighborsClassifier(n_neighbors=best_k)
    knn_2d.fit(X2_train, y2_train)
    plot_decision_boundaries(X, y_series.values, knn_2d, feature_names=list(X_df.columns[:2]))

if __name__ == "__main__":
    main()
