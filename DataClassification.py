import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'death': [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 
         0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
         0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],

    'lvedAvg': [1.19e+02, 6.97e+01, 98, 8.89e+01, 90, 1.20e+02, 98, 101, 76, 93, 87, 
           7.75e+01, 130, 121, 9.23e+01, 128, 00, 1.06e+02, 90, 9.01e+01, 59, 
           82, 7.27e+01, 8.84e+01, 1.03e+02, 1.06e+02, 112, 1.08e+02, 75, 138, 
           7.88e+01, 106, 9.42e+01, 1.16e+02, 9.43e+01, 84, 00, 9.67e+01, 
           1.20e+02, 9.26e+01, 110, 1.30e+02, 94, 93, 117, 62, 129, 00, 
           6.87e+01, 6.77e+01, 71, 7.67e+01, 144, 117, 98, 00, 1.09e+02, 
           7.73e+01, 6.53e+01, 9.93e+01, 130, 1.04e+02, 1.14e+02, 1.62e+02, 
           1.04e+02, 9.07e+01, 9.53e+01, 98, 113, 90, 00, 1.04e+02, 101, 
           5.63e+01, 1.08e+02, 8.13e+01, 7.53e+01, 9.83e+01, 9.95e+01, 
           1.23e+02, 1.16e+02],

    'lvesAvg': [68, 3.13e+01, 38, 4.21e+01, 44, 5.42e+01, 3.93e+01, 58, 3.17e+01, 
           39, 3.15e+01, 3.35e+01, 7.33e+01, 67, 4.17e+01, 68, 00, 3.97e+01, 
           35, 4.69e+01, 2.25e+01, 3.34e+01, 31.4, 3.47e+01, 51, 4.85e+01, 5.46e+01, 
           4.79e+01, 2.84e+01, 6.03e+01, 3.17e+01, 48, 41, 5.04e+01, 4.43e+01, 
           38, 00, 4.33e+01, 6.34e+01, 4.33e+01, 5.05e+01, 67, 47, 46, 63, 24, 
           7.15e+01, 00, 30, 31, 2.93e+01, 33, 6.67e+01, 6.17e+01, 40, 00, 55, 
           4.03e+01, 2.47e+01, 41, 42, 5.33e+01, 5.83e+01, 1.08e+02, 42, 35, 
           3.63e+01, 49, 5.07e+01, 2.93e+01, 00, 42, 4.83e+01, 24, 6.13e+01, 
           3.73e+01, 3.77e+01, 4.93e+01, 4.25e+01, 4.93e+01, 6.24e+0]      
})

# Remove rows where lvedAvg or lvesAvg is zero
data_cleaned = data[(data['lvedAvg'] != 0) & (data['lvesAvg'] != 0)]

# Print the cleaned dataset
print(data_cleaned)

# Define features (X) and target (y)
X = data_cleaned[['lvedAvg', 'lvesAvg']]
y = data_cleaned['death']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear')  
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the decision boundary 
def plot_decision_boundary(X, y, model):
    # Create a mesh grid for plotting
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict over the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Define custom colors for the classes
    colors = {0: 'black', 1: '#ceb891'}  
    fig, ax = plt.subplots(facecolor='white')  # Set background color to white
    
    # Scatter plot with custom colors
    for class_value in np.unique(y):
        indices = np.where(y == class_value)[0]
        ax.scatter(X.iloc[indices, 0],
                   X.iloc[indices, 1],
                   c=colors[class_value],
                   edgecolor='k',
                   marker='o',
                   label=f'Death = {class_value}')
        
    plt.xlabel('LVED')
    plt.ylabel('LVES')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.show()

# Call the function to visualize decision boundaries for training data
plot_decision_boundary(X_train, y_train, svm_model)