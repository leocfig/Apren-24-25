from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

# Loading the dataset
data = loadarff('diabetes.arff')
df = pd.DataFrame(data[0])

# Decoding the byte string class labels
df['Outcome'] = df['Outcome'].str.decode('utf-8')

# Separate the features from the target
X = df.drop(columns=['Outcome'])        # Features (8 biological features)
y = df['Outcome']                       # Target (normal/diabetes)

# Define the parameters for the samples_split
min_samples_splits = [2, 5,10, 20, 30, 50, 100]

# Initialize lists to store the accuracies
train_accuracies = []
test_accuracies = []

# Stratified 80-20 training-testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=1)

# 
for min_samples in min_samples_splits:
    training_accs = []
    testing_accs = []

    # Run 10 times for averaging
    for _ in range(10):

        # Initialize a Decision Tree Classifier with the current number min_samples
        classifier = DecisionTreeClassifier(min_samples_split=min_samples, random_state=1)

        # Fit the model
        classifier.fit(X_train, y_train)

        # Predict on training sets
        train_pred = classifier.predict(X_train)

        # Predict on testing sets
        test_pred = classifier.predict(X_test)

        # Calculate accuracies
        training_acc = accuracy_score(y_train, train_pred)
        testing_acc = accuracy_score(y_test, test_pred)
        
        # Store accuracies
        training_accs.append(training_acc)
        testing_accs.append(testing_acc)
    
    # Average accuracies for current min_samples_split

    # Calculate mean for training accuracies
    train_mean = sum(training_accs) / len(training_accs) if training_accs else 0
    train_accuracies.append(train_mean)

    # Calculate mean for testing accuracies
    test_mean = sum(testing_accs) / len(testing_accs) if testing_accs else 0
    test_accuracies.append(test_mean)


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(min_samples_splits, train_accuracies, label='Training Accuracy')
plt.plot(min_samples_splits, test_accuracies, label='Testing Accuracy')
plt.title('Training and Testing Accuracies vs Minimum Samples Split')
plt.xlabel('Minimum Samples Split')
plt.ylabel('Accuracy')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()  # Stops elements from being cut off
plt.show()


