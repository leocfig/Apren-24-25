from sklearn.tree import plot_tree, DecisionTreeClassifier
from scipy.io.arff import loadarff
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset
data = loadarff('diabetes.arff')
df = pd.DataFrame(data[0])

# Decoding the byte string class labels
df['Outcome'] = df['Outcome'].str.decode('utf-8')

# Separate the features from the target
X = df.drop(columns=['Outcome'])        # Features (8 biological features)
y = df['Outcome']                       # Target (normal/diabetes)

# Initialize a Decision Tree Classifier with the required maximum depth
classifier = DecisionTreeClassifier(max_depth=3, random_state=1)

# Fit the model
classifier.fit(X, y)


# Plot the results
plt.figure(figsize=(12, 6))
plot_tree(classifier, feature_names=X.columns, class_names=['Normal', 'Diabetes'], filled=True)
plt.title('Decision Tree (Max Depth = 3)')
plt.show()


