from scipy.io.arff import loadarff
from sklearn.feature_selection import f_classif
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
data = loadarff('diabetes.arff')
df = pd.DataFrame(data[0])

# Decoding the byte string class labels
df['Outcome'] = df['Outcome'].str.decode('utf-8')

# Separate the features from the target
X = df.drop(columns=['Outcome'])        # Features (8 biological features)
y = df['Outcome']                       # Target (normal/diabetes)

# Using ANOVA to determine the input variables with the worst 
# and best discriminative power
f_values = f_classif(X, y)[0]
best_power = X.columns[f_values.argmax()]
worst_power = X.columns[f_values.argmin()]

print(f'The input variable with the best discriminative power is {best_power}')
print(f'The input variable with the worst discriminative power is {worst_power}')


# Plotting the class-conditional probability density function of each feature

plt.figure(figsize=(12, 6))

# Plot for the best discriminative feature
plt.subplot(1, 2, 1)
sns.kdeplot(X[best_power][y == '0'], label='Normal', fill=True)
sns.kdeplot(X[best_power][y == '1'], label='Diabetes', fill=True)
plt.title(f'Class-Conditional PDF of {best_power}')
plt.xlabel(best_power)
plt.ylabel('Density')
plt.legend(title='Outcome')

# Plot for the worst discriminative feature
plt.subplot(1, 2, 2)
sns.kdeplot(X[worst_power][y == '0'], label='Normal', fill=True)
sns.kdeplot(X[worst_power][y == '1'], label='Diabetes', fill=True)
plt.title(f'Class-Conditional PDF of {worst_power}')
plt.xlabel(worst_power)
plt.ylabel('Density')
plt.legend(title='Outcome')

plt.show()