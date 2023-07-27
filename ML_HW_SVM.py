#=================================================
#   ML_HW4_FoadMoslem_401129902
#   Foad Moslem - PhD Student - Aerodynamics
#   Using Python 3.9.16 & Spyder IDE
#=================================================

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


#%% Libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


#%% Pre-processing the data

# Load the dataset
data = pd.read_csv("./mitbih.csv")


# Check data
data
data.info() # get a quick description of the data
data.describe() # shows a summary of the numerical attributes


# Missing values
print(data.isnull().sum()) #check missing data
data.fillna(data.mean(), inplace=True) # Replace missing values with mean


# Duplicated data
print(data.duplicated().sum()) # check duplicated data | there are some duplicate rows in the dataset since some of the values are True.


# Check data again
data
data.info() # get a quick description of the data
data.describe() # shows a summary of the numerical attributes



#%% Split the data into features and labels
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X = np.array(X)
y = np.array(y)

# Split the data into train and test
""" Randomly select 70% of the data as training data """
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.7, random_state=42)



#%% (LSVM) Linear SVM Algorithm
""" (a) Build an SVM learner with linear SVM algorithm and train data set. 
Calculate the classification accuracy on train and test sets. Then specify 
the number of support vectors. """

# Create an SVM learner with linear kernel
LSVM = svm.SVC(kernel="linear")

# Train the learner on the train set
LSVM.fit(X_train, y_train)

# Predict the labels on the train and test sets
LSVM_y_train_pred = LSVM.predict(X_train)
LSVM_y_test_pred = LSVM.predict(X_test)

# Calculate the accuracy scores on the train and test sets
LSVM_train_acc = accuracy_score(y_train, LSVM_y_train_pred)
LSVM_test_acc = accuracy_score(y_test, LSVM_y_test_pred)

# Print the results
print("Train accuracy (LSVM):", LSVM_train_acc)
print("Test accuracy (LSVM):", LSVM_test_acc)
print("Number of support vectors (LSVM):", len(LSVM.support_))



#%% (NLSVM_soft) Nonlinear SVM Algorithm In SVM Soft Mode
""" (b) Build an SVM learner with the nonlinear SVM algorithm in SVM soft mode. 
Adjust the SVM soft parameter using the validation data you create from the 
training set to give the best accuracy on the validation set. Plot the curve 
of changes of classification accuracy on validation data based on C value. 
Calculate the accuracy of the classification on the train and test sets in 
this best selected value. Then specify the number of support vectors and 
compare with the previous state."""

# Train SVM model with nonlinear kernel in soft mode
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
NLSVM_soft_val_accs = []
for C in C_values:
    NLSVM_soft = svm.SVC(kernel='rbf', C=C)
    NLSVM_soft.fit(X_train, y_train)
    NLSVM_soft_val_accs.append(NLSVM_soft.score(X_val, y_val))

# Plot curve of changes of classification accuracy on validation data based on C value
plt.plot(C_values, NLSVM_soft_val_accs)
plt.xlabel('C values')
plt.ylabel('Validation accuracy')
plt.show()

# Select best C value based on validation accuracy
NLSVM_soft_best_C = C_values[np.argmax(NLSVM_soft_val_accs)]

# Train SVM model with best C value on full training set
NLSVM_soft_bestC = svm.SVC(kernel='rbf', C=NLSVM_soft_best_C)
NLSVM_soft_bestC.fit(X_train, y_train)

# Predict the labels on the train and test sets
NLSVM_soft_y_train_pred = NLSVM_soft_bestC.predict(X_train)
NLSVM_soft_y_test_pred = NLSVM_soft_bestC.predict(X_test)

# Calculate classification accuracy on train and test sets with best C value
NLSVM_soft_train_acc_bestC = accuracy_score(y_train, NLSVM_soft_y_train_pred)
NLSVM_soft_test_acc_bestC = accuracy_score(y_test, NLSVM_soft_y_test_pred)

# Print results
print("Best C (NLSVM_soft):", NLSVM_soft_best_C)
print("Train accuracy (NLSVM_soft):", NLSVM_soft_train_acc_bestC)
print("Test accuracy (NLSVM_soft):", NLSVM_soft_test_acc_bestC)
print("Number of support vectors (NLSVM_soft):", NLSVM_soft_bestC.n_support_)



#%% (NLSVM_kernel) Nonlinear SVM Algorithm In Kernel Mode
""" (c) Build an SVM learner with the nonlinear SVM algorithm in kernel mode. 
Choose RBF kernel and polynomial kernel. You can choose the best kernel based 
on the accuracy of the validation set. Calculate the classification accuracy 
on train and test sets in this best selected kernel. Then specify the number 
of support vectors."""

# Train SVM model with RBF kernel in kernel mode
NLSVM_kernel_rbf = svm.SVC(kernel='rbf')
NLSVM_kernel_rbf.fit(X_train, y_train)

# Train SVM model with polynomial kernel in kernel mode
NLSVM_kernel_poly = svm.SVC(kernel='poly')
NLSVM_kernel_poly.fit(X_train, y_train)

# Calculate classification accuracy on validation sets for both kernels
val_acc_NLSVM_kernel_rbf = NLSVM_kernel_rbf.score(X_val, y_val)
val_acc_NLSVM_kernel_poly = NLSVM_kernel_poly.score(X_val, y_val)

# Select best kernel based on validation accuracy
if val_acc_NLSVM_kernel_rbf > val_acc_NLSVM_kernel_poly:
    best_NLSVM_kernel = 'rbf'
    NLSVM_kernel_best = NLSVM_kernel_rbf
else:
    best_NLSVM_kernel = 'poly'
    NLSVM_kernel_best = NLSVM_kernel_poly

# Predict the labels on the train and test sets
NLSVM_kernel_y_train_pred = NLSVM_kernel_best.predict(X_train)
NLSVM_kernel_y_test_pred = NLSVM_kernel_best.predict(X_test)

# Calculate classification accuracy on train and test sets with best kernel
NLSVM_kernel_train_acc_best = accuracy_score(y_train, NLSVM_kernel_y_train_pred)
NLSVM_kernel_test_acc_best = accuracy_score(y_test, NLSVM_kernel_y_test_pred)

# Print results
print("Best kernel (NLSVM_kernel):", best_NLSVM_kernel)
print("Train accuracy (NLSVM_kernel):", NLSVM_kernel_train_acc_best)
print("Test accuracy (NLSVM_kernel):", NLSVM_kernel_test_acc_best)
print("Number of support vectors (NLSVM_kernel):", NLSVM_kernel_best.n_support_)



#%% (NLSVM_Combined) Nonlinear SVM Algorithm In The Combined Mode of Kernel & SVM Soft
""" (d) Build an SVM learner with the non-linear SVM algorithm in the combined
mode of kernel and SVM soft. Choose the best kernel from the previous step. 
Adjust the SVM soft parameter using the validation data to give the best 
accuracy on the validation set. Calculate the classification accuracy on the 
train and test sets at this best value. Then specify the number of support 
vectors."""

""" **************** """
""" same as part (b) """
""" **************** """



#%% (NLSVM_Combined_cv) Nonlinear SVM Algorithm In The Combined Mode of Kernel & SVM Soft with 5-fold-cross validation
""" (e) Repeat the previous section with 5-fold-cross validation and compare 
results."""

# Train SVM model with best kernel from previous step in combined mode of kernel and SVM soft with 5-fold-cross validation
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
NLSVM_Combined_cv_val_accs = []
for C in C_values:
    NLSVM_Combined_cv = svm.SVC(kernel='rbf', C=C)
    NLSVM_Combined_cv_val_accs.append(np.mean(cross_val_score(NLSVM_Combined_cv, X_train, y_train, cv=5)))

# Select best C value based on validation accuracy
NLSVM_Combined_cv_best_C = C_values[np.argmax(NLSVM_Combined_cv_val_accs)]

# Train SVM model with best kernel and best C value on full training set in combined mode of kernel and SVM soft
NLSVM_Combined_cv_bestC = svm.SVC(kernel='rbf', C=NLSVM_Combined_cv_best_C)
NLSVM_Combined_cv_bestC.fit(X_train, y_train)

# Predict the labels on the train and test sets
NLSVM_Combined_cv_y_train_pred = NLSVM_Combined_cv_bestC.predict(X_train)
NLSVM_Combined_cv_y_test_pred = NLSVM_Combined_cv_bestC.predict(X_test)

# Calculate classification accuracy on train and test sets with best kernel and best C value
NLSVM_Combined_cv_train_acc_best = accuracy_score(y_train, NLSVM_Combined_cv_y_train_pred)
NLSVM_Combined_cv_test_acc_best = accuracy_score(y_test, NLSVM_Combined_cv_y_test_pred)

# Print results
print("Best kernel (NLSVM_Combined_cv):", NLSVM_Combined_cv_best_C)
print("Train accuracy (NLSVM_Combined_cv):", NLSVM_Combined_cv_train_acc_best)
print("Test accuracy (NLSVM_Combined_cv):", NLSVM_Combined_cv_test_acc_best)
print("Number of support vectors (NLSVM_Combined_cv):", NLSVM_Combined_cv_bestC.n_support_)


