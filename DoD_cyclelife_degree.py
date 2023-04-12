import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Generate some example data
DoD = [10, 20, 30, 40, 50, 60, 65, 70, 75, 80]
cyclelife = [170000, 48000, 21050, 11400, 6400, 4150, 3500, 3000, 2700, 2500]


# Split the data into training and testing sets
DoD_train, DoD_test, cyclelife_train, cyclelife_test = train_test_split(DoD, cyclelife, test_size=0.2, random_state=42)
DoD_train = np.array(DoD_train)
cyclelife_train = np.array(cyclelife_train)
DoD_test = np.array(DoD_test)
cyclelife_test = np.array(cyclelife_test)

# Loop over different polynomial degrees and evaluate their performance
best_degree = 1
best_score = -np.inf
for degree in range(1, 20):
    # Fit a polynomial regression model
    poly = PolynomialFeatures(degree=degree)
    X_train = poly.fit_transform(DoD_train.reshape(-1, 1))
    model = LinearRegression().fit(X_train, cyclelife_train)
    
    # Evaluate the model on the testing set
    X_test = poly.fit_transform(DoD_test.reshape(-1, 1))
    score = model.score(X_test, cyclelife_test)
    
    # Update the best degree if this degree has a better score
    if score > best_score:
        best_degree = degree
        best_score = score
        
print(f"Best polynomial degree: {best_degree}")
