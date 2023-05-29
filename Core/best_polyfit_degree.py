import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

def MyFun(X, Y):


    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Loop over different polynomial degrees and evaluate their performance
    best_degree = 1
    best_score = -np.inf
    for degree in range(1, 20):
        # Fit a polynomial regression model
        poly = PolynomialFeatures(degree=degree)
        X_TRAIN = poly.fit_transform(X_train.reshape(-1, 1))
        model = LinearRegression().fit(X_TRAIN, Y_train)
        
        # Evaluate the model on the testing set
        X_TEST = poly.fit_transform(X_test.reshape(-1, 1))
        score = model.score(X_TEST, Y_test)
        
        # Update the best degree if this degree has a better score
        if score > best_score:
            best_degree = degree
            best_score = score
            
    return best_degree