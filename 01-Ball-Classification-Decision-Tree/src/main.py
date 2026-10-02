"""
------------------------------------------------------------
Project Name        : Ball Classification

Dataset Information :

Surface Encoding    :   Rough  = 1
                        Smooth = 0

Ball Type Encoding  :   Tennis Ball  = 1
                        Cricket Ball = 2

Features            :   Weight  -> Weight of the ball (grams)
                        Surface -> Surface type of the ball

Target              :   Ball Type

Machine Learning Information :

Algorithm Used      :   Decision Tree Classifier

Library             :   Scikit-Learn

Problem Type        :   Classification

Author              :   Ishwari Vijaykumar Surve

Date                :   02/10/2026
------------------------------------------------------------
"""

############################################################
# Required Python Packages
############################################################

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


############################################################
# Constants
############################################################

# Surface : Rough = 1, Smooth = 0
# Label   : Tennis = 1, Cricket = 2

TEST_SIZE = 2
RANDOM_STATE = 42


############################################################
# Function Name : display_data
# Description   : Display the case study title
# Input         : Nothing
# Output        : Prints the heading
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def display_data():
    print("------------------------------------------------------------")
    print("---------- Ball Classification Case Study ------------------")
    print("------------------------------------------------------------")


############################################################
# Function Name : load_data
# Description   : Load the encoded dataset
#                 Features : [Weight, Surface]
#                 Surface  : Rough = 1, Smooth = 0
#                 Labels   : Tennis = 1, Cricket = 2
# Input         : Nothing
# Output        : Features, Labels
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def load_data():
    """Return features and labels."""

    # Independent Variables
    features = [
        [35, 1], [47, 1], [90, 0], [48, 1], [90, 0],
        [35, 1], [92, 0], [35, 1], [35, 1], [35, 1],
        [96, 0], [43, 1], [110, 0], [35, 1], [95, 0]
    ]

    # Dependent Variables
    labels = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]

    return features, labels


############################################################
# Function Name : split_dataset
# Description   : Split the dataset into training and testing
# Input         : Features, Labels
# Output        : train_x, test_x, train_y, test_y
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def split_dataset(features, labels):
    """Split dataset into training and testing data."""

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    return train_x, test_x, train_y, test_y


############################################################
# Function Name : build_model
# Description   : Create the Decision Tree model
# Input         : Nothing
# Output        : Decision Tree model
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def build_model():
    """Build and return a Decision Tree classifier."""

    return tree.DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )


############################################################
# Function Name : train_model
# Description   : Train the model using training data
# Input         : Model, train_x, train_y
# Output        : Trained model
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def train_model(model, train_x, train_y):
    """Train the Decision Tree model."""

    model.fit(train_x, train_y)

    return model


############################################################
# Function Name : evaluate_model
# Description   : Calculate model accuracy on test data
# Input         : Trained model, test_x, test_y
# Output        : Accuracy score
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def evaluate_model(model, test_x, test_y):
    """Calculate and return model accuracy."""

    predictions = model.predict(test_x)

    return accuracy_score(test_y, predictions)


############################################################
# Function Name : predict_ball
# Description   : Predict the type of a new sports ball
# Input         : Trained model, weight, surface
#                 Surface : Rough = 1, Smooth = 0
# Output        : Prediction message
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def predict_ball(model, weight, surface):
    """Predict the ball type for a new sample."""

    result = model.predict([[weight, surface]])[0]

    if result == 1:
        return "Object looks like Tennis Ball"

    return "Object looks like Cricket Ball"


############################################################
# Function Name : display_footer
# Description   : Display the project completion message
# Input         : Nothing
# Output        : Prints the footer
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def display_footer():
    print("------------------------------------------------------------")
    print("Ball Classification Case Study Completed Successfully")
    print("------------------------------------------------------------")


############################################################
# Function Name : main
# Description   : Main function from where execution starts
# Author        : Ishwari Vijaykumar Surve
# Date          : 02/10/2026
############################################################

def main():

    display_data()

    # 1. Load dataset
    features, labels = load_data()

    # 2. Split dataset into training and testing data
    train_x, test_x, train_y, test_y = split_dataset(
        features,
        labels
    )

    # 3. Build and train Decision Tree model
    model = build_model()
    model = train_model(model, train_x, train_y)

    # 4. Evaluate model
    accuracy = evaluate_model(model, test_x, test_y)

    print(f"Model Accuracy :: {accuracy * 100:.2f} %")

    # 5. Predict new balls
    print(predict_ball(model, 37, 1))
    print(predict_ball(model, 94, 0))

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":
    main()
