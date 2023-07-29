# Hiep Pham

# For data frames
import pandas as pd
# For visualizations
import matplotlib.pyplot as pyplot
from pandas.plotting import scatter_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# For prediction model
from sklearn import metrics, model_selection, linear_model

# Data we are using from kaggle, https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data
# Data is wrangled by truncating the sex and island column. Sex is indicated by beak size, and island is unknown in
#   this scenario
penguin_data = "https://raw.githubusercontent.com/HiepHPham/MachineLearningModel/penguins_data_set.csv"
print("\n-----------------")
print("Penguin data has been pulled from kaggle.")
# Set headers for the data_set:
# Culmen_length_mm, culmen_depth_mm, flipper_length_mm, and body_mass_g will be independent variables
# 'Species' will be the dependent variables
penguin_headers = ['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("\n-----------------")
print("Penguin data has been assigned headers.")
# Dataframe to read in the penguin data set, and combine with the headers
penguin_data_frame = pd.read_csv(penguin_data, names=penguin_headers)
print("\n-----------------")
print("Penguin data frame has been created:")
print(penguin_data_frame)

# NON-DESCRIPTIVE METHOD
# Logistic regression assigns each dependent variable a probability and chooses the variable with the highest
#   probability
penguin_prediction_model = linear_model.LogisticRegression(max_iter=100000)
print("\n-----------------")
print("Penguin prediction model has been created using Logistic Regression.")

# Data processing, break into dependent and independent subsets
y_dependent_variable = penguin_data_frame.values[:, 0]
x_independent_variable = penguin_data_frame.values[:, 1:5]
print("\n-----------------")
print("Penguin data has been separated into dependent and independent variables. We will be using penguin dimensions "
      "(independent variables) to predict the penguin species (dependent variable).")

# train_test_split returns 4 variables: training and test data for independent & dependent
x_independent_variable_train, x_independent_variable_test, y_dependent_variable_train, y_dependent_variable_test = \
    model_selection.train_test_split(x_independent_variable, y_dependent_variable, test_size=0.2)
print("\n-----------------")
print("Training and testing data has been created. 80% of the data will be for training and the remaining 20% will be"
      " for testing.")

# Train model, produces function that maps penguin descriptors (independent) to species (dependent)
# Uses the created training data returned from the train_test_split function
penguin_prediction_model.fit(x_independent_variable_train, y_dependent_variable_train)
print("\n-----------------")
print("Penguin prediction model has been fit using the created training data.")

# Making a prediction model based on created test data returned from the train_test_split function
y_dependent_variable_predict = penguin_prediction_model.predict(x_independent_variable_test)

# Metrics to calculate accuracy using created independent test data returned from the train_test_split function
print("\n-----------------")
print(f'This is the confidence level for this prediction model: '
      f'\n{metrics.accuracy_score(y_dependent_variable_test, y_dependent_variable_predict)}')


# DESCRIPTIVE METHOD
# Pandas tool, display data frame as histogram. Shows count of penguin species in the dataset
print("\n-----------------")
print("Creating histogram to show count of penguin species in dataset:")
pd.Series(penguin_data_frame.values[:, 0]).value_counts().plot(kind='bar')

# DESCRIPTIVE METHOD
# Scatter matrix to show relations between the penguin descriptors
print("\n-----------------")
print("Creating scatter matrix to show the relations between the penguin descriptors:")
scatter_matrix(penguin_data_frame)

# DESCRIPTIVE METHOD
# Confusion model to assure confidence in the prediction model. Shows correct and incorrect labeling.
print("\n-----------------")
print("Creating confusion matrix to assure confidence in the prediction model (diagonal is good):")
confusion_matrix = metrics.confusion_matrix(y_dependent_variable_test, y_dependent_variable_predict,
                                            labels=penguin_prediction_model.classes_)
confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                  display_labels=penguin_prediction_model.classes_)
confusion_matrix_display.plot()

print("\n*************")
print("Close all graphs to begin using The Penguin Identifier.")
print("*************\n")
# Show the created charts
pyplot.show()

# Creating start flag for while loop to enable repeated penguin identifying
continue_flag = "1"
# Until user enters "2" to quit
while continue_flag != "2":
    # Program introduction
    print("----------------------------------\n"
          "The Penguin Identifier\n"
          "----------------------------------\n"
          "This program will identify a penguin species based on provided measurements.\n"
          "----------------------------------\n"
          "Provide the following measurements in order separated by a space: \n"
          "culmen_length_mm culmen_depth_mm flipper_length_mm body_mass_g\n"
          "----------------------------------\n"
          "For example: \n"
          "'49.9 16.1 213 5400' will predict 'Gentoo'.\n"
          "'39.7 17.7 193 3200' will predict 'Adelie'.\n"
          "'54.2 20.8 201 4300' will predict 'Chinstrap'.\n"
          "----------------------------------\n")
    # Redisplay confidence level
    print(f'The confidence level for this prediction model is: '
          f'\n{metrics.accuracy_score(y_dependent_variable_test, y_dependent_variable_predict)}')
    print("\n-----------------")

    # Obtain user intention. (1) is continue, and (2) is to quit
    continue_flag = input("Would you like to continue?\n"
                          "Enter '1' to continue.\n"
                          "Enter '2' to quit.\n")
    # If the user wishes to continue
    if continue_flag == "1":
        # Try block for if the user enters anything but 4 numbers
        try:
            # Create an input list based on the inputs by the user
            input_list = [float(x) for x in input("Enter four numbers separated by a space:\n").split()]
            # Print the prediction model prediction using the input list.
            print(penguin_prediction_model.predict([input_list]))
        # If the user enters anything but 4 numbers, a ValueError will be caught instead of stopping the program
        except ValueError:
            # Print Invalid input, and passes to let the user try again.
            print("Invalid input.")
    # If the user wishes to quit the program
    elif continue_flag == "2":
        # Quits out of the while loop and ends the program
        quit()
    # If the user enters anything else but 1 or 2
    else:
        # Print Invalid input, and passes to let the user try again.
        print("Invalid input.")

# Program end message
print("Penguin Identifier complete. We hope this tool was useful to you.")
