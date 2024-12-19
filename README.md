![Titanic-sinking](https://user-images.githubusercontent.com/59442907/97293330-89abad00-1872-11eb-8b51-2c2e93a6de33.jpg)

# Titanic Survival Predictor ML Model 
### This webapp predicts whether a passenger (with characteristics as your input) can survive or not!


## Screenshots-
![screenshot](https://user-images.githubusercontent.com/59442907/97587747-bf42c880-1a21-11eb-8d54-59730337c737.jpg)


# Technical Details
This is an example repository to show Django can be used as a front end for predicting an output of a machine learning model.

Below are some of the technical details about how the project can be replicated and used for your specific purpose. 

# Prerequisites
* Python: ~= 3.10.0
* Django: ~= 5.1.4
* pandas: ~= 2.2.3
* NumPy: ~= 2.2.0
* scikit-learn: ~= 1.6.0
* conda: ~= 4.8

# Step 1. Setup dev enviornment:
1. Create a new conda environment:

   ```
   conda create --name myenv python=3.10
   ```

2. Activate that new conda environment:

   ```
   conda activate myenv
   ```

3. From the directory where requirements.txt is located install dependncies:

    ```
    pip install -r requirements.txt
    ```
Now your enviornment is setup so moving on to creating the ml model.

# Step 2: Preparing the Data
Before training the model, you need to prepare the data. The training data is located in a CSV file named `train.csv` and test data is located as `test.csv`.

The data files contains the following columns:
- `pclass` (passenger class)
- `sex` (passenger gender)
- `age` (passenger age)
- `sibsp` (number of siblings/spouses aboard the Titanic)
- `parch` (number of parents/children aboard the Titanic)
- `fare` (passenger fare)
- `embarked` (port of embarkation)
- `survived` (whether the passenger survived or not)

The `sex` column contains the values `male` or `female`. The `embarked` column contains the values `S`, `C`, or `Q`, representing the South Hampton, Cherbourg, or Queenstown ports respectively.

# Step 3: Training the Model
Since this project is about how to use Django, no fancy ML application is done and the code is quite vanila as to just get a pickled model out.

To train the model, you need to run the `ml_model.py` script. This script performs the following steps:
1. Reads the `train.csv` file.
2. Renames the columns to lowercase.
3. Selects the relevant columns from the dataset.
4. Maps the `sex` column to binary values (0 for male, 1 for female).
5. Fills missing values in the `age` column with the mean of the non-missing values.
6. Creates dummy variables for the `embarked` column.
7. Scales the dataset using the `MinMaxScaler` from scikit-learn.
8. Splits the dataset into training and testing sets.
9. Trains a logistic regression model on the training set.
10. Saves the trained model and the scaler to the `ml_model.sav` and `scaler.sav` files, respectively.

# Step 4: Running the Web App

To run the web app, you need to follow these steps:
1. Make sure you have Django installed.
2. Navigate to the `titanic` directory.
3. Run the following command to start the development server:
   ```
   python manage.py runserver
   ```
4. Open your web browser and go to `http://localhost:8000/`.
5. Fill in the passenger characteristics in the form and click the "Predict" button.
6. The web app will display the prediction result as shown in the screenshot above.


# Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

# Acknowledgments
This project uses the Titanic dataset from the Kaggle competition.
