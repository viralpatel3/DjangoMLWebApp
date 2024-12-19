from django.shortcuts import render
import pickle

def home(request):
    """
    Render the home page of the Titanic Prediction application.

    This function handles requests to the home page and renders the index.html template.

    Parameters:
    request (HttpRequest): The HTTP request object sent by the client.

    Returns:
    HttpResponse: A rendered HTML response containing the home page content.
    """
    return render(request, 'index.html')

def generate_prediction(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    """
    Predicts survival on the Titanic based on passenger information.

    This function loads a pre-trained machine learning model and scaler,
    transforms the input data, and makes a prediction on whether the
    passenger survived or not.

    Parameters:
    pclass (int): Passenger class (1, 2, or 3)
    sex (int): Gender of the passenger (0 for male, 1 for female)
    age (int): Age of the passenger
    sibsp (int): Number of siblings/spouses aboard
    parch (int): Number of parents/children aboard
    fare (float): Fare paid for the ticket
    C (int): Whether the passenger embarked from Cherbourg (0 or 1)
    Q (int): Whether the passenger embarked from Queenstown (0 or 1)
    S (int): Whether the passenger embarked from Southampton (0 or 1)

    Returns:
    str: 'yes' if the model predicts survival, 'no' if it predicts death,
         'error' if the prediction is neither 0 nor 1
    """
    model = pickle.load(open('ml_model.sav', 'rb'))
    scaled = pickle.load(open('scaler.sav', 'rb'))

    prediction = model.predict(scaled.transform([
        [pclass, sex, age, sibsp, parch, fare, C, Q, S]
    ]))

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def result(request):
    """
    Process the form submission and render the prediction result.

    This function handles the GET request containing passenger information,
    calls the prediction function, and renders the result page.

    Parameters:
    request (HttpRequest): The HTTP request object containing GET parameters:
        - pclass (str): Passenger class (1, 2, or 3)
        - sex (str): Gender of the passenger (0 for male, 1 for female)
        - age (str): Age of the passenger
        - sibsp (str): Number of siblings/spouses aboard
        - parch (str): Number of parents/children aboard
        - fare (str): Fare paid for the ticket
        - embC (str): Whether the passenger embarked from Cherbourg (0 or 1)
        - embQ (str): Whether the passenger embarked from Queenstown (0 or 1)
        - embS (str): Whether the passenger embarked from Southampton (0 or 1)

    Returns:
    HttpResponse: A rendered HTML response displaying the prediction result.
    """
    pclass = int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])
    embC = int(request.GET['embC'])
    embQ = int(request.GET['embQ'])
    embS = int(request.GET['embS'])

    result = generate_prediction(pclass, sex, age, sibsp,
                            parch, fare, embC, embQ, embS)

    return render(request, 'result.html', {'result': result})
