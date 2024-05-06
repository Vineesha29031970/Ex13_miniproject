# Ex.No: 13 Learning – Use Supervised Learning
# DATE: 22/04/2024
# REGISTER NUMBER : 212221040180
# AIM:
To write a program to train the classifier for Diabetes Prediction.

# Algorithm:
1.Start the program.

2.Import required Python libraries, including NumPy, Pandas, Google Colab, Gradio, and various scikit-learn modules.

3.Mount Google Drive using Google Colab's 'drive.mount()' method to access the data file located in Google Drive.

4.Install the Gradio library using 'pip install gradio'.

5.Load the diabetes dataset from a CSV file ('diabetes.csv') using Pandas.

6.Separate the target variable ('Outcome') from the input features and Scale the input features using the StandardScaler from scikit-learn.

7.Create a multi-layer perceptron (MLP) classifier model using scikit-learn's 'MLPClassifier'.

8.Train the model using the training data (x_train and y_train).

9.Define a function named 'diabetes' that takes input parameters for various features and Use the trained machine learning model to predict the outcome based on the input features.

10.Create a Gradio interface using 'gr.Interface' and Specify the function to be used to make predictions based on user inputs.

11.Launch the Gradio web application, enabling sharing, to allow users to input their data and get predictions regarding diabetes risk.

12.Stop the program.

# Program:
```
import numpy as np
import pandas as pd
```

```
pip install gradio
pip install typing-extensions --upgrade
import gradio as gr
```
<img width="573" alt="image" src="https://github.com/Vineesha29031970/Ex13_miniproject/assets/133136880/76805d37-0fcb-417d-b0e2-ef6fbd9acc39">

```
print(data.columns)
```

<img width="614" alt="image" src="https://github.com/Vineesha29031970/Ex13_miniproject/assets/133136880/177d8f0a-fb6e-47f1-ba39-9108cefda91d">

```
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
```

<img width="606" alt="image" src="https://github.com/Vineesha29031970/Ex13_miniproject/assets/133136880/0826c44d-0992-4922-aaf7-581a9e1c3262">


```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y)

#scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#instatiate model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))
```
<img width="589" alt="image" src="https://github.com/Vineesha29031970/Ex13_miniproject/assets/133136880/2a0425fc-448e-44b8-8702-56b0f1b07acd">

```
#create a function for gradio
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```

<img width="605" alt="image" src="https://github.com/Vineesha29031970/Ex13_miniproject/assets/133136880/0706f94e-4647-458e-8f91-33d8d3071613">



<img width="300" alt="image" src="https://github.com/Vineesha29031970/Ex13_miniproject/assets/133136880/4b155061-6c83-424f-b4af-00b6b11f8e07">

# Result:
Thus the system was trained successfully and the prediction was carried out.










