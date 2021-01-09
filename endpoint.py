import requests
import json
scoring_uri='http://a777970f-63ef-40af-a16f-0466710559e4.southcentralus.azurecontainer.io/score'

data = {"data":
        [
          {
            "Pregnancies": 5,
            "Glucose": 155,
            "BloodPressure": 90,
            "SkinThickness": 35,
            "Insulin": 135,
            "BMI": 36.6,
            "DiabetesPedigreeFunction": 0.7,
            "Age": 55,
                      },
          {
            "Pregnancies": 3,
            "Glucose": 145,
            "BloodPressure": 95,
            "SkinThickness": 32,
            "Insulin": 155,
            "BMI": 34.3,
            "DiabetesPedigreeFunction": 0.625,
            "Age": 65,
          },
      ]
    }
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())