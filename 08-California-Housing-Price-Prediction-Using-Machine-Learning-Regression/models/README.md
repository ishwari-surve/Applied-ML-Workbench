# Models

This folder keeps the trained model of the California Housing Price Prediction case study.

## File
- `california_housing_model.pkl`

## About the model
- The best ensemble model is saved here. In the tested run it is the Random Forest.
- It was trained on 80% of the data (16,512 areas).
- It is saved with joblib, and the file is about 32 MB.

## The model file is not uploaded to GitHub
- The `.gitignore` of this repository blocks `*.pkl` files, because model files are created by the code and can be large.
- This README keeps the folder on GitHub.
- To create the model, run the program from the case study folder:

      pip install -r requirements.txt
      python src/main.py

  The file `california_housing_model.pkl` is created in this folder (Step 11 of the program).

## How to use the saved model

    import joblib
    import pandas as pd

    model = joblib.load("models/california_housing_model.pkl")

    area = pd.DataFrame([{
        "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841,
        "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556,
        "Latitude": 37.88, "Longitude": -122.23
    }])

    price = model.predict(area)[0] * 100000
    print(f"Predicted price : {price:,.0f} dollars")

## Notes
- The prediction is in units of 100,000 dollars, so the code above multiplies it by 100,000.
- The prices in this dataset are cut off at 500,000 dollars, so the model cannot predict above that value.
- Use the same column names and order as in the dataset.
