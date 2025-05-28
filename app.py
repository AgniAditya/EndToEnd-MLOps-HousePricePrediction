from fastapi import FastAPI
from Data_analysis.data_cleaning import DataDivision
from steps.ingest_data import ingestdata
import joblib
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

test_data = {
    'Title': '2 BHK Apartment for Sale in Sector 62',
    'Bathroom': 2,
    'Carpet Area': 850,
    'location': 'Noida, Sector 62',
    'Transaction': 'Resale',
    'Furnishing': 'Semi-Furnished',
    'Balcony': 2,
    'facing': 'East',
    'Price (in rupees)': 7500000,
    'Status': 'Ready to Move',
    'Society': 'Green Valley Residency',
    'Floor': '5 out of 10'
}

df = ingestdata(test_data)

label_encoder = LabelEncoder()

test_to_numeric = ['Title','location','Transaction','Furnishing','facing','Status','Floor','Society']

for x in test_to_numeric:
    df[x] = label_encoder.fit_transform(df[x])

model = joblib.load('./models/model.pkl')
print('Loaded')

# Make prediction
prediction = model.predict(df)
print(f"Predicted price: {prediction[0]:,.2f} rupees")