from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Data Pelatihan 
data = {
    'earliest_cr_yr': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
    'installment': [250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
    'loan_amnt': [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    'emp_length': [5, 10, 15, 20, 25, 0, 1, 2, 3, 4],
    'loan_status': ['good', 'risky', 'good', 'risky', 'good', 'risky', 'risky', 'risky', 'risky', 'risky']
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Mengonversi label menjadi biner
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x == 'good' else 0)

# Memisahkan fitur dan label
X = df[['earliest_cr_yr', 'installment', 'loan_amnt', 'emp_length']]
y = df['loan_status']

# Membagi data menjadi data pelatihan dan data validasi
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=14)

# Melatih model Random Forest
rf = RandomForestClassifier(random_state=14)
rf.fit(train_X, train_y)

# Menyimpan model ke dalam file
model_filename = 'rf_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rf, file)

# Memuat model yang telah dilatih
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Mendapatkan data dari form
        data = request.form
        earliest_cr_yr = int(data['earliest_cr_yr'])
        installment = float(data['installment'])
        loan_amnt = float(data['loan_amnt'])
        emp_length = int(data['emp_length'])
        
        # Ekstraksi fitur dari data
        features = [earliest_cr_yr, installment, loan_amnt, emp_length]
        
        # Konversi fitur menjadi array numpy
        features_array = np.array(features).reshape(1, -1)
        
        # Melakukan prediksi
        prediction = model.predict(features_array)
        prediction_proba = model.predict_proba(features_array)
        
        # Mengkonversi prediksi menjadi label
        output = 'good' if prediction[0] == 1 else 'risky'
        proba = prediction_proba[0][prediction[0]]
        
        # Mengembalikan hasil prediksi dalam bentuk JSON
        return render_template('predict.html', prediction_text='Loan is {} with probability {:.2f}%'.format(output, proba * 100))
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
