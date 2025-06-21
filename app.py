#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train_model.py
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Contoh data latih: [luas, kamar, lokasi]
X = np.array([
    [100, 3, 0],
    [150, 4, 1],
    [120, 3, 2],
    [80, 2, 0],
    [200, 5, 1],
    [130, 3, 1],
    [90, 2, 2]
])

# Harga dalam jutaan
y = np.array([500, 800, 700, 400, 1000, 750, 450])

# Latih model
model = LinearRegression()
model.fit(X, y)

# Simpan model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model harga rumah disimpan sebagai model.pkl")


# In[ ]:


# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        luas = float(data['luas'])
        kamar = int(data['kamar'])
        lokasi = int(data['lokasi'])  # 0 = pusat kota, 1 = pinggiran, 2 = desa
        fitur = np.array([[luas, kamar, lokasi]])
        prediksi = model.predict(fitur)[0]
        return jsonify({'prediction': f'{prediksi:.2f} juta'})
    except:
        return jsonify({'prediction': 'Input tidak valid'})

if __name__ == '__main__':
    app.run(debug=True)

