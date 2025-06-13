#!/usr/bin/env python
# coding: utf-8

# In[148]:


import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.layers import Normalization,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score




df_final = pd.read_csv("battery_health_dataset.csv")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_final["health_status"])

df = pd.get_dummies(df_final,columns=["charging_pattern"])
df = df.astype(int,errors="ignore")
print(df.head())
x_train = df.drop(["health_status"],axis=1)
print(x_train)

norm_l = Normalization()
norm_l.adapt(x_train.values)
x_train_norm = norm_l(x_train)


model = Sequential(
    [
        Dense(units=64,activation="relu"),
        Dense(units=32,activation="relu"),
        Dense(units=4,activation="linear")
    ]
)

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train_norm,y_train,epochs=10)

st.title("🔋 Battery Health Prediction App")


col1, spacer, col2 = st.columns([4, 0.4, 5])

with col1:  # 🧾 Form goes left
    with st.form("battery_form"):
        charge_cycles = st.number_input("Charge Cycles", min_value=0)
        voltage = st.number_input("Voltage (V)", format="%.2f")
        temperature = st.number_input("Temperature (°C)", format="%.2f")
        internal_resistance = st.number_input("Internal Resistance (Ohms)", format="%.2f")
        discharge_rate = st.number_input("Discharge Rate (C)", format="%.2f")
        age_months = st.number_input("Age (Months)", min_value=0)
        capacity_percent = st.number_input("Capacity (%)", min_value=0.0, max_value=100.0, format="%.1f")
        charging_pattern = st.selectbox("Charging Pattern", ["Fast", "Slow", "Overnight"])
        submit = st.form_submit_button("Submit")

if submit:
    with col2:  # ✅ Results go right
        st.success("✅ Submitted successfully!")

        # One-hot encode manually
        if charging_pattern == "Fast":
            cpf, cps, cpo = 1, 0, 0
        elif charging_pattern == "Slow":
            cpf, cps, cpo = 0, 1, 0
        else:
            cpf, cps, cpo = 0, 0, 1

        x_test = pd.DataFrame([[charge_cycles, voltage, temperature, internal_resistance, discharge_rate,
                                age_months, capacity_percent, cpf, cps, cpo]],
                              columns=['charge_cycles', 'voltage', 'temperature', 'internal_resistance',
                                       'discharge_rate', 'age_months', 'capacity_percent',
                                       'charging_pattern_fast', 'charging_pattern_slow', 'charging_pattern_overnight'])

        # Normalize and predict
        x_test = norm_l(x_test.values)
        prediction_test = model.predict(x_test)
        category = np.argmax(prediction_test[0])
        st.subheader("🔍 Prediction")
        if category == 0:
            cat = "The state of the battery is critical. Immediate attention is required!"
            st.error(f"Predicted Health Category: **{cat}**")
        elif category == 1:
            cat = "Battery is looking good"
            st.success(f"Predicted Health Category: **{cat}**")
        elif category == 2:
            cat = "Condition of the battery is moderate."
            st.info(f"Predicted Health Category: **{cat}**")
        elif category == 3:
            cat = "The Battery needs maintenance..."
            st.warning(f"Predicted Health Category: **{cat}**")

        # Training accuracy
        pred_train = model.predict(x_train_norm)
        pred_labels = np.argmax(pred_train, axis=1)
        acc = accuracy_score(y_train, pred_labels)
        st.subheader("📊 Model Accuracy")
        st.write(f"Accuracy on training : **{acc * 100:.2f}%**")

        


