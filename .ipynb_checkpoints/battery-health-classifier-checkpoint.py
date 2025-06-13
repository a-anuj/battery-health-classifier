{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "5a027946-2ab2-4305-a40e-48be01281796",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"-1\"\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import streamlit as st\n",
    "from tensorflow.keras.layers import Normalization,Dense\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.losses import SparseCategoricalCrossentropy\n",
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "1b9987ef-079e-4214-90b0-b79781b7f423",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   charge_cycles charging_pattern  voltage  temperature  internal_resistance  \\\n",
      "0           1176        overnight     3.97         57.7                57.91   \n",
      "1            910             fast     3.55         46.7                77.31   \n",
      "2           1344             slow     3.45         47.1                41.68   \n",
      "3           1180             slow     3.48         34.5                36.18   \n",
      "4           1145             slow     4.28         43.7                45.11   \n",
      "\n",
      "   discharge_rate  age_months  capacity_percent      health_status  \n",
      "0            1.51          42              99.7           Critical  \n",
      "1            0.95          59              68.8  Needs Maintenance  \n",
      "2            2.02          20              69.2           Critical  \n",
      "3            1.43           9              74.0           Critical  \n",
      "4            2.08          17              95.0           Critical  \n"
     ]
    }
   ],
   "source": [
    "df_final = pd.read_csv(\"battery_health_dataset.csv\")\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "8ad8abc6-3f58-4fea-a977-ba55b37f4a61",
   "metadata": {},
   "outputs": [],
   "source": [
    "label_encoder = LabelEncoder()\n",
    "y_train = label_encoder.fit_transform(df[\"health_status\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "62592aef-9436-4cb7-a5be-e995b6a26cbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.get_dummies(df_final,columns=[\"charging_pattern\"])\n",
    "df = df.astype(int,errors=\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "bdd46607-3578-465b-912e-a3059818d979",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   charge_cycles  voltage  temperature  internal_resistance  discharge_rate  \\\n",
      "0           1176        3           57                   57               1   \n",
      "1            910        3           46                   77               0   \n",
      "2           1344        3           47                   41               2   \n",
      "3           1180        3           34                   36               1   \n",
      "4           1145        4           43                   45               2   \n",
      "\n",
      "   age_months  capacity_percent      health_status  charging_pattern_fast  \\\n",
      "0          42                99           Critical                      0   \n",
      "1          59                68  Needs Maintenance                      1   \n",
      "2          20                69           Critical                      0   \n",
      "3           9                74           Critical                      0   \n",
      "4          17                95           Critical                      0   \n",
      "\n",
      "   charging_pattern_overnight  charging_pattern_slow  \n",
      "0                           1                      0  \n",
      "1                           0                      0  \n",
      "2                           0                      1  \n",
      "3                           0                      1  \n",
      "4                           0                      1  \n"
     ]
    }
   ],
   "source": [
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "cbcdef02-3e07-4a4f-aec4-b63bb428bc9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     charge_cycles  voltage  temperature  internal_resistance  discharge_rate  \\\n",
      "0             1176        3           57                   57               1   \n",
      "1              910        3           46                   77               0   \n",
      "2             1344        3           47                   41               2   \n",
      "3             1180        3           34                   36               1   \n",
      "4             1145        4           43                   45               2   \n",
      "..             ...      ...          ...                  ...             ...   \n",
      "995            885        3           46                   37               0   \n",
      "996            773        3           55                   55               2   \n",
      "997            274        3           43                   23               2   \n",
      "998            581        3           45                   85               0   \n",
      "999            162        3           26                   86               0   \n",
      "\n",
      "     age_months  capacity_percent  charging_pattern_fast  \\\n",
      "0            42                99                      0   \n",
      "1            59                68                      1   \n",
      "2            20                69                      0   \n",
      "3             9                74                      0   \n",
      "4            17                95                      0   \n",
      "..          ...               ...                    ...   \n",
      "995          53                86                      1   \n",
      "996          30                55                      0   \n",
      "997          26                86                      1   \n",
      "998          16                52                      0   \n",
      "999          13                59                      0   \n",
      "\n",
      "     charging_pattern_overnight  charging_pattern_slow  \n",
      "0                             1                      0  \n",
      "1                             0                      0  \n",
      "2                             0                      1  \n",
      "3                             0                      1  \n",
      "4                             0                      1  \n",
      "..                          ...                    ...  \n",
      "995                           0                      0  \n",
      "996                           0                      1  \n",
      "997                           0                      0  \n",
      "998                           1                      0  \n",
      "999                           0                      1  \n",
      "\n",
      "[1000 rows x 10 columns]\n"
     ]
    }
   ],
   "source": [
    "x_train = df.drop([\"health_status\"],axis=1)\n",
    "print(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "bef3e654-4596-42a8-947b-06204bf8fb11",
   "metadata": {},
   "outputs": [],
   "source": [
    "norm_l = Normalization()\n",
    "norm_l.adapt(x_train.values)\n",
    "x_train_norm = norm_l(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "2cd4d95a-fd88-4422-ac27-e355192424e9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tf.Tensor(\n",
      "[[ 0.89376193 -0.5465358   1.5210199  ... -0.71294856  1.4152749\n",
      "  -0.70181006]\n",
      " [ 0.25736344 -0.5465358   0.56566    ...  1.4026257  -0.7065765\n",
      "  -0.70181006]\n",
      " [ 1.2956978  -0.5465358   0.6525109  ... -0.71294856 -0.7065765\n",
      "   1.424887  ]\n",
      " ...\n",
      " [-1.264251   -0.5465358   0.30510727 ...  1.4026257  -0.7065765\n",
      "  -0.70181006]\n",
      " [-0.529761   -0.5465358   0.4788091  ... -0.71294856  1.4152749\n",
      "  -0.70181006]\n",
      " [-1.5322082  -0.5465358  -1.1713581  ... -0.71294856 -0.7065765\n",
      "   1.424887  ]], shape=(1000, 10), dtype=float32)\n"
     ]
    }
   ],
   "source": [
    "print(x_train_norm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "73aebf4f-99f2-40b5-8c85-c4a5b602ef07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 2ms/step - loss: 1.1628   \n",
      "Epoch 2/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 2ms/step - loss: 0.8999 \n",
      "Epoch 3/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - loss: 0.7457 \n",
      "Epoch 4/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 2ms/step - loss: 0.6682 \n",
      "Epoch 5/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - loss: 0.5903 \n",
      "Epoch 6/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - loss: 0.4927 \n",
      "Epoch 7/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - loss: 0.4682 \n",
      "Epoch 8/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 2ms/step - loss: 0.4186 \n",
      "Epoch 9/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - loss: 0.3836 \n",
      "Epoch 10/10\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - loss: 0.3307 \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x796651623440>"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = Sequential(\n",
    "    [\n",
    "        Dense(units=64,activation=\"relu\"),\n",
    "        Dense(units=32,activation=\"relu\"),\n",
    "        Dense(units=4,activation=\"linear\")\n",
    "    ]\n",
    ")\n",
    "\n",
    "model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))\n",
    "model.fit(x_train_norm,y_train,epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "6a3ebbbb-d785-48aa-be50-0d4631722583",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 37ms/step\n",
      "[ 1.3995166  -2.182235   -0.68738425  2.6996267 ] , Category : 3\n"
     ]
    }
   ],
   "source": [
    "x_test = pd.DataFrame([[673, 3.1, 39.9, 64.21, 0.66, 43, 64.6, 1, 0, 0]],\n",
    "                      columns=['charge_cycles', 'voltage', 'temperature', 'internal_resistance',\n",
    "                               'discharge_rate', 'age_months', 'capacity_percent',\n",
    "                               'charging_pattern_fast', 'charging_pattern_slow', 'charging_pattern_overnight'])\n",
    "x_test = norm_l(x_test.values)\n",
    "prediction_test = model.predict(x_test)\n",
    "for i in range(len(prediction_test)):\n",
    "    print(f\"{prediction_test[i]} , Category : {np.argmax(prediction_test[i])}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "51b23e79-3eaa-4a51-9683-07f4ce6a6ea6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 2ms/step\n",
      "[0. 3. 0. 0. 0. 0. 3. 2. 0. 3. 0. 0. 0. 3. 3. 0. 3. 2. 0. 0. 3. 0. 0. 3.\n",
      " 0. 0. 0. 3. 2. 3. 0. 2. 0. 2. 3. 0. 2. 0. 3. 0. 3. 0. 0. 0. 3. 2. 3. 0.\n",
      " 0. 0. 3. 3. 2. 2. 0. 0. 3. 0. 0. 0. 0. 0. 3. 3. 0. 2. 0. 0. 0. 0. 0. 0.\n",
      " 0. 3. 3. 3. 2. 3. 0. 3. 3. 0. 2. 0. 0. 2. 3. 2. 0. 0. 3. 3. 3. 0. 0. 3.\n",
      " 0. 0. 0. 1. 0. 0. 0. 3. 0. 0. 3. 0. 3. 3. 3. 0. 0. 2. 0. 3. 0. 3. 0. 0.\n",
      " 0. 0. 2. 3. 0. 3. 0. 0. 3. 2. 0. 0. 2. 0. 2. 3. 0. 3. 2. 0. 0. 0. 0. 0.\n",
      " 0. 0. 0. 3. 0. 2. 0. 0. 0. 0. 3. 0. 0. 0. 0. 3. 0. 0. 3. 0. 0. 0. 0. 0.\n",
      " 0. 2. 0. 0. 0. 0. 3. 3. 0. 0. 0. 3. 0. 0. 3. 0. 0. 0. 0. 0. 0. 0. 3. 3.\n",
      " 0. 0. 0. 0. 0. 0. 3. 3. 3. 0. 3. 3. 0. 1. 0. 0. 0. 3. 0. 0. 3. 0. 0. 0.\n",
      " 2. 3. 2. 3. 3. 0. 0. 3. 0. 0. 0. 0. 3. 0. 3. 0. 3. 3. 0. 0. 0. 0. 0. 0.\n",
      " 2. 0. 3. 3. 0. 0. 0. 3. 0. 0. 0. 0. 0. 3. 0. 3. 1. 3. 0. 0. 2. 0. 0. 0.\n",
      " 3. 0. 0. 3. 3. 0. 0. 3. 0. 0. 3. 0. 0. 0. 3. 3. 0. 0. 3. 3. 2. 0. 0. 0.\n",
      " 3. 0. 0. 3. 3. 3. 0. 0. 0. 0. 0. 0. 3. 0. 3. 2. 0. 0. 0. 0. 3. 1. 1. 0.\n",
      " 0. 3. 0. 0. 0. 0. 3. 1. 3. 0. 2. 0. 0. 3. 0. 0. 0. 3. 0. 0. 1. 0. 0. 0.\n",
      " 0. 2. 0. 0. 0. 0. 3. 0. 3. 0. 3. 0. 0. 3. 0. 0. 3. 0. 0. 0. 2. 1. 0. 3.\n",
      " 0. 3. 3. 0. 3. 3. 0. 3. 0. 0. 0. 0. 2. 0. 0. 3. 0. 0. 2. 0. 3. 3. 3. 3.\n",
      " 0. 3. 3. 3. 3. 0. 3. 0. 0. 0. 0. 3. 0. 2. 3. 0. 0. 0. 3. 3. 0. 1. 3. 0.\n",
      " 0. 0. 3. 3. 0. 3. 0. 0. 3. 0. 0. 3. 3. 0. 0. 0. 3. 3. 3. 3. 3. 2. 3. 0.\n",
      " 0. 3. 2. 3. 0. 0. 3. 2. 0. 0. 0. 0. 0. 3. 0. 0. 0. 0. 3. 0. 0. 0. 0. 0.\n",
      " 0. 3. 3. 3. 2. 0. 0. 0. 0. 3. 0. 0. 0. 0. 0. 1. 0. 0. 3. 0. 3. 0. 3. 0.\n",
      " 3. 3. 0. 0. 0. 3. 3. 0. 3. 0. 0. 0. 0. 3. 0. 0. 2. 0. 3. 0. 0. 3. 0. 2.\n",
      " 3. 0. 3. 0. 3. 0. 0. 0. 0. 3. 3. 3. 0. 0. 2. 0. 2. 0. 3. 3. 0. 3. 0. 0.\n",
      " 3. 2. 3. 0. 3. 0. 2. 0. 0. 3. 0. 0. 0. 3. 0. 2. 3. 3. 3. 0. 0. 0. 2. 3.\n",
      " 0. 2. 0. 0. 3. 0. 0. 0. 0. 3. 3. 0. 3. 0. 0. 1. 2. 3. 0. 1. 0. 3. 3. 0.\n",
      " 0. 0. 0. 1. 0. 0. 0. 3. 0. 3. 3. 0. 0. 3. 0. 1. 0. 0. 0. 3. 0. 3. 0. 0.\n",
      " 0. 0. 3. 3. 3. 3. 0. 0. 0. 3. 0. 0. 0. 0. 0. 0. 0. 0. 0. 3. 0. 0. 0. 0.\n",
      " 2. 0. 0. 0. 2. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 3. 0.\n",
      " 0. 2. 0. 0. 0. 0. 0. 3. 2. 0. 2. 0. 0. 0. 0. 3. 3. 0. 3. 3. 3. 3. 0. 3.\n",
      " 0. 3. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 3. 0. 0.\n",
      " 2. 0. 0. 3. 0. 0. 0. 0. 0. 0. 3. 0. 0. 3. 3. 3. 3. 0. 3. 3. 0. 0. 3. 3.\n",
      " 3. 3. 0. 0. 0. 2. 0. 2. 3. 3. 0. 0. 3. 0. 0. 3. 0. 0. 0. 3. 0. 1. 0. 0.\n",
      " 2. 3. 0. 3. 0. 3. 3. 3. 3. 0. 3. 0. 3. 3. 2. 3. 3. 0. 0. 0. 3. 3. 1. 0.\n",
      " 0. 0. 0. 3. 3. 0. 3. 0. 0. 3. 3. 0. 1. 0. 2. 3. 3. 0. 0. 3. 0. 0. 0. 0.\n",
      " 2. 0. 0. 3. 3. 3. 0. 3. 0. 0. 0. 3. 3. 3. 0. 0. 0. 3. 3. 0. 2. 3. 0. 3.\n",
      " 0. 3. 0. 3. 3. 0. 0. 0. 3. 3. 0. 0. 0. 3. 3. 1. 3. 0. 0. 0. 0. 1. 0. 1.\n",
      " 3. 2. 0. 3. 0. 0. 3. 3. 0. 3. 3. 0. 3. 0. 0. 0. 2. 0. 0. 3. 0. 0. 0. 3.\n",
      " 0. 3. 0. 0. 3. 3. 0. 0. 0. 3. 0. 0. 3. 3. 3. 0. 3. 2. 0. 2. 3. 3. 3. 0.\n",
      " 0. 3. 0. 3. 3. 0. 3. 1. 0. 3. 3. 3. 3. 2. 1. 0. 0. 2. 3. 2. 0. 0. 3. 0.\n",
      " 0. 3. 0. 0. 0. 0. 0. 0. 0. 3. 2. 3. 0. 0. 0. 2. 0. 0. 3. 3. 0. 3. 0. 1.\n",
      " 2. 3. 0. 1. 1. 3. 1. 0. 0. 3. 3. 2. 0. 3. 0. 0. 0. 0. 0. 0. 3. 2. 0. 0.\n",
      " 0. 3. 0. 3. 0. 3. 0. 0. 2. 0. 1. 0. 0. 0. 3. 0. 3. 0. 1. 2. 0. 3. 3. 0.\n",
      " 0. 2. 3. 3. 0. 0. 3. 0. 3. 0. 0. 3. 0. 1. 0. 3.]\n",
      "[0 3 0 0 0 0 3 2 0 3 0 0 0 3 3 0 3 2 0 0 3 0 0 3 0 0 0 3 2 3 3 2 0 2 3 0 2\n",
      " 0 3 0 3 0 3 0 3 2 3 0 0 0 3 3 2 2 0 0 3 3 0 0 0 0 3 3 0 2 0 0 0 0 0 0 0 3\n",
      " 3 3 1 3 0 3 3 0 2 0 0 2 3 2 0 0 3 3 3 0 0 2 0 0 0 1 0 0 0 3 0 0 3 0 2 3 3\n",
      " 0 0 2 0 3 0 3 0 0 0 0 2 3 0 3 3 0 3 3 0 0 2 0 2 3 0 3 1 0 0 0 3 0 0 0 0 2\n",
      " 0 1 0 0 0 0 2 0 0 0 0 3 0 0 3 0 0 0 0 0 0 2 0 0 0 0 3 3 0 0 0 3 0 0 2 0 0\n",
      " 0 0 0 0 0 3 3 0 0 0 0 0 0 3 3 3 0 3 3 0 2 0 0 0 3 0 0 3 0 0 0 2 3 2 3 3 0\n",
      " 0 3 0 0 0 0 3 0 2 0 3 3 0 3 0 0 0 0 2 3 3 2 3 0 3 3 0 0 0 0 0 3 0 3 1 3 0\n",
      " 0 2 0 0 0 3 0 0 3 2 0 0 0 0 0 3 0 0 0 3 0 0 0 3 3 1 0 0 0 3 0 0 2 3 2 0 0\n",
      " 0 0 0 0 3 0 3 2 0 0 0 0 3 1 1 0 0 0 0 0 0 0 3 2 3 0 2 0 3 3 0 0 0 3 0 0 1\n",
      " 0 0 0 0 1 0 0 0 0 2 0 3 0 3 0 0 2 0 0 3 0 0 0 1 1 0 3 0 3 3 0 3 0 0 3 0 3\n",
      " 0 0 2 0 0 3 0 0 2 0 3 3 2 3 0 3 2 3 3 0 2 0 0 0 3 3 0 1 3 0 0 0 2 3 0 2 3\n",
      " 0 0 0 3 3 0 3 0 0 3 0 0 3 3 0 3 0 3 3 3 3 3 2 3 3 0 3 2 3 0 0 3 2 3 0 0 0\n",
      " 0 3 0 0 0 0 3 0 0 0 0 0 3 0 3 3 1 0 0 0 0 3 0 0 0 0 0 1 0 0 3 0 3 0 3 0 3\n",
      " 3 0 0 0 2 2 0 3 0 0 0 0 3 0 0 2 0 3 0 0 3 3 2 3 3 2 0 3 0 0 0 0 3 3 3 0 0\n",
      " 2 0 1 0 3 3 3 3 0 0 3 1 3 0 3 0 2 0 0 3 0 0 3 3 0 2 3 3 3 0 0 0 2 3 0 2 0\n",
      " 0 2 3 0 0 0 3 3 0 3 0 0 1 2 3 0 1 0 2 3 0 0 3 0 1 3 0 0 2 0 3 3 0 0 3 0 2\n",
      " 0 0 0 2 0 3 0 0 0 0 3 3 3 3 0 0 0 3 0 0 0 0 0 0 0 0 0 3 0 0 0 0 2 0 0 0 2\n",
      " 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 3 3 0 0 2 0 0 0 0 0 3 2 0 2 0 0 0 0 3 3 0\n",
      " 3 3 3 3 0 3 0 3 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 2 0 0 3 0 0 0\n",
      " 0 0 0 3 0 0 3 3 3 3 0 3 0 0 0 3 3 3 3 0 0 0 2 3 2 3 0 0 0 3 0 0 3 0 0 0 3\n",
      " 0 1 0 0 2 3 0 3 0 3 3 3 3 0 2 0 3 3 2 3 3 3 0 0 3 3 1 0 0 0 0 0 3 3 3 0 3\n",
      " 2 3 0 2 0 1 3 3 0 0 3 0 0 0 0 2 0 3 2 3 3 0 3 0 0 0 3 3 2 0 0 0 3 3 0 2 3\n",
      " 0 3 3 2 0 3 3 0 0 0 2 3 0 0 0 3 3 1 3 0 0 0 0 1 0 1 3 2 0 3 0 0 3 3 0 3 3\n",
      " 0 2 0 0 0 1 0 0 3 0 0 0 3 0 2 0 3 3 3 0 0 0 2 0 0 3 3 0 0 3 1 0 2 0 3 3 0\n",
      " 0 3 0 3 3 0 3 2 0 3 3 2 3 2 1 0 0 2 3 2 3 0 3 0 0 3 3 0 0 0 0 0 0 3 1 3 0\n",
      " 0 0 1 0 0 2 3 0 0 0 1 2 3 0 1 1 0 1 0 0 3 3 2 0 3 0 0 0 3 0 0 3 2 0 0 0 3\n",
      " 0 3 0 3 0 0 2 0 1 0 3 0 3 0 2 0 1 3 0 3 3 3 0 2 3 3 0 0 3 0 3 0 0 3 3 1 0\n",
      " 3]\n"
     ]
    }
   ],
   "source": [
    "pred = model.predict(x_train_norm)\n",
    "pred_pr = np.zeros(len(pred))\n",
    "for i in range(len(pred)):\n",
    "    pred_pr[i] = np.argmax(pred[i])\n",
    "print(pred_pr)\n",
    "print(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "4b644374-f2a1-43a4-9a9f-f40a5b636eca",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "acc = accuracy_score(y_train, pred_pr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "585a8c78-2e9b-433a-ad19-0d9a6401e69d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.892\n"
     ]
    }
   ],
   "source": [
    "print(acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "id": "2a1e4dc2-7033-4f7d-9636-20894e0ad14a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-13 18:43:59.424 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-13 18:43:59.502 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /home/anuj/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-06-13 18:43:59.502 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title(\"🔋 Battery Health Prediction App\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "69412df2-2b51-423f-9644-f164858bdd2d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
