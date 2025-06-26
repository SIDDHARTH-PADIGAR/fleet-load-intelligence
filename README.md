# Vehicle Load Prediction using Engine Telemetry

A machine learning system to predict vehicle **load weight** and detect **overload conditions** using **engine telemetry data** and **baseline dynamo profiles** per vehicle.

---

## Features

-  Predict **vehicle weight** using regression
-  Classify **load status** as `Normal` or `Overload`
-  Analyze **engine strain** vs. baseline dynamo profile
-  Works with real-time or simulated telemetry
-  Handles extreme cases & gear/rpm mismatches
-  Easily extendable to fleets

---

## Input Data

The model uses a mix of **raw telemetry** and **derived features**:

| Feature         | Description                          |
|-----------------|--------------------------------------|
| `torque`        | Engine torque (Nm)                   |
| `rpm`           | Engine revolutions per minute        |
| `gear`          | Current gear                         |
| `speed`         | Vehicle speed (km/h)                 |
| `elevation`     | Road gradient (%)                    |
| `voltage`       | Battery voltage (V)                  |
| + engineered features | e.g., `stress_index`, `power_density`, etc.

---

## Model Architecture

The system is composed of two machine learning models, each optimized for a specific task:

---

### 1. Load Estimation Model  
**Algorithm:** `XGBoost Regressor`

| Metric                     | Value                        |
|----------------------------|------------------------------|
| Target                     | Estimated load (tonnes)      |
| Input Features             | 12 telemetry + derived features |
| Mean Absolute Error (MAE)  | **0.21 tonnes**              |
| R² Score                   | **0.994**                    |
| Inference Time             | ~3 ms per instance           |
| Notes                      | High accuracy, robust generalization |

**Top Feature Importance:**

| Feature Name              | Description                                  |
|---------------------------|----------------------------------------------|
| `torque`                  | Base engine output                           |
| `power_kw`                | Torque × RPM converted to kilowatts          |
| `stress_index`            | Torque load adjusted for elevation and speed |
| `rpm_pg`                  | RPM normalized by gear                       |
| `torque_elevation_ratio`  | Torque normalized by gradient                |

---

### 2. Load Status Classification Model  
**Algorithm:** `Random Forest Classifier`

| Metric                  | Value                            |
|-------------------------|----------------------------------|
| Target                  | Load status (`Normal` / `Overload`) |
| Accuracy                | **99%**                          |
| Precision (Overload)    | **1.00**                         |
| Recall (Overload)       | **0.17**                         |
| F1 Score (Overload)     | **0.29**                         |
| Notes                   | Needs better overload representation |

**Class Distribution (Training Data):**

| Class      | Samples |
|------------|---------|
| Normal     | 4994    |
| Overload   | 6       |

**Strengths:**
- Very high precision (zero false positives).
- Low variance due to ensemble method.
- Interpretable, easy to debug.

**Limitations:**
- Extremely low recall on `Overload` due to class imbalance.
- May benefit from SMOTE or adjusted classification threshold.

---

### Combined Logic Flow

1. The **regression model** estimates current vehicle load in tonnes.
2. The **classification model** uses stress indicators and gear dynamics to determine overload status.
3. A per-vehicle **baseline dynamo profile** is used to detect abnormal torque-per-tonne strain, flagging issues even if the classifier predicts "Normal".

---

## Baseline Dynamo Profiles

Each vehicle (e.g., `truck_001`) has an expected **torque/tonne ratio**, based on healthy performance.  
The model compares predicted load to current torque to detect **abnormal strain**.

Example:
```txt
Expected: 19.4 Nm/tonne
Actual:   28.9 Nm/tonne
Abnormal strain detected
````

---

## Sample Output (Full Prediction)

```txt
Vehicle: truck_001
Torque: 390 Nm | RPM: 1800 | Gear: 4 | Speed: 28 km/h | Elevation: 10.5%
Voltage: 27.2 V | Stress Index: 146.25

Model Prediction:
Load Status: Overload (62.0%)
Estimated Weight: 13.47 tonnes

Baseline Comparison:
Expected Torque/Tonne: 19.39
Actual Torque/Tonne: 28.95
Abnormal strain detected
```

---

## Stress Testing

Tested across edge cases:

* Extreme torque (800+ Nm)
* Gear-speed mismatches
* Borderline overload
* High weight + low stress
* Low torque on steep inclines

Result: **Model remained stable and explainable.**

---

## Model Saving

Trained models are saved using `joblib`:

```python
joblib.dump(classifier_model, "model_fleet_dynamo.pkl")
joblib.dump(regression_model, "regressor_fleet_dynamo.pkl")
```

---

##  Next Steps

* [ ] Balance training data (overload class)
* [ ] Add real vehicle dynamo data
* [ ] Build dashboard / alert system
* [ ] Integrate live telemetry API

---

## Built With

* Python 3.11
* XGBoost
* Scikit-learn
* Pandas / NumPy
* Matplotlib / Seaborn

---
