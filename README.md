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

| Task           | Algorithm         | Notes                      |
|----------------|-------------------|----------------------------|
| Load Estimation | XGBoost Regressor | MAE ≈ 0.21 tonnes, R² ≈ 0.99 |
| Load Status     | Random Forest     | High precision, needs more overload samples |

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
