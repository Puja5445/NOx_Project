import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Load data
df = pd.read_excel("Data/Furnace7_NOx_data.xlsx")

# Features
X = df[
    [
        "Fuel_gas_flow_to_upper_Burner",
        "Bridgewall_Temp_Avg",
        "Adiabatic Flame Temperature",
    ]
]

# Target
y = df["Furnace7_Nox_In_Ng_Per_J"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVR(kernel="rbf")
model.fit(X_train, y_train)


# Function for API
def predict_nox(fuel, temp, aft):
    input_data = pd.DataFrame(
        [
            {
                "Fuel_gas_flow_to_upper_Burner": fuel,
                "Bridgewall_Temp_Avg": temp,
                "Adiabatic Flame Temperature": aft,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    return prediction