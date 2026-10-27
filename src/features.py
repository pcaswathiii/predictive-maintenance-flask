import numpy as np

def add_features(df):
    """Create the same derived features used during model training."""
    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    df["torque_wear"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    return df