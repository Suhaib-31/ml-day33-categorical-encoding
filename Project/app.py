import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("===== Categorical Encoding =====\n")

# Load Dataset
df = pd.read_csv("employees.csv")

print("Original Dataset:\n")
print(df)

# ----------------------------
# Label Encoding
# ----------------------------

label_encoder = LabelEncoder()

df["Gender_Encoded"] = label_encoder.fit_transform(
    df["Gender"]
)

print("\nAfter Label Encoding:\n")
print(df)

# ----------------------------
# One-Hot Encoding
# ----------------------------

one_hot_df = pd.get_dummies(
    df,
    columns=["Department"]
)

print("\nAfter One-Hot Encoding:\n")
print(one_hot_df)

# Save Output
one_hot_df.to_csv(
    "encoded_data.csv",
    index=False
)

print("\nEncoding Completed Successfully!")