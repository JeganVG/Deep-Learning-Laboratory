import torch
import pandas as pd

# Custom function to read the data from a file
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Custom function to preprocess the data
def preprocess_data(df):
    # Perform any necessary preprocessing steps, such as feature scaling, encoding categorical variables, etc.
    return df

# Load the dataset using the custom functions
file_path = "D:\\MEPCO\\SEMESTER 6\\Deep Learning\\Datasets\\Iris.csv"
df = read_data(file_path)
df = preprocess_data(df)

# Convert the pandas dataframe to a PyTorch tensor
X = torch.tensor(df.drop("Species", axis=1).values, dtype=torch.float32)
# from sklearn.preprocessing import LabelEncoder as lb
# print(df['Species'])
# df["Species"] = lb.fit_transform(df["Species"].values)
y = torch.tensor(df["Species"].values, dtype=torch.long)
print(y)
# Use the preprocessed data for further analysis or modeling
