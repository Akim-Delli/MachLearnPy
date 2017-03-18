import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())

# accessing the underlying Numpy Array of the DataFrame via values
print(df.values)

# drop rows with missing values
print(df.dropna())

# drop columns with missing values
print(df.dropna(axis=1))

# only drop rows where all columns are NaN
print(df.dropna(how='all'))

# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific colums (here: 'C')
print(df.dropna(subset=['C']))
