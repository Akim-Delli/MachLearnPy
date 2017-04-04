import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])

df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
df

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

print(X)

ohe = OneHotEncoder(categorical_features=[0])
oheArr = ohe.fit_transform(X).toarray()
print(oheArr)

dummy = pd.get_dummies(df[['price', 'color', 'size']])
print(dummy)
