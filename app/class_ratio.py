import pandas as pd

df = pd.read_csv('../dataset2/train.csv')
print(df['status'].value_counts())