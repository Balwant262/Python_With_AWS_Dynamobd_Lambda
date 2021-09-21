import pandas as pd
df = pd.read_csv('Indian-Male-Names.csv')
print(df)
words_list = df['name'].tolist()
print(words_list)