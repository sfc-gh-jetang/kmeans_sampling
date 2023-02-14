import pandas as pd

features_csv = pd.read_csv('features_list.csv')
res = []
for feature in features_csv['col_name']:
    res.append(feature)

print(res)