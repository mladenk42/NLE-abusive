import pandas as pd
import json
file_name = '/import/cogsci/ravi/datasets/Embeddia/STY_24sata_comments_hr_001.csv'

# file_name = '../../../data/unsup/24h/classify/cro_test.csv'

df = pd.read_csv(file_name)
df['length'] = df.content.str.len()

data = {}
data['max'] = int(df["length"].max())
data['min'] = int(df["length"].min())
data['mean'] = int(df["length"].mean())
data['mode'] = int(df["length"].mode())
data['median'] = int(df["length"].median())

gr_mean = len(df[df['length'] > data['mean']])
gr_mean2 = len(df[df['length'] > 2*data['mean']  ])
gr_mean1 = len(df[df['length'] > 1.5*data['mean']  ])

print(gr_mean)
print(type(gr_mean))
# print(gr_mean.values,type(gr_mean.values))


data['gr_mean'] = gr_mean
data['gr_mean1'] = gr_mean1
data['gr_mean2'] = gr_mean2

# print(df["length"].max(), df["length"].min(), df["length"].mean(),df["length"].mode(), df["length"].median())
# print(df['length'])

with open('dataLen.json','w') as fid:
    json.dump(data,fid)

print(data)
# print(df[df['length']==967].content.values)