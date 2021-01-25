import pandas as pd
from sklearn.model_selection import train_test_split

lsbgs = 'random_LSBGs_all.csv'
artifacts_2 = 'random_negative_all_2.csv'

# load csv
df_lsbgs = pd.read_csv(lsbgs)
df_artifacts_2 = pd.read_csv(artifacts_2)
cols = ['ra', 'dec']

# drop unwanted cols
df_lsbgs = df_lsbgs[cols]
df_artifacts_2 = df_artifacts_2[cols]

# set labels
df_lsbgs['label'] = 1
df_artifacts_2['label'] = 0

# concatenate and shuffle final dataframe
df = pd.concat([df_lsbgs, df_artifacts_2])
df = df.sample(frac=1).reset_index(drop=True)

# splt train, validation and test sets according to Tanoglidis
df_train, df_other = train_test_split(df, test_size=0.25)
df_val, df_test = train_test_split(df_other, test_size=0.5)

print('train:\n', df_train['label'].value_counts(), '\n')
print('val:\n', df_val['label'].value_counts(), '\n')
print('test:\n', df_test['label'].value_counts(), '\n')

df_train.to_csv('train.csv')
df_val.to_csv('val.csv')
df_test.to_csv('test.csv')
