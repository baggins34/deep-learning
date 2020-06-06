import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

titanic = sns.load_dataset("titanic")  # Download Titanic dataset from the internet
print(titanic.info())
print(titanic.columns)
print(titanic.describe(include='all'))
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.columns)
print(test.columns, "The size: ", test.shape)


print(titanic[(titanic.sex == "female")
        & (titanic['class'].isin(['First', 'Third']))
        & (titanic.age > 30)
        & (titanic.survived == 0)])


towns_dic = {'name':['Southampton', 'Cherbourg', 'Queenstown', 'Montevideo'],
             'country': ['United Kingdom', 'France', 'United Kingdom', 'Uruguay'],
             'population': [236900, 37121, 12347, 1305000],
             'age': [np.random.randint(500, 1000) for _ in range(4)]}
towns_df = pd.DataFrame(towns_dic)
merge = titanic.merge(towns_df, how='left', left_on='embark_town', right_on='name', indicator=True,
              suffixes=('_passenger', '_city'))
print(merge.head())

sns.distplot(titanic.age.dropna())
plt.show()

g= sns.FacetGrid(titanic, row= 'survived', col='class')
g.map(sns.distplot, 'age')
plt.show()


sns.jointplot(data=titanic, x='age', y='fare', kind='reg', color='g')
plt.show()


sns.heatmap(titanic.corr(), annot=True, fmt=".1f")
plt.show()