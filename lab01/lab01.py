import urllib.request
import tarfile
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle

# dwonloading and preparing data
urllib.request.urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz", "housing.tgz")

tar = tarfile.open("housing.tgz")
tar.extractall()
tar.close()

with open("housing.csv", 'rb') as f_in, gzip.open("housing.csv.gz", 'wb') as f_out:
    f_out.writelines(f_in)
df = pd.read_csv('housing.csv.gz')


# info about data
df.head()   # kilka pierwszych wierszy
df.info()   # informacje o kolumnach
df.value_counts('ocean_proximity')    # ile wierszy o danej wartości kolumny
df.ocean_proximity.describe()         # statystyki dla danej kolumny

# visualization
hist = df.hist(bins=50, figsize=(20,15))

for i in range(3):
  fig1 = hist[i][i].get_figure()
  fig1.savefig("obraz1.png")

plot2 = df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7,4))
fig2 = plot2.get_figure()
fig2.savefig("obraz2.png")

plot3 = df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(7,3), colorbar="True",
        s=df["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"))
fig3 = plot3.get_figure()
fig3.savefig("obraz3.png")


# correlation matrix
kor = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)
kor.reset_index(name="wspolczynnik_korelacji").rename(columns={'index': "atrybut"}).to_csv('kor.csv', index=False)
sns.pairplot(df)


# dividing into training and testing data
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(len(train_set))
print(len(test_set))
# print(train_test)
print()
# print(test_set)
print(train_set.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))
print(test_set.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))

with open('train_set.pkl', 'wb') as file:
  pickle.dump(train_set, file)
with open('test_set.pkl', 'wb') as file:
  pickle.dump(test_set, file)





