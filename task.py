import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns


def load_jm1_dataset(path):
    data, meta = arff.loadarff(path)

    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")

    return df

file_path = "jm1.arff"

df = load_jm1_dataset(file_path)


print("Перші 5 рядків набору даних:")
print(df.head())

print("\nІнформація про набір даних:")
print(df.info())

print("\nКількість пропусків:")
print(df.isnull().sum())


# перетворення у числовий формат
df["defects"] = df["defects"].map({"true": 1, "false": 0})

# обчислення кореляцій
corr_matrix = df.corr(numeric_only=True)

# кореляція ознак із defects
target_corr = corr_matrix["defects"].sort_values(ascending=False)

print("\nКореляція ознак із defects:")
print(target_corr)

top_features = target_corr[1:11]

plt.figure(figsize=(10, 6))

sns.barplot(x=top_features.values,y=top_features.index)

plt.title("Найбільш впливові метрики на defects")
plt.xlabel("Кореляція")
plt.ylabel("Метрики")

plt.tight_layout()
plt.show()

# теплова мапа
plt.figure(figsize=(14, 10))

sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Матриця кореляцій")

plt.tight_layout()
plt.show()
