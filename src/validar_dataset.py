import pandas as pd

df = pd.read_csv("../data/processed/dataset_satisfaccion_cliente.csv")

print("Dimensiones:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)

print("\nPorcentaje de nulos:")
print(df.isnull().mean().sort_values(ascending=False).head(15))

print("\nPorcentaje de duplicados:")
print(df.duplicated().mean())

print("\nBalance de clases:")
print(df["satisfaccion_alta"].value_counts(normalize=True))

print("\nCorrelaciones con satisfacción:")
print(
    df.corr(numeric_only=True)["satisfaccion_alta"]
    .sort_values(ascending=False)
    .head(15)
)