# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/EstudioCrediticio_TrainP.csv')

# Configurar opciones de visualización
sns.set(style="whitegrid")
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Verificar los tipos de datos de cada columna
print("Tipos de datos del DataFrame:")
print(df.dtypes)

# Mostrar las primeras filas del DataFrame
print("\nPrimeras filas del DataFrame:")
print(df.head())

# Descripción estadística de las variables numéricas
print("\nDescripción estadística de las variables numéricas:")
print(df.describe())

# Lista de variables categóricas y numéricas
variables_categoricas = ['SituacionLaboral', 'NivelEducativo', 'EstadoCivil', 'EstadoVivienda', 'ObjetoCredito']
variables_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
variables_numericas = [var for var in variables_numericas if var not in ['Id']]

# Análisis de valores faltantes
print("\nPorcentaje de valores faltantes por columna:")
print((df.isnull().mean() * 100).sort_values(ascending=False))

# Visualización de valores faltantes
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Mapa de calor de valores faltantes')
plt.show()

# Imputación de valores faltantes
# Para variables numéricas, imputamos con la mediana
for var in variables_numericas:
    if df[var].isnull().sum() > 0:
        median_value = df[var].median()
        df[var].fillna(median_value, inplace=True)
        print(f"Valores faltantes en '{var}' imputados con la mediana ({median_value}).")

# Para variables categóricas, imputamos con la moda
for var in variables_categoricas:
    if df[var].isnull().sum() > 0:
        mode_value = df[var].mode()[0]
        df[var].fillna(mode_value, inplace=True)
        print(f"Valores faltantes en '{var}' imputados con la moda ('{mode_value}').")

# Verificar que no hay valores faltantes
print("\nValores faltantes después de la imputación:")
print(df.isnull().sum())

# Análisis univariado de variables numéricas
for var in variables_numericas:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[var], kde=True, bins=30)
    plt.title(f'Distribución de {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.show()

# Análisis univariado de variables categóricas
for var in variables_categoricas:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=var, data=df, order=df[var].value_counts().index)
    plt.title(f'Conteo de {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()

# Análisis bivariado: relación con la variable objetivo 'CreditoAprobado'
# Para variables numéricas
for var in variables_numericas:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='CreditoAprobado', y=var, data=df)
    plt.title(f'{var} vs CreditoAprobado')
    plt.xlabel('CreditoAprobado')
    plt.ylabel(var)
    plt.show()

# Para variables categóricas
for var in variables_categoricas:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=var, hue='CreditoAprobado', data=df, order=df[var].value_counts().index)
    plt.title(f'{var} vs CreditoAprobado')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()

# Matriz de correlación de variables numéricas
plt.figure(figsize=(12, 10))
corr = df[variables_numericas].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de correlación entre variables numéricas')
plt.show()

# Identificar variables altamente correlacionadas (multicolinealidad)
umbral = 0.8
variables_correlacionadas = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > umbral:
            colname_i = corr.columns[i]
            colname_j = corr.columns[j]
            variables_correlacionadas.add((colname_i, colname_j))
print("\nPares de variables altamente correlacionadas (correlación > 0.8):")
for pair in variables_correlacionadas:
    print(pair)

# Detección de outliers utilizando el método IQR
for var in variables_numericas:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[var] < limite_inferior) | (df[var] > limite_superior)]
    porcentaje_outliers = (len(outliers) / len(df)) * 100
    print(f"\nVariable '{var}': {len(outliers)} outliers detectados ({porcentaje_outliers:.2f}%)")
    # Visualización de outliers
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[var])
    plt.title(f'Outliers en {var}')
    plt.xlabel(var)
    plt.show()

# Análisis de la variable objetivo 'CreditoAprobado'
print("\nDistribución de la variable objetivo 'CreditoAprobado':")
print(df['CreditoAprobado'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='CreditoAprobado', data=df)
plt.title('Distribución de CreditoAprobado')
plt.xlabel('CreditoAprobado')
plt.ylabel('Frecuencia')
plt.show()

# Si la variable 'ScoreRiesgo' es continua, analizamos su distribución
plt.figure(figsize=(10, 4))
sns.histplot(df['ScoreRiesgo'], kde=True, bins=30)
plt.title('Distribución de ScoreRiesgo')
plt.xlabel('ScoreRiesgo')
plt.ylabel('Frecuencia')
plt.show()