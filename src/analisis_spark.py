import os

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    Imputer
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    ClusteringEvaluator,
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)


# ============================================================
# 1. Crear SparkSession
# ============================================================

spark = SparkSession.builder \
    .appName("Proyecto_Satisfaccion_Cliente_Spark_MLlib") \
    .master("local[*]") \
    .config("spark.sql.debug.maxToStringFields", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# 2. Cargar dataset
# ============================================================

df = spark.read.csv(
    "../data/processed/dataset_satisfaccion_cliente.csv",
    header=True,
    inferSchema=True
)

print("\n=== ESQUEMA DEL DATASET ===")
df.printSchema()

print("\n=== DIMENSIONES INICIALES ===")
print("Filas:", df.count())
print("Columnas:", len(df.columns))


# ============================================================
# 3. Limpieza básica
# ============================================================

df = df.dropDuplicates()

print("\n=== DIMENSIONES DESPUÉS DE ELIMINAR DUPLICADOS ===")
print("Filas:", df.count())
print("Columnas:", len(df.columns))


# ============================================================
# 4. Definir variables
# ============================================================

label_col = "satisfaccion_alta"

numeric_cols = [
    "edad",
    "antiguedad_meses",
    "nivel_digitalizacion",
    "frecuencia_uso_mensual",
    "sesiones_app_mensual",
    "llamadas_callcenter_3m",
    "reclamos_6m",
    "tickets_abiertos_6m",
    "tickets_resueltos_6m",
    "cancelaciones_previas",
    "cambios_plan_12m",
    "uso_chatbot_mensual",
    "visitas_web_mensual",
    "tiempo_espera_promedio_min",
    "tiempo_resolucion_promedio_h",
    "numero_escalamientos",
    "cumplimiento_sla",
    "disponibilidad_servicio_pct",
    "interrupciones_servicio_3m",
    "calidad_atencion_score",
    "tiempo_primera_respuesta_min",
    "resolucion_primer_contacto",
    "valor_plan_mensual",
    "pagos_atrasados_6m",
    "monto_total_pagado_12m",
    "descuentos_recibidos_12m",
    "cargos_extra_6m",
    "saldo_pendiente",
    "consumo_servicio_promedio",
    "mes_registro",
    "trimestre",
    "es_fin_de_semana",
    "temporada_alta",
    "meses_desde_ultimo_reclamo",
    "tasa_resolucion_tickets",
    "reclamos_por_mes",
    "uso_digital_total",
    "intensidad_contacto",
    "ratio_pagos_atrasados",
    "indice_problemas_servicio",
    "indice_valor_cliente",
    "indice_experiencia"
]

categorical_cols = [
    "genero",
    "nivel_ingreso",
    "segmento_cliente",
    "tipo_cliente",
    "zona_residencia",
    "ocupacion",
    "prioridad_ticket",
    "dia_semana_ultima_interaccion",
    "id_region",
    "id_sede"
]


# ============================================================
# 5. Imputación, indexación, codificación y vectorización
# ============================================================

imputed_cols = [c + "_imp" for c in numeric_cols]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=imputed_cols
).setStrategy("median")

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=c + "_idx",
        outputCol=c + "_ohe"
    )
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=imputed_cols + [c + "_ohe" for c in categorical_cols],
    outputCol="features_raw",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=False,
    withStd=True
)

preprocessing_stages = [imputer] + indexers + encoders + [assembler, scaler]


# ============================================================
# 6. Análisis no supervisado: K-Means
# ============================================================

print("\n=== ANÁLISIS NO SUPERVISADO: K-MEANS ===")

silhouette_results = []

for k in range(2, 9):
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        seed=33,
        maxIter=30
    )

    pipeline_kmeans = Pipeline(stages=preprocessing_stages + [kmeans])
    model_kmeans = pipeline_kmeans.fit(df)
    predictions_kmeans = model_kmeans.transform(df)

    evaluator_cluster = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette"
    )

    silhouette = evaluator_cluster.evaluate(predictions_kmeans)
    silhouette_results.append((k, silhouette))

    print(f"K={k} | Silhouette={silhouette:.4f}")


best_k = max(silhouette_results, key=lambda x: x[1])[0]

print("\nMejor número de clusters según Silhouette:", best_k)

kmeans_final = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=best_k,
    seed=33,
    maxIter=30
)

pipeline_kmeans_final = Pipeline(stages=preprocessing_stages + [kmeans_final])
model_kmeans_final = pipeline_kmeans_final.fit(df)
clusters = model_kmeans_final.transform(df)

print("\n=== TAMAÑO DE CLUSTERS ===")
clusters.groupBy("cluster").count().orderBy("cluster").show()

print("\n=== PERFIL DE CLUSTERS ===")
clusters.groupBy("cluster").avg(
    "reclamos_6m",
    "tiempo_espera_promedio_min",
    "tiempo_resolucion_promedio_h",
    "calidad_atencion_score",
    "disponibilidad_servicio_pct",
    "pagos_atrasados_6m",
    "indice_problemas_servicio",
    "indice_experiencia",
    "satisfaccion_alta"
).orderBy("cluster").show(truncate=False)


# ============================================================
# 7. División train/test
# ============================================================

train, test = df.randomSplit([0.8, 0.2], seed=33)

print("\n=== DIVISIÓN TRAIN / TEST ===")
print("Train:", train.count())
print("Test:", test.count())


# ============================================================
# 8. Función de evaluación
# ============================================================

def evaluar_modelo(predictions, nombre_modelo):
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy"
    )

    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1"
    )

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    accuracy = evaluator_acc.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    auc = evaluator_auc.evaluate(predictions)

    print(f"\n=== MÉTRICAS: {nombre_modelo} ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    print("\nMatriz de confusión:")
    predictions.groupBy(label_col, "prediction").count().orderBy(label_col, "prediction").show()

    return {
        "Modelo": nombre_modelo,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }


# ============================================================
# 9. Modelos supervisados
# ============================================================

print("\n=== ANÁLISIS SUPERVISADO ===")

modelos = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=50
    ),
    "Decision Tree": DecisionTreeClassifier(
        featuresCol="features",
        labelCol=label_col,
        seed=33,
        maxDepth=6
    ),
    "Random Forest": RandomForestClassifier(
        featuresCol="features",
        labelCol=label_col,
        seed=33,
        numTrees=100,
        maxDepth=8
    ),
    "GBT": GBTClassifier(
        featuresCol="features",
        labelCol=label_col,
        seed=33,
        maxIter=50,
        maxDepth=5
    )
}

resultados = []

for nombre, modelo in modelos.items():
    pipeline = Pipeline(stages=preprocessing_stages + [modelo])
    fitted_model = pipeline.fit(train)
    predictions = fitted_model.transform(test)

    metricas = evaluar_modelo(predictions, nombre)
    resultados.append(metricas)


# ============================================================
# 10. Tabla comparativa
# ============================================================

print("\n=== TABLA COMPARATIVA DE MODELOS ===")

for r in resultados:
    print(
        f"{r['Modelo']} | "
        f"Accuracy={r['Accuracy']:.4f} | "
        f"Precision={r['Precision']:.4f} | "
        f"Recall={r['Recall']:.4f} | "
        f"F1={r['F1']:.4f} | "
        f"AUC={r['AUC']:.4f}"
    )

mejor_modelo_f1 = max(resultados, key=lambda x: x["F1"])
mejor_modelo_auc = max(resultados, key=lambda x: x["AUC"])

print("\nMejor modelo según F1-score:", mejor_modelo_f1["Modelo"])
print("Mejor modelo según AUC:", mejor_modelo_auc["Modelo"])


# ============================================================
# 11. Cerrar Spark
# ============================================================

spark.stop()