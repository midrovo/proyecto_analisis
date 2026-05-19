import numpy as np
import pandas as pd

np.random.seed(33)

n = 15000

df = pd.DataFrame({
    "id_cliente": [f"C{str(i).zfill(6)}" for i in range(n)],
    "id_servicio": np.random.choice([f"S{i}" for i in range(1, 16)], n),
    "id_region": np.random.choice([f"R{i}" for i in range(1, 8)], n),
    "id_sede": np.random.choice([f"SEDE{i}" for i in range(1, 21)], n),
    "id_agente": np.random.choice([f"A{i}" for i in range(1, 101)], n),
    "id_campania": np.random.choice([f"CAM{i}" for i in range(1, 11)], n),
    "id_ticket_reciente": [f"T{str(i).zfill(7)}" for i in range(n)]
})

df["edad"] = np.clip(np.random.normal(38, 12, n), 18, 75).round()
df["genero"] = np.random.choice(["F", "M", "Otro"], n, p=[0.49, 0.49, 0.02])
df["nivel_ingreso"] = np.random.choice(["bajo", "medio", "alto"], n, p=[0.35, 0.50, 0.15])
df["segmento_cliente"] = np.random.choice(["basico", "estandar", "premium"], n, p=[0.45, 0.40, 0.15])
df["antiguedad_meses"] = np.clip(np.random.gamma(3, 12, n), 1, 120).round()
df["tipo_cliente"] = np.random.choice(["nuevo", "regular", "leal"], n, p=[0.25, 0.55, 0.20])
df["nivel_digitalizacion"] = np.random.beta(3, 2, n)
df["zona_residencia"] = np.random.choice(["urbana", "suburbana", "rural"], n, p=[0.65, 0.25, 0.10])
df["ocupacion"] = np.random.choice(
    ["empleado", "independiente", "estudiante", "jubilado", "otro"],
    n,
    p=[0.45, 0.25, 0.15, 0.07, 0.08]
)

df["frecuencia_uso_mensual"] = np.random.poisson(18, n)
df["sesiones_app_mensual"] = np.random.poisson(12, n)
df["llamadas_callcenter_3m"] = np.random.poisson(2, n)
df["reclamos_6m"] = np.random.poisson(1.2, n)
df["tickets_abiertos_6m"] = df["reclamos_6m"] + np.random.poisson(1, n)
df["tickets_resueltos_6m"] = np.maximum(
    0,
    df["tickets_abiertos_6m"] - np.random.binomial(2, 0.25, n)
)
df["cancelaciones_previas"] = np.random.binomial(1, 0.08, n)
df["cambios_plan_12m"] = np.random.poisson(0.6, n)
df["uso_chatbot_mensual"] = np.random.poisson(4, n)
df["visitas_web_mensual"] = np.random.poisson(8, n)

df["tiempo_espera_promedio_min"] = np.random.gamma(2, 4, n)
df["tiempo_resolucion_promedio_h"] = np.random.gamma(2, 6, n)
df["numero_escalamientos"] = np.random.poisson(0.7, n)
df["prioridad_ticket"] = np.random.choice(["baja", "media", "alta"], n, p=[0.55, 0.35, 0.10])
df["cumplimiento_sla"] = np.random.binomial(1, 0.78, n)
df["disponibilidad_servicio_pct"] = np.clip(np.random.normal(96, 3, n), 80, 100)
df["interrupciones_servicio_3m"] = np.random.poisson(0.9, n)
df["calidad_atencion_score"] = np.clip(np.random.beta(5, 2, n) * 10, 1, 10)
df["tiempo_primera_respuesta_min"] = np.random.gamma(2, 5, n)
df["resolucion_primer_contacto"] = np.random.binomial(1, 0.68, n)

df["valor_plan_mensual"] = np.random.lognormal(3.5, 0.35, n)
df["pagos_atrasados_6m"] = np.random.poisson(0.8, n)
df["monto_total_pagado_12m"] = df["valor_plan_mensual"] * 12 + np.random.normal(0, 50, n)
df["descuentos_recibidos_12m"] = np.random.gamma(1.5, 15, n)
df["cargos_extra_6m"] = np.random.gamma(1.2, 10, n)
df["saldo_pendiente"] = np.random.lognormal(2.5, 0.7, n)
df["consumo_servicio_promedio"] = np.random.gamma(4, 8, n)

df["mes_registro"] = np.random.randint(1, 13, n)
df["trimestre"] = ((df["mes_registro"] - 1) // 3) + 1
df["dia_semana_ultima_interaccion"] = np.random.choice(
    ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"],
    n
)
df["es_fin_de_semana"] = df["dia_semana_ultima_interaccion"].isin(["sabado", "domingo"]).astype(int)
df["temporada_alta"] = df["mes_registro"].isin([11, 12, 1]).astype(int)
df["meses_desde_ultimo_reclamo"] = np.clip(np.random.exponential(4, n), 0, 24)

df["tasa_resolucion_tickets"] = df["tickets_resueltos_6m"] / (df["tickets_abiertos_6m"] + 1)
df["reclamos_por_mes"] = df["reclamos_6m"] / 6
df["uso_digital_total"] = (
    df["sesiones_app_mensual"] +
    df["uso_chatbot_mensual"] +
    df["visitas_web_mensual"]
)
df["intensidad_contacto"] = (
    df["llamadas_callcenter_3m"] +
    df["tickets_abiertos_6m"] +
    df["reclamos_6m"]
)
df["ratio_pagos_atrasados"] = df["pagos_atrasados_6m"] / (df["antiguedad_meses"] + 1)
df["indice_problemas_servicio"] = (
    df["interrupciones_servicio_3m"] +
    df["numero_escalamientos"] +
    df["reclamos_6m"]
)
df["indice_valor_cliente"] = df["monto_total_pagado_12m"] - df["descuentos_recibidos_12m"]
df["indice_experiencia"] = (
    df["calidad_atencion_score"] +
    df["disponibilidad_servicio_pct"] / 10 -
    df["tiempo_espera_promedio_min"] / 10
)

score = (
    0.40 * df["calidad_atencion_score"]
    + 0.04 * df["disponibilidad_servicio_pct"]
    + 1.20 * df["resolucion_primer_contacto"]
    + 1.10 * df["cumplimiento_sla"]
    + 0.80 * df["tasa_resolucion_tickets"]
    - 0.55 * df["reclamos_6m"]
    - 0.08 * df["tiempo_espera_promedio_min"]
    - 0.05 * df["tiempo_resolucion_promedio_h"]
    - 0.45 * df["pagos_atrasados_6m"]
    - 0.65 * df["interrupciones_servicio_3m"]
    - 0.50 * df["numero_escalamientos"]
    + np.random.normal(0, 1.2, n)
)

prob = 1 / (1 + np.exp(-((score - score.mean()) / score.std())))
df["satisfaccion_alta"] = np.random.binomial(1, prob)

df["score_satisfaccion"] = np.clip(
    1 + 9 * prob + np.random.normal(0, 0.7, n),
    1,
    10
)

# Nulos: 5%
columnas_nulos = [
    "nivel_ingreso",
    "ocupacion",
    "calidad_atencion_score",
    "tiempo_resolucion_promedio_h",
    "saldo_pendiente"
]

for col in columnas_nulos:
    idx = df.sample(frac=0.05, random_state=33).index
    df.loc[idx, col] = np.nan

# Outliers: 2%
columnas_outliers = [
    "tiempo_espera_promedio_min",
    "tiempo_resolucion_promedio_h",
    "saldo_pendiente",
    "cargos_extra_6m"
]

for col in columnas_outliers:
    idx = df.sample(frac=0.02, random_state=33).index
    df.loc[idx, col] = df.loc[idx, col] * 5

# Duplicados: 1%
duplicados = df.sample(frac=0.01, random_state=33)
df_final = pd.concat([df, duplicados], ignore_index=True)

df_final.to_csv("../data/processed/dataset_satisfaccion_cliente.csv", index=False)
df_final.to_parquet("../data/processed/dataset_satisfaccion_cliente.parquet", index=False)

print("Dataset generado correctamente")
print(df_final.shape)