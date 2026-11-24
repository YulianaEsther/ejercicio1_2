
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 80

# ---------------- Ejercicio 1: satisfacción del cliente ----------------
valoracion = rng.integers(1, 6, N)                                  # 1 a 5
tiempo_espera = rng.integers(2, 61, N)                              # minutos
problemas = rng.choice([0, 1, 2, 3], size=N, p=[0.5, 0.25, 0.15, 0.1])

puntaje = valoracion - 0.06 * tiempo_espera - 0.9 * problemas + rng.normal(0, 0.5, N)
satisfecho = (puntaje >= 1.8).astype(int)

clientes = pd.DataFrame(
    {
        "valoracion": valoracion,
        "tiempo_espera": tiempo_espera,
        "problemas_reportados": problemas,
        "satisfecho": satisfecho,
    }
)
clientes.to_csv("clientes.csv", index=False)

# ---------------- Ejercicio 2: inversión en software ----------------
empleados = rng.integers(1, 61, N)
horas_manuales = rng.integers(5, 81, N)                             # horas por semana
presupuesto = rng.integers(100, 5001, N)                            # soles al mes

puntaje2 = (
    0.03 * empleados + 0.05 * horas_manuales + 0.0006 * presupuesto + rng.normal(0, 0.6, N)
)
invierte = (puntaje2 >= np.median(puntaje2)).astype(int)

empresarios = pd.DataFrame(
    {
        "empleados": empleados,
        "horas_manuales_semana": horas_manuales,
        "presupuesto_tecnologia": presupuesto,
        "invierte": invierte,
    }
)
empresarios.to_csv("empresarios.csv", index=False)

print("Listo: clientes.csv y empresarios.csv creados con", N, "registros cada uno.")
print("\nClientes por clase:\n", clientes["satisfecho"].value_counts())
print("\nEmpresarios por clase:\n", empresarios["invierte"].value_counts())