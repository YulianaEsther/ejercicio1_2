
import base64
import io
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")  # dibuja los gráficos en memoria (sin ventana)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PUERTO = 8000
K = 3
TAMANO_PRUEBA = 0.25
CARPETA = os.path.dirname(os.path.abspath(__file__))

# Nombres bonitos para mostrar en pantalla (si una columna no está aquí,
# se muestra su nombre cambiando "_" por espacios).
NOMBRES = {
    "valoracion": "Valoración (1–5)",
    "tiempo_espera": "Tiempo de espera (min)",
    "problemas_reportados": "Problemas reportados",
    "calificacion_atencion": "Calificación de atención (1–10)",
    "precio_pagado": "Precio pagado (S/)",
    "resolvio_problema": "Problema resuelto (0 = no, 1 = sí)",
    "empleados": "Número de empleados",
    "horas_manuales_semana": "Horas de trabajo manual por semana",
    "presupuesto_tecnologia": "Presupuesto en tecnología (S/ al mes)",
    "ingresos_mensuales": "Ingresos mensuales (S/)",
    "numero_empleados": "Número de empleados",
    "procesos_manuales": "Procesos manuales",
}

EJERCICIOS = {
    "1": {
        "archivo": "clientes.csv",
        "objetivo": "satisfecho",
        "positivo": "El cliente está SATISFECHO con el servicio.",
        "negativo": "El cliente NO está satisfecho con el servicio.",
    },
    "2": {
        "archivo": "empresarios.csv",
        "objetivo": "invierte",
        "positivo": "El empresario SÍ invertiría en el software.",
        "negativo": "El empresario NO invertiría en el software.",
    },
}
ETIQUETAS = ["No", "Sí"]


def nombre_bonito(columna):
    return NOMBRES.get(columna, columna.replace("_", " ").capitalize())


# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
def crear_grafico(df, columnas, objetivo, X_train_esc, X_test_esc, y_train, y_test):
    """Gráficos de Matplotlib: k vs exactitud y dispersión por clase (PNG en base64)."""
    valores_k = list(range(1, min(16, len(y_train))))
    exactitudes = []
    for k in valores_k:
        m = KNeighborsClassifier(n_neighbors=k).fit(X_train_esc, y_train)
        exactitudes.append(accuracy_score(y_test, m.predict(X_test_esc)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(valores_k, exactitudes, marker="o", color="#6d74c9")
    ax1.axvline(K, color="#9aa3b5", linestyle="--", label=f"k usado = {K}")
    ax1.set_title("Exactitud según el valor de k")
    ax1.set_xlabel("k (número de vecinos)")
    ax1.set_ylabel("Exactitud")
    ax1.set_xticks(valores_k)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for clase, color in ((0, "#9aa3b5"), (1, "#6d74c9")):
        mascara = df[objetivo] == clase
        ax2.scatter(
            df.loc[mascara, columnas[0]],
            df.loc[mascara, columnas[1]],
            c=color,
            label=ETIQUETAS[clase],
            edgecolors="k",
            alpha=0.85,
        )
    ax2.set_title("Datos por clase")
    ax2.set_xlabel(nombre_bonito(columnas[0]))
    ax2.set_ylabel(nombre_bonito(columnas[1]))
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def preparar(config):
    ruta = os.path.join(CARPETA, config["archivo"])
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró {config['archivo']} en {CARPETA}")

    df = pd.read_csv(ruta)
    objetivo = config["objetivo"]
    if objetivo not in df.columns:
        raise ValueError(f"{config['archivo']} no tiene la columna '{objetivo}'")
    if len(df) < 10:
        raise ValueError(f"{config['archivo']} tiene muy pocas filas ({len(df)}). Usa 30 o más.")

    X = df.drop(columns=[objetivo])
    y = df[objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANO_PRUEBA, random_state=42, stratify=y
    )

    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    k = min(K, len(X_train))
    modelo = KNeighborsClassifier(n_neighbors=k).fit(X_train_esc, y_train)
    y_pred = modelo.predict(X_test_esc)

    caracteristicas = []
    for columna in X.columns:
        es_entero = pd.api.types.is_integer_dtype(X[columna])
        mediana = float(X[columna].median())
        caracteristicas.append(
            {
                "columna": columna,
                "nombre": nombre_bonito(columna),
                "defecto": int(round(mediana)) if es_entero else round(mediana, 1),
                "paso": 1 if es_entero else 0.1,
                "min": float(X[columna].min()),
                "max": float(X[columna].max()),
            }
        )

    return {
        "config": config,
        "columnas": list(X.columns),
        "escalador": escalador,
        "modelo": modelo,
        "k": k,
        "y_train": y_train.to_numpy(),
        "estado": {
            "k": k,
            "exactitud": round(accuracy_score(y_test, y_pred) * 100, 1),
            "registros": int(len(df)),
            "caracteristicas": caracteristicas,
            "etiquetas": ETIQUETAS,
            "matriz": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
            "prueba": int(TAMANO_PRUEBA * 100),
            "grafico": crear_grafico(
                df, list(X.columns), objetivo, X_train_esc, X_test_esc, y_train, y_test
            ),
        },
    }


def predecir(m, valores):
    fila = pd.DataFrame(
        [[float(valores[c]) for c in m["columnas"]]], columns=m["columnas"]
    )
    fila_esc = m["escalador"].transform(fila)
    distancias, indices = m["modelo"].kneighbors(fila_esc)
    clases = m["y_train"][indices[0]]

    votos_si = int((clases == 1).sum())
    votos_no = int(len(clases) - votos_si)
    prediccion = int(m["modelo"].predict(fila_esc)[0])

    return {
        "prediccion": prediccion,
        "mensaje": m["config"]["positivo"] if prediccion == 1 else m["config"]["negativo"],
        "votos_si": votos_si,
        "votos_no": votos_no,
        "vecinos": [
            {
                "n": i + 1,
                "clase": ETIQUETAS[int(clase)],
                "distancia": round(float(dist), 3),
            }
            for i, (clase, dist) in enumerate(zip(clases, distancias[0]))
        ],
    }


# ------------------------------------------------------------
# Página web
# ------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SENATI | KNN Lab</title>
<style>
  :root { --morado:#6d74c9; --gris:#7b8496; --texto:#2b3040; --suave:#8a90a0; --borde:#e3e6ee; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color:var(--texto); background:#fff; }
  header { background:var(--gris); color:#fff; padding:18px 40px; display:flex; justify-content:space-between; align-items:center; }
  header .marca { font-size:20px; } header .marca b { font-weight:700; }
  header .etiqueta { font-size:13px; }
  main { max-width:960px; margin:0 auto; padding:32px 24px 60px; }
  h1 { margin:0 0 6px; font-size:28px; }
  .subtitulo { color:var(--suave); font-size:14px; margin:0 0 22px; }
  .tabs { display:flex; gap:10px; margin-bottom:28px; }
  .tab { border:0; background:transparent; color:var(--texto); font-weight:600; font-size:13px; padding:11px 18px; border-radius:4px; cursor:pointer; }
  .tab.activo { background:var(--morado); color:#fff; }
  .kpis { display:grid; grid-template-columns:repeat(3, 1fr); gap:20px; margin-bottom:34px; }
  .kpi .rotulo { color:var(--suave); font-size:13px; }
  .kpi .valor { color:var(--morado); font-size:28px; font-weight:700; }
  .columnas { display:grid; grid-template-columns:1fr 1fr; gap:40px; }
  h2 { font-size:17px; margin:0 0 6px; }
  .nota { color:var(--suave); font-size:12.5px; margin:0 0 14px; }
  label { display:block; font-size:13px; color:var(--suave); margin-bottom:16px; }
  label small { display:block; font-size:11px; opacity:.8; }
  input { display:block; width:100%; margin-top:6px; padding:12px; border:1px solid var(--borde); border-radius:4px; font-size:14px; color:var(--texto); background:#fafbfd; }
  button.ejecutar { width:100%; padding:13px; border:0; border-radius:4px; background:var(--morado); color:#fff; font-weight:600; font-size:14px; cursor:pointer; }
  button.ejecutar:hover { filter:brightness(1.08); }
  .resultado { margin-top:22px; font-size:12px; color:var(--suave); }
  .resultado .mensaje { font-size:18px; font-weight:700; margin-top:6px; }
  .resultado.si .mensaje { color:#2f8f5b; } .resultado.no .mensaje { color:#c0504d; }
  table { border-collapse:collapse; width:100%; font-size:13px; margin-bottom:28px; }
  th, td { text-align:center; padding:9px 6px; } th { color:var(--suave); font-weight:600; }
  td.fila { font-weight:700; }
  .vecino { color:var(--suave); font-size:13px; margin:9px 0; }
  .barra { height:8px; background:var(--borde); border-radius:4px; overflow:hidden; margin:14px 0 8px; }
  .barra div { height:100%; background:var(--morado); }
  .graficos { margin-top:40px; } .graficos img { width:100%; border:1px solid var(--borde); border-radius:6px; }
  .error { color:#c0504d; font-weight:600; }
  @media (max-width:760px) { .columnas { grid-template-columns:1fr; } .kpis { grid-template-columns:1fr; } header { padding:16px 20px; } }
</style>
</head>
<body>
<header>
  <div class="marca"><b>SENATI</b> / KNN Lab</div>
  <div class="etiqueta">Machine Learning · Clasificación</div>
</header>
<main>
  <h1>Laboratorio de predicción inteligente</h1>
  <p class="subtitulo" id="subtitulo">Clasificación supervisada mediante los vecinos más cercanos.</p>

  <div class="tabs">
    <button class="tab activo" data-ej="1">01 · Satisfacción del cliente</button>
    <button class="tab" data-ej="2">02 · Inversión en software</button>
  </div>

  <p class="error" id="error"></p>

  <div class="kpis">
    <div class="kpi"><div class="rotulo">Exactitud</div><div class="valor" id="exactitud">–</div></div>
    <div class="kpi"><div class="rotulo">Registros</div><div class="valor" id="registros">–</div></div>
    <div class="kpi"><div class="rotulo">Vecinos (K)</div><div class="valor" id="vecinos">–</div></div>
  </div>

  <div class="columnas">
    <section>
      <h2>Simulador de predicción</h2>
      <p class="nota">Cambia los valores y ejecuta el algoritmo.</p>
      <div id="campos"></div>
      <button class="ejecutar" id="ejecutar">Ejecutar predicción KNN</button>
      <div class="resultado" id="resultado"></div>
    </section>

    <section>
      <h2>Evaluación del modelo</h2>
      <p class="nota" id="notaMatriz">Matriz de confusión</p>
      <table id="matriz"></table>

      <h2>Votación de los vecinos</h2>
      <div id="votacion"></div>
      <p class="nota">Las variables se estandarizan antes de calcular la distancia euclidiana.</p>
    </section>
  </div>

  <section class="graficos">
    <h2>Gráficos (Matplotlib)</h2>
    <img id="grafico" alt="Gráficos del modelo">
  </section>
</main>

<script>
  const $ = (id) => document.getElementById(id);
  let ejercicio = "1";
  let estado = null;

  async function cargar(ej) {
    ejercicio = ej;
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("activo", t.dataset.ej === ej));
    $("error").textContent = "";
    const respuesta = await fetch("/api/estado/" + ej);
    estado = await respuesta.json();
    if (estado.error) { $("error").textContent = estado.error; return; }
    pintar();
    await predecir();
  }

  function pintar() {
    $("exactitud").textContent = estado.exactitud.toFixed(1) + "%";
    $("registros").textContent = estado.registros;
    $("vecinos").textContent = estado.k;
    $("subtitulo").textContent = "Clasificación supervisada mediante los " + estado.k + " vecinos más cercanos.";
    $("campos").innerHTML = estado.caracteristicas.map(c =>
      `<label>${c.nombre}
         <input type="number" id="campo_${c.columna}" value="${c.defecto}" step="${c.paso}">
         <small>Rango en los datos: ${c.min} a ${c.max}</small>
       </label>`).join("");
    $("notaMatriz").textContent = "Matriz de confusión · conjunto de prueba (" + estado.prueba + "%)";
    const e = estado.etiquetas, m = estado.matriz;
    $("matriz").innerHTML =
      `<tr><th>Real / Predicho</th><th>${e[0]}</th><th>${e[1]}</th></tr>
       <tr><td class="fila">${e[0]}</td><td>${m[0][0]}</td><td>${m[0][1]}</td></tr>
       <tr><td class="fila">${e[1]}</td><td>${m[1][0]}</td><td>${m[1][1]}</td></tr>`;
    $("grafico").src = "data:image/png;base64," + estado.grafico;
    $("resultado").innerHTML = "";
  }

  async function predecir() {
    const valores = {};
    for (const c of estado.caracteristicas) valores[c.columna] = parseFloat($("campo_" + c.columna).value);
    if (Object.values(valores).some(v => Number.isNaN(v))) { alert("Completa todos los campos con números."); return; }

    const respuesta = await fetch("/api/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ej: ejercicio, valores })
    });
    const d = await respuesta.json();
    if (d.error) { $("error").textContent = d.error; return; }

    $("resultado").className = "resultado " + (d.prediccion === 1 ? "si" : "no");
    $("resultado").innerHTML = "RESULTADO DE LA PREDICCIÓN<div class='mensaje'>" + d.mensaje + "</div>";

    const total = d.votos_si + d.votos_no;
    $("votacion").innerHTML =
      d.vecinos.map(v => `<div class="vecino">Vecino ${v.n} · ${v.clase} · distancia ${v.distancia.toFixed(3)}</div>`).join("") +
      `<div class="barra"><div style="width:${(d.votos_si / total) * 100}%"></div></div>
       <div class="nota">Sí: ${d.votos_si} votos · No: ${d.votos_no} votos</div>`;
  }

  document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => cargar(t.dataset.ej)));
  $("ejecutar").addEventListener("click", predecir);
  cargar("1");
</script>
</body>
</html>
"""


# ------------------------------------------------------------
# Servidor
# ------------------------------------------------------------
MODELOS = {}


class Manejador(BaseHTTPRequestHandler):
    def _responder(self, codigo, contenido, tipo="application/json; charset=utf-8"):
        datos = contenido.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        ruta = urlparse(self.path).path
        if ruta == "/":
            self._responder(200, HTML, "text/html; charset=utf-8")
        elif ruta.startswith("/api/estado/"):
            ej = ruta.rsplit("/", 1)[1]
            if ej in MODELOS:
                self._responder(200, json.dumps(MODELOS[ej]["estado"]))
            else:
                self._responder(404, json.dumps({"error": "Ejercicio no encontrado"}))
        else:
            self._responder(404, json.dumps({"error": "No encontrado"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/predecir":
            self._responder(404, json.dumps({"error": "No encontrado"}))
            return
        try:
            largo = int(self.headers.get("Content-Length", 0))
            cuerpo = json.loads(self.rfile.read(largo))
            resultado = predecir(MODELOS[cuerpo["ej"]], cuerpo["valores"])
            self._responder(200, json.dumps(resultado))
        except Exception as error:
            self._responder(400, json.dumps({"error": f"No se pudo predecir: {error}"}))

    def log_message(self, formato, *args):
        pass  # evita llenar la terminal de mensajes


def main():
    for ej, config in EJERCICIOS.items():
        try:
            MODELOS[ej] = preparar(config)
            print(f"Ejercicio {ej}: listo ({MODELOS[ej]['estado']['exactitud']}% de exactitud)")
        except Exception as error:
            print(f"Error en el ejercicio {ej}: {error}")
            sys.exit(1)

    servidor = ThreadingHTTPServer(("localhost", PUERTO), Manejador)
    url = f"http://localhost:{PUERTO}"
    print(f"\nKNN Lab funcionando en {url}  (Ctrl + C para detener)")
    webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()