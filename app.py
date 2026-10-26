# -*- coding: utf-8 -*-
# =====================================================================================
# DASHBOARD ANALÍTICO · DATASET POS RÚSTICA (CAPA SILVER)
# Curso: Analítica Descriptiva y Diagnóstica con Visualización y Storytelling (UNMSM)
# Ejecución : streamlit run app.py
# Requisitos: pip install streamlit pandas numpy scipy plotly seaborn matplotlib pyarrow
# Archivo de datos esperado en la misma carpeta: dataset_pos_silver.parquet
# =====================================================================================

# ------------------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ------------------------------------------------------------------------------------
import os                                    # Permite comprobar si el archivo parquet existe en disco
import numpy as np                           # Cálculo numérico vectorizado (arreglos, raíces, infinitos)
import pandas as pd                          # Manipulación de datos tabulares (DataFrame, groupby, pivot)
import streamlit as st                       # Framework que construye la interfaz web del dashboard
import plotly.express as px                  # Gráficos interactivos de alto nivel (barras, mapas de calor)
import plotly.graph_objects as go            # Gráficos Plotly de bajo nivel (control total de trazas)
import seaborn as sns                        # Mapa de calor de la matriz de correlación
import matplotlib.pyplot as plt              # Figura base sobre la que seaborn dibuja
from scipy import stats                      # Pruebas estadísticas (t, chi-cuadrado, correlaciones)

# ------------------------------------------------------------------------------------
# 2. CONFIGURACIÓN GENERAL DE LA PÁGINA Y ESTILO VISUAL
# ------------------------------------------------------------------------------------
st.set_page_config(                          # DEBE ser la primera instrucción de Streamlit del script
    page_title="Analítica POS Rústica",      # Título que se muestra en la pestaña del navegador
    page_icon="📊",                          # Ícono de la pestaña
    layout="wide",                           # Usa todo el ancho de la pantalla (aspecto profesional)
    initial_sidebar_state="expanded",        # La barra lateral de filtros inicia desplegada
)

RUTA_PARQUET = "dataset_pos_silver.parquet"  # Nombre del archivo local con la capa silver
COLOR_PRIMARIO = "#7A0019"                   # Color vino institucional (identidad UNMSM)
COLOR_SECUNDARIO = "#C8A24A"                 # Color dorado de acento
COLOR_NEUTRO = "#9AA3AF"                     # Gris para elementos de apoyo
DIAS_NOMBRE = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]  # Etiquetas de días (0=Lunes)

st.markdown(                                 # Inyecta CSS para dar aspecto de "tarjetas" a las métricas
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}             /* Reduce márgenes de la página */
    div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E5E7EB;
        border-left: 5px solid #7A0019; padding: 12px 14px; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);}                                /* Tarjeta blanca con borde vino */
    div[data-testid="stMetricLabel"] p {font-size: 0.80rem; font-weight: 600;}  /* Etiqueta de KPI más compacta */
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}                              /* Separación entre pestañas */
    .stTabs [data-baseweb="tab"] {background: #F3F4F6; border-radius: 8px 8px 0 0; padding: 8px 14px;}
    h1, h2, h3 {color: #7A0019;}                                               /* Títulos en color vino */
    </style>
    """,
    unsafe_allow_html=True,                  # Necesario para que Streamlit acepte etiquetas HTML/CSS
)

# ------------------------------------------------------------------------------------
# 3. FUNCIONES UTILITARIAS (cálculos y formato reutilizables)
# ------------------------------------------------------------------------------------
def mostrar_tabla(d, **kw):
    """Muestra un DataFrame a todo el ancho, compatible con versiones nuevas y antiguas de Streamlit."""
    try:                                                 # Primero intenta la sintaxis moderna
        st.dataframe(d, width="stretch", **kw)           # width="stretch" = ocupar todo el ancho disponible
    except Exception:                                    # Si la versión instalada es antigua...
        st.dataframe(d, use_container_width=True, **kw)  # ...usa el parámetro clásico equivalente


def estilo(fig, alto=380):
    """Aplica un formato uniforme (plantilla limpia, márgenes, fuente) a todas las figuras Plotly."""
    fig.update_layout(template="plotly_white", height=alto,          # Fondo blanco y altura fija
                      margin=dict(l=10, r=10, t=45, b=10),           # Márgenes compactos
                      font=dict(family="Arial", size=12),            # Tipografía legible
                      legend=dict(orientation="h", y=-0.2))          # Leyenda horizontal bajo el gráfico
    return fig                                                       # Devuelve la figura ya formateada


def fmt_p(p):
    """Formatea un p-valor como se reporta en informes académicos."""
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"             # Evita mostrar 0.0000


def fuerza_correlacion(r):
    """Traduce |r| a la escala estándar de interpretación vista en clase."""
    a = abs(r)                                                       # El signo solo indica dirección, no fuerza
    if np.isnan(a): return "No calculable"                          # Protege contra valores NaN
    if a == 0: return "Nula"                                        # Variables totalmente independientes
    if a < 0.30: return "Débil"                                     # Rango 0.01 – 0.29
    if a < 0.50: return "Moderada"                                  # Rango 0.30 – 0.49
    if a < 0.70: return "Moderadamente fuerte"                      # Rango 0.50 – 0.69
    if a < 1.0: return "Fuerte / Alta"                              # Rango 0.70 – 0.99
    return "Perfecta"                                               # |r| = 1


def estado_semaforo(valor, meta):
    """Semáforo R/A/V: verde si cumple la meta, amarillo hasta 1.5×meta, rojo si la excede más."""
    if valor <= meta: return "🟢 Óptimo"                            # Cumple la meta
    if valor <= meta * 1.5: return "🟡 Alerta"                      # Zona de tolerancia
    return "🔴 Crítico"                                             # Fuera de tolerancia


def ic_media(x, conf):
    """Intervalo de confianza t para una media: devuelve (media, límite inferior, límite superior)."""
    x = np.asarray(x, dtype=float)                                   # Convierte a arreglo numérico
    n = len(x)                                                       # Tamaño de muestra
    m = x.mean()                                                     # Media muestral
    if n < 2: return m, np.nan, np.nan                               # Sin variabilidad no hay intervalo
    se = x.std(ddof=1) / np.sqrt(n)                                  # Error estándar = s / √n
    t_crit = stats.t.ppf(1 - (1 - conf) / 2, n - 1)                  # Valor crítico t bilateral
    return m, m - t_crit * se, m + t_crit * se                       # Media ± margen de error


def ic_diferencia_welch(a, b, conf):
    """IC de la diferencia de medias (a-b) sin asumir varianzas iguales (Welch–Satterthwaite)."""
    na, nb = len(a), len(b)                                          # Tamaños de cada grupo
    va, vb = a.var(ddof=1), b.var(ddof=1)                            # Varianzas muestrales
    se = np.sqrt(va / na + vb / nb)                                  # Error estándar de la diferencia
    gl = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))  # Grados de libertad de Welch
    t_crit = stats.t.ppf(1 - (1 - conf) / 2, gl)                     # Valor crítico t con esos grados de libertad
    dif = a.mean() - b.mean()                                        # Diferencia observada
    return dif, dif - t_crit * se, dif + t_crit * se, gl             # Diferencia, límites e información de gl


def cohen_d(a, b):
    """Tamaño del efecto d de Cohen (diferencia estandarizada con desviación combinada)."""
    na, nb = len(a), len(b)                                          # Tamaños de muestra
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))  # Desviación combinada
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan          # d = diferencia / desviación combinada


def etiqueta_efecto(d):
    """Clasifica la magnitud práctica del efecto según |d|."""
    d = abs(d)                                                       # Solo importa la magnitud
    if np.isnan(d): return "No calculable"
    if d < 0.2: return "Trivial"                                     # Prácticamente nulo
    if d < 0.5: return "Pequeño"
    if d < 0.8: return "Mediano"
    return "Grande"


def calcular_meses(pre_f):
    """Cuenta los días con datos por mes y marca como 'completo' el mes con ≥90% de sus días."""
    dias = pre_f.groupby("mes")["fecha_transaccion"].nunique()       # Días distintos con ventas por mes
    completo = pd.Series(dias.values >= 0.9 * dias.index.days_in_month, index=dias.index)  # Bandera de mes completo
    return dias, completo                                            # Devuelve ambos resultados


def tukey(s):
    """Límites de Tukey: Q1-1.5·IQR y Q3+1.5·IQR para detectar valores atípicos."""
    q1, q3 = s.quantile([0.25, 0.75]).values                         # Primer y tercer cuartil
    iqr = q3 - q1                                                    # Rango intercuartílico
    return q1, q3, iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr               # Cuartiles, IQR y límites


# ------------------------------------------------------------------------------------
# 4. CARGA, LIMPIEZA Y PREPROCESAMIENTO (BACKEND, SE EJECUTA UNA SOLA VEZ Y QUEDA EN CACHÉ)
# ------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando y depurando el dataset (solo la primera vez)...")
def obtener_datos(ruta):
    """Lee el parquet, aplica limpieza idempotente y construye la tabla agregada por precuenta."""
    log = []                                                         # Bitácora de transformaciones (reproducibilidad)
    df = pd.read_parquet(ruta)                                       # Lee el archivo parquet completo en memoria
    log.append(f"Lectura inicial: {len(df):,} filas × {df.shape[1]} columnas.")  # Registra dimensiones

    df.columns = df.columns.str.strip().str.lower()                  # Estandariza nombres a minúsculas sin espacios
    requeridas = ["id_precuenta", "id_tienda", "id_cajero", "hora_precuenta", "cant_adultos", "cant_ninos",
                  "cantidad", "precio_unitario", "descuento_total", "categoria_familia"]  # Columnas imprescindibles
    faltan = [c for c in requeridas if c not in df.columns]          # Detecta columnas ausentes
    if faltan:                                                       # Si falta alguna...
        raise ValueError(f"El parquet no contiene las columnas requeridas: {faltan}")  # ...detiene con mensaje claro

    n0 = len(df)                                                     # Filas antes de deduplicar
    df = df.drop_duplicates()                                        # Elimina filas duplicadas exactas (distorsionan SUM/COUNT)
    log.append(f"Duplicados eliminados: {n0 - len(df):,}.")          # Registra cuántos se eliminaron

    for c in ["fecha_proceso", "hora_precuenta", "hora_pedido"]:     # Recorre las columnas de fecha/hora
        if c in df.columns:                                          # Solo si existen en el archivo
            df[c] = pd.to_datetime(df[c], errors="coerce")           # Convierte a datetime; lo inválido pasa a NaT
    n1 = len(df)                                                     # Filas antes de filtrar fechas inválidas
    df = df[df["hora_precuenta"].notna()]                            # Sin fecha de transacción no hay análisis temporal
    log.append(f"Filas sin fecha de precuenta descartadas: {n1 - len(df):,}.")  # Registra el descarte

    for c in ["cant_adultos", "cant_ninos", "cantidad"]:             # Conteos enteros
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")  # Nulos → 0; entero con signo (-1 = anulación)
    for c in ["precio_unitario", "descuento_total"]:                 # Importes decimales
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64")  # Nulos → 0.0
    log.append("Tipos numéricos normalizados (enteros con signo e importes float64); nulos imputados con 0.")

    df["id_cajero"] = df["id_cajero"].astype("string").fillna("SIN_CAJERO").astype("category")  # Cajero nulo → etiqueta explícita
    df["id_tienda"] = df["id_tienda"].astype("string").astype("category")  # Tienda como categoría (ahorra memoria)
    for c in ["categoria_familia", "nombre_subcategoria"]:           # Textos con espacios laterales ('CERVEZAS ')
        if c in df.columns:                                          # Solo si la columna existe
            df[c] = df[c].astype("string").str.strip().astype("category")  # Recorta espacios y vuelve a categoría
    log.append("Textos de categoría recortados (.str.strip) y convertidos a 'category'.")

    df["fecha_transaccion"] = df["hora_precuenta"].dt.normalize()    # Fecha OFICIAL de venta (reemplaza a fecha_proceso, desfasada)
    df["total_detalle_bruto"] = df["cantidad"] * df["precio_unitario"]  # Venta teórica antes de descuentos
    df["total_detalle_neto"] = df["total_detalle_bruto"] - df["descuento_total"]  # Dinero real que cuadra con caja
    df["es_anulacion"] = df["cantidad"] < 0                          # Verdadero si la línea es una anulación posventa
    df["tiene_descuento"] = df["descuento_total"] > 0                # Verdadero si la línea tuvo descuento/promoción
    df["comensales"] = df["cant_adultos"] + df["cant_ninos"]         # Variable derivada: personas en la mesa
    df["mes"] = df["fecha_transaccion"].dt.to_period("M").dt.to_timestamp()  # Primer día del mes (agregación temporal)
    df["dia_num"] = df["fecha_transaccion"].dt.dayofweek.astype("int8")  # 0=Lunes … 6=Domingo
    log.append("Variables derivadas creadas: fecha_transaccion, total_detalle_bruto/neto, es_anulacion, "
               "tiene_descuento, comensales, mes y dia_num.")

    g = df.groupby("id_precuenta", sort=False, observed=True)        # Agrupa las líneas por cuenta (unidad de análisis del ticket)
    pre = g.agg(                                                     # Resume cada precuenta en una sola fila
        id_tienda=("id_tienda", "first"),                            # Tienda donde ocurrió
        fecha_transaccion=("fecha_transaccion", "first"),            # Fecha de la venta
        hora_precuenta=("hora_precuenta", "first"),                  # Instante de apertura de la cuenta
        ticket_neto=("total_detalle_neto", "sum"),                   # Ticket neto = suma de líneas netas
        ticket_bruto=("total_detalle_bruto", "sum"),                 # Ticket bruto
        descuento=("descuento_total", "sum"),                        # Descuento total de la cuenta
        comensales=("comensales", "first"),                          # Personas (se repite en cada línea → first)
        n_lineas=("cantidad", "size"),                               # Número de líneas de pedido
        n_anulaciones=("es_anulacion", "sum"),                       # Líneas anuladas en la cuenta
        tiene_descuento=("tiene_descuento", "max"),                  # Verdadero si alguna línea tuvo descuento
        mes=("mes", "first"),                                        # Mes contable
        dia_num=("dia_num", "first"),                                # Día 0–6
    ).reset_index()                                                  # id_precuenta vuelve a ser columna
    pre["id_tienda"] = pre["id_tienda"].astype(str)                  # Texto simple para evitar categorías vacías al agrupar
    pre["tiene_descuento"] = pre["tiene_descuento"].astype(bool)     # Garantiza tipo booleano
    pre["tipo_dia"] = np.where(pre["dia_num"] >= 5, "Sábado y Domingo", "Lunes a Viernes")  # Segmento de calendario
    pre["dia_nombre"] = pre["dia_num"].map(dict(enumerate(DIAS_NOMBRE)))  # Nombre legible del día
    log.append(f"Tabla por precuenta construida: {len(pre):,} cuentas "
               "(total_precuenta NO se suma porque se repite en cada línea).")

    calidad = pd.DataFrame({"tipo": df.dtypes.astype(str),           # Tipo de dato de cada columna
                            "nulos": df.isna().sum()})               # Nulos restantes tras la limpieza
    return df, pre, {"log": log, "calidad": calidad}                 # Entrega todo a la aplicación


# ------------------------------------------------------------------------------------
# 5. FUNCIONES DE AGREGACIÓN EN CACHÉ (evitan recalcular millones de filas en cada clic)
# ------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def agg_categorias(tiendas, f1, f2):
    """Ingreso neto por tienda × categoría_familia dentro del filtro activo."""
    df_, _, _ = obtener_datos(RUTA_PARQUET)                          # Recupera el dataset ya cargado (caché de recurso)
    m = df_["id_tienda"].isin(tiendas) & df_["fecha_transaccion"].between(f1, f2)  # Máscara de filtros
    sub = df_.loc[m, ["id_tienda", "categoria_familia", "total_detalle_neto"]]     # Solo las columnas necesarias
    out = (sub.groupby(["id_tienda", "categoria_familia"], observed=True)["total_detalle_neto"]
              .sum().reset_index())                                  # GROUP BY + SUM
    return out.astype({"id_tienda": str, "categoria_familia": str})  # Texto simple para el resto del tablero


@st.cache_data(show_spinner=False)
def agg_anulaciones(tiendas, f1, f2):
    """Líneas y anulaciones por tienda × mes y por cajero."""
    df_, _, _ = obtener_datos(RUTA_PARQUET)                          # Dataset cacheado
    m = df_["id_tienda"].isin(tiendas) & df_["fecha_transaccion"].between(f1, f2)  # Máscara de filtros
    sub = df_.loc[m, ["id_tienda", "mes", "id_cajero", "es_anulacion", "total_detalle_bruto"]].copy()  # Subconjunto
    sub["monto_anulado"] = sub["total_detalle_bruto"].where(sub["es_anulacion"], 0.0)  # Importe solo de anulaciones
    por_mes = (sub.groupby(["id_tienda", "mes"], observed=True)
                  .agg(lineas=("es_anulacion", "size"), anulaciones=("es_anulacion", "sum"),
                       monto_anulado=("monto_anulado", "sum")).reset_index())  # COUNT y SUM por tienda-mes
    por_cajero = (sub.groupby("id_cajero", observed=True)
                     .agg(lineas=("es_anulacion", "size"), anulaciones=("es_anulacion", "sum"))
                     .reset_index())                                 # COUNT y SUM por cajero
    por_mes["id_tienda"] = por_mes["id_tienda"].astype(str)          # Texto simple
    por_cajero["id_cajero"] = por_cajero["id_cajero"].astype(str)    # Texto simple
    return por_mes, por_cajero                                       # Devuelve ambas tablas


@st.cache_data(show_spinner="Calculando correlaciones...")
def corr_resultados(tiendas, f1, f2):
    """Pearson, Spearman y Kendall entre comensales y ticket, más matrices y controles de confusión."""
    _, pre_, _ = obtener_datos(RUTA_PARQUET)                         # Tabla por precuenta cacheada
    m = pre_["id_tienda"].isin(tiendas) & pre_["fecha_transaccion"].between(f1, f2)  # Filtros activos
    ok = m & (pre_["comensales"] > 0) & (pre_["ticket_neto"] > 0)    # Cuentas válidas: con comensales y ticket positivo
    b = pre_.loc[ok, ["id_tienda", "comensales", "ticket_neto", "n_lineas", "descuento", "hora_precuenta"]].copy()
    b["hora_dia"] = b["hora_precuenta"].dt.hour                      # Hora de apertura (0–23) como variable numérica
    b = b.drop(columns="hora_precuenta")                             # Ya no se necesita la marca completa
    out = {"n": len(b)}                                              # Tamaño de la base de correlación
    if len(b) < 30:                                                  # Con menos de 30 cuentas no es confiable
        return out
    vars_ = ["comensales", "n_lineas", "descuento", "hora_dia", "ticket_neto"]  # Variables de la matriz
    muestra = b.sample(min(20000, len(b)), random_state=42)          # Muestra reproducible para Kendall (computacionalmente costoso)
    out["pearson_m"] = b[vars_].corr(method="pearson")               # Matriz de Pearson (relación lineal)
    out["spearman_m"] = b[vars_].corr(method="spearman")             # Matriz de Spearman (relación monótona por rangos)
    out["kendall_m"] = muestra[vars_].corr(method="kendall")         # Matriz de Kendall (sobre la muestra)
    out["pearson_r"], out["pearson_p"] = stats.pearsonr(b["comensales"], b["ticket_neto"])      # r y p de Pearson
    out["spearman_rho"], out["spearman_p"] = stats.spearmanr(b["comensales"], b["ticket_neto"])  # ρ y p de Spearman
    out["kendall_tau"], out["kendall_p"] = stats.kendalltau(muestra["comensales"], muestra["ticket_neto"])  # τ y p de Kendall
    _, _, _, li, ls = tukey(b["ticket_neto"])                        # Límites de Tukey del ticket
    sin = b[(b["ticket_neto"] >= li) & (b["ticket_neto"] <= ls)]     # Base sin valores atípicos
    out["pearson_sin_outliers"] = stats.pearsonr(sin["comensales"], sin["ticket_neto"])[0]  # Pearson tras quitar outliers
    out["por_tienda"] = pd.Series({t: stats.spearmanr(g["comensales"], g["ticket_neto"])[0]
                                   for t, g in b.groupby("id_tienda")})  # Spearman dentro de cada tienda (control de confusión)
    out["muestra_scatter"] = b.sample(min(5000, len(b)), random_state=1)[["comensales", "ticket_neto"]]  # Puntos para graficar
    return out


def test_brecha(pre_f, conf):
    """Compara el ticket neto de Sábado y Domingo vs Lunes a Viernes con t de Welch y Mann-Whitney."""
    pv = pre_f[pre_f["ticket_neto"] > 0]                             # Solo cuentas con ticket positivo
    a = pv.loc[pv["tipo_dia"] == "Sábado y Domingo", "ticket_neto"].values  # Grupo A: fin de jornada de descanso
    b = pv.loc[pv["tipo_dia"] == "Lunes a Viernes", "ticket_neto"].values   # Grupo B: días hábiles
    if len(a) < 2 or len(b) < 2:                                     # Se necesitan ambos grupos con datos
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)                    # t de Welch (no asume varianzas iguales)
    dif, lo, hi, gl = ic_diferencia_welch(a, b, conf)                # Diferencia con su intervalo de confianza
    rng = np.random.default_rng(42)                                  # Generador reproducible
    sa = rng.choice(a, size=min(len(a), 100000), replace=False)      # Submuestra A para la prueba no paramétrica
    sb = rng.choice(b, size=min(len(b), 100000), replace=False)      # Submuestra B
    _, p_u = stats.mannwhitneyu(sa, sb, alternative="two-sided")     # U de Mann-Whitney (alternativa robusta a outliers)
    return {"media_a": a.mean(), "media_b": b.mean(), "n_a": len(a), "n_b": len(b),
            "brecha_pct": (a.mean() / b.mean() - 1) * 100, "t": t, "p": p, "gl": gl,
            "dif": dif, "lo": lo, "hi": hi, "d": cohen_d(a, b), "p_u": p_u,
            "ic_a": ic_media(a, conf), "ic_b": ic_media(b, conf)}    # Todo el resultado en un diccionario


# ------------------------------------------------------------------------------------
# 6. CARGA DE DATOS Y VALIDACIÓN DE EXISTENCIA DEL ARCHIVO
# ------------------------------------------------------------------------------------
if not os.path.exists(RUTA_PARQUET):                                 # Verifica que el parquet esté junto a app.py
    st.error(f"No se encontró '{RUTA_PARQUET}'. Colócalo en la misma carpeta que app.py y recarga.")
    st.stop()                                                        # Detiene la ejecución del script

try:                                                                 # Intenta cargar y limpiar los datos
    df, pre, info = obtener_datos(RUTA_PARQUET)                      # Ejecuta (o recupera de caché) el preprocesamiento
except Exception as e:                                               # Si ocurre cualquier problema...
    st.error(f"Error al preparar los datos: {e}")                    # ...lo muestra de forma legible
    st.stop()                                                        # ...y detiene la aplicación

# ------------------------------------------------------------------------------------
# 7. BARRA LATERAL: FILTROS Y PARÁMETROS ESTADÍSTICOS
# ------------------------------------------------------------------------------------
with st.sidebar:                                                     # Todo lo que sigue se dibuja en la barra lateral
    st.markdown("## 🎛️ Panel de control")                            # Título del panel
    tiendas_todas = sorted(map(str, df["id_tienda"].unique()))       # Lista ordenada de tiendas disponibles
    tiendas_sel = st.multiselect("Tiendas", tiendas_todas, default=tiendas_todas)  # Filtro multiselección
    fmin = pre["fecha_transaccion"].min().date()                     # Primera fecha del dataset
    fmax = pre["fecha_transaccion"].max().date()                     # Última fecha del dataset
    rango = st.date_input("Rango de fechas de venta", value=(fmin, fmax), min_value=fmin, max_value=fmax)  # Calendario
    st.markdown("---")                                               # Separador visual
    st.markdown("### Parámetros de inferencia")                      # Subtítulo
    alpha = st.selectbox("Nivel de significancia (α)", [0.05, 0.01, 0.001], index=2,
                         help="Con millones de registros se recomienda α = 0.001 y reportar el tamaño del efecto.")
    conf = st.selectbox("Nivel de confianza del IC", [0.90, 0.95, 0.99], index=1)  # Confianza de los intervalos
    st.markdown("### Parámetros de KPI")                             # Subtítulo
    meta_anul = st.number_input("Meta de anulación (% de líneas)", 0.1, 20.0, 2.0, 0.1)  # Meta editable del KPI 3
    umbral_caida = st.slider("Umbral de caída mensual (%)", 1, 40, 10)  # Umbral de alerta del KPI 4
    st.caption("Los valores por defecto son supuestos de trabajo: ajústalos a la meta real del negocio.")

if len(tiendas_sel) == 0:                                            # Si el usuario deseleccionó todas las tiendas
    st.warning("Selecciona al menos una tienda.")
    st.stop()
if not isinstance(rango, (tuple, list)) or len(rango) != 2:          # El calendario requiere fecha inicial y final
    st.info("Selecciona una fecha inicial y una final.")
    st.stop()

tiendas_sel = tuple(tiendas_sel)                                     # Tupla (hashable) para las funciones en caché
f1, f2 = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])              # Fechas del filtro como Timestamp
pre_f = pre[pre["id_tienda"].isin(tiendas_sel) & pre["fecha_transaccion"].between(f1, f2)]  # Precuentas filtradas
if pre_f.empty:                                                      # Sin datos no hay nada que graficar
    st.warning("No hay ventas con los filtros seleccionados.")
    st.stop()

# ------------------------------------------------------------------------------------
# 8. ENCABEZADO Y CÁLCULO DE LOS 6 KPIs PRINCIPALES
# ------------------------------------------------------------------------------------
st.title("📊 Analítica Descriptiva y Diagnóstica · POS Rústica")     # Título principal
st.caption(f"{len(pre_f):,} precuentas · {len(tiendas_sel)} tiendas · "
           f"{f1:%Y-%m-%d} a {f2:%Y-%m-%d}")                         # Subtítulo con el alcance del filtro

pv = pre_f[pre_f["ticket_neto"] > 0]                                 # Precuentas válidas (ticket positivo)
mediana_g = pv["ticket_neto"].median()                               # KPI 1: ticket neto mediano
cv_g = pv["ticket_neto"].std() / pv["ticket_neto"].mean() * 100      # Coeficiente de variación global

cat_f = agg_categorias(tiendas_sel, f1, f2)                          # Ingreso por tienda × categoría
tot_cat = cat_f.groupby("categoria_familia")["total_detalle_neto"].sum().sort_values(ascending=False)  # Ingreso por categoría
top3_g = tot_cat.head(3).sum() / tot_cat.sum() * 100                 # KPI 2: % del ingreso en las 3 categorías líderes

por_mes_an, por_cajero_an = agg_anulaciones(tiendas_sel, f1, f2)     # Datos de anulación
tasa_g = por_mes_an["anulaciones"].sum() / por_mes_an["lineas"].sum() * 100  # KPI 3: tasa global de anulación

dias_datos, completo = calcular_meses(pre_f)                         # Meses completos vs parciales
tot_mes = pre_f.groupby("mes")["ticket_neto"].sum()                  # Ingreso neto mensual total
tot_ok = tot_mes.where(completo.reindex(tot_mes.index))              # Anula meses incompletos (evita falsas caídas)
mom_tot = ((tot_ok / tot_ok.shift(1) - 1) * 100).replace([np.inf, -np.inf], np.nan).dropna()  # Variación mensual %
mom_ult = mom_tot.iloc[-1] if len(mom_tot) else np.nan               # KPI 4: último crecimiento mensual válido

cr = corr_resultados(tiendas_sel, f1, f2)                            # KPI 5: resultados de correlación
rho = cr.get("spearman_rho", np.nan)                                 # ρ de Spearman (NaN si no hay datos suficientes)

res_b = test_brecha(pre_f, conf)                                     # KPI 6: prueba de brecha de ticket

k1, k2, k3, k4, k5, k6 = st.columns(6)                               # Seis columnas para seis KPIs
k1.metric("Ticket neto mediano", f"{mediana_g:,.2f}", f"CV {cv_g:.0f}%", delta_color="off")
k2.metric("Concentración Top-3 categorías", f"{top3_g:.1f}%", "del ingreso neto", delta_color="off")
k3.metric("Tasa de anulación", f"{tasa_g:.2f}%", f"meta ≤ {meta_anul:.1f}%",
          delta_color="inverse" if tasa_g > meta_anul else "normal")
k4.metric("Crecimiento mensual del ingreso",
          "n/d" if np.isnan(mom_ult) else f"{mom_ult:+.1f}%",
          "" if np.isnan(mom_ult) else f"mes {mom_tot.index[-1]:%Y-%m}", delta_color="off")
k5.metric("Correlación ρ comensales–ticket", "n/d" if np.isnan(rho) else f"{rho:.3f}",
          "" if np.isnan(rho) else fuerza_correlacion(rho), delta_color="off")
if res_b:                                                            # Solo si la prueba pudo calcularse
    k6.metric("Brecha ticket Sáb-Dom vs Lun-Vie", f"{res_b['brecha_pct']:+.1f}%",
              "significativa" if res_b["p"] <= alpha else "no significativa", delta_color="off")
else:
    k6.metric("Brecha ticket Sáb-Dom vs Lun-Vie", "n/d", "sin datos en ambos grupos", delta_color="off")

with st.expander("🧹 Bitácora de preprocesamiento (backend) y calidad de datos"):  # Panel plegable de auditoría
    for linea in info["log"]:                                        # Muestra cada paso de limpieza registrado
        st.write("•", linea)
    mostrar_tabla(info["calidad"])                                   # Tipos de dato y nulos finales por columna

# ------------------------------------------------------------------------------------
# 9. PESTAÑAS TEMÁTICAS (una por KPI, nombradas con el tema técnico)
# ------------------------------------------------------------------------------------
tabs = st.tabs([
    "📈 Ticket Neto · Tendencia Central y Dispersión",
    "🧩 Mix de Carta · Segmentación y Cruces",
    "🚦 Tasa de Anulación · Semáforo de KPI",
    "💰 Ingreso Neto · Agregación y Crecimiento",
    "🔗 Comensales vs Ticket · Correlación",
    "🧪 Brecha de Ticket · Pruebas de Hipótesis",
])

# ===================== PESTAÑA 1 · ESTADÍSTICA DESCRIPTIVA =====================
with tabs[0]:
    st.subheader("KPI 1 · Ticket Neto Mediano por Precuenta")
    st.info("**Pregunta SMART:** ¿Cuál es el ticket neto mediano por precuenta en cada tienda durante el periodo "
            "filtrado y cuántas tiendas superan un CV del 35%, para priorizar en el próximo trimestre la "
            "estandarización de la carta y los precios en las tiendas más heterogéneas?")
    st.caption(f"Se excluyen {len(pre_f) - len(pv):,} precuentas con ticket ≤ 0 (anulaciones totales). "
               "Se usa la mediana como KPI porque la distribución del ticket es asimétrica.")

    def resumen_serie(s):                                            # Calcula todas las medidas descriptivas de una serie
        q1, q3, iqr, li, ls = tukey(s)                               # Cuartiles y límites de Tukey
        media, desv = s.mean(), s.std()                              # Media y desviación estándar muestral
        return pd.Series({
            "n": len(s), "media": media, "mediana": s.median(),      # Tendencia central
            "moda": s.round(0).mode().iloc[0],                       # Moda (valor redondeado a la unidad)
            "desv_est": desv, "cv_pct": desv / media * 100,          # Dispersión absoluta y relativa
            "q1": q1, "q3": q3, "iqr": iqr, "p90": s.quantile(0.9),  # Posición
            "min": s.min(), "max": s.max(),                          # Extremos
            "asimetria": s.skew(), "curtosis": s.kurt(),             # Forma de la distribución
            "pct_outliers": ((s < li) | (s > ls)).mean() * 100,      # % de valores atípicos (Tukey)
            "bigote_inf": s[s >= li].min(), "bigote_sup": s[s <= ls].max()})  # Extremos del boxplot

    tabla = pv.groupby("id_tienda")["ticket_neto"].apply(resumen_serie).unstack()  # Resumen por tienda
    tabla.loc["TOTAL (selección)"] = resumen_serie(pv["ticket_neto"])  # Fila global
    cols_vista = ["n", "media", "mediana", "moda", "desv_est", "cv_pct", "q1", "q3", "iqr", "p90",
                  "min", "max", "asimetria", "curtosis", "pct_outliers"]  # Columnas visibles
    mostrar_tabla(tabla[cols_vista].round(2))                        # Tabla de estadísticos descriptivos

    sesgo = "positivo (cola larga a la derecha: la media supera a la mediana)" \
        if pv["ticket_neto"].mean() > pv["ticket_neto"].median() else "nulo o negativo"
    n_cv = int((tabla.drop(index="TOTAL (selección)")["cv_pct"] > 35).sum())  # Tiendas con CV > 35%
    st.markdown(f"**Lectura:** sesgo {sesgo}. **{n_cv}** de {len(tabla) - 1} tiendas presentan CV > 35% "
                "(alta heterogeneidad del ticket).")

    c1, c2 = st.columns(2)                                           # Dos columnas de gráficos
    with c1:
        tope = pv["ticket_neto"].quantile(0.99)                      # Se recorta en P99 para ver bien la forma
        conteo, bordes = np.histogram(pv["ticket_neto"].clip(upper=tope), bins=60)  # Histograma numérico
        centros = (bordes[:-1] + bordes[1:]) / 2                     # Punto medio de cada intervalo
        fig = go.Figure(go.Bar(x=centros, y=conteo, marker_color=COLOR_PRIMARIO, name="Histograma"))  # Barras
        fig.add_trace(go.Scatter(x=centros, y=conteo, mode="lines", line=dict(color=COLOR_SECUNDARIO, width=3),
                                 name="Polígono de frecuencias"))    # Línea que une los puntos medios
        fig.add_vline(x=pv["ticket_neto"].mean(), line_dash="dash", line_color="black", annotation_text="Media")
        fig.add_vline(x=pv["ticket_neto"].median(), line_dash="dot", line_color="green", annotation_text="Mediana")
        fig.update_layout(title="Distribución del ticket neto (recortada en P99)", xaxis_title="Ticket neto",
                          yaxis_title="Precuentas")
        st.plotly_chart(estilo(fig))                                 # Dibuja el gráfico en Streamlit
    with c2:
        fig = go.Figure()                                            # Boxplot construido con cuartiles precalculados
        for t, fila in tabla.drop(index="TOTAL (selección)").iterrows():  # Una caja por tienda
            fig.add_trace(go.Box(x=[t], q1=[fila["q1"]], median=[fila["mediana"]], q3=[fila["q3"]],
                                 lowerfence=[fila["bigote_inf"]], upperfence=[fila["bigote_sup"]],
                                 mean=[fila["media"]], marker_color=COLOR_PRIMARIO, showlegend=False))
        fig.update_layout(title="Diagrama de caja por tienda (sin atípicos)", yaxis_title="Ticket neto")
        st.plotly_chart(estilo(fig))

    c3, c4 = st.columns(2)
    with c3:
        cv_df = tabla.drop(index="TOTAL (selección)")["cv_pct"].reset_index()  # CV por tienda
        cv_df.columns = ["tienda", "cv"]                             # Nombres claros
        cv_df["nivel"] = pd.cut(cv_df["cv"], [-np.inf, 15, 35, np.inf],
                                labels=["Baja (<15%)", "Moderada (15–35%)", "Alta (>35%)"])  # Regla práctica del CV
        fig = px.bar(cv_df, x="tienda", y="cv", color="nivel",
                     color_discrete_map={"Baja (<15%)": "#2E8B57", "Moderada (15–35%)": COLOR_SECUNDARIO,
                                         "Alta (>35%)": COLOR_PRIMARIO},
                     title="Coeficiente de variación del ticket por tienda")
        fig.add_hline(y=35, line_dash="dash", annotation_text="Umbral 35%")
        st.plotly_chart(estilo(fig))
    with c4:
        bordes_f = np.unique(np.round(pv["ticket_neto"].quantile([0, .2, .4, .6, .8, .95, 1]).values, 2))  # Cortes por cuantiles
        fr = pd.cut(pv["ticket_neto"], bins=bordes_f, include_lowest=True).value_counts().sort_index()  # Frecuencia absoluta
        tf = pd.DataFrame({"Intervalo de ticket": fr.index.astype(str), "Frec. absoluta": fr.values})  # Tabla base
        tf["Frec. relativa (%)"] = (tf["Frec. absoluta"] / tf["Frec. absoluta"].sum() * 100).round(2)  # Relativa
        tf["Frec. acumulada (%)"] = tf["Frec. relativa (%)"].cumsum().round(2)  # Acumulada
        st.markdown("**Distribución de frecuencias del ticket**")
        mostrar_tabla(tf)
        _, _, iqr_g, li_g, ls_g = tukey(pv["ticket_neto"])           # Límites globales de Tukey
        st.caption(f"Método IQR (Tukey): IQR = {iqr_g:,.2f}; atípico si ticket < {li_g:,.2f} o > {ls_g:,.2f}. "
                   "No se eliminan: se investigan y se reportan análisis con y sin ellos.")

# ===================== PESTAÑA 2 · SEGMENTACIÓN, AGRUPACIÓN Y CRUCES =====================
with tabs[1]:
    st.subheader("KPI 2 · Concentración del Ingreso en las 3 Categorías Líderes")
    st.info("**Pregunta SMART:** ¿Qué porcentaje del ingreso neto de cada tienda proviene de sus 3 categorías "
            "principales y en qué cuartil de ingreso se ubica cada tienda, para focalizar el mix de carta y las "
            "promociones del próximo trimestre?")
    ing_t = cat_f.groupby("id_tienda")["total_detalle_neto"].sum()   # Ingreso neto por tienda
    top3_pct = cat_f.groupby("id_tienda")["total_detalle_neto"].apply(
        lambda s: s.nlargest(3).sum() / s.sum() * 100)               # % del ingreso de las 3 primeras categorías
    top_filas = cat_f.sort_values("total_detalle_neto", ascending=False).groupby("id_tienda").head(3)  # Top-3 por tienda
    nombres_top = top_filas.groupby("id_tienda")["categoria_familia"].agg(lambda s: ", ".join(s))  # Nombres de las 3
    seg = pd.DataFrame({"Ingreso neto": ing_t, "Participación (%)": ing_t / ing_t.sum() * 100,
                        "Top-3 categorías (%)": top3_pct, "Categorías líderes": nombres_top})  # Tabla integrada
    if len(ing_t) >= 4:                                              # Los cuartiles requieren al menos 4 tiendas
        seg["Cuartil de ingreso"] = pd.qcut(ing_t.rank(method="first"), 4,
                                            labels=["Q1 · Bajo", "Q2 · Medio-bajo", "Q3 · Medio-alto", "Q4 · Alto"])
    else:
        st.caption("Se necesitan ≥ 4 tiendas seleccionadas para segmentar por cuartiles.")
    mostrar_tabla(seg.sort_values("Ingreso neto", ascending=False).round(2))  # Tabla ordenada por ingreso

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(top3_pct.reset_index(), x="id_tienda", y="total_detalle_neto",
                     labels={"id_tienda": "Tienda", "total_detalle_neto": "% del ingreso en Top-3"},
                     title="Concentración Top-3 de categorías por tienda", color_discrete_sequence=[COLOR_PRIMARIO])
        st.plotly_chart(estilo(fig))
    with c2:
        par = tot_cat.reset_index()                                  # Diagrama de Pareto de categorías
        par.columns = ["categoria", "ingreso"]
        par = par.head(15)                                           # 15 categorías principales
        par["acum"] = par["ingreso"].cumsum() / tot_cat.sum() * 100  # % acumulado sobre el total
        fig = go.Figure(go.Bar(x=par["categoria"], y=par["ingreso"], marker_color=COLOR_PRIMARIO, name="Ingreso"))
        fig.add_trace(go.Scatter(x=par["categoria"], y=par["acum"], yaxis="y2", mode="lines+markers",
                                 line=dict(color=COLOR_SECUNDARIO), name="% acumulado"))
        fig.update_layout(title="Pareto de categorías (Top 15)",
                          yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="% acumulado"))
        st.plotly_chart(estilo(fig))

    piv = cat_f.pivot_table(index="id_tienda", columns="categoria_familia", values="total_detalle_neto",
                            aggfunc="sum", fill_value=0)             # Tabla dinámica tienda × categoría
    piv_pct = piv.div(piv.sum(axis=1), axis=0) * 100                 # % por fila (composición del ingreso por tienda)
    fig = px.imshow(piv_pct, aspect="auto", color_continuous_scale="Reds",
                    labels=dict(color="% del ingreso"), title="Tabla dinámica: composición del ingreso por tienda × categoría")
    st.plotly_chart(estilo(fig, 430))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Segmentación por desviación estándar (μ ± σ) del ticket**")
        mu, sd = pv["ticket_neto"].mean(), pv["ticket_neto"].std()   # Media y desviación del ticket
        cortes = [-np.inf, mu - sd, mu, mu + sd, mu + 2 * sd, np.inf]  # Fronteras de los segmentos
        etiquetas = ["Crítico (< μ−1σ)", "Riesgo [μ−1σ, μ)", "Esperado [μ, μ+1σ)",
                     "Alta demanda [μ+1σ, μ+2σ)", "Saturado (≥ μ+2σ)"]  # Nombres diagnósticos
        sg = pd.cut(pv["ticket_neto"], bins=cortes, labels=etiquetas, right=False).value_counts().reindex(etiquetas)
        sg = pd.DataFrame({"Segmento": sg.index, "Precuentas": sg.values,
                           "%": (sg.values / sg.sum() * 100).round(2)})  # Tabla de resultados
        mostrar_tabla(sg)
        st.caption("Como el ticket no puede ser negativo, el segmento 'Crítico' puede quedar vacío (límite natural 0).")
    with c4:
        st.markdown("**Rangos personalizados de ticket**")
        txt = st.text_input("Límites separados por coma", "0,5,10,20,40,80")  # El usuario define sus propios cortes
        try:
            lims = sorted(set(float(x) for x in txt.split(",") if x.strip()))  # Convierte el texto a números únicos ordenados
            rp = pd.cut(pv["ticket_neto"], bins=lims + [np.inf], right=False).value_counts().sort_index()
            rp = pd.DataFrame({"Rango": rp.index.astype(str), "Precuentas": rp.values,
                               "%": (rp.values / rp.sum() * 100).round(2)})
            mostrar_tabla(rp)
        except ValueError:
            st.warning("Escribe números separados por coma, por ejemplo: 0,5,10,20")

    st.markdown("**Tabla cruzada (cross-tabulation): tienda × precuentas con descuento**")
    ct = pd.crosstab(pre_f["id_tienda"], pre_f["tiene_descuento"].map({True: "Con descuento", False: "Sin descuento"}),
                     margins=True, margins_name="Total")             # Frecuencias absolutas con totales
    ct_pct = (pd.crosstab(pre_f["id_tienda"], pre_f["tiene_descuento"].map({True: "Con descuento", False: "Sin descuento"}),
                          normalize="index") * 100).round(2)         # % por fila
    cc1, cc2 = st.columns(2)
    with cc1:
        mostrar_tabla(ct)
    with cc2:
        mostrar_tabla(ct_pct.add_suffix(" (%)"))
    ing_td = pre_f.pivot_table(index="id_tienda", columns="tipo_dia", values="ticket_neto", aggfunc="sum")  # Agrupación múltiple
    fig = px.bar(ing_td.reset_index().melt(id_vars="id_tienda", var_name="Tipo de día", value_name="Ingreso neto"),
                 x="id_tienda", y="Ingreso neto", color="Tipo de día", barmode="group",
                 color_discrete_sequence=[COLOR_PRIMARIO, COLOR_SECUNDARIO],
                 title="Agrupación múltiple: ingreso neto por tienda y tipo de día")
    st.plotly_chart(estilo(fig))

# ===================== PESTAÑA 3 · KPI CON FICHA TÉCNICA Y SEMÁFORO =====================
with tabs[2]:
    st.subheader("KPI 3 · Tasa de Anulación Posventa (%)")
    pregunta3 = (f"¿Qué porcentaje de las líneas de pedido de cada tienda corresponde a anulaciones (cantidad < 0) "
                 f"cada mes y qué tiendas superan la meta de {meta_anul:.1f}%, para activar una auditoría de "
                 "cajeros y procedimientos de anulación durante el mes siguiente?")
    st.info("**Pregunta SMART:** " + pregunta3)
    with st.expander("📋 Ficha técnica estándar del KPI (The KPI Institute)", expanded=False):
        ficha = pd.DataFrame([
            ("1. Identificación", "ID del KPI", "KPI-OPE-003"),
            ("1. Identificación", "Nombre del KPI", "% Tasa de Anulación Posventa"),
            ("2. Definición y Uso", "Definición conceptual", "Porcentaje de líneas de pedido con cantidad negativa sobre el total de líneas."),
            ("2. Definición y Uso", "Propósito estratégico", "Detectar fuga de ingresos y riesgo operativo en caja."),
            ("2. Definición y Uso", "Pregunta clave (KPQ) – SMART", pregunta3),
            ("3. Métodos de cálculo", "Fórmula matemática", "(Líneas con cantidad < 0 / Total de líneas) × 100"),
            ("3. Métodos de cálculo", "Unidad de medida", "Porcentaje (%)"),
            ("4. Enfoque e impacto", "Tipo de indicador", "Lagging (histórico) / Outcome"),
            ("4. Enfoque e impacto", "Perspectiva BSC", "Procesos internos"),
            ("5. Perfil de datos", "Fuente de origen", "dataset_pos_silver.parquet · columna es_anulacion"),
            ("5. Perfil de datos", "Frecuencia de reporte", "Mensual"),
            ("5. Perfil de datos", "Custodio del dato", "Analista de datos / administrador del POS"),
            ("6. Metas y rangos", "Meta objetivo", f"≤ {meta_anul:.1f}%"),
            ("6. Metas y rangos", "Semáforo (R/A/V)",
             f"Verde ≤ {meta_anul:.1f}% · Amarillo hasta {meta_anul * 1.5:.1f}% · Rojo > {meta_anul * 1.5:.1f}%"),
            ("7. Gobernanza", "Dueño del KPI (owner)", "Gerente de Operaciones"),
        ], columns=["Sección", "Campo técnico", "Especificación / contenido"])
        mostrar_tabla(ficha)

    por_t = por_mes_an.groupby("id_tienda")[["lineas", "anulaciones", "monto_anulado"]].sum()  # SUM por tienda
    por_t["tasa_pct"] = por_t["anulaciones"] / por_t["lineas"] * 100  # Tasa de anulación
    por_t["estado"] = por_t["tasa_pct"].apply(lambda v: estado_semaforo(v, meta_anul))  # Semáforo
    c1, c2 = st.columns([3, 2])
    with c1:
        colores = por_t["estado"].map(lambda e: "#2E8B57" if "🟢" in e else ("#E0A100" if "🟡" in e else "#B00020"))
        fig = go.Figure(go.Bar(x=por_t.index, y=por_t["tasa_pct"], marker_color=colores))
        fig.add_hline(y=meta_anul, line_dash="dash", annotation_text=f"Meta {meta_anul:.1f}%")
        fig.update_layout(title="Tasa de anulación por tienda (semáforo)", yaxis_title="% de líneas anuladas")
        st.plotly_chart(estilo(fig))
    with c2:
        mostrar_tabla(por_t.rename(columns={"lineas": "Líneas", "anulaciones": "Anulaciones",
                                            "monto_anulado": "Monto anulado", "tasa_pct": "Tasa %",
                                            "estado": "Estado"}).round(2))
    mm = por_mes_an.assign(tasa=por_mes_an["anulaciones"] / por_mes_an["lineas"] * 100)  # Tasa por tienda-mes
    hm = mm.pivot(index="id_tienda", columns="mes", values="tasa")   # Matriz tienda × mes
    hm.columns = [c.strftime("%Y-%m") for c in hm.columns]           # Etiquetas legibles de mes
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(color="Tasa %"),
                    title="Mapa de calor de la tasa de anulación (tienda × mes)")
    st.plotly_chart(estilo(fig, 360))
    st.markdown("**Diagnóstico por cajero (Top 10 con mayor tasa)**")
    min_l = st.number_input("Mínimo de líneas por cajero para ser evaluado", 50, 5000, 500, 50)  # Evita tasas inestables
    cj = por_cajero_an[por_cajero_an["lineas"] >= min_l].copy()      # Cajeros con volumen suficiente
    cj["tasa_pct"] = cj["anulaciones"] / cj["lineas"] * 100          # Tasa por cajero
    cj["estado"] = cj["tasa_pct"].apply(lambda v: estado_semaforo(v, meta_anul))  # Semáforo
    mostrar_tabla(cj.sort_values("tasa_pct", ascending=False).head(10).round(2))  # Top 10

# ===================== PESTAÑA 4 · AGREGACIÓN TEMPORAL Y CRECIMIENTO =====================
with tabs[3]:
    st.subheader("KPI 4 · Crecimiento Mensual del Ingreso Neto (%)")
    st.info(f"**Pregunta SMART:** ¿Cuál es la variación porcentual mensual del ingreso neto por tienda y cuáles "
            f"registran caídas superiores al {umbral_caida}% en meses completos, para reasignar las acciones "
            "comerciales y de dotación del siguiente mes?")
    ing_tm = (pre_f.groupby(["id_tienda", "mes"])
                   .agg(ingreso_neto=("ticket_neto", "sum"), precuentas=("id_precuenta", "count"),
                        ticket_prom=("ticket_neto", "mean")).reset_index())  # SUM, COUNT y AVG por tienda-mes
    piv_i = ing_tm.pivot(index="mes", columns="id_tienda", values="ingreso_neto").sort_index()  # mes × tienda
    mask_comp = completo.reindex(piv_i.index).fillna(False).astype(bool)  # Bandera de mes completo alineada
    piv_ok = piv_i.copy()
    piv_ok.loc[~mask_comp, :] = np.nan                               # Meses parciales no se comparan
    mom = ((piv_ok / piv_ok.shift(1) - 1) * 100).replace([np.inf, -np.inf], np.nan)  # Variación mensual %

    c1, c2 = st.columns([3, 2])
    with c1:
        tot_plot = piv_i.sum(axis=1)                                 # Ingreso total por mes (incluye parciales)
        colores = [COLOR_PRIMARIO if mask_comp.loc[m] else COLOR_NEUTRO for m in tot_plot.index]  # Gris = parcial
        fig = go.Figure(go.Bar(x=[m.strftime("%Y-%m") for m in tot_plot.index], y=tot_plot.values,
                               marker_color=colores))
        fig.update_layout(title="Ingreso neto mensual total (gris = mes parcial)", yaxis_title="Ingreso neto")
        st.plotly_chart(estilo(fig))
    with c2:
        d_tab = pd.DataFrame({"Días con datos": dias_datos, "Días del mes": dias_datos.index.days_in_month,
                              "Mes completo": completo})             # Control de completitud
        d_tab.index = d_tab.index.strftime("%Y-%m")
        mostrar_tabla(d_tab)

    hm = mom.T                                                       # tienda × mes
    hm.columns = [c.strftime("%Y-%m") for c in hm.columns]
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    labels=dict(color="MoM %"), title="Variación mensual del ingreso neto por tienda (%)")
    st.plotly_chart(estilo(fig, 360))

    caidas = mom.stack().reset_index()                               # Formato largo (se descartan NaN)
    caidas.columns = ["mes", "tienda", "variacion_pct"]
    caidas = caidas[caidas["variacion_pct"] < -umbral_caida].sort_values("variacion_pct")  # Solo caídas relevantes
    caidas["mes"] = caidas["mes"].dt.strftime("%Y-%m")
    st.markdown(f"**Alertas: tienda-mes con caída mayor al {umbral_caida}%** ({len(caidas)} casos)")
    mostrar_tabla(caidas.round(2))

    st.markdown("**Agregación por tienda (SUM · COUNT · AVG · GROUP BY)**")
    ag = pre_f.groupby("id_tienda").agg(ingreso_neto=("ticket_neto", "sum"), precuentas=("id_precuenta", "count"),
                                        ticket_prom=("ticket_neto", "mean"),
                                        comensales=("comensales", "sum")).reset_index()  # Cuatro agregaciones
    ag["ingreso_por_comensal"] = ag["ingreso_neto"] / ag["comensales"].replace(0, np.nan)  # Ratio derivado
    c3, c4 = st.columns(2)
    with c3:
        mostrar_tabla(ag.round(2))
    with c4:
        fig = px.bar(ag.sort_values("ingreso_neto"), x="ingreso_neto", y="id_tienda", orientation="h",
                     color_discrete_sequence=[COLOR_PRIMARIO], title="Ingreso neto acumulado por tienda")
        st.plotly_chart(estilo(fig))

# ===================== PESTAÑA 5 · CORRELACIÓN VS CAUSALIDAD =====================
with tabs[4]:
    st.subheader("KPI 5 · Correlación (Spearman) entre Comensales y Ticket Neto")
    st.info("**Pregunta SMART:** ¿Qué intensidad tiene la correlación monótona entre el número de comensales y el "
            "ticket neto por precuenta en el periodo filtrado y supera |ρ| = 0.50 (moderadamente fuerte), para "
            "justificar combos o promociones por tamaño de mesa en el próximo trimestre?")
    if "spearman_m" not in cr:
        st.warning("No hay suficientes precuentas válidas para calcular correlaciones.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Pearson (r) · lineal", f"{cr['pearson_r']:.3f}", f"{fuerza_correlacion(cr['pearson_r'])} · {fmt_p(cr['pearson_p'])}", delta_color="off")
        m2.metric("Spearman (ρ) · monótona", f"{cr['spearman_rho']:.3f}", f"{fuerza_correlacion(cr['spearman_rho'])} · {fmt_p(cr['spearman_p'])}", delta_color="off")
        m3.metric("Kendall (τ) · muestra 20 mil", f"{cr['kendall_tau']:.3f}", f"{fuerza_correlacion(cr['kendall_tau'])} · {fmt_p(cr['kendall_p'])}", delta_color="off")
        st.caption(f"Base: {cr['n']:,} precuentas válidas. Con tantos registros el p-valor será casi siempre ≈ 0: "
                   "la decisión debe apoyarse en la magnitud del coeficiente, no solo en la significancia.")
        dif = abs(cr["pearson_r"] - cr["spearman_rho"])              # Diferencia entre coeficientes
        if dif > 0.10:
            st.warning(f"Pearson y Spearman difieren en {dif:.2f}: hay sensibilidad a valores atípicos o relación no lineal. "
                       f"Pearson sin atípicos (Tukey) = {cr['pearson_sin_outliers']:.3f}.")
        else:
            st.success(f"Pearson y Spearman son coherentes (diferencia {dif:.2f}). "
                       f"Pearson sin atípicos = {cr['pearson_sin_outliers']:.3f}.")

        c1, c2 = st.columns(2)
        with c1:
            metodo = st.radio("Matriz de correlación", ["Pearson", "Spearman", "Kendall (muestra)"], horizontal=True)
            mat = {"Pearson": cr["pearson_m"], "Spearman": cr["spearman_m"], "Kendall (muestra)": cr["kendall_m"]}[metodo]
            fig_h, ax = plt.subplots(figsize=(6, 4.6))               # Figura de matplotlib
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                        linewidths=0.5, ax=ax)                       # Mapa de calor seaborn
            ax.set_title(f"Matriz de correlación · {metodo}")
            st.pyplot(fig_h)
            plt.close(fig_h)                                         # Libera memoria de la figura
        with c2:
            ms = cr["muestra_scatter"]                               # Muestra de 5 mil puntos
            fig = px.scatter(ms, x="comensales", y="ticket_neto", opacity=0.35,
                             color_discrete_sequence=[COLOR_PRIMARIO],
                             title="Comensales vs ticket neto (muestra de 5 000)")
            pend, inter = np.polyfit(ms["comensales"], ms["ticket_neto"], 1)  # Recta de mínimos cuadrados
            xs = np.array([ms["comensales"].min(), ms["comensales"].max()])
            fig.add_trace(go.Scatter(x=xs, y=pend * xs + inter, mode="lines", name="Tendencia lineal",
                                     line=dict(color=COLOR_SECUNDARIO, width=3)))
            st.plotly_chart(estilo(fig, 420))

        st.markdown("**Control de variables de confusión: Spearman dentro de cada tienda**")
        pt = cr["por_tienda"].reset_index()
        pt.columns = ["tienda", "rho"]
        fig = px.bar(pt, x="tienda", y="rho", color_discrete_sequence=[COLOR_PRIMARIO])
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="ρ = 0.50")
        fig.update_layout(yaxis_range=[-1, 1], title="Estratificación por tienda: ¿la relación se mantiene?")
        st.plotly_chart(estilo(fig, 320))
        st.warning("**Correlación ≠ causalidad.** Más comensales se asocia con mayor ticket, pero no prueba que "
                   "agregar personas lo cause: la hora, el tipo de día o la tienda pueden actuar como tercer factor. "
                   "Para afirmar causalidad se requiere secuencia temporal, mecanismo plausible y un experimento A/B.")
        st.markdown("**Escala estándar de interpretación (valor absoluto)**")
        mostrar_tabla(pd.DataFrame({
            "Rango |coeficiente|": ["0.00", "0.01 – 0.29", "0.30 – 0.49", "0.50 – 0.69", "0.70 – 0.99", "1.00"],
            "Fuerza": ["Nula", "Débil", "Moderada", "Moderadamente fuerte", "Fuerte / Alta", "Perfecta"],
            "Nota": ["Independencia", "Tendencia sutil", "Relación clara", "Patrón evidente",
                     "Una variable predice a la otra", "Rara en datos reales"]}))
        st.caption("Advertencias: el signo no resta fuerza; Kendall suele dar valores menores que Pearson/Spearman; "
                   "la exigencia del umbral depende de la disciplina.")

# ===================== PESTAÑA 6 · PRUEBAS DE HIPÓTESIS E INFERENCIA =====================
with tabs[5]:
    st.subheader("KPI 6 · Brecha Porcentual de Ticket Neto: Sábado y Domingo vs Lunes a Viernes")
    st.info(f"**Pregunta SMART:** ¿Difiere significativamente (α = {alpha}) el ticket neto promedio entre sábado–domingo "
            f"y lunes–viernes y cuál es la brecha porcentual con su IC al {conf:.0%}, para decidir el próximo "
            "trimestre una política de dotación y promociones diferenciada por tipo de día?")
    st.caption("H₀: μ(Sáb–Dom) = μ(Lun–Vie) · H₁: μ(Sáb–Dom) ≠ μ(Lun–Vie). Prueba bilateral por defecto.")
    if not res_b:
        st.warning("Se necesitan precuentas en ambos grupos de días para comparar.")
    else:
        r = res_b
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Ticket medio Sáb–Dom", f"{r['media_a']:,.2f}", f"n = {r['n_a']:,}", delta_color="off")
        a2.metric("Ticket medio Lun–Vie", f"{r['media_b']:,.2f}", f"n = {r['n_b']:,}", delta_color="off")
        a3.metric("Brecha", f"{r['brecha_pct']:+.2f}%", f"d de Cohen = {r['d']:.3f} ({etiqueta_efecto(r['d'])})", delta_color="off")
        a4.metric("t de Welch", f"{r['t']:.2f}", fmt_p(r["p"]), delta_color="off")
        sig = r["p"] <= alpha                                        # ¿Se rechaza H0?
        dire = "mayor" if r["dif"] > 0 else "menor"                  # Dirección del efecto
        st.markdown(
            f"**Redacción del informe:** {'Existe' if sig else 'No existe'} evidencia estadísticamente significativa "
            f"{'para rechazar' if sig else 'para rechazar'} H₀ (t({r['gl']:.0f}) = {r['t']:.2f}, {fmt_p(r['p'])}, α = {alpha}). "
            f"La diferencia de medias es {r['dif']:+.2f} (IC {conf:.0%}: {r['lo']:+.2f} a {r['hi']:+.2f}); el ticket de "
            f"sábado y domingo es {dire} que el de lunes a viernes. Mann-Whitney (robusto a atípicos): {fmt_p(r['p_u'])}.")
        if sig and etiqueta_efecto(r["d"]) in ("Trivial", "Pequeño"):
            st.warning("Paradoja de Big Data: la diferencia es estadísticamente significativa pero el efecto es "
                       f"{etiqueta_efecto(r['d']).lower()}. Evalúa su relevancia práctica antes de decidir.")
        fig = go.Figure()
        for nombre, (m_, lo_, hi_) in [("Sábado y Domingo", r["ic_a"]), ("Lunes a Viernes", r["ic_b"])]:
            fig.add_trace(go.Bar(x=[nombre], y=[m_], error_y=dict(type="data", array=[hi_ - m_], arrayminus=[m_ - lo_]),
                                 marker_color=COLOR_PRIMARIO if nombre.startswith("S") else COLOR_SECUNDARIO,
                                 name=nombre))
        fig.update_layout(title=f"Ticket neto medio con IC {conf:.0%}", yaxis_title="Ticket neto", showlegend=False)
        st.plotly_chart(estilo(fig, 340))

    st.markdown("---")
    st.markdown("### Prueba t de una muestra: ticket vs valor de referencia")
    x1 = pv["ticket_neto"].values                                    # Ticket de todas las cuentas válidas
    ref = st.number_input("Valor de referencia (ticket objetivo)", value=float(round(x1.mean() * 1.05, 2)), step=0.1)
    alt = st.radio("Tipo de prueba", ["Bilateral (≠)", "Unilateral izquierda (<)", "Unilateral derecha (>)"], horizontal=True)
    alt_map = {"Bilateral (≠)": "two-sided", "Unilateral izquierda (<)": "less", "Unilateral derecha (>)": "greater"}
    t1 = stats.ttest_1samp(x1, ref, alternative=alt_map[alt])        # Prueba t contra el valor de referencia
    m_, lo_, hi_ = ic_media(x1, conf)                                # IC de la media
    st.markdown(f"Media observada = **{m_:,.3f}** (IC {conf:.0%}: {lo_:,.3f} – {hi_:,.3f}) · "
                f"t({len(x1) - 1:,}) = **{t1.statistic:.2f}**, {fmt_p(t1.pvalue)} → "
                f"**{'se rechaza H₀' if t1.pvalue <= alpha else 'no se rechaza H₀'}** con α = {alpha}.")

    st.markdown("### Prueba t pareada: mismo grupo de tiendas en dos momentos")
    piv_t = pv.pivot_table(index="id_tienda", columns="mes", values="ticket_neto", aggfunc="mean")  # Ticket medio tienda × mes
    meses_ok = [m for m in piv_t.columns if bool(completo.get(m, False))]  # Solo meses completos
    if len(meses_ok) >= 2 and len(piv_t) >= 2:
        mA, mB = meses_ok[0], meses_ok[-1]                           # Primer y último mes completo
        par = piv_t[[mA, mB]].dropna()                               # Tiendas con dato en ambos momentos
        if len(par) >= 2:
            tp = stats.ttest_rel(par[mB], par[mA])                   # t pareada: después − antes
            st.markdown(f"Ticket medio por tienda, **{mA:%Y-%m}** → **{mB:%Y-%m}**: "
                        f"diferencia media = {(par[mB] - par[mA]).mean():+.3f} · t({len(par) - 1}) = {tp.statistic:.2f}, "
                        f"{fmt_p(tp.pvalue)} → **{'se rechaza H₀' if tp.pvalue <= alpha else 'no se rechaza H₀'}** (n = {len(par)} tiendas).")
        else:
            st.caption("Menos de 2 tiendas con datos en ambos meses.")
    else:
        st.caption("Se requieren al menos 2 meses completos y 2 tiendas en el filtro.")

    st.markdown("### Chi-cuadrado de independencia: descuento vs variable categórica")
    var_c = st.radio("Cruzar 'tiene descuento' con:", ["Tienda", "Tipo de día"], horizontal=True)
    col_c = {"Tienda": "id_tienda", "Tipo de día": "tipo_dia"}[var_c]
    ctab = pd.crosstab(pre_f[col_c], pre_f["tiene_descuento"])       # Tabla de contingencia observada
    if ctab.shape[0] >= 2 and ctab.shape[1] >= 2:
        chi2, p_chi, gl_chi, esp = stats.chi2_contingency(ctab)      # Estadístico, p, grados de libertad y esperados
        n_t = ctab.values.sum()
        v_cr = np.sqrt(chi2 / (n_t * (min(ctab.shape) - 1)))         # V de Cramér (tamaño del efecto)
        ok5 = (esp >= 5).mean() * 100                                # % de celdas con esperado ≥ 5 (supuesto)
        pct = (ctab.div(ctab.sum(axis=1), axis=0) * 100)             # % por fila
        st.markdown(f"χ²({gl_chi}) = **{chi2:,.1f}**, {fmt_p(p_chi)}, V de Cramér = **{v_cr:.3f}** → "
                    f"**{'se rechaza independencia' if p_chi <= alpha else 'no se rechaza independencia'}** (α = {alpha}). "
                    f"Celdas con esperado ≥ 5: {ok5:.0f}%.")
        if True in pct.columns:
            fig = px.bar(pct[True].reset_index(), x=col_c, y=True, color_discrete_sequence=[COLOR_PRIMARIO],
                         labels={True: "% de precuentas con descuento"}, title=f"% de cuentas con descuento según {var_c.lower()}")
            st.plotly_chart(estilo(fig, 320))
    else:
        st.caption("Se necesitan al menos 2 categorías y ambos estados de descuento.")

    st.markdown("### Chi-cuadrado de bondad de ajuste: ¿las cuentas se reparten según los días disponibles?")
    obs = pre_f.groupby("dia_num").size().reindex(range(7), fill_value=0)  # Cuentas observadas por día
    dias_dato = pre_f.groupby("dia_num")["fecha_transaccion"].nunique().reindex(range(7), fill_value=0)  # Días con datos
    if (dias_dato > 0).all():                                        # Requiere los 7 días presentes
        esp_d = dias_dato / dias_dato.sum() * obs.sum()              # Esperado: proporcional a los días disponibles
        chi_g, p_g = stats.chisquare(obs.values, f_exp=esp_d.values)  # Prueba de bondad de ajuste
        w = np.sqrt(chi_g / obs.sum())                               # Tamaño del efecto w de Cohen
        st.markdown(f"χ²(6) = **{chi_g:,.1f}**, {fmt_p(p_g)}, w = **{w:.3f}** → "
                    f"**{'la distribución difiere de la esperada' if p_g <= alpha else 'sin evidencia de diferencia'}**.")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=DIAS_NOMBRE, y=obs.values, name="Observado", marker_color=COLOR_PRIMARIO))
        fig.add_trace(go.Bar(x=DIAS_NOMBRE, y=esp_d.values, name="Esperado", marker_color=COLOR_SECUNDARIO))
        fig.update_layout(barmode="group", title="Precuentas por día: observado vs esperado")
        st.plotly_chart(estilo(fig, 320))
    else:
        st.caption("El filtro no contiene los siete días de la jornada; amplía el rango de fechas.")

    st.markdown("---")
    st.caption("Interpretación correcta del IC: si repitiéramos el muestreo muchas veces, el "
               f"{conf:.0%} de los intervalos construidos contendría el parámetro verdadero; el parámetro es fijo, "
               "el intervalo es el que varía. Con muestras masivas, reporta siempre tamaño del efecto e IC.")
