import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from pathlib import Path


st.set_page_config(page_title="Simulación Ventas — Noviembre 2025", layout="wide")

PALETTE_PRIMARY = "#667eea"
PALETTE_ACCENT = "#764ba2"


@st.cache_data
def load_inferencia(path: str = None):
	# Resolver ruta: si la ruta dada no existe, buscarla relativa a la raíz del repo
	if path:
		p = Path(path)
		if not p.exists():
			repo_root = Path(__file__).resolve().parents[1]
			p = repo_root / path
	else:
		repo_root = Path(__file__).resolve().parents[1]
		p = repo_root / "data" / "processed" / "inferencia_df_transformado.csv"

	if not p.exists():
		raise FileNotFoundError(f"Archivo de inferencia no encontrado en: {p}")

	df = pd.read_csv(p, parse_dates=["fecha"]) 
	return df


@st.cache_resource
def load_model(path: str = None):
	if path:
		p = Path(path)
		if not p.exists():
			repo_root = Path(__file__).resolve().parents[1]
			p = repo_root / path
	else:
		repo_root = Path(__file__).resolve().parents[1]
		p = repo_root / "models" / "modelo_final.joblib"

	if not p.exists():
		raise FileNotFoundError(f"Modelo no encontrado en: {p}")

	return joblib.load(p)


def format_eur(x):
	try:
		return f"€{x:,.2f}"
	except Exception:
		return x


def simulate_recursive(df: pd.DataFrame, model, features: list):
	df = df.sort_values("fecha").reset_index(drop=True).copy()

	lag_cols = [f"unidades_vendidas_lag{i}" for i in range(1, 8)]
	# soportar ambas variantes de nombre de media móvil (ma7 o mm7)
	mm_candidates = ["unidades_vendidas_ma7", "unidades_vendidas_mm7"]
	mm_col = next((c for c in mm_candidates if c in df.columns), None)
	if mm_col is None:
		raise KeyError(f"Falta columna obligatoria de media móvil: {mm_candidates}")

	for c in lag_cols + [mm_col]:
		if c not in df.columns:
			raise KeyError(f"Falta columna obligatoria en el dataframe: {c}")

	# Inicializar last_lags desde el día 1 (vienen de octubre)
	last_lags = [float(x) if pd.notna(x) else 0.0 for x in df.loc[0, lag_cols].values]

	preds = []

	n = len(df)
	for i in range(n):
		# Para día 1 mantenemos los lags tal cual. Para días posteriores, actualizamos los lags con predicciones previas
		if i > 0:
			for k in range(1, 8):
				df.at[i, f"unidades_vendidas_lag{k}"] = last_lags[k - 1]
			# actualizar la columna real de media móvil encontrada
			df.at[i, mm_col] = float(np.mean(last_lags))

		# Comprobar que las columnas de features estén presentes
		missing = [f for f in features if f not in df.columns]
		if missing:
			raise KeyError(f"El modelo requiere columnas ausentes: {missing}")

		X_row = df.loc[i, features].to_numpy().reshape(1, -1)
		# Asegurar tipo numérico
		X_row = X_row.astype(float)

		pred = float(model.predict(X_row)[0])
		preds.append(pred)

		df.at[i, "pred_unidades"] = pred
		df.at[i, "ingresos_pred"] = pred * float(df.at[i, "precio_venta"]) if pd.notna(df.at[i, "precio_venta"]) else 0.0

		# actualizar last_lags para el siguiente día
		last_lags = [pred] + last_lags[:6]

	return df


def main():
	st.markdown(f"<h1 style='color:{PALETTE_ACCENT};'>📈 Simulación Ventas — Noviembre 2025</h1>", unsafe_allow_html=True)

	# Cargar recursos
	try:
		with st.spinner("Cargando datos y modelo..."):
			inferencia_df = load_inferencia("data/processed/inferencia_df_transformado.csv")
			model = load_model("models/modelo_final.joblib")
	except Exception as e:
		st.error(f"Error cargando datos o modelo: {e}")
		st.stop()

	# Controles en sidebar
	st.sidebar.title("Controles de Simulación")

	productos = inferencia_df["nombre"].unique().tolist()
	producto = st.sidebar.selectbox("Producto", productos)

	ajuste_desc = st.sidebar.slider("Ajuste de descuento (%)", -50, 50, 0, step=5)
	escenario = st.sidebar.radio("Escenario de competencia", ["Actual (0%)", "Competencia -5%", "Competencia +5%"], index=0)

	simular = st.sidebar.button("Simular Ventas")

	st.sidebar.markdown("---")
	st.sidebar.markdown("Selecciona controles y pulsa **Simular Ventas**")

	# Mostrar información básica
	st.markdown("---")
	st.write(f"**Producto seleccionado:** {producto}")

	if not simular:
		st.info("Ajusta los controles en la barra lateral y pulsa 'Simular Ventas' para ejecutar la predicción recursiva.")
		return

	# Copiar y filtrar
	df = inferencia_df[inferencia_df["nombre"] == producto].copy()
	if df.shape[0] == 0:
		st.error("No hay datos para el producto seleccionado.")
		return

	# Recalcular precios según controles
	descuento_frac = ajuste_desc / 100.0
	df["precio_venta"] = df["precio_base"] * (1 - descuento_frac)
	# recomputar porc_descuento respecto a precio_base
	df["porc_descuento"] = np.where(df["precio_base"] > 0, 1 - df["precio_venta"] / df["precio_base"], 0)

	# Ajustar competencia
	escenario_map = {"Actual (0%)": 0.0, "Competencia -5%": -0.05, "Competencia +5%": 0.05}
	pct = escenario_map.get(escenario, 0.0)

	comp_cols = [c for c in ["Amazon", "Decathlon", "Deporvillage"] if c in df.columns]
	if comp_cols:
		df[comp_cols] = df[comp_cols] * (1 + pct)
		df["precio_competencia"] = df[comp_cols].mean(axis=1)
	else:
		if "precio_competencia" in df.columns:
			df["precio_competencia"] = df["precio_competencia"] * (1 + pct)
		else:
			st.error("No se encontró columna 'precio_competencia' ni columnas de competencia individuales.")
			st.stop()

	# recalcular ratio precio
	with np.errstate(divide="ignore", invalid="ignore"):
		df["ratioprecio"] = df["precio_venta"] / df["precio_competencia"]
		df["ratioprecio"].replace([np.inf, -np.inf], np.nan, inplace=True)

	# Preparar features del modelo y validaciones
	try:
		feature_names = list(model.feature_names_in_)
	except Exception:
		st.error("El modelo cargado no expone 'feature_names_in_'. Asegúrate que fue entrenado con scikit-learn >=1.0.")
		st.stop()

	missing_cols = [f for f in feature_names if f not in df.columns]
	if missing_cols:
		st.error(f"El dataframe no contiene todas las columnas que el modelo espera: {missing_cols}")
		st.stop()

	# Ejecutar predicción recursiva con spinner
	with st.spinner("Ejecutando predicción recursiva día a día... Esto puede tardar unos segundos."):
		try:
			resultado = simulate_recursive(df, model, feature_names)
		except Exception as e:
			st.error(f"Error durante la simulación recursiva: {e}")
			st.stop()

	# KPIs
	total_unidades = resultado["pred_unidades"].sum()
	total_ingresos = resultado["ingresos_pred"].sum()
	precio_promedio = resultado["precio_venta"].mean()
	descuento_promedio = resultado["porc_descuento"].mean()

	k1, k2, k3, k4 = st.columns(4)
	k1.metric("Unidades totales proyectadas", f"{total_unidades:,.0f}")
	k2.metric("Ingresos proyectados", format_eur(total_ingresos))
	k3.metric("Precio medio de venta", format_eur(precio_promedio))
	k4.metric("Descuento promedio", f"{descuento_promedio:.0%}")

	st.markdown("---")

	# Gráfico diario con seaborn
	fig, ax = plt.subplots(figsize=(10, 4))
	sns.lineplot(data=resultado, x="dia", y="pred_unidades", color=PALETTE_ACCENT, ax=ax)
	ax.set_xlabel("Día de noviembre")
	ax.set_ylabel("Unidades vendidas (predicción)")
	ax.set_title(f"Predicción diaria — {producto}")
	# Marcar Black Friday (día 28)
	if (resultado["dia"] == 28).any():
		bf_val = float(resultado.loc[resultado["dia"] == 28, "pred_unidades"].iloc[0])
		ax.axvline(28, color="red", linestyle="--", linewidth=1.2)
		ax.scatter([28], [bf_val], color="red", s=80, zorder=5)
		ax.annotate("Black Friday 🔥", xy=(28, bf_val), xytext=(28, bf_val * 1.05), color="red")

	st.pyplot(fig)

	st.markdown("---")

	# Tabla detallada
	display_cols = ["fecha", "dia_semana_nombre", "precio_venta", "precio_competencia", "porc_descuento", "pred_unidades", "ingresos_pred", "dia"]
	table_df = resultado[display_cols].copy()
	table_df["precio_venta"] = table_df["precio_venta"].round(2)
	table_df["precio_competencia"] = table_df["precio_competencia"].round(2)
	table_df["pred_unidades"] = table_df["pred_unidades"].round(0)
	table_df["ingresos_pred"] = table_df["ingresos_pred"].round(2)

	# Añadir columna de evento para Black Friday
	table_df["evento"] = table_df["dia"].apply(lambda x: "🔥 Black Friday" if x == 28 else "")

	# Formateo con Styler
	styled = (
		table_df.style
		.format({"precio_venta": "€{:.2f}", "precio_competencia": "€{:.2f}", "porc_descuento": "{:.0%}", "pred_unidades": "{:.0f}", "ingresos_pred": "€{:.2f}"})
		.apply(lambda row: ["background-color:#fff2cc" if row["dia"] == 28 else "" for _ in row], axis=1)
	)

	st.write("**Detalle diario (resaltado Black Friday)**")
	st.write(styled)

	st.markdown("---")

	# Comparativa de escenarios (manteniendo descuento)
	st.write("**Comparativa de escenarios de competencia**")
	escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
	resultados_esc = []
	for esc in escenarios:
		tmp = inferencia_df[inferencia_df["nombre"] == producto].copy()
		tmp["precio_venta"] = tmp["precio_base"] * (1 - descuento_frac)
		pct_esc = escenario_map.get(esc, 0.0)
		if comp_cols:
			tmp[comp_cols] = tmp[comp_cols] * (1 + pct_esc)
			tmp["precio_competencia"] = tmp[comp_cols].mean(axis=1)
		else:
			tmp["precio_competencia"] = tmp["precio_competencia"] * (1 + pct_esc)
		with np.errstate(divide="ignore", invalid="ignore"):
			tmp["ratioprecio"] = tmp["precio_venta"] / tmp["precio_competencia"]
		try:
			out = simulate_recursive(tmp, model, feature_names)
		except Exception as e:
			st.error(f"Error en simulación comparativa ({esc}): {e}")
			return
		resultados_esc.append((esc, out["pred_unidades"].sum(), out["ingresos_pred"].sum()))

	c1, c2, c3 = st.columns(3)
	for col, res in zip((c1, c2, c3), resultados_esc):
		esc_name, unidades_tot, ingresos_tot = res
		col.metric(esc_name, f"{unidades_tot:,.0f}", format_eur(ingresos_tot))

	st.markdown("---")
	st.info("Simulación completada. Puedes cambiar controles y volver a simular.")


if __name__ == "__main__":
	main()

