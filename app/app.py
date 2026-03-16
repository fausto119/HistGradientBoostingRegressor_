import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="📊 Simulador de Ventas Nov 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-label { color: rgba(255,255,255,0.9); font-size: 14px; }
    .metric-value { color: white; font-size: 32px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ========================= FUNCIONES AUXILIARES =========================

@st.cache_resource
def cargar_modelo_y_datos():
    """Carga el modelo entrenado y el dataframe de inferencia"""
    try:
        modelo = joblib.load('models/model_final.joblib')
        df_inferencia = pd.read_csv('data/procesed/inferencia_df.csv')
        return modelo, df_inferencia
    except Exception as e:
        st.error(f"❌ Error cargando archivos: {e}")
        return None, None

def obtener_productos_unicos(df):
    """Obtiene lista de productos únicos del dataframe"""
    if 'nombre' in df.columns:
        return sorted(df['nombre'].unique().tolist())
    return []

def actualizar_lags(row_actual, prediccion_anterior, lags_anteriores):
    """
    Actualiza los lags recursivamente para el siguiente día
    """
    nuevos_lags = {}
    
    # lag_1 toma el valor de la predicción anterior
    nuevos_lags['unidades_vendidas_lag_1'] = prediccion_anterior
    
    # Los demás lags se desplazan
    for i in range(2, 8):
        col_anterior = f'unidades_vendidas_lag_{i-1}'
        col_nuevo = f'unidades_vendidas_lag_{i}'
        if col_anterior in lags_anteriores:
            nuevos_lags[col_nuevo] = lags_anteriores[col_anterior]
        else:
            nuevos_lags[col_nuevo] = row_actual.get(col_nuevo, 0)
    
    return nuevos_lags

def actualizar_ma7(predicciones_ultimas_7):
    """Calcula la media móvil de 7 días con las últimas predicciones"""
    return np.mean(predicciones_ultimas_7[-7:]) if len(predicciones_ultimas_7) >= 1 else 0

def hacer_predicciones_recursivas(df_producto, modelo, descuento_pct, escenario_competencia):
    """
    Realiza predicciones recursivas día por día actualizando lags
    """
    df_trabajo = df_producto.copy().sort_values('fecha').reset_index(drop=True)
    predicciones = []
    detalles_diarios = []
    predicciones_ultimas_7 = []
    
    # Obtener columnas que el modelo necesita
    feature_columns = modelo.feature_names_in_
    
    for idx, row in df_trabajo.iterrows():
        # Preparar la fila para predicción
        row_pred = row.copy()
        
        # Aplicar descuento al precio_base si existe
        if 'precio_base' in row_pred.index:
            precio_venta = row_pred['precio_base'] * (1 - descuento_pct / 100)
            row_pred['precio_venta'] = precio_venta
        
        # Aplicar escenario de competencia
        if escenario_competencia == "Competencia -5%":
            row_pred['precio_competencia'] = row_pred.get('precio_competencia', 0) * 0.95

        elif escenario_competencia == "Competencia +5%":
            row_pred['precio_competencia'] = row_pred.get('precio_competencia', 0) * 1.05

        if row_pred.get('precio_competencia', 0) > 0:
             row_pred['ratio_precio'] = row_pred['precio_venta'] / row_pred['precio_competencia']
    
        # Recalcular ratio_precio si es necesario
        if 'precio_venta' in row_pred.index and 'precio_competencia' in row_pred.index:
            if row_pred['precio_competencia'] > 0:
                row_pred['ratio_precio'] = row_pred['precio_venta'] / row_pred['precio_competencia']
        
        # Actualizar descuento_porcentaje
        row_pred['porcentaje_descuento'] = descuento_pct
        
        # Actualizar media móvil
        row_pred['unidades_vendidas_ma_7'] = actualizar_ma7(predicciones_ultimas_7)
        
        # Seleccionar solo las columnas que el modelo necesita
        features_disponibles = [col for col in feature_columns if col in row_pred.index]
        X = row_pred[features_disponibles].values.reshape(1, -1)
        
        # Hacer predicción
        prediccion = modelo.predict(X)[0]
        prediccion = max(0, prediccion)  # Asegurar valores positivos
        
        predicciones.append(prediccion)
        predicciones_ultimas_7.append(prediccion)
        
        # Guardar detalles del día
        fecha_obj = pd.to_datetime(row_pred.get('fecha', datetime(2025, 11, 1)))
        dia_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo'][fecha_obj.weekday()]
        
        ingreso_diario = prediccion * row_pred.get('precio_venta', row_pred.get('precio_base', 0))
        
        detalles_diarios.append({
            'fecha': fecha_obj,
            'dia_semana': dia_semana,
            'precio_venta': row_pred.get('precio_venta', row_pred.get('precio_base', 0)),
            'precio_competencia': row_pred.get('precio_competencia', 0),
            'descuento_porcentaje': descuento_pct,
            'unidades_predichas': prediccion,
            'ingresos_diarios': ingreso_diario,
            'es_black_friday': row_pred.get('es_black_friday', 0) == 1
        })
        
        # Actualizar lags para el siguiente día si no es el último
        if idx < len(df_trabajo) - 1:
            lags_actuales = {col: row_pred[col] for col in row_pred.index if 'lag' in col}
            nuevos_lags = actualizar_lags(row_pred, prediccion, lags_actuales)
            
            # Aplicar los nuevos lags a la siguiente fila
            for col, valor in nuevos_lags.items():
                if col in df_trabajo.columns:
                    df_trabajo.at[idx + 1, col] = valor
    
    return predicciones, pd.DataFrame(detalles_diarios)

# ========================= CARGAR DATOS =========================
modelo, df_inferencia = cargar_modelo_y_datos()

if modelo is None or df_inferencia is None:
    st.error("No se pueden cargar los archivos necesarios")
    st.stop()

# ========================= SIDEBAR =========================
st.sidebar.markdown("## 🎮 Controles de Simulación")

# Obtener productos únicos
productos = obtener_productos_unicos(df_inferencia)

if not productos:
    st.error("No se encontraron productos en el dataset")
    st.stop()

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "📦 Selecciona un producto",
    productos,
    index=0
)

# Slider de descuento
descuento = st.sidebar.slider(
    "💰 Ajuste de descuento (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    help="Ajusta el descuento sobre el precio base (-50% a +50%)"
)

# Selector de escenario competencia
st.sidebar.markdown("### 🏪 Escenario de Competencia")
escenario = st.sidebar.radio(
    "Selecciona un escenario:",
    options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    help="Simula cambios en los precios de la competencia"
)

# Botón para simular
simular = st.sidebar.button("🚀 Simular Ventas", use_container_width=True, type="primary")

# ========================= ÁREA PRINCIPAL =========================

st.markdown("# 📊 Simulador de Predicciones - Noviembre 2025")
st.markdown(f"### 📈 Análisis para: **{producto_seleccionado}**")
st.divider()

if simular:
    # Filtrar datos para el producto seleccionado
    df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
    
    if df_producto.empty:
        st.error(f"No hay datos disponibles para {producto_seleccionado}")
    else:
        with st.spinner("⏳ Procesando predicciones recursivas..."):
            # Hacer predicciones recursivas
            predicciones, df_detalles = hacer_predicciones_recursivas(
                df_producto, 
                modelo, 
                descuento, 
                escenario
            )
        
        # Calcular KPIs
        unidades_totales = sum(predicciones)
        ingresos_totales = df_detalles['ingresos_diarios'].sum()
        precio_promedio = df_detalles['precio_venta'].mean()
        descuento_promedio = df_detalles['descuento_porcentaje'].mean()
        
        # Mostrar KPIs
        st.markdown("### 📊 KPIs Principales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📦 Unidades Totales",
                f"{unidades_totales:,.0f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "💵 Ingresos Proyectados",
                f"€{ingresos_totales:,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "💲 Precio Promedio",
                f"€{precio_promedio:,.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                "🏷️ Descuento Promedio",
                f"{descuento_promedio:.1f}%",
                delta=None
            )
        
        st.divider()
        
        # Gráfico de predicción diaria
        st.markdown("### 📈 Predicción Diaria de Ventas")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Datos para el gráfico
        dias = range(1, len(predicciones) + 1)
        
        # Línea principal
        ax.plot(dias, predicciones, linewidth=2.5, color='#667eea', marker='o', markersize=5)
        ax.fill_between(dias, predicciones, alpha=0.2, color='#667eea')
        
        # Marcar Black Friday (día 28)
        if len(predicciones) >= 28:
            ax.axvline(x=28, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(28, predicciones[27], 'o', color='#FF6B6B', markersize=12, label='Black Friday')
            ax.annotate('🎉 Black Friday', xy=(28, predicciones[27]), xytext=(28, predicciones[27] * 1.15),
                       ha='center', fontsize=11, fontweight='bold', color='#FF6B6B',
                       arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5))
        
        ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title('Proyección de Ventas Diarias - Noviembre 2025', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(range(1, 31, 2))
        sns.set_style("whitegrid")
        plt.tight_layout()
        
        st.pyplot(fig)
        
        st.divider()
        
        # Tabla detallada
        st.markdown("### 📋 Detalle Diario de Predicciones")
        
        df_tabla = df_detalles.copy()
        df_tabla['fecha'] = pd.to_datetime(df_tabla['fecha']).dt.strftime('%d/%m/%Y')
        df_tabla['precio_venta'] = df_tabla['precio_venta'].apply(lambda x: f"€{x:,.2f}")
        df_tabla['precio_competencia'] = df_tabla['precio_competencia'].apply(lambda x: f"€{x:,.2f}")
        df_tabla['descuento_porcentaje'] = df_tabla['descuento_porcentaje'].apply(lambda x: f"{x:.1f}%")
        df_tabla['unidades_predichas'] = df_tabla['unidades_predichas'].apply(lambda x: f"{x:,.0f}")
        df_tabla['ingresos_diarios'] = df_tabla['ingresos_diarios'].apply(lambda x: f"€{x:,.2f}")
        
        df_tabla = df_tabla.rename(columns={
            'fecha': 'Fecha',
            'dia_semana': 'Día',
            'precio_venta': 'Precio Venta',
            'precio_competencia': 'Precio Competencia',
            'descuento_porcentaje': 'Descuento',
            'unidades_predichas': 'Unidades',
            'ingresos_diarios': 'Ingresos'
        })
        
        # Destacar Black Friday
        def highlight_black_friday(row):
            if df_detalles.loc[df_tabla.index.get_loc(row.name), 'es_black_friday']:
                return ['background-color: #FFE5E5'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            df_tabla.drop('es_black_friday', axis=1, errors='ignore'),
            use_container_width=True,
            height=500
        )
        
        st.divider()
        
        # Comparativa de escenarios
        st.markdown("### 🔄 Comparativa de Escenarios de Competencia")
        
        col1, col2, col3 = st.columns(3)
        
        escenarios_nombres = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        escenarios_data = []
        
        for esc in escenarios_nombres:
            with st.spinner(f"Calculando escenario: {esc}"):
                _, df_temp = hacer_predicciones_recursivas(df_producto, modelo, descuento, esc)
                unidades = df_temp['unidades_predichas'].sum()
                ingresos = df_temp['ingresos_diarios'].sum()
                escenarios_data.append({'unidades': unidades, 'ingresos': ingresos})
        
        columnas = [col1, col2, col3]
        for idx, (col, nombre) in enumerate(zip(columnas, escenarios_nombres)):
            with col:
                st.markdown(f"#### {nombre}")
                st.metric("Unidades", f"{escenarios_data[idx]['unidades']:,.0f}")
                st.metric("Ingresos", f"€{escenarios_data[idx]['ingresos']:,.2f}")
        
        st.success("✅ Simulación completada correctamente")

else:
    st.info("👉 Configura los parámetros en el sidebar y haz clic en 'Simular Ventas' para comenzar")