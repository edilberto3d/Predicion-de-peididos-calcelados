# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

# 1. Inicialización de la aplicación Flask
app = Flask(__name__)

# Configurar logging para monitorear la aplicación en producción
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 2. Carga del modelo entrenado (.pkl)
#    Asegúrate de haber guardado el modelo correcto (el 'modelo_rf_basico').
try:

    modelo_cargado = joblib.load('modelo_random_forest_entrenado.pkl')
    app.logger.info("✅ Modelo de clasificación cargado correctamente.")
except FileNotFoundError:
    app.logger.error("❌ Error Crítico: No se encontró el archivo 'modelo_cancelacion_basico.pkl'.")
    app.logger.error("Asegúrate de que el archivo del modelo esté en el mismo directorio que app.py.")
    modelo_cargado = None
except Exception as e:
    app.logger.error(f"❌ Error inesperado al cargar el modelo: {e}")
    modelo_cargado = None

# 3. Definición de las características que el modelo espera
#    Estas deben coincidir exactamente con las usadas en el entrenamiento.
CARACTERISTICAS_ESPERADAS = [
    'num__total_a_pagar',
    'num__total_cantidad_productos',
    'num__total_productos_distintos',
    'num__stock_minimo_del_pedido',
    'num__total_categorias_distintas',
    'num__tasa_cancelaciones_historicas_cliente'
]

# 4. Creación del endpoint de predicción
@app.route('/predecir', methods=['POST'])
def predecir_cancelacion():
    
    if modelo_cargado is None:
        app.logger.error("Modelo no disponible para predicciones.")
        return jsonify({'error': 'El modelo de predicción no está disponible. Revisa los logs del servidor.'}), 503

    data = request.get_json()
    if not data:
        app.logger.warning("No se recibieron datos en la petición.")
        return jsonify({'error': 'No se recibieron datos en la petición. Se esperaba un JSON.'}), 400

    app.logger.info(f"Petición recibida para predicción: {data}")

    if not all(feature in data for feature in CARACTERISTICAS_ESPERADAS):
        faltantes = [feature for feature in CARACTERISTICAS_ESPERADAS if feature not in data]
        app.logger.warning(f"Petición inválida. Faltan características: {faltantes}")
        return jsonify({
            'error': 'Petición inválida. Faltan características requeridas.',
            'caracteristicas_faltantes': faltantes
        }), 400

    try:
        datos_para_predecir = {key: [data[key]] for key in CARACTERISTICAS_ESPERADAS}
        df_prediccion = pd.DataFrame(datos_para_predecir)

        app.logger.info(f"DataFrame creado para la predicción:\n{df_prediccion.to_string()}")

        # 5. Realizar la predicción de PROBABILIDADES
        prediccion_probabilidades = modelo_cargado.predict_proba(df_prediccion)
        probabilidad_cancelacion = float(prediccion_probabilidades[0][1])

    
        UMBRAL_RIESGO = 0.85 

        # 7. Aplicar el umbral para determinar la clase y la etiqueta final
        if probabilidad_cancelacion > UMBRAL_RIESGO:
            resultado_clase = 1
            etiqueta_prediccion = "Cancelado" # O puedes usar "Riesgo Alto"
        else:
            resultado_clase = 0
            etiqueta_prediccion = "No Cancelado" # O puedes usar "Aceptable"

        app.logger.info(f"Predicción exitosa: Clase={resultado_clase} ({etiqueta_prediccion}) con Probabilidad={probabilidad_cancelacion:.4f} y Umbral={UMBRAL_RIESGO}")
        
        # 8. Devolver el resultado como JSON
        return jsonify({
            'prediccion_texto': etiqueta_prediccion,
            'prediccion_clase': resultado_clase,
            'probabilidad_de_cancelacion': round(probabilidad_cancelacion, 4)
        })

    except (ValueError, TypeError) as e:
        app.logger.error(f"Error en los tipos de datos recibidos: {e}")
        return jsonify({'error': f'Error en los datos de entrada. Asegúrate de que los valores sean numéricos. Detalle: {e}'}), 400
    except Exception as e:
        app.logger.error(f"Ocurrió un error inesperado durante la predicción: {e}")
        return jsonify({'error': 'Ocurrió un error interno en el servidor.'}), 500

# Ruta de bienvenida para verificar que la API está funcionando
@app.route('/predecir', methods=['GET'])
def index():
    return "<h1>API de Predicción de Cancelaciones</h1><p>El servicio está activo. Usa el endpoint /predecir para obtener una predicción.</p>"

# Iniciar el servidor de Flask
if __name__ == '__main__':
    # Es recomendable usar un servidor de producción como Gunicorn o Waitress en lugar del servidor de desarrollo de Flask.
    app.run(host='0.0.0.0', port=5001, debug=False)
