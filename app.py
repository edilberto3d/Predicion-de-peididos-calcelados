# app.py
# Importar las librerías necesarias
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# 1. Inicialización de la aplicación Flask
app = Flask(__name__)

# Configurar un sistema de logging para monitorear la aplicación en producción
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 2. Carga del modelo entrenado (.pkl)
# Es una buena práctica cargar el modelo una sola vez al iniciar la aplicación.
try:
    # Asegúrate de que el nombre del archivo coincida exactamente con el que guardaste.
    modelo_cargado = joblib.load('modelo_random_forest_entrenado.pkl')
    app.logger.info("✅ Modelo de clasificación cargado correctamente.")
except FileNotFoundError:
    app.logger.error("❌ Error Crítico: No se encontró el archivo 'modelo_random_forest_entrenado.pkl'.")
    app.logger.error("Asegúrate de que el archivo del modelo esté en el mismo directorio que app.py.")
    modelo_cargado = None
except Exception as e:
    app.logger.error(f"❌ Error inesperado al cargar el modelo: {e}")
    modelo_cargado = None

# 3. Definición de las características que el modelo espera
# Esto es crucial para la validación y para crear el DataFrame correctamente.
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
    """
    Endpoint para predecir la probabilidad de cancelación de un pedido.
    Espera un JSON con las 6 características requeridas por el modelo.
    """
    # Verificar si el modelo se cargó correctamente al inicio.
    if modelo_cargado is None:
        # Si el modelo no está disponible, no podemos hacer predicciones.
        return jsonify({'error': 'El modelo de predicción no está disponible. Revisa los logs del servidor.'}), 503 # Service Unavailable

    # Obtener los datos JSON de la petición.
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No se recibieron datos en la petición. Se esperaba un JSON.'}), 400

    app.logger.info(f"Petición recibida para predicción: {data}")

    # Validar que todas las características necesarias están en el JSON recibido.
    if not all(feature in data for feature in CARACTERISTICAS_ESPERADAS):
        faltantes = [feature for feature in CARACTERISTICAS_ESPERADAS if feature not in data]
        app.logger.warning(f"Petición inválida. Faltan las siguientes características: {faltantes}")
        return jsonify({
            'error': 'Petición inválida. Faltan características requeridas.',
            'caracteristicas_faltantes': faltantes
        }), 400

    try:
        # Crear un DataFrame de pandas con los datos recibidos.
        # El modelo espera los datos en un formato específico (DataFrame), con los nombres de columna correctos.
        # Usamos un diccionario para asegurar el orden y los nombres de las columnas.
        datos_para_predecir = {key: [data[key]] for key in CARACTERISTICAS_ESPERADAS}
        df_prediccion = pd.DataFrame(datos_para_predecir)
        
        app.logger.info(f"DataFrame creado para la predicción:\n{df_prediccion.to_string()}")

        # 5. Realizar la predicción
        # .predict() da la clase (0 o 1)
        prediccion_clase = modelo_cargado.predict(df_prediccion)
        
        # .predict_proba() da la probabilidad de cada clase. Es más útil.
        # Devuelve un array como [[prob_clase_0, prob_clase_1]]
        prediccion_probabilidades = modelo_cargado.predict_proba(df_prediccion)

        # Extraer los resultados
        resultado_clase = int(prediccion_clase[0])
        probabilidad_cancelacion = float(prediccion_probabilidades[0][1]) # Probabilidad de la clase '1' (cancelación)

        # Mapear el resultado numérico a una etiqueta legible
        etiqueta_prediccion = "Pedido probablemente Cancelado" if resultado_clase == 1 else "Pedido probablemente No Cancelado"

        app.logger.info(f"Predicción exitosa: {etiqueta_prediccion} con probabilidad {probabilidad_cancelacion:.2f}")

        # 6. Devolver el resultado como JSON
        return jsonify({
            'prediccion_texto': etiqueta_prediccion,
            'prediccion_clase': resultado_clase, # 1 para cancelación, 0 para no cancelación
            'probabilidad_de_cancelacion': round(probabilidad_cancelacion, 4)
        })

    except (ValueError, TypeError) as e:
        app.logger.error(f"Error en los tipos de datos recibidos: {e}")
        return jsonify({'error': f'Error en los datos de entrada. Asegúrate de que todos los valores sean numéricos. Detalle: {e}'}), 400
    except Exception as e:
        app.logger.error(f"Ocurrió un error inesperado durante la predicción: {e}")
        return jsonify({'error': 'Ocurrió un error interno en el servidor.'}), 500

# Ruta de bienvenida para verificar que la API está funcionando
@app.route('/', methods=['GET'])
def index():
    return "<h1>API de Predicción de Cancelaciones</h1><p>El servicio está activo. Usa el endpoint /predecir para obtener una predicción.</p>"

# 7. Iniciar el servidor de Flask
if __name__ == '__main__':
    # host='0.0.0.0' permite que la API sea accesible desde fuera del contenedor (necesario para producción)
    # El puerto 5001 es un buen ejemplo para no usar el puerto por defecto (5000)
    app.run(host='0.0.0.0', port=5001, debug=False) # debug=False es importante para producción

