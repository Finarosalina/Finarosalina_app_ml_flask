from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Cargar modelo y escalador
model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Obtener los 11 valores del formulario
        features = [
            float(request.form['val1']),
            float(request.form['val2']),
            float(request.form['val3']),
            float(request.form['val4']),
            float(request.form['val5']),
            float(request.form['val6']),
            float(request.form['val7']),
            float(request.form['val8']),
            float(request.form['val9']),
            float(request.form['val10']),
            float(request.form['val11']),
        ]
        
        # Escalar las características
        features_scaled = scaler.transform([features])
        
        # Predecir la calidad del vino
        prediction = model.predict(features_scaled)

        # Interpretar la predicción
        if prediction == 0:
            result = "Este vino probablemente sea de baja calidad."
        elif prediction == 1:
            result = "Este vino probablemente sea de calidad media."
        else:
            result = "Este vino probablemente sea de alta calidad."

        return render_template('index.html', prediction=result)
    
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
