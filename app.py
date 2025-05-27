from flask import Flask, render_template, request, flash
import numpy as np

app = Flask(__name__)
app.secret_key = "clave_super_segura_cambia_esto"  # Cambia para producción

def eliminacion_gauss(A, b):
    n = len(b)
    A = A.copy()
    b = b.copy()
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError("Pivote cero detectado.")
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def regresion_lineal(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        raise ValueError("No se puede calcular pendiente, denominador cero.")
    m = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - m * sum_x) / n
    y_pred = m * x + b
    mse = np.mean((y - y_pred)**2)
    return m, b, mse

@app.route("/", methods=["GET", "POST"])
def index():
    gauss_result = None
    regression_result = None
    gauss_used = False
    regression_used = False

    if request.method == "POST":
        if "solve_gauss" in request.form:
            gauss_used = True
            try:
                matrix_a_str = request.form["matrix_a"]
                vector_b_str = request.form["vector_b"]

                rows = matrix_a_str.strip().split("\n")
                A = np.array([list(map(float, row.split())) for row in rows])
                b = np.array(list(map(float, vector_b_str.strip().split())))

                if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
                    raise ValueError("Dimensiones incompatibles entre matriz y vector.")

                x = eliminacion_gauss(A, b)
                gauss_result = f"Solución: {np.round(x,6)}"

            except Exception as e:
                flash(f"Error en Eliminación de Gauss: {e}", "danger")

        elif "calculate_regression" in request.form:
            regression_used = True
            try:
                x_str = request.form["x_points"]
                y_str = request.form["y_points"]

                x = np.array(list(map(float, x_str.strip().split())))
                y = np.array(list(map(float, y_str.strip().split())))

                if len(x) != len(y) or len(x) < 2:
                    raise ValueError("Se necesitan al menos 2 puntos con x e y correspondientes.")

                m, b, mse = regresion_lineal(x, y)
                regression_result = f"Recta: y = {m:.4f}x + {b:.4f} | Error cuadrático medio: {mse:.6f}"

            except Exception as e:
                flash(f"Error en Regresión Lineal: {e}", "danger")

    return render_template(
        "index.html",
        gauss_result=gauss_result,
        regression_result=regression_result,
        gauss_used=gauss_used,
        regression_used=regression_used
    )

if __name__ == "__main__":
    app.run(debug=True)
