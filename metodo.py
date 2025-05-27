import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from numpy.linalg import LinAlgError

class NumericalMethodsApp:
    def __init__(self, root):
        """Inicializa la aplicación con la ventana principal."""
        self.root = root
        self.root.title("Aplicaciones de Métodos Numéricos")
        self.root.geometry("600x400")
        
        # Crear pestañas para los temas
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, expand=True)
        
        # Pestaña para Álgebra Lineal
        self.linear_algebra_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.linear_algebra_frame, text="Álgebra Lineal")
        
        # Pestaña para Optimización y Regresión
        self.optimization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.optimization_frame, text="Optimización y Regresión")
        
        # Configurar componentes de Álgebra Lineal
        self.setup_linear_algebra_tab()
        
        # Configurar componentes de Optimización y Regresión
        self.setup_optimization_tab()
    
    def setup_linear_algebra_tab(self):
        """Configura la pestaña de Álgebra Lineal con entrada y botón para Gauss."""
        tk.Label(self.linear_algebra_frame, text="Método de Eliminación de Gauss").pack(pady=5)
        
        tk.Label(self.linear_algebra_frame, text="Matriz A (separada por espacios, filas por líneas):").pack()
        self.matrix_a_entry = tk.Text(self.linear_algebra_frame, height=4, width=50)
        self.matrix_a_entry.pack(pady=5)
        
        tk.Label(self.linear_algebra_frame, text="Vector b (separado por espacios):").pack()
        self.vector_b_entry = tk.Entry(self.linear_algebra_frame, width=50)
        self.vector_b_entry.pack(pady=5)
        
        tk.Button(self.linear_algebra_frame, text="Resolver con Gauss", command=self.solve_gauss).pack(pady=10)
        self.result_gauss = tk.Label(self.linear_algebra_frame, text="")
        self.result_gauss.pack(pady=5)
    
    def setup_optimization_tab(self):
        """Configura la pestaña de Optimización con entrada y botón para Regresión Lineal."""
        tk.Label(self.optimization_frame, text="Regresión Lineal (Mínimos Cuadrados)").pack(pady=5)
        
        tk.Label(self.optimization_frame, text="Puntos x (separados por espacios):").pack()
        self.x_points_entry = tk.Entry(self.optimization_frame, width=50)
        self.x_points_entry.pack(pady=5)
        
        tk.Label(self.optimization_frame, text="Puntos y (separados por espacios):").pack()
        self.y_points_entry = tk.Entry(self.optimization_frame, width=50)
        self.y_points_entry.pack(pady=5)
        
        tk.Button(self.optimization_frame, text="Calcular Regresión", command=self.linear_regression).pack(pady=10)
        self.result_regression = tk.Label(self.optimization_frame, text="")
        self.result_regression.pack(pady=5)
    
    def solve_gauss(self):
        """Implementa la Eliminación de Gauss para resolver Ax = b."""
        try:
            # Leer matriz A
            matrix_a_str = self.matrix_a_entry.get("1.0", tk.END).strip()
            rows = matrix_a_str.split('\n')
            A = np.array([list(map(float, row.split())) for row in rows])
            
            # Leer vector b
            b = np.array(list(map(float, self.vector_b_entry.get().split())))
            
            # Validar dimensiones
            if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
                raise ValueError("Dimensiones incompatibles.")
            
            # Copiar matriz y vector para no modificar originales
            n = len(b)
            A = A.copy()
            b = b.copy()
            
            # Eliminación hacia adelante
            for i in range(n):
                if A[i, i] == 0:
                    raise ValueError("Pivote cero detectado.")
                for j in range(i + 1, n):
                    factor = A[j, i] / A[i, i]
                    A[j, i:] -= factor * A[i, i:]
                    b[j] -= factor * b[i]
            
            # Sustitución hacia atrás
            x = np.zeros(n)
            for i in range(n - 1, -1, -1):
                x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
            
            self.result_gauss.config(text=f"Solución: {x}")
            
            # Prueba de validación
            self.validate_gauss(A, b, x)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo resolver: {str(e)}")
    
    def validate_gauss(self, A_orig, b_orig, x):
        """Valida la solución de Gauss verificando Ax = b."""
        A = A_orig.copy()
        b = b_orig.copy()
        computed_b = np.dot(A, x)
        if np.allclose(computed_b, b):
            messagebox.showinfo("Validación", "La solución es correcta (Ax ≈ b).")
        else:
            messagebox.showwarning("Validación", "La solución no satisface Ax = b.")
    
    def linear_regression(self):
        """Implementa la regresión lineal por mínimos cuadrados."""
        try:
            # Leer puntos x e y
            x = np.array(list(map(float, self.x_points_entry.get().split())))
            y = np.array(list(map(float, self.y_points_entry.get().split())))
            
            # Validar entrada
            if len(x) != len(y) or len(x) < 2:
                raise ValueError("Se necesitan al menos 2 puntos con x e y correspondientes.")
            
            # Calcular parámetros de la recta y = mx + b
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x**2)
            
            m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            b = (sum_y - m * sum_x) / n
            
            self.result_regression.config(text=f"Recta: y = {m:.4f}x + {b:.4f}")
            
            # Validación con ejemplo
            self.validate_regression(x, y, m, b)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo calcular: {str(e)}")
    
    def validate_regression(self, x, y, m, b):
        """Valida la regresión calculando el error cuadrático medio."""
        y_pred = m * x + b
        mse = np.mean((y - y_pred)**2)
        messagebox.showinfo("Validación", f"Error cuadrático medio: {mse:.6f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NumericalMethodsApp(root)
    root.mainloop()