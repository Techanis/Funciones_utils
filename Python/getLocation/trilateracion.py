import numpy as np
from scipy.optimize import least_squares



def get_XY(anchor_coords, distances):
    """
    Estima la posición 2D (x, y) de un objetivo mediante trilateración no lineal por mínimos cuadrados.

    Esta función calcula las coordenadas cartesianas más probables de un punto desconocido
    a partir de las posiciones conocidas de varios anclajes (anchors) y las distancias medidas
    desde el objetivo hacia cada uno. El cálculo se realiza minimizando la suma de los cuadrados
    de los residuales entre las distancias predichas y las distancias medidas, utilizando
    `scipy.optimize.least_squares`.

    Parámetros
    ----------
    anchor_coords : array_like de forma (N, 2)
        Coordenadas (x_i, y_i) de los N anclajes conocidos.
    distances : array_like de forma (N,)
        Distancias medidas desde el punto desconocido hacia cada anclaje.

    Retorna
    -------
    estimated_pos : ndarray de forma (2,)
        Posición estimada [x_est, y_est] del punto objetivo en el mismo sistema de referencia
        que los anclajes.

    Notas
    -----
    - La estimación inicial se toma como el centroide de los anclajes.
    - Se utiliza el método Trust Region Reflective ('trf') con tolerancias estrictas de convergencia.
    - Imprime información de diagnóstico como la norma de los residuales, número de evaluaciones
      de la función y el estado final del optimizador.

    Ejemplos
    --------
    >>> anchors = [(0, 0), (10, 0), (0, 10), (10, 10)]
    >>> distances = [7.1, 5.0, 6.8, 3.2]
    >>> get_XY(anchors, distances)
    Posición estimada: [4.85 5.22]
    Norma L2 de residuales: 0.013
    Evaluaciones de función: 25
    Éxito: True | Mensaje: 'Optimización terminada exitosamente.'
    array([4.85, 5.22])
    """

    anchors = np.array(anchor_coords)  # Convert to numpy array if not already
    measured_distances = np.array(distances)

    # Define residual function for least_squares
    def trilateration_residuals(p, anchors, distances):
        """
        p: array-like, shape (2,) -> [x, y] estimate of position
        anchors: array, shape (N, 2) with anchor coordinates
        distances: array, shape (N,) with measured ranges r_i

        Returns vector of residuals: f_i(x, y) = (predicted_distance_i - measured_distance_i)
        """
        x, y = p
        # Predicted distances from current estimate to each anchor
        predicted = np.sqrt(np.sum((anchors - np.array([x, y]))**2, axis=1))
        # Residuals (vector of length N)
        return predicted - distances

    # Initial guess x0 for [x, y]
    # A simple choice: centroid of anchors
    x0 = np.mean(anchors, axis=0) 

    # Call least_squares
    result = least_squares(
        fun=trilateration_residuals,   # residual function
        x0=x0,                         # initial guess [x0, y0]
        args=(anchors, measured_distances),  # extra args passed to residual function
        method='trf',                  # trust-region reflective (default)
        ftol=1e-10,                    # function tolerance
        xtol=1e-10,                    # step tolerance
        gtol=1e-10                     # gradient norm tolerance
    )

    # Extract estimated position and some diagnostics
    estimated_pos = result.x       # [x_est, y_est]
    residual_norm = np.linalg.norm(result.fun)  # ||f(x)||_2
    num_iterations = result.nfev   # number of function evaluations

    # print("Estimated position: ", estimated_pos)
    # print("Residual L2 norm:   ", residual_norm)
    # print("Function evals:     ", num_iterations)
    # print("Success:", result.success, "| Message:", result.message)

    return estimated_pos

