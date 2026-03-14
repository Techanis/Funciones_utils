import numpy as np
from trilateracion import get_XY

def test_get_XY_con_ruido():
    # Semilla fija para que el test sea reproducible
    np.random.seed(123)

    # Anclas en las esquinas de un cuadrado 10x10
    anchors = np.array([(0, 0), (10, 0), (0, 10), (10, 10)])

    # Punto real (ground truth)
    true_pos = np.array([3.5, 6.2])

    # Distancias ideales
    true_distances = np.linalg.norm(anchors - true_pos, axis=1)

    # Añadir ruido gaussiano (media 0, sigma en metros, por ejemplo 0.1 m)
    noise_sigma = 1
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=true_distances.shape)
    noisy_distances = true_distances + noise

    # Posición estimada a partir de distancias ruidosas
    estimated = get_XY(anchors, noisy_distances)

    # Error máximo permitido (por ejemplo, 0.3 m)
    max_position_error = 0.5
    error = np.linalg.norm(estimated - true_pos)
    
    print(f"Posición real:     {true_pos}")
    print(f"Posición estimada: {estimated}")
    print(f"Error total:       {error:.3f} m (umbral {max_position_error} m)")

    assert error <= max_position_error, (
        f"Error demasiado grande: {error:.3f} m (umbral {max_position_error} m)"
    )