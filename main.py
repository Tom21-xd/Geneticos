# ============================================================================
# IMPORTS
# ============================================================================
import random
import numpy as np
from datetime import datetime

# Streamlit y visualización
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ============================================================================
# PARÁMETROS DEL ALGORITMO GENÉTICO (valores por defecto)
# ============================================================================
num_generaciones = 100
tamaño_poblacion = 50
prob_mutacion_inicial = 0.5
prob_mutacion_final = 0.05
elitismo_ratio_inicial = 0.1
elitismo_ratio_final = 0.3
prob_cruce = 0.8
penalizacion_inicial = -10
num_gen_penalizadas = 2

# ============================================================================
# PARÁMETROS DEL REACTOR Y RESTRICCIONES DE SEGURIDAD
# ============================================================================
LIMITES = {
    "temperatura": (50, 200),
    "presion": (1, 10),
    "flujo": (10, 100)
}

R = 8.314
Ea = 50000
A = 1e10
C0 = 2.0

PESO_RENDIMIENTO = 0.5
PESO_COSTO = 0.25
PESO_ENERGIA = 0.25

# ============================================================================
# MODELO DEL REACTOR - FUNCIONES DE EVALUACIÓN
# ============================================================================

def calcular_rendimiento(temperatura, presion, flujo):
    """Calcula el rendimiento del reactor usando cinética de Arrhenius"""
    T_kelvin = temperatura + 273.15
    k = A * np.exp(-Ea / (R * T_kelvin))
    tau = 100 / flujo
    conversion = (k * tau) / (1 + k * tau)
    factor_presion = 1 + 0.05 * (presion - 1)
    rendimiento = conversion * factor_presion * 100

    if temperatura > 180:
        rendimiento *= 0.9
    if presion > 8:
        rendimiento *= 0.95

    return min(rendimiento, 100)

def calcular_costo(temperatura, presion, flujo):
    """Calcula el costo de operación"""
    costo_reactivos = flujo * 0.5
    costo_presion = (presion - 1) * 2
    costo_mantenimiento = max(0, (temperatura - 150) * 0.1)
    return costo_reactivos + costo_presion + costo_mantenimiento

def calcular_consumo_energia(temperatura, presion, flujo):
    """Calcula el consumo energético"""
    energia_calentamiento = flujo * (temperatura - 25) * 0.08
    energia_bombeo = flujo * presion * 0.01
    return energia_calentamiento + energia_bombeo

def verificar_restricciones(individuo, limites=None, conversion_min=30):
    """Verifica que el individuo cumple con las restricciones de seguridad"""
    if limites is None:
        limites = LIMITES

    temperatura, presion, flujo = individuo

    if not (limites["temperatura"][0] <= temperatura <= limites["temperatura"][1]):
        return False
    if not (limites["presion"][0] <= presion <= limites["presion"][1]):
        return False
    if not (limites["flujo"][0] <= flujo <= limites["flujo"][1]):
        return False

    rendimiento = calcular_rendimiento(temperatura, presion, flujo)
    if rendimiento < conversion_min:
        return False

    return True

def evaluar_individuo(individuo, penalizacion=0, limites=None, pesos=None):
    """Función de fitness multiobjetivo"""
    if limites is None:
        limites = LIMITES
    if pesos is None:
        pesos = {"rendimiento": PESO_RENDIMIENTO, "costo": PESO_COSTO, "energia": PESO_ENERGIA}

    temperatura, presion, flujo = individuo

    if not verificar_restricciones(individuo, limites):
        return -1000

    rendimiento = calcular_rendimiento(temperatura, presion, flujo)
    costo = calcular_costo(temperatura, presion, flujo)
    energia = calcular_consumo_energia(temperatura, presion, flujo)

    rendimiento_norm = rendimiento / 100
    costo_norm = max(0, 1 - (costo / 100))
    energia_norm = max(0, 1 - (energia / 300))

    fitness = (pesos["rendimiento"] * rendimiento_norm +
               pesos["costo"] * costo_norm +
               pesos["energia"] * energia_norm)

    return fitness * 100 + penalizacion

# ============================================================================
# OPERADORES GENÉTICOS
# ============================================================================

def crear_individuo(limites=None):
    """Crea un individuo aleatorio dentro de los rangos permitidos"""
    if limites is None:
        limites = LIMITES
    temperatura = random.uniform(*limites["temperatura"])
    presion = random.uniform(*limites["presion"])
    flujo = random.uniform(*limites["flujo"])
    return [temperatura, presion, flujo]

def crear_poblacion(tamaño=None, limites=None):
    """Crea la población inicial"""
    if tamaño is None:
        tamaño = tamaño_poblacion
    if limites is None:
        limites = LIMITES

    poblacion = []
    intentos = 0
    max_intentos = tamaño * 10

    while len(poblacion) < tamaño and intentos < max_intentos:
        individuo = crear_individuo(limites)
        if verificar_restricciones(individuo, limites):
            poblacion.append(individuo)
        intentos += 1

    while len(poblacion) < tamaño:
        individuo = [120, 3, 50]
        poblacion.append(individuo)

    return poblacion

def seleccion_torneo(poblacion, fitness_poblacion, k=3):
    """Selección por torneo"""
    indices_torneo = random.sample(range(len(poblacion)), k)
    mejor_idx = max(indices_torneo, key=lambda i: fitness_poblacion[i])
    return poblacion[mejor_idx].copy()

def seleccion_ruleta(poblacion, fitness_poblacion):
    """Selección por aptitud/ruleta (fitness proporcional)"""
    # Manejar fitness negativos
    min_fitness = min(fitness_poblacion)
    if min_fitness < 0:
        # Ajustar para que todos sean positivos
        fitness_ajustado = [f - min_fitness + 1 for f in fitness_poblacion]
    else:
        fitness_ajustado = fitness_poblacion.copy()

    # Calcular suma total
    suma_total = sum(fitness_ajustado)
    if suma_total == 0:
        # Si todos tienen fitness 0, selección aleatoria
        return random.choice(poblacion).copy()

    # Calcular probabilidades
    probabilidades = [f / suma_total for f in fitness_ajustado]

    # Método de la ruleta con selección acumulativa
    r = random.random()
    acumulado = 0

    for i, prob in enumerate(probabilidades):
        acumulado += prob
        if r <= acumulado:
            return poblacion[i].copy()

    # Por si acaso (errores de redondeo)
    return poblacion[-1].copy()

def seleccion_ranking(poblacion, fitness_poblacion, presion=2):
    """Selección por ranking lineal"""
    n = len(poblacion)
    # Ordenar por fitness
    indices_ordenados = sorted(range(n), key=lambda i: fitness_poblacion[i])

    # Asignar probabilidades basadas en ranking
    probabilidades = []
    for i in range(n):
        prob = (2 - presion) / n + (2 * i * (presion - 1)) / (n * (n - 1))
        probabilidades.append(prob)

    # Selección por ruleta con probabilidades de ranking
    r = random.random()
    acumulado = 0

    for i, prob in enumerate(probabilidades):
        acumulado += prob
        if r <= acumulado:
            return poblacion[indices_ordenados[i]].copy()

    return poblacion[indices_ordenados[-1]].copy()

def cruce_blx_alpha(padre1, padre2, alpha=0.5):
    """Cruce BLX-α (Blend Crossover)"""
    hijo1 = []
    hijo2 = []

    for gen1, gen2 in zip(padre1, padre2):
        cmin = min(gen1, gen2)
        cmax = max(gen1, gen2)
        rango = cmax - cmin

        limite_inf = cmin - rango * alpha
        limite_sup = cmax + rango * alpha

        hijo1.append(random.uniform(limite_inf, limite_sup))
        hijo2.append(random.uniform(limite_inf, limite_sup))

    return hijo1, hijo2

def mutar(individuo, prob_mutacion, generacion, num_gen, limites=None):
    """Mutación gaussiana con intensidad decreciente"""
    if limites is None:
        limites = LIMITES

    if random.random() < prob_mutacion:
        sigma_inicial = 0.3
        sigma_final = 0.05
        sigma = sigma_inicial - (sigma_inicial - sigma_final) * (generacion / num_gen)

        for i in range(len(individuo)):
            if random.random() < 0.4:
                nombres = ["temperatura", "presion", "flujo"]
                limite_inf, limite_sup = limites[nombres[i]]
                rango = limite_sup - limite_inf
                individuo[i] += random.gauss(0, sigma * rango)
                individuo[i] = max(limite_inf, min(limite_sup, individuo[i]))

def reparar_individuo(individuo, limites=None):
    """Repara un individuo para que esté dentro de los límites"""
    if limites is None:
        limites = LIMITES

    nombres = ["temperatura", "presion", "flujo"]
    for i, nombre in enumerate(nombres):
        limite_inf, limite_sup = limites[nombre]
        individuo[i] = max(limite_inf, min(limite_sup, individuo[i]))

# ============================================================================
# ALGORITMO GENÉTICO PRINCIPAL
# ============================================================================

def algoritmo_genetico(params=None, progress_callback=None, generation_callback=None):
    """Algoritmo genético principal con soporte para visualización en tiempo real"""
    # Usar parámetros por defecto si no se proporcionan
    if params is None:
        params = {
            "num_generaciones": num_generaciones,
            "tamaño_poblacion": tamaño_poblacion,
            "prob_mutacion_inicial": prob_mutacion_inicial,
            "prob_mutacion_final": prob_mutacion_final,
            "elitismo_ratio_inicial": elitismo_ratio_inicial,
            "elitismo_ratio_final": elitismo_ratio_final,
            "prob_cruce": prob_cruce,
            "penalizacion_inicial": penalizacion_inicial,
            "num_gen_penalizadas": num_gen_penalizadas,
            "limites": LIMITES,
            "pesos": {"rendimiento": PESO_RENDIMIENTO, "costo": PESO_COSTO, "energia": PESO_ENERGIA},
            "conversion_min": 30,
            "metodo_seleccion": "torneo",  # Por defecto torneo
            "torneo_k": 3
        }

    poblacion = crear_poblacion(params["tamaño_poblacion"], params["limites"])
    poblacion_anterior = None  # Para calcular vectores de movimiento

    mejor_fitness_historia = []
    promedio_fitness_historia = []
    mejor_individuo_global = None
    mejor_fitness_global = float('-inf')
    mejor_generacion = 0
    soluciones_por_generacion = []
    evol_parametros = {"temp": [], "pres": [], "flujo": [], "rend": [], "costo": [], "energia": []}

    for generacion in range(params["num_generaciones"]):
        penalizacion = params["penalizacion_inicial"] if generacion < params["num_gen_penalizadas"] else 0

        prob_mutacion = params["prob_mutacion_inicial"] - \
                       (params["prob_mutacion_inicial"] - params["prob_mutacion_final"]) * \
                       (generacion / params["num_generaciones"])

        elitismo_ratio = params["elitismo_ratio_inicial"] + \
                        (params["elitismo_ratio_final"] - params["elitismo_ratio_inicial"]) * \
                        (generacion / params["num_generaciones"])

        fitness_poblacion = [evaluar_individuo(ind, penalizacion, params["limites"], params["pesos"])
                           for ind in poblacion]

        mejor_fitness_gen = max(fitness_poblacion)
        promedio_fitness = np.mean([f for f in fitness_poblacion if f > -1000])

        if generacion == 0:
            mejor_fitness_historia.append(mejor_fitness_gen)
        else:
            mejor_fitness_historia.append(max(mejor_fitness_historia[-1], mejor_fitness_gen))

        promedio_fitness_historia.append(promedio_fitness)

        idx_mejor_gen = fitness_poblacion.index(mejor_fitness_gen)
        soluciones_generacion = []
        for ind, fit in zip(poblacion, fitness_poblacion):
            temp, pres, fl = ind
            rend = calcular_rendimiento(temp, pres, fl)
            cost = calcular_costo(temp, pres, fl)
            energ = calcular_consumo_energia(temp, pres, fl)
            soluciones_generacion.append({
                'individuo': ind.copy(),
                'fitness': fit,
                'rendimiento': rend,
                'costo': cost,
                'energia': energ
            })
        soluciones_por_generacion.append(soluciones_generacion)

        mejor_ind_gen = poblacion[idx_mejor_gen]
        evol_parametros["temp"].append(mejor_ind_gen[0])
        evol_parametros["pres"].append(mejor_ind_gen[1])
        evol_parametros["flujo"].append(mejor_ind_gen[2])
        evol_parametros["rend"].append(calcular_rendimiento(*mejor_ind_gen))
        evol_parametros["costo"].append(calcular_costo(*mejor_ind_gen))
        evol_parametros["energia"].append(calcular_consumo_energia(*mejor_ind_gen))

        if mejor_fitness_gen > mejor_fitness_global:
            mejor_fitness_global = mejor_fitness_gen
            mejor_individuo_global = poblacion[idx_mejor_gen].copy()
            mejor_generacion = generacion

        if progress_callback:
            progress_callback(generacion + 1, params["num_generaciones"])

        # Guardar población actual para la siguiente generación
        poblacion_anterior = [ind.copy() for ind in poblacion]

        num_elite = max(1, int(params["tamaño_poblacion"] * elitismo_ratio))
        indices_ordenados = sorted(range(len(fitness_poblacion)),
                                   key=lambda i: fitness_poblacion[i],
                                   reverse=True)
        elite = [poblacion[i].copy() for i in indices_ordenados[:num_elite]]

        nueva_poblacion = elite.copy()
        eventos_reproduccion = []  # Capturar eventos de reproducción

        evento_count = 0
        while len(nueva_poblacion) < params["tamaño_poblacion"] and evento_count < 5:  # Limitar a 5 eventos para visualización
            # Seleccionar padres según el método elegido
            if params.get("metodo_seleccion", "torneo") == "torneo":
                padre1 = seleccion_torneo(poblacion, fitness_poblacion,
                                        k=params.get("torneo_k", 3))
                padre2 = seleccion_torneo(poblacion, fitness_poblacion,
                                        k=params.get("torneo_k", 3))
            elif params.get("metodo_seleccion") == "ruleta":
                padre1 = seleccion_ruleta(poblacion, fitness_poblacion)
                padre2 = seleccion_ruleta(poblacion, fitness_poblacion)
            else:  # ranking
                padre1 = seleccion_ranking(poblacion, fitness_poblacion)
                padre2 = seleccion_ranking(poblacion, fitness_poblacion)

            # Guardar posiciones de padres antes del cruce
            padre1_original = padre1.copy()
            padre2_original = padre2.copy()

            hubo_cruce = random.random() < params["prob_cruce"]
            if hubo_cruce:
                hijo1, hijo2 = cruce_blx_alpha(padre1, padre2)
            else:
                hijo1, hijo2 = padre1.copy(), padre2.copy()

            hijo1_antes_mutacion = hijo1.copy()
            hijo2_antes_mutacion = hijo2.copy()

            mutar(hijo1, prob_mutacion, generacion, params["num_generaciones"], params["limites"])
            mutar(hijo2, prob_mutacion, generacion, params["num_generaciones"], params["limites"])

            reparar_individuo(hijo1, params["limites"])
            reparar_individuo(hijo2, params["limites"])

            # Registrar evento de reproducción
            eventos_reproduccion.append({
                "padre1": padre1_original,
                "padre2": padre2_original,
                "hijo1": hijo1.copy(),
                "hijo2": hijo2.copy(),
                "hubo_cruce": hubo_cruce,
                "muto_hijo1": hijo1 != hijo1_antes_mutacion,
                "muto_hijo2": hijo2 != hijo2_antes_mutacion
            })

            nueva_poblacion.extend([hijo1, hijo2])
            evento_count += 1

        # Completar población sin registrar eventos
        while len(nueva_poblacion) < params["tamaño_poblacion"]:
            if params.get("metodo_seleccion", "torneo") == "torneo":
                padre1 = seleccion_torneo(poblacion, fitness_poblacion,
                                        k=params.get("torneo_k", 3))
                padre2 = seleccion_torneo(poblacion, fitness_poblacion,
                                        k=params.get("torneo_k", 3))
            elif params.get("metodo_seleccion") == "ruleta":
                padre1 = seleccion_ruleta(poblacion, fitness_poblacion)
                padre2 = seleccion_ruleta(poblacion, fitness_poblacion)
            else:
                padre1 = seleccion_ranking(poblacion, fitness_poblacion)
                padre2 = seleccion_ranking(poblacion, fitness_poblacion)

            if random.random() < params["prob_cruce"]:
                hijo1, hijo2 = cruce_blx_alpha(padre1, padre2)
            else:
                hijo1, hijo2 = padre1.copy(), padre2.copy()

            mutar(hijo1, prob_mutacion, generacion, params["num_generaciones"], params["limites"])
            mutar(hijo2, prob_mutacion, generacion, params["num_generaciones"], params["limites"])

            reparar_individuo(hijo1, params["limites"])
            reparar_individuo(hijo2, params["limites"])

            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion[:params["tamaño_poblacion"]]

        # Calcular fitness de la nueva población para visualización correcta
        fitness_nueva_poblacion = [evaluar_individuo(ind, penalizacion, params["limites"], params["pesos"])
                                   for ind in poblacion]

        # Encontrar el mejor individuo de la nueva población
        mejor_fitness_nueva = max(fitness_nueva_poblacion)
        idx_mejor_nueva = fitness_nueva_poblacion.index(mejor_fitness_nueva)

        # Callback para visualización en tiempo real (después de reproducción)
        if generation_callback:
            generation_callback({
                "generacion": generacion,
                "poblacion": [ind.copy() for ind in poblacion],
                "poblacion_anterior": [ind.copy() for ind in poblacion_anterior] if poblacion_anterior else None,
                "fitness": fitness_nueva_poblacion.copy(),
                "mejor_fitness": mejor_fitness_nueva,
                "promedio_fitness": np.mean([f for f in fitness_nueva_poblacion if f > -1000]),
                "mejor_individuo": poblacion[idx_mejor_nueva].copy(),
                "eventos_reproduccion": eventos_reproduccion
            })

    return {
        "mejor_individuo": mejor_individuo_global,
        "mejor_fitness": mejor_fitness_global,
        "mejor_generacion": mejor_generacion,
        "historia_fitness": mejor_fitness_historia,
        "historia_promedio": promedio_fitness_historia,
        "evol_parametros": evol_parametros,
        "soluciones_por_generacion": soluciones_por_generacion
    }

# ============================================================================
# VISUALIZACIÓN 3D EN TIEMPO REAL
# ============================================================================

def crear_frames_fusion_animada(eventos_reprod, poblacion, limites, num_pasos=10):
    """
    Crea múltiples frames interpolados mostrando la fusión animada de padres → hijos
    num_pasos: cantidad de frames intermedios para suavizar la transición
    """
    frames_fusion = []

    # Población de fondo (constante en todos los frames)
    temps_pob = [ind[0] for ind in poblacion]
    presiones_pob = [ind[1] for ind in poblacion]
    flujos_pob = [ind[2] for ind in poblacion]

    # Crear frames de interpolación
    for paso in range(num_pasos + 1):
        t = paso / num_pasos  # Factor de interpolación (0 a 1)

        # Easing para movimiento más natural (ease-in-out)
        t_eased = t * t * (3 - 2 * t)  # Suavizado cúbico

        frame_data = []

        # Población de fondo
        frame_data.append(go.Scatter3d(
            x=temps_pob,
            y=presiones_pob,
            z=flujos_pob,
            mode='markers',
            marker=dict(size=4, color='gray', opacity=0.1),
            name='Población',
            showlegend=False,
            hoverinfo='skip'
        ))

        # Interpolar posiciones de padres e hijos
        padres_x, padres_y, padres_z = [], [], []
        particulas_x, particulas_y, particulas_z = [], [], []
        particulas_color = []
        particulas_size = []

        for evento in eventos_reprod:
            padre1 = evento["padre1"]
            padre2 = evento["padre2"]
            hijo1 = evento["hijo1"]
            hijo2 = evento["hijo2"]

            # Fase 1 (t: 0 → 0.5): Padres se acercan entre sí
            if t <= 0.5:
                t1 = t * 2  # Escalar a 0-1

                # Padres moviéndose hacia el punto medio
                punto_medio1 = [(padre1[i] + padre2[i]) / 2 for i in range(3)]

                p1_actual = [padre1[i] + (punto_medio1[i] - padre1[i]) * t1 for i in range(3)]
                p2_actual = [padre2[i] + (punto_medio1[i] - padre2[i]) * t1 for i in range(3)]

                particulas_x.extend([p1_actual[0], p2_actual[0]])
                particulas_y.extend([p1_actual[1], p2_actual[1]])
                particulas_z.extend([p1_actual[2], p2_actual[2]])
                particulas_color.extend(['blue', 'cyan'])

                # Tamaño reduce a medida que se fusionan
                size_fusion = 10 * (1 - t1 * 0.5)
                particulas_size.extend([size_fusion, size_fusion])

            # Fase 2 (t: 0.5 → 1.0): Aparecen los hijos desde el punto de fusión
            else:
                t2 = (t - 0.5) * 2  # Escalar a 0-1

                # Punto de fusión (centro entre padres)
                punto_fusion = [(padre1[i] + padre2[i]) / 2 for i in range(3)]

                # Hijos emergen desde el punto de fusión
                h1_actual = [punto_fusion[i] + (hijo1[i] - punto_fusion[i]) * t2 for i in range(3)]
                h2_actual = [punto_fusion[i] + (hijo2[i] - punto_fusion[i]) * t2 for i in range(3)]

                particulas_x.extend([h1_actual[0], h2_actual[0]])
                particulas_y.extend([h1_actual[1], h2_actual[1]])
                particulas_z.extend([h1_actual[2], h2_actual[2]])
                particulas_color.extend(['lime', 'green'])

                # Tamaño crece desde el centro
                size_nacimiento = 5 + 7 * t2
                particulas_size.extend([size_nacimiento, size_nacimiento])

        # Dibujar partículas (padres o hijos según la fase)
        frame_data.append(go.Scatter3d(
            x=particulas_x,
            y=particulas_y,
            z=particulas_z,
            mode='markers',
            marker=dict(
                size=particulas_size,
                color=particulas_color,
                symbol='circle',
                line=dict(color='white', width=1),
                opacity=0.8
            ),
            name='Fusión',
            showlegend=False
        ))

        # Agregar líneas de energía/fusión
        if 0.3 < t < 0.7:  # Solo durante la fusión
            lineas_x, lineas_y, lineas_z = [], [], []
            for evento in eventos_reprod:
                padre1 = evento["padre1"]
                padre2 = evento["padre2"]
                punto_medio = [(padre1[i] + padre2[i]) / 2 for i in range(3)]

                lineas_x.extend([padre1[0], punto_medio[0], None])
                lineas_y.extend([padre1[1], punto_medio[1], None])
                lineas_z.extend([padre1[2], punto_medio[2], None])

                lineas_x.extend([padre2[0], punto_medio[0], None])
                lineas_y.extend([padre2[1], punto_medio[1], None])
                lineas_z.extend([padre2[2], punto_medio[2], None])

            frame_data.append(go.Scatter3d(
                x=lineas_x,
                y=lineas_y,
                z=lineas_z,
                mode='lines',
                line=dict(color='rgba(255, 255, 0, 0.6)', width=3),
                name='Energía',
                showlegend=False,
                hoverinfo='skip'
            ))

        frames_fusion.append(frame_data)

    return frames_fusion

def crear_frame_reproduccion(evento, poblacion, limites, frame_tipo="padres"):
    """Crea un frame mostrando el proceso de reproducción"""

    # Extraer todos los individuos de la población
    temps_pob = [ind[0] for ind in poblacion]
    presiones_pob = [ind[1] for ind in poblacion]
    flujos_pob = [ind[2] for ind in poblacion]

    # Padres
    padre1 = evento["padre1"]
    padre2 = evento["padre2"]
    hijo1 = evento["hijo1"]
    hijo2 = evento["hijo2"]

    frame_data = []

    # Población de fondo (semi-transparente)
    frame_data.append(go.Scatter3d(
        x=temps_pob,
        y=presiones_pob,
        z=flujos_pob,
        mode='markers',
        marker=dict(size=5, color='gray', opacity=0.2),
        name='Población',
        showlegend=False,
        hoverinfo='skip'
    ))

    if frame_tipo == "padres":
        # Mostrar solo padres seleccionados
        frame_data.append(go.Scatter3d(
            x=[padre1[0], padre2[0]],
            y=[padre1[1], padre2[1]],
            z=[padre1[2], padre2[2]],
            mode='markers',
            marker=dict(size=15, color=['blue', 'cyan'], symbol='circle',
                       line=dict(color='white', width=2)),
            text=['Padre 1', 'Padre 2'],
            name='Padres',
            showlegend=True
        ))

        # Línea conectando padres
        frame_data.append(go.Scatter3d(
            x=[padre1[0], padre2[0]],
            y=[padre1[1], padre2[1]],
            z=[padre1[2], padre2[2]],
            mode='lines',
            line=dict(color='yellow', width=5, dash='dash'),
            name='Cruce',
            showlegend=False
        ))

    elif frame_tipo == "hijos":
        # Mostrar padres y hijos
        frame_data.append(go.Scatter3d(
            x=[padre1[0], padre2[0]],
            y=[padre1[1], padre2[1]],
            z=[padre1[2], padre2[2]],
            mode='markers',
            marker=dict(size=12, color=['blue', 'cyan'], opacity=0.5),
            name='Padres',
            showlegend=False
        ))

        # Hijos
        frame_data.append(go.Scatter3d(
            x=[hijo1[0], hijo2[0]],
            y=[hijo1[1], hijo2[1]],
            z=[hijo1[2], hijo2[2]],
            mode='markers',
            marker=dict(size=15, color=['lime', 'green'], symbol='diamond',
                       line=dict(color='yellow', width=3)),
            text=['Hijo 1' + (' 🧬 Mutado' if evento['muto_hijo1'] else ''),
                  'Hijo 2' + (' 🧬 Mutado' if evento['muto_hijo2'] else '')],
            name='Hijos',
            showlegend=True
        ))

        # Líneas de herencia
        for i, (padre, hijo, color) in enumerate([
            (padre1, hijo1, 'rgba(0, 255, 0, 0.5)'),
            (padre2, hijo2, 'rgba(0, 200, 0, 0.5)')
        ]):
            frame_data.append(go.Scatter3d(
                x=[padre[0], hijo[0]],
                y=[padre[1], hijo[1]],
                z=[padre[2], hijo[2]],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False
            ))

    return frame_data

def crear_grafica_3d_generacion(poblacion, poblacion_anterior, fitness, generacion, limites):
    """Crea una gráfica 3D para una generación específica (para tiempo real)"""

    # Extraer coordenadas
    temps = [ind[0] for ind in poblacion]
    presiones = [ind[1] for ind in poblacion]
    flujos = [ind[2] for ind in poblacion]

    # Normalizar fitness para colores
    fitness_norm = [(f + 1000) / 11 if f > -1000 else 0 for f in fitness]

    # Encontrar el mejor individuo
    idx_mejor = fitness.index(max(fitness))

    # Preparar vectores de movimiento
    vectores_x = []
    vectores_y = []
    vectores_z = []

    if poblacion_anterior and len(poblacion_anterior) == len(poblacion):
        for i in range(len(poblacion)):
            dx = poblacion[i][0] - poblacion_anterior[i][0]
            dy = poblacion[i][1] - poblacion_anterior[i][1]
            dz = poblacion[i][2] - poblacion_anterior[i][2]

            movimiento = (dx**2 + dy**2 + dz**2)**0.5
            if movimiento > 0.5:
                vectores_x.extend([poblacion_anterior[i][0], poblacion[i][0], None])
                vectores_y.extend([poblacion_anterior[i][1], poblacion[i][1], None])
                vectores_z.extend([poblacion_anterior[i][2], poblacion[i][2], None])

    # Separar el mejor individuo del resto
    temps_resto = [temps[i] for i in range(len(temps)) if i != idx_mejor]
    presiones_resto = [presiones[i] for i in range(len(presiones)) if i != idx_mejor]
    flujos_resto = [flujos[i] for i in range(len(flujos)) if i != idx_mejor]
    fitness_norm_resto = [fitness_norm[i] for i in range(len(fitness_norm)) if i != idx_mejor]

    # Crear figura
    fig = go.Figure()

    # Trace 0: Vectores
    if vectores_x:
        fig.add_trace(go.Scatter3d(
            x=vectores_x,
            y=vectores_y,
            z=vectores_z,
            mode='lines',
            line=dict(color='rgba(0, 255, 255, 0.5)', width=4),
            name='Vectores',
            showlegend=True,
            hoverinfo='skip'
        ))

    # Trace 1: Población (sin el mejor)
    fig.add_trace(go.Scatter3d(
        x=temps_resto,
        y=presiones_resto,
        z=flujos_resto,
        mode='markers',
        marker=dict(
            size=10,
            color=fitness_norm_resto,
            colorscale='Viridis',
            colorbar=dict(title="Fitness", x=1.02),
            opacity=0.85,
            line=dict(color='white', width=1),
            cmin=0,
            cmax=100
        ),
        text=[f'Ind {i}<br>Fitness: {fitness[i]:.2f}<br>T: {temps[i]:.1f}°C<br>P: {presiones[i]:.1f}bar<br>F: {flujos[i]:.1f}L/min'
              for i in range(len(poblacion)) if i != idx_mejor],
        hoverinfo='text',
        name=f'Población ({len(temps_resto)} individuos)'
    ))

    # Trace 2: Mejor individuo
    fig.add_trace(go.Scatter3d(
        x=[temps[idx_mejor]],
        y=[presiones[idx_mejor]],
        z=[flujos[idx_mejor]],
        mode='markers',
        marker=dict(
            size=18,
            color='gold',
            symbol='diamond',
            line=dict(color='red', width=4)
        ),
        text=f'⭐ MEJOR<br>Fitness: {fitness[idx_mejor]:.2f}<br>T: {temps[idx_mejor]:.1f}°C<br>P: {presiones[idx_mejor]:.1f}bar<br>F: {flujos[idx_mejor]:.1f}L/min',
        hoverinfo='text',
        name='★ Mejor'
    ))

    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f'🧬 Simulación en Tiempo Real - Generación {generacion + 1}',
            font=dict(size=18, color='white')
        ),
        scene=dict(
            xaxis=dict(
                title='Temperatura (°C)',
                range=[limites["temperatura"][0], limites["temperatura"][1]],
                backgroundcolor="rgb(20, 24, 54)",
                gridcolor="rgb(50, 50, 50)"
            ),
            yaxis=dict(
                title='Presión (bar)',
                range=[limites["presion"][0], limites["presion"][1]],
                backgroundcolor="rgb(20, 24, 54)",
                gridcolor="rgb(50, 50, 50)"
            ),
            zaxis=dict(
                title='Flujo (L/min)',
                range=[limites["flujo"][0], limites["flujo"][1]],
                backgroundcolor="rgb(20, 24, 54)",
                gridcolor="rgb(50, 50, 50)"
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor="rgb(10, 10, 10)"
        ),
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        paper_bgcolor="rgb(10, 10, 10)",
        plot_bgcolor="rgb(10, 10, 10)",
        font=dict(color='white')
    )

    return fig

def crear_frames_reproduccion_fluida(poblacion_anterior, poblacion_nueva, fitness_anterior, fitness_nueva, generacion, limites, num_steps=5):
    """Crea frames intermedios fluidos que muestran el proceso de reproducción"""
    frames_intermedios = []

    # Para cada paso de interpolación
    for step in range(num_steps):
        t = (step + 1) / num_steps  # 0.2, 0.4, 0.6, 0.8, 1.0

        # Interpolar posiciones entre población anterior y nueva
        temps_interp = []
        presiones_interp = []
        flujos_interp = []
        fitness_interp = []

        for i in range(len(poblacion_nueva)):
            if i < len(poblacion_anterior):
                # Interpolación suave con easing
                ease_t = t * t * (3 - 2 * t)  # smoothstep easing

                temp = poblacion_anterior[i][0] * (1 - ease_t) + poblacion_nueva[i][0] * ease_t
                pres = poblacion_anterior[i][1] * (1 - ease_t) + poblacion_nueva[i][1] * ease_t
                flujo = poblacion_anterior[i][2] * (1 - ease_t) + poblacion_nueva[i][2] * ease_t
                fit = fitness_anterior[i] * (1 - ease_t) + fitness_nueva[i] * ease_t
            else:
                # Nuevos individuos aparecen gradualmente
                temp = poblacion_nueva[i][0]
                pres = poblacion_nueva[i][1]
                flujo = poblacion_nueva[i][2]
                fit = fitness_nueva[i] * t  # Aparecen gradualmente

            temps_interp.append(temp)
            presiones_interp.append(pres)
            flujos_interp.append(flujo)
            fitness_interp.append(fit)

        # Normalizar fitness
        fitness_norm_interp = [(f + 1000) / 11 if f > -1000 else 0 for f in fitness_interp]

        # Encontrar mejor
        idx_mejor = fitness_interp.index(max(fitness_interp))

        # Separar mejor del resto
        temps_resto = [temps_interp[i] for i in range(len(temps_interp)) if i != idx_mejor]
        presiones_resto = [presiones_interp[i] for i in range(len(presiones_interp)) if i != idx_mejor]
        flujos_resto = [flujos_interp[i] for i in range(len(flujos_interp)) if i != idx_mejor]
        fitness_norm_resto = [fitness_norm_interp[i] for i in range(len(fitness_norm_interp)) if i != idx_mejor]

        # Crear frame
        frame_data = [
            go.Scatter3d(x=[], y=[], z=[], mode='lines', name='Vectores'),  # Sin vectores durante transición
            go.Scatter3d(
                x=temps_resto,
                y=presiones_resto,
                z=flujos_resto,
                mode='markers',
                marker=dict(
                    size=10,
                    color=fitness_norm_resto,
                    colorscale='Viridis',
                    colorbar=dict(title="Fitness", x=1.02),
                    opacity=0.85,
                    line=dict(color='white', width=1),
                    cmin=0,
                    cmax=100
                ),
                name=f'Población ({len(temps_resto)} individuos)'
            ),
            go.Scatter3d(
                x=[temps_interp[idx_mejor]],
                y=[presiones_interp[idx_mejor]],
                z=[flujos_interp[idx_mejor]],
                mode='markers',
                marker=dict(size=18, color='gold', symbol='diamond', line=dict(color='red', width=4)),
                name='★ Mejor'
            )
        ]

        frames_intermedios.append(go.Frame(
            data=frame_data,
            name=f"gen{generacion}_step{step}",
            layout=go.Layout(
                title_text=f'🧬 Reproducción en progreso - Generación {generacion + 1} ({int(t*100)}%)'
            )
        ))

    return frames_intermedios

def crear_animacion_3d_completa(historial_generaciones, limites, mostrar_reproduccion=True):
    """Crea una animación 3D completa con todos los frames de la evolución, incluyendo reproducción fluida"""

    frames = []
    frame_counter = 0
    generacion_a_frame = {}  # Mapea número de generación al índice del frame final

    for idx_gen, gen_data in enumerate(historial_generaciones):
        poblacion = gen_data["poblacion"]
        poblacion_anterior = gen_data["poblacion_anterior"]
        fitness = gen_data["fitness"]
        generacion = gen_data["generacion"]

        # VALIDACIÓN: Saltar frames con datos vacíos o inconsistentes
        if not poblacion or not fitness or len(poblacion) != len(fitness):
            continue

        # Si hay población anterior y queremos mostrar reproducción, crear frames intermedios
        if mostrar_reproduccion and poblacion_anterior and idx_gen > 0:
            # Obtener fitness de la generación anterior
            fitness_anterior = historial_generaciones[idx_gen - 1]["fitness"]

            # Crear frames de transición fluida (más frames = más fluido)
            frames_transicion = crear_frames_reproduccion_fluida(
                poblacion_anterior,
                poblacion,
                fitness_anterior,
                fitness,
                generacion,
                limites,
                num_steps=20  # 10 pasos intermedios para máxima fluidez
            )
            frames.extend(frames_transicion)
            frame_counter += len(frames_transicion)

        # Extraer coordenadas
        temps = [ind[0] for ind in poblacion]
        presiones = [ind[1] for ind in poblacion]
        flujos = [ind[2] for ind in poblacion]

        # Normalizar fitness para colores
        fitness_norm = [(f + 1000) / 11 if f > -1000 else 0 for f in fitness]

        # Encontrar el mejor individuo
        idx_mejor = fitness.index(max(fitness))

        # Preparar vectores de movimiento
        vectores_x = []
        vectores_y = []
        vectores_z = []

        if poblacion_anterior:
            for i in range(min(len(poblacion), len(poblacion_anterior))):
                dx = poblacion[i][0] - poblacion_anterior[i][0]
                dy = poblacion[i][1] - poblacion_anterior[i][1]
                dz = poblacion[i][2] - poblacion_anterior[i][2]

                movimiento = (dx**2 + dy**2 + dz**2)**0.5
                if movimiento > 0.5:
                    vectores_x.extend([poblacion_anterior[i][0], poblacion[i][0], None])
                    vectores_y.extend([poblacion_anterior[i][1], poblacion[i][1], None])
                    vectores_z.extend([poblacion_anterior[i][2], poblacion[i][2], None])

        # Crear frame con los datos de esta generación
        frame_data = []

        # Trace 0: Vectores
        if vectores_x:
            frame_data.append(go.Scatter3d(
                x=vectores_x,
                y=vectores_y,
                z=vectores_z,
                mode='lines',
                line=dict(color='rgba(0, 255, 255, 0.5)', width=4),
                name='Vectores',
                showlegend=True,
                hoverinfo='skip'
            ))
        else:
            frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name='Vectores'))

        # Separar el mejor individuo del resto
        temps_resto = [temps[i] for i in range(len(temps)) if i != idx_mejor]
        presiones_resto = [presiones[i] for i in range(len(presiones)) if i != idx_mejor]
        flujos_resto = [flujos[i] for i in range(len(flujos)) if i != idx_mejor]
        fitness_norm_resto = [fitness_norm[i] for i in range(len(fitness_norm)) if i != idx_mejor]

        # Trace 1: Población (sin el mejor)
        frame_data.append(go.Scatter3d(
            x=temps_resto,
            y=presiones_resto,
            z=flujos_resto,
            mode='markers',
            marker=dict(
                size=10,
                color=fitness_norm_resto,
                colorscale='Viridis',
                colorbar=dict(title="Fitness", x=1.02),
                opacity=0.85,
                line=dict(color='white', width=1),
                cmin=0,
                cmax=100
            ),
            text=[f'Ind {i}<br>Fitness: {fitness[i]:.2f}<br>T: {temps[i]:.1f}°C<br>P: {presiones[i]:.1f}bar<br>F: {flujos[i]:.1f}L/min'
                  for i in range(len(poblacion)) if i != idx_mejor],
            hoverinfo='text',
            name=f'Población ({len(temps_resto)} individuos)'
        ))

        # Trace 2: Mejor individuo (más grande y visible)
        frame_data.append(go.Scatter3d(
            x=[temps[idx_mejor]],
            y=[presiones[idx_mejor]],
            z=[flujos[idx_mejor]],
            mode='markers',
            marker=dict(
                size=18,
                color='gold',
                symbol='diamond',
                line=dict(color='red', width=4)
            ),
            text=f'⭐ MEJOR<br>Fitness: {fitness[idx_mejor]:.2f}<br>T: {temps[idx_mejor]:.1f}°C<br>P: {presiones[idx_mejor]:.1f}bar<br>F: {flujos[idx_mejor]:.1f}L/min',
            hoverinfo='text',
            name='★ Mejor'
        ))

        # Frame final de la generación (población completa después de reproducción)
        generacion_a_frame[generacion] = frame_counter  # Guardar el índice
        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_counter),
            layout=go.Layout(
                title_text=f'🧬 Generación {generacion + 1} - Completa'
            )
        ))
        frame_counter += 1

    # Crear figura con el primer frame
    primera_gen = historial_generaciones[0]
    temps0 = [ind[0] for ind in primera_gen["poblacion"]]
    presiones0 = [ind[1] for ind in primera_gen["poblacion"]]
    flujos0 = [ind[2] for ind in primera_gen["poblacion"]]
    fitness0 = primera_gen["fitness"]
    fitness_norm0 = [(f + 1000) / 11 if f > -1000 else 0 for f in fitness0]
    idx_mejor0 = fitness0.index(max(fitness0))

    # Separar el mejor del resto para el frame inicial
    temps0_resto = [temps0[i] for i in range(len(temps0)) if i != idx_mejor0]
    presiones0_resto = [presiones0[i] for i in range(len(presiones0)) if i != idx_mejor0]
    flujos0_resto = [flujos0[i] for i in range(len(flujos0)) if i != idx_mejor0]
    fitness_norm0_resto = [fitness_norm0[i] for i in range(len(fitness_norm0)) if i != idx_mejor0]

    fig = go.Figure(
        data=[
            go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='rgba(0, 255, 255, 0.5)', width=4), name='Vectores'),
            go.Scatter3d(
                x=temps0_resto, y=presiones0_resto, z=flujos0_resto,
                mode='markers',
                marker=dict(size=10, color=fitness_norm0_resto, colorscale='Viridis',
                           colorbar=dict(title="Fitness", x=1.02), opacity=0.85,
                           line=dict(color='white', width=1), cmin=0, cmax=100),
                name=f'Población ({len(temps0_resto)} individuos)'
            ),
            go.Scatter3d(
                x=[temps0[idx_mejor0]], y=[presiones0[idx_mejor0]], z=[flujos0[idx_mejor0]],
                mode='markers',
                marker=dict(size=18, color='gold', symbol='diamond', line=dict(color='red', width=4)),
                name='★ Mejor'
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f'🧬 Evolución de la Población - Generación 1',
                font=dict(size=18, color='white')
            ),
            scene=dict(
                xaxis=dict(
                    title='Temperatura (°C)',
                    range=[limites["temperatura"][0], limites["temperatura"][1]],
                    backgroundcolor="rgb(20, 24, 54)",
                    gridcolor="rgb(50, 50, 50)"
                ),
                yaxis=dict(
                    title='Presión (bar)',
                    range=[limites["presion"][0], limites["presion"][1]],
                    backgroundcolor="rgb(20, 24, 54)",
                    gridcolor="rgb(50, 50, 50)"
                ),
                zaxis=dict(
                    title='Flujo (L/min)',
                    range=[limites["flujo"][0], limites["flujo"][1]],
                    backgroundcolor="rgb(20, 24, 54)",
                    gridcolor="rgb(50, 50, 50)"
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                bgcolor="rgb(10, 10, 10)"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True,
                                      "transition": {"duration": 450, "easing": "cubic-in-out"}}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ],
                x=0.05, y=1.12, xanchor="left", yanchor="top",
                bgcolor="rgba(50, 50, 50, 0.8)",
                bordercolor="white",
                borderwidth=1
            )],
            sliders=[dict(
                active=0,
                yanchor="bottom",
                y=-0.15,
                xanchor="left",
                currentvalue=dict(
                    prefix="Generación: ",
                    visible=True,
                    xanchor="center",
                    font=dict(size=14, color="white")
                ),
                pad=dict(b=10, t=50),
                len=0.85,
                x=0.075,
                bgcolor="rgba(50, 50, 50, 0.8)",
                bordercolor="white",
                borderwidth=2,
                ticklen=5,
                tickcolor="white",
                steps=[dict(args=[[str(generacion_a_frame[gen])], {"frame": {"duration": 0, "redraw": True},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}],
                           method="animate",
                           label=f"Gen {gen + 1}") for gen in sorted(generacion_a_frame.keys())]
            )],
            height=800,
            margin=dict(l=50, r=50, t=120, b=150),
            showlegend=True,
            paper_bgcolor="rgb(10, 10, 10)",
            plot_bgcolor="rgb(10, 10, 10)",
            font=dict(color='white')
        ),
        frames=frames
    )

    return fig

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main_streamlit():
    """Interfaz Streamlit"""
    st.set_page_config(
        page_title="Optimización de Reactor Continuo",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("⚗️ Optimización de Reactor Continuo con Algoritmos Genéticos")
    st.markdown("### Optimización multiobjetivo de parámetros de operación")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")

        st.subheader("Parámetros del Algoritmo")
        num_gen = st.slider("Generaciones", 10, 200, 100, 10)
        tam_pob = st.slider("Tamaño de población", 20, 100, 50, 10)

        st.subheader("Mutación Adaptativa")
        mut_ini = st.slider("Mutación inicial", 0.1, 0.9, 0.5, 0.05)
        mut_fin = st.slider("Mutación final", 0.01, 0.3, 0.05, 0.01)

        st.subheader("Elitismo Adaptativo")
        elit_ini = st.slider("Elitismo inicial", 0.05, 0.3, 0.1, 0.05)
        elit_fin = st.slider("Elitismo final", 0.1, 0.5, 0.3, 0.05)

        st.subheader("Método de Selección")
        metodo_sel = st.selectbox(
            "Tipo de selección",
            ["torneo", "ruleta", "ranking"],
            index=0,
            help="Torneo: Competencia entre k individuos\n"
                 "Ruleta: Probabilidad proporcional al fitness\n"
                 "Ranking: Basado en posición ordenada"
        )

        if metodo_sel == "torneo":
            torneo_k = st.slider("Tamaño del torneo (k)", 2, 10, 3, 1,
                                help="Mayor k = Mayor presión selectiva")
        else:
            torneo_k = 3  # Valor por defecto

        st.subheader("Otros Parámetros")
        p_cruce = st.slider("Probabilidad de cruce", 0.5, 1.0, 0.8, 0.05)
        penal_ini = st.slider("Penalización inicial", -20, 0, -10, 1)
        num_gen_penal = st.slider("Generaciones penalizadas", 0, 10, 2, 1)

        st.divider()

        st.subheader("Límites del Reactor")
        col1, col2 = st.columns(2)
        with col1:
            temp_min = st.number_input("Temp. mín (°C)", 0, 150, 50)
            temp_max = st.number_input("Temp. máx (°C)", 100, 300, 200)
        with col2:
            pres_min = st.number_input("Presión mín (bar)", 0.5, 5.0, 1.0, 0.5)
            pres_max = st.number_input("Presión máx (bar)", 5.0, 20.0, 10.0, 0.5)

        flujo_min = st.number_input("Flujo mín (L/min)", 5, 50, 10)
        flujo_max = st.number_input("Flujo máx (L/min)", 50, 200, 100)

        conv_min = st.slider("Conversión mínima (%)", 10, 50, 30)

        st.divider()

        st.subheader("Pesos de Objetivos")
        p_rend = st.slider("Rendimiento", 0.0, 1.0, 0.5, 0.05)
        p_cost = st.slider("Costo", 0.0, 1.0, 0.25, 0.05)
        p_ener = st.slider("Energía", 0.0, 1.0, 0.25, 0.05)

        suma = p_rend + p_cost + p_ener
        if abs(suma - 1.0) > 0.01:
            st.warning(f"⚠️ Suma: {suma:.2f} (debe ser 1.0)")

        st.divider()

        st.subheader("Animación 3D")
        visualizar_3d = st.checkbox("Activar animación 3D", value=True, help="Crea una animación fluida después de ejecutar el algoritmo")
        if visualizar_3d:
            col1, col2 = st.columns(2)
            with col1:
                velocidad_animacion = st.slider("Velocidad de animación (ms/frame)", 30, 500, 100, 10, help="Duración de cada frame. Menor = más rápido, más fluido.")
            with col2:
                mostrar_reproduccion = st.checkbox("Mostrar reproducción fluida", value=True, help="Interpola suavemente entre generaciones (10 frames intermedios)")
        else:
            velocidad_animacion = 50
            mostrar_reproduccion = True

    params = {
        "num_generaciones": num_gen,
        "tamaño_poblacion": tam_pob,
        "prob_mutacion_inicial": mut_ini,
        "prob_mutacion_final": mut_fin,
        "elitismo_ratio_inicial": elit_ini,
        "elitismo_ratio_final": elit_fin,
        "prob_cruce": p_cruce,
        "penalizacion_inicial": penal_ini,
        "num_gen_penalizadas": num_gen_penal,
        "metodo_seleccion": metodo_sel,  # Agregar método de selección
        "torneo_k": torneo_k,  # Agregar k para torneo
        "limites": {
            "temperatura": (temp_min, temp_max),
            "presion": (pres_min, pres_max),
            "flujo": (flujo_min, flujo_max)
        },
        "pesos": {
            "rendimiento": p_rend,
            "costo": p_cost,
            "energia": p_ener
        },
        "conversion_min": conv_min
    }

    if st.button("🚀 Ejecutar Optimización", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Historial de generaciones para animación
        historial_generaciones = []

        def update_progress(gen, total):
            progress_bar.progress(gen / total)
            status_text.text(f"Generación {gen}/{total}")

        # Callback para capturar generaciones (sin visualización en tiempo real)
        def capturar_generacion(data_gen):
            gen_num = data_gen["generacion"]
            pob = data_gen["poblacion"]
            fit = data_gen["fitness"]

            # Hacer copia profunda para evitar que las referencias se sobrescriban
            copia_profunda = {
                "generacion": gen_num,
                "poblacion": [ind.copy() for ind in pob],
                "poblacion_anterior": [ind.copy() for ind in data_gen["poblacion_anterior"]] if data_gen["poblacion_anterior"] else None,
                "fitness": fit.copy(),
                "mejor_fitness": data_gen["mejor_fitness"],
                "promedio_fitness": data_gen["promedio_fitness"],
                "mejor_individuo": data_gen["mejor_individuo"].copy(),
                "eventos_reproduccion": data_gen["eventos_reproduccion"]
            }

            historial_generaciones.append(copia_profunda)

        resultados = algoritmo_genetico(params, update_progress, capturar_generacion if visualizar_3d else None)

        progress_bar.empty()
        status_text.empty()

        st.session_state["resultados"] = resultados
        st.session_state["params"] = params
        st.session_state["historial_generaciones"] = historial_generaciones
        st.session_state["velocidad_animacion"] = velocidad_animacion
        st.session_state["mostrar_reproduccion"] = mostrar_reproduccion

        st.success("✅ Optimización completada!")

    if "resultados" in st.session_state:
        resultados = st.session_state["resultados"]
        params = st.session_state["params"]

        temp_opt, pres_opt, flujo_opt = resultados["mejor_individuo"]
        rend_opt = calcular_rendimiento(temp_opt, pres_opt, flujo_opt)
        costo_opt = calcular_costo(temp_opt, pres_opt, flujo_opt)
        energia_opt = calcular_consumo_energia(temp_opt, pres_opt, flujo_opt)

        st.divider()
        st.header("📊 Resultados")

        # Mostrar método de selección usado
        metodo_usado = params.get("metodo_seleccion", "torneo")
        if metodo_usado == "torneo":
            metodo_desc = f"Torneo (k={params.get('torneo_k', 3)})"
        elif metodo_usado == "ruleta":
            metodo_desc = "Ruleta (Fitness Proporcional)"
        else:
            metodo_desc = "Ranking Lineal"

        st.info(f"🎯 **Método de Selección:** {metodo_desc} | "
                f"**Generación óptima:** {resultados['mejor_generacion'] + 1}/{params['num_generaciones']}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Fitness", f"{resultados['mejor_fitness']:.2f}")
        with col2:
            st.metric("⚡ Rendimiento", f"{rend_opt:.2f} %")
        with col3:
            st.metric("💰 Costo", f"{costo_opt:.2f} $/min")
        with col4:
            st.metric("🔋 Energía", f"{energia_opt:.2f} kW")

        # Gráficas con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=resultados["historia_fitness"], mode='lines', name='Mejor', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(y=resultados["historia_promedio"], mode='lines', name='Promedio', line=dict(color='blue', width=2, dash='dash')))

        # Título dinámico con método de selección
        titulo_metodo = f"Evolución del Fitness - Método: {metodo_desc}"
        fig.update_layout(title=titulo_metodo, xaxis_title="Generación", yaxis_title="Fitness", height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = go.Figure(go.Bar(
                x=['Temperatura', 'Presión', 'Flujo'],
                y=[temp_opt, pres_opt, flujo_opt],
                text=[f'{temp_opt:.2f}°C', f'{pres_opt:.2f}bar', f'{flujo_opt:.2f}L/min'],
                textposition='outside',
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1']
            ))
            fig_bar.update_layout(title="Parámetros Óptimos", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_obj = go.Figure(go.Bar(
                x=['Rendimiento', 'Costo', 'Energía'],
                y=[rend_opt, costo_opt, energia_opt],
                text=[f'{rend_opt:.2f}%', f'${costo_opt:.2f}/min', f'{energia_opt:.2f}kW'],
                textposition='outside',
                marker_color=['#2ecc71', '#e74c3c', '#f39c12']
            ))
            fig_obj.update_layout(title="Objetivos", height=400)
            st.plotly_chart(fig_obj, use_container_width=True)

        # Gráfica de Composición del Fitness por Generación (barras apiladas)
        st.subheader("Composición del Fitness por Generación")

        evol_params = resultados["evol_parametros"]
        generaciones_list = list(range(1, len(resultados["historia_fitness"]) + 1))

        # Calcular alturas de cada componente
        heights_rend = []
        heights_cost = []
        heights_energ = []

        for fitness_val, rend, cost, energ in zip(resultados["historia_fitness"],
                                                    evol_params["rend"],
                                                    evol_params["costo"],
                                                    evol_params["energia"]):
            # Normalizar proporcionalmente al fitness
            heights_rend.append(fitness_val * (rend / 100) * 0.5)
            heights_cost.append(fitness_val * (1 - cost/100) * 0.25)
            heights_energ.append(fitness_val * (1 - energ/300) * 0.25)

        fig_comp = go.Figure()

        # Barras apiladas
        fig_comp.add_trace(go.Bar(
            x=generaciones_list,
            y=heights_rend,
            name='Rendimiento',
            marker_color='#2ecc71'
        ))
        fig_comp.add_trace(go.Bar(
            x=generaciones_list,
            y=heights_cost,
            name='Costo',
            marker_color='#e74c3c'
        ))
        fig_comp.add_trace(go.Bar(
            x=generaciones_list,
            y=heights_energ,
            name='Energía',
            marker_color='#f39c12'
        ))

        # Marcar la mejor generación
        mejor_gen = resultados["mejor_generacion"]
        fig_comp.add_vline(
            x=mejor_gen + 1,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Mejor Gen {mejor_gen + 1}",
            annotation_position="top"
        )

        fig_comp.update_layout(
            barmode='stack',
            title="Composición del Fitness (Rendimiento + Costo + Energía)",
            xaxis_title="Generación",
            yaxis_title="Composición del Fitness",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_comp, use_container_width=True)

        # Evolución de Parámetros
        st.subheader("Evolución de Parámetros del Reactor")
        col1, col2 = st.columns(2)

        with col1:
            fig_temp_pres = go.Figure()
            fig_temp_pres.add_trace(go.Scatter(
                x=generaciones_list,
                y=evol_params["temp"],
                mode='lines+markers',
                name='Temperatura (°C)',
                line=dict(color='red', width=2),
                yaxis='y'
            ))
            fig_temp_pres.add_trace(go.Scatter(
                x=generaciones_list,
                y=evol_params["pres"],
                mode='lines+markers',
                name='Presión (bar)',
                line=dict(color='blue', width=2, dash='dash'),
                yaxis='y2'
            ))
            fig_temp_pres.update_layout(
                title="Evolución de Temperatura y Presión",
                xaxis_title="Generación",
                yaxis=dict(title=dict(text="Temperatura (°C)", font=dict(color='red')), tickfont=dict(color='red')),
                yaxis2=dict(title=dict(text="Presión (bar)", font=dict(color='blue')), tickfont=dict(color='blue'), overlaying='y', side='right'),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_temp_pres, use_container_width=True)

        with col2:
            fig_flujo = go.Figure()
            fig_flujo.add_trace(go.Scatter(
                x=generaciones_list,
                y=evol_params["flujo"],
                mode='lines+markers',
                name='Flujo',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ))
            fig_flujo.add_hline(y=flujo_opt, line_dash="dash", line_color="red", annotation_text=f"Óptimo: {flujo_opt:.2f}")
            fig_flujo.update_layout(
                title="Evolución del Flujo",
                xaxis_title="Generación",
                yaxis_title="Flujo (L/min)",
                height=400
            )
            st.plotly_chart(fig_flujo, use_container_width=True)

        # Evolución de Objetivos
        st.subheader("Evolución de Objetivos")
        fig_obj_evol = go.Figure()
        fig_obj_evol.add_trace(go.Scatter(
            x=generaciones_list,
            y=evol_params["rend"],
            mode='lines+markers',
            name='Rendimiento (%)',
            line=dict(color='green', width=2),
            marker=dict(size=3),
            yaxis='y'
        ))
        fig_obj_evol.add_trace(go.Scatter(
            x=generaciones_list,
            y=evol_params["costo"],
            mode='lines+markers',
            name='Costo ($/min)',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=3),
            yaxis='y2'
        ))
        fig_obj_evol.add_trace(go.Scatter(
            x=generaciones_list,
            y=evol_params["energia"],
            mode='lines+markers',
            name='Energía (kW)',
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=3),
            yaxis='y3'
        ))
        fig_obj_evol.update_layout(
            title="Evolución de Rendimiento, Costo y Energía",
            xaxis_title="Generación",
            yaxis=dict(title=dict(text="Rendimiento (%)", font=dict(color='green')), tickfont=dict(color='green')),
            yaxis2=dict(title=dict(text="Costo ($/min)", font=dict(color='red')), tickfont=dict(color='red'), overlaying='y', side='right'),
            yaxis3=dict(title=dict(text="Energía (kW)", font=dict(color='orange')), tickfont=dict(color='orange'), overlaying='y', side='right', position=0.85),
            height=450,
            hovermode='x unified'
        )
        st.plotly_chart(fig_obj_evol, use_container_width=True)

        # Análisis de Sensibilidad
        st.subheader("Análisis de Sensibilidad y Trade-offs")
        col1, col2 = st.columns(2)

        with col1:
            # Rendimiento vs Temperatura
            temps_test = np.linspace(params["limites"]["temperatura"][0], params["limites"]["temperatura"][1], 100)
            rends_test = [calcular_rendimiento(t, pres_opt, flujo_opt) for t in temps_test]

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=temps_test, y=rends_test, mode='lines', name='Rendimiento', line=dict(color='blue', width=2)))
            fig_sens.add_vline(x=temp_opt, line_dash="dash", line_color="red", annotation_text=f"T óptima: {temp_opt:.2f}°C")
            fig_sens.update_layout(
                title="Rendimiento vs Temperatura",
                xaxis_title="Temperatura (°C)",
                yaxis_title="Rendimiento (%)",
                height=400
            )
            st.plotly_chart(fig_sens, use_container_width=True)

        with col2:
            # Costo-Energía vs Flujo
            flujos_test = np.linspace(params["limites"]["flujo"][0], params["limites"]["flujo"][1], 50)
            costos_test = [calcular_costo(temp_opt, pres_opt, f) for f in flujos_test]
            energias_test = [calcular_consumo_energia(temp_opt, pres_opt, f) for f in flujos_test]

            fig_tradeoff = go.Figure()
            fig_tradeoff.add_trace(go.Scatter(
                x=flujos_test,
                y=costos_test,
                mode='lines',
                name='Costo ($/min)',
                line=dict(color='red', width=2),
                yaxis='y'
            ))
            fig_tradeoff.add_trace(go.Scatter(
                x=flujos_test,
                y=energias_test,
                mode='lines',
                name='Energía (kW)',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            fig_tradeoff.add_vline(x=flujo_opt, line_dash="dash", line_color="green", annotation_text=f"Flujo óptimo: {flujo_opt:.2f}")
            fig_tradeoff.update_layout(
                title="Costo y Energía vs Flujo",
                xaxis_title="Flujo (L/min)",
                yaxis=dict(title=dict(text="Costo ($/min)", font=dict(color='red')), tickfont=dict(color='red')),
                yaxis2=dict(title=dict(text="Energía (kW)", font=dict(color='blue')), tickfont=dict(color='blue'), overlaying='y', side='right'),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True)

        # Espacio de Búsqueda (Temperatura vs Presión)
        st.subheader("Espacio de Búsqueda (Temperatura vs Presión)")
        temps_mesh = np.linspace(params["limites"]["temperatura"][0], params["limites"]["temperatura"][1], 30)
        pres_mesh = np.linspace(params["limites"]["presion"][0], params["limites"]["presion"][1], 30)

        Z_fitness = []
        for p in pres_mesh:
            row = []
            for t in temps_mesh:
                fitness_val = evaluar_individuo([t, p, flujo_opt], 0, params["limites"], params["pesos"])
                row.append(fitness_val)
            Z_fitness.append(row)

        fig_heatmap = go.Figure(data=go.Contour(
            z=Z_fitness,
            x=temps_mesh,
            y=pres_mesh,
            colorscale='Viridis',
            colorbar=dict(title="Fitness")
        ))
        fig_heatmap.add_trace(go.Scatter(
            x=[temp_opt],
            y=[pres_opt],
            mode='markers',
            name='Óptimo',
            marker=dict(size=15, color='red', symbol='star', line=dict(color='white', width=2))
        ))
        fig_heatmap.update_layout(
            title=f"Espacio de Búsqueda (Flujo fijo = {flujo_opt:.1f} L/min)",
            xaxis_title="Temperatura (°C)",
            yaxis_title="Presión (bar)",
            height=500
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Exportar
        df = pd.DataFrame({
            "Generación": range(1, len(resultados["historia_fitness"]) + 1),
            "Mejor Fitness": resultados["historia_fitness"],
            "Fitness Promedio": resultados["historia_promedio"],
            "Temperatura": resultados["evol_parametros"]["temp"],
            "Presión": resultados["evol_parametros"]["pres"],
            "Flujo": resultados["evol_parametros"]["flujo"]
        })

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar CSV",
            csv,
            "resultados_optimizacion.csv",
            "text/csv",
            use_container_width=True,
            key="download_csv_results"
        )

        if "historial_generaciones" in st.session_state and st.session_state["historial_generaciones"]:
            st.divider()
            st.header("🧬 Evolución de la Población")

            historial_gen = st.session_state["historial_generaciones"]
            vel_anim = st.session_state.get("velocidad_animacion", 100)
            mostrar_reprod = st.session_state.get("mostrar_reproduccion", True)

            with st.spinner("Generando animación 3D..."):
                fig_animacion = crear_animacion_3d_completa(historial_gen, params["limites"], mostrar_reproduccion=mostrar_reprod)

                # Configurar transiciones MUY suaves
                fig_animacion.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = vel_anim
                fig_animacion.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = int(vel_anim * 0.95)
                fig_animacion.layout.updatemenus[0].buttons[0].args[1]["transition"]["easing"] = "cubic-in-out"

            st.plotly_chart(fig_animacion, use_container_width=True, key=f"animacion_3d_{len(historial_gen)}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    main_streamlit()