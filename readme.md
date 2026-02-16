# Agente Jugador Autónomo mediante Aprendizaje por Refuerzo

## Descripción
Este proyecto implementa un agente inteligente capaz de aprender de forma autónoma a jugar un videojuego 2D utilizando técnicas de Aprendizaje por Refuerzo (Reinforcement Learning).  
El agente no cuenta con reglas predefinidas sobre cómo jugar; en su lugar, aprende a través de la interacción con el entorno, recibiendo recompensas y penalizaciones según sus acciones.

El videojuego seleccionado es un entorno dinámico tipo *Flappy Bird simplificado*, donde el agente debe evitar obstáculos y maximizar su tiempo de supervivencia.

---

## Objetivo del Proyecto
Diseñar e implementar un agente jugador autónomo que:
- Perciba el estado del entorno
- Tome decisiones basadas en una política aprendida
- Mejore su desempeño progresivamente sin intervención humana directa

---

## Tecnologías Utilizadas
- Python 3
- Aprendizaje por Refuerzo (Q-Learning)
- Entorno 2D personalizado
- Librerías estándar (NumPy, opcionalmente Pygame para visualización)

---

## Descripción del Entorno

### Estados
El estado del agente está compuesto por variables discretizadas que describen el entorno, tales como:
- Altura del agente
- Velocidad vertical
- Distancia al próximo obstáculo
- Posición del hueco del obstáculo

### Acciones
El agente puede ejecutar las siguientes acciones:
- No hacer nada (caer)
- Saltar

### Recompensas
- +1 por cada instante que el agente permanece con vida
- +10 por superar un obstáculo
- -100 por colisión

---

## Algoritmo de Aprendizaje

Se utiliza el algoritmo **Q-Learning**, un método de Aprendizaje por Refuerzo basado en valores, que permite al agente aprender una política óptima mediante la actualización iterativa de una tabla Q.

La actualización de la función Q se define como:

Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]

Donde:
- α es la tasa de aprendizaje
- γ es el factor de descuento
- r es la recompensa recibida

---

## Entrenamiento
El agente es entrenado durante múltiples episodios, comenzando con un comportamiento altamente exploratorio. A medida que avanza el entrenamiento, el agente reduce la exploración y explota el conocimiento adquirido.

---

## Resultados Esperados
- Mejora progresiva en la supervivencia del agente
- Incremento en la recompensa acumulada por episodio
- Desarrollo de una política estable y eficiente

---

## Ejecución
1. Clonar el repositorio
2. Instalar dependencias
3. Ejecutar el archivo principal de entrenamiento

---

## Trabajo Futuro
- Incorporar redes neuronales (Deep Q-Learning)
- Aumentar la complejidad del entorno
- Comparar diferentes estrategias de exploración

---

## Autor
Proyecto desarrollado con fines académicos como parte de un curso universitario de Inteligencia Artificial.
