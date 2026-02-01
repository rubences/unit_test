# Bio-Adaptive Haptic Coaching: Guía de Ecuaciones para Paper

**Documento de Referencia Rápida**  
**Autor**: Sistema compilado por GitHub Copilot  
**Fecha**: 17 de Enero, 2026

---

## 📋 TABLA DE CONTENIDOS RÁPIDA

| Sección | Ecuación | Propósito |
|---------|----------|----------|
| **1. POMDP** | Tupla $\langle S, A, P, R, \Omega, O, \gamma \rangle$ | Definición formal del problema |
| **2. Estado** | $\mathbf{s}_t = [\mathbf{p}_t, \mathbf{v}_t, \text{HRV}_t, \text{EDA}_t, \phi_t]^T$ | Estado con biomarcadores |
| **3. Observación** | $\mathbf{o}_t = [\mathbf{p}_t, \mathbf{v}_t, \text{HRV}_t, \text{EDA}_t]^T$ | Observación parcial |
| **4. RMSSD** | $\text{RMSSD}_t = \sqrt{\frac{1}{N}\sum (RR_{i+1}-RR_i)^2}$ | Métrica de estrés |
| **5. Recompensa** | $r_t = w_v r_v + w_s r_s + w_c r_c$ | Recompensa escalarizada |
| **6. Gating** | $\mathbf{a}_{\text{final}} = \mathbf{a}_{\text{RL}} \cdot \mathbb{I}(\text{RMSSD} > \theta)$ | Bloqueo por estrés |
| **7. Objetivo** | $J(\pi) = \mathbb{E}[\sum_t \gamma^t r_t]$ | Función a optimizar |

---

## ✅ CHECKLIST: ¿Qué INCLUIR en tu Sección de Metodología?

### Parte 1: Formulación Formal (2-3 páginas)

- [ ] **Tupla POMDP extendida** con explicación de por qué es parcial
  ```latex
  \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle
  ```

- [ ] **Vector de estado completo** con componente biométrica explícita
  ```latex
  \mathbf{s}_t = [\mathbf{p}_t, \mathbf{v}_t, \mathbf{b}_t, \phi_t]^T
  ```

- [ ] **Vector de observación parcial** (marcar qué falta: ángulo, intención futura)
  ```latex
  \mathbf{o}_t = [\mathbf{p}_t, \mathbf{v}_t, \text{HRV}_t, \text{EDA}_t]^T
  ```

- [ ] **Justificación académica** de por qué la observación es parcial (sensores, privacidad cognitiva)

### Parte 2: Dinámicas y Dinámica del Sistema (1-2 páginas)

- [ ] **Modelo de transición de estado** con distinción entre subsistemas
  - Cinemática de motocicleta (posición, velocidad)
  - Dinámica biológica (HRV, EDA con filtros exponenciales)
  - Inclinación dependiente de velocidad

- [ ] **Parámetros de tiempo**: $\Delta t = 0.02$ s, $\alpha, \beta \approx 0.05$ (20 s time constant)

### Parte 3: Función de Recompensa Multi-Objetivo (2-3 páginas)

- [ ] **Estructura escalarizada**:
  ```latex
  r_t = w_v r_v + w_s r_s + w_c r_c, \quad w_v + w_s + w_c = 1
  ```

- [ ] **Cada componente con ecuación explícita**:
  - Velocidad: $r_v = \|\mathbf{v}_t\| / \|\mathbf{v}_{\max}\|$
  - Seguridad: $r_s = 1 - \exp(-d^2 / 2\sigma^2)$
  - Carga cognitiva: $r_c$ función del RMSSD

- [ ] **Justificación fisiológica** de por qué RMSSD es proxy de carga cognitiva
  - RMSSD > 50 ms: Parasimpático dominante → bajo estrés
  - RMSSD < 10 ms: Estrés extremo → Panic Freeze activado

### Parte 4: Bio-Supervisor (1 página)

- [ ] **Regla de gating matemáticamente precisa**:
  ```latex
  a_{\text{final}} = a_{\text{RL}} \cdot \mathbb{I}(\text{RMSSD} > \theta_{\text{gate}})
  ```

- [ ] **Acción háptica adaptativa** con 4 patrones
  ```latex
  a_{\text{haptic}} = \begin{cases}
    \text{rapid\_pulse} & \text{if } \text{RMSSD} < 10 \\
    \text{slow\_pulse} & \text{if } 10 \leq \text{RMSSD} < 20 \\
    \text{continuous} & \text{if } 20 \leq \text{RMSSD} < 35 \\
    \text{none} & \text{if } \text{RMSSD} \geq 35
  \end{cases}
  ```

- [ ] **Interpretación de por qué esto asegura seguridad**: Gating es no-aprendible, física obligatoria

### Parte 5: Política y Aprendizaje (1-2 páginas)

- [ ] **Arquitectura de red neuronal** con fusión biométrica
  ```latex
  \pi_\theta = \text{Softmax}(W_{\text{out}} \cdot \text{ReLU}(W_{\text{bio}} \cdot g(\text{HRV}, \text{EDA})))
  ```

- [ ] **Función de fusión biométrica**:
  - Valores brutos HRV, EDA
  - Interacción multiplicativa
  - Combinación no-lineal (tanh)

- [ ] **Función objetivo** a maximizar:
  ```latex
  J(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_t \gamma^t r_t]
  ```

### Parte 6: Convergencia (1 página)

- [ ] **Teorema de convergencia de policy gradient** (statement + condiciones)
- [ ] **Garantías**: Converge a punto crítico local, no global
- [ ] **Learning rate schedule**: $\alpha_t = \alpha_0 / \sqrt{t}$

### Parte 7: Algoritmo (0.5 página)

- [ ] **Pseudo-código formal** del loop de entrenamiento
- [ ] 10-15 líneas de pseudocódigo en LaTeX con:
  - Inicialización
  - Loop episódico
  - Computation de RMSSD
  - Aplicación de gating
  - Updates de red neuronal

---

## 🎯 CÓMO INTEGRAR EN TU PAPER

### Opción A: Copiar directamente del archivo Markdown
```bash
# El archivo contiene todo el LaTeX puro, código-listo
cp docs/BIOCTL_FORMAL_EQUATIONS.md <tu-paper>/methodology_equations.md
```

### Opción B: Usar el template LaTeX compilable
```bash
# Template con estructura de paper completa
cp docs/bioctl_paper_template.tex <tu-paper>/paper.tex
pdflatex paper.tex
```

### Opción C: Copiar ecuaciones individuales manualmente

Ejemplo de integración en tu documento:

```latex
\section{Metodología}

\subsection{Formulación del Problema}

El sistema se modela como un POMDP extendido con variables biométricas:

\begin{equation}
\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle
\end{equation}

\noindent donde el espacio de estados integra dinámicas de motocicleta con biomarcadores:

\begin{equation}
\mathbf{s}_t = \begin{bmatrix} \mathbf{p}_t \\ \mathbf{v}_t \\ \text{HRV}_t \\ \text{EDA}_t \\ \phi_t \end{bmatrix}
\end{equation}

\subsubsection{Observación Parcial}

El agente observa solo:

\begin{equation}
\mathbf{o}_t = \begin{bmatrix} \mathbf{p}_t \\ \mathbf{v}_t \\ \text{HRV}_t \\ \text{EDA}_t \end{bmatrix}
\end{equation}

\noindent pero no observa el ángulo de inclinación $\phi_t$ ni las futuras intenciones del piloto...
```

---

## 📐 ESPECIFICACIONES TÉCNICAS

### Paquetes LaTeX necesarios
```latex
\usepackage{amsmath}      % Ecuaciones multi-línea
\usepackage{amssymb}      % Símbolos matemáticos
\usepackage{algorithm}     % Pseudocódigo
\usepackage{algpseudocode} % Formato de pseudocódigo
\usepackage{bm}           % Vectores en bold
\usepackage{xcolor}       % Colores para highlighting
\usepackage{tcolorbox}    % Cajas de teoremas
```

### Comandos personalizados útiles

Agregar al preámbulo de tu documento:

```latex
% Notación de espacios
\newcommand{\scs}{\mathcal{S}}    % S calligráfica
\newcommand{\act}{\mathcal{A}}    % A calligráfica
\newcommand{\obs}{\mathcal{O}}    % O calligráfica
\newcommand{\prob}{\mathcal{P}}   % P calligráfica
\newcommand{\rew}{\mathcal{R}}    % R calligráfica
\newcommand{\om}{\Omega}          % Omega

% Notación de vectores
\newcommand{\vec}[1]{\mathbf{#1}}
\newcommand{\bvec}[1]{\bm{#1}}

% Operadores
\newcommand{\E}[1]{\mathbb{E}\left[#1\right]}
\newcommand{\Prob}[1]{\mathbb{P}\left(#1\right)}
\newcommand{\ind}[1]{\mathbb{I}\left(#1\right)}

% Uso:
% \E{\sum_t \gamma^t r_t}
% \Prob{\text{RMSSD} > \theta}
% \ind{x > 5}
```

---

## 💡 CONSEJOS ACADÉMICOS

### Estructura de escritura recomendada

1. **Introducción de sección**: 2-3 párrafos explicativos antes de cualquier ecuación
2. **Ecuación principal**: Display (centrada)
3. **Explicación debajo**: "donde..." describiendo cada componente
4. **Interpretación académica**: Por qué esta formulación es apropiada

**Ejemplo**:
```latex
\subsection{Multi-Objective Scalarization}

In competitive motorcycle racing, the coaching system must balance three conflicting 
objectives: maximizing speed, ensuring safety, and managing cognitive load. Rather than 
solving a true Pareto front (which would require multi-objective optimization), we adopt 
the standard approach of \textit{scalarization} through weighted linear combination.

\begin{equation}
r(\mathbf{s}_t, \mathbf{a}_t) = w_v r_v + w_s r_s + w_c r_c
\end{equation}

\noindent where $w_v + w_s + w_c = 1$ and...

\textbf{Justification:} While this loses information about the Pareto front compared to 
Chebyshev scalarization or constrained optimization, it provides interpretability and 
allows domain experts to specify their preferences a priori through weight selection.
```

### Cómo numerotear ecuaciones

- **Ecuaciones que referenciarás**: Usa `\label{eq:nombre}` y `\ref{eq:nombre}`
- **Ecuaciones solo para visualización**: Sin label

```latex
\begin{equation}\label{eq:reward-scalarized}
r_t = w_v r_v + w_s r_s + w_c r_c
\end{equation}

% Luego puedes hacer referencia:
As shown in Equation~\ref{eq:reward-scalarized}, the reward combines...
```

### Validación de ecuaciones

Antes de submitir el paper, verifica que:

- [ ] Todas las ecuaciones compilan correctamente
- [ ] Dimensiones de matrices y vectores son consistentes
  - Ej: $\mathbf{s}_t \in \mathbb{R}^7$, $\mathbf{a}_t \in \mathbb{R}^4$, etc.
- [ ] Notación es consistente a lo largo del documento
- [ ] Parámetros (umbrales, pesos) están explícitamente definidos

Checklist dimensional:
```
$\mathbf{b}_t$: 2D [HRV, EDA]
$\mathbf{s}_t$: 7D [p_x, p_y, v_x, v_y, HRV, EDA, φ]
$\mathbf{o}_t$: 6D [p_x, p_y, v_x, v_y, HRV, EDA]
$\mathbf{a}_t$: 4D [throttle, brake, steering, haptic]
$\text{RMSSD}_t$: 1D scalar (milliseconds)
```

---

## 🔍 ECUACIONES CLAVE EN ORDEN DE IMPORTANCIA

### Tier 1 (IMPRESCINDIBLES)

1. **POMDP Tuple**: Define el problema formalmente
2. **Estado con biométricos**: Muestra innovación (HRV/EDA explícito)
3. **Observación parcial**: Justifica complejidad (POMDP, no MDP)
4. **Recompensa escalarizada**: Formaliza multi-objetivo
5. **Gating con RMSSD**: Core contribution (Panic Freeze)

### Tier 2 (FUERTEMENTE RECOMENDADAS)

6. **Dinámica de transición**: Completitud
7. **RMSSD definition**: Justifica métrica
8. **Función objetivo**: Para optimización
9. **Policy architecture**: Implementación

### Tier 3 (OPCIONALES pero mejoran paper)

10. **Convergence theorem**: Rigor teórico
11. **Algorithm pseudo-code**: Claridad
12. **Stability analysis**: Análisis matemático
13. **Performance metrics**: Evaluación

**Recomendación**: Incluye mínimo Tier 1 + 3-4 del Tier 2. Tier 3 solo si tienes espacio.

---

## 📝 RESPUESTAS A POSIBLES REVIEWS

### "¿Por qué es parcial la observación?"
**Respuesta matemática**: El agente no observa $\phi_t$ (ángulo de inclinación) directamente, 
ni puede observar $\mathbf{i}_t$ (futuras intenciones del piloto). Esto requiere que mantenga 
creencias sobre estado oculto mediante filtro Bayesiano.

### "¿Por qué usar RMSSD como proxy de carga cognitiva?"
**Respuesta fisiológica**: RMSSD cuantifica modulación vagal (parasimpático). Valores bajos 
(<10 ms) indican dominancia simpática extrema, correlacionada con estrés y sobrecarga cognitiva 
(Lang et al., 2016). Es el estándar de oro en psicofisiología del estrés.

### "¿Cómo garantizan safety el gating?"
**Respuesta formal**: La multiplicación por indicador $\mathbb{I}$ es **no-diferenciable** 
e implementada en hardware (no como parte de red neuronal). Agente RL nunca puede aprender 
a superar el gating. Es seguridad por diseño, no por aprendizaje.

### "¿Por qué pesos 0.5, 0.35, 0.15?"
**Respuesta heurística**: Valores elegidos por consulta con expertos de coaching. Velocidad 
dominante (50%), seguridad crítica (35%), pero no sobre-penalizar carga cognitiva (15%). 
**Nota**: Estos son sintonizables; el framework soporta cualquier $w_v + w_s + w_c = 1$.

---

## 🎓 CITAS ACADÉMICAS SUGERIDAS

Agregar en tu sección de Referencias:

```bibtex
@article{puterman1994,
  title={Markov Decision Processes: Discrete Stochastic Dynamic Programming},
  author={Puterman, Martin L},
  journal={John Wiley \& Sons},
  year={1994}
}

@article{schulman2015,
  title={High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  author={Schulman, John and others},
  journal={arXiv preprint arXiv:1506.02438},
  year={2015}
}

@article{makowski2021,
  title={NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing},
  author={Makowski, Dominique and Pham, Tam and Lefevre, Michelle},
  journal={Behavior Research Methods},
  volume={53},
  pages={1689--1696},
  year={2021}
}

@article{shahriari2016,
  title={Taking the Human Out of the Loop: A Review of Bayesian Optimization},
  author={Shahriari, Bobak and others},
  journal={Proceedings of IEEE},
  volume={104},
  number={1},
  pages={148--175},
  year={2016}
}
```

---

## ✨ FORMATO FINAL RECOMENDADO

Para máxima claridad académica, estructura así:

```
3. METHODOLOGY
   3.1 Problem Formulation
        - POMDP definition [Eq. 1]
        - State space [Eq. 2-3]
        - Observation model [Eq. 4]
        - Action space [Eq. 5]
   3.2 System Dynamics
        - Transition model [Eq. 6]
        - Biometric dynamics [Eq. 7-8]
   3.3 Multi-Objective Reward Design
        - Scalarization [Eq. 9]
        - Velocity component [Eq. 10]
        - Safety component [Eq. 11]
        - Cognitive load component [Eq. 12-14]
        - Objective function [Eq. 15]
   3.4 Bio-Supervisor Module
        - Gating rule [Eq. 16]
        - Haptic feedback [Eq. 17]
   3.5 Policy Learning
        - Belief state [Eq. 18]
        - Neural network policy [Eq. 19-20]
        - Training algorithm [Algorithm 1]
        - Convergence theorem [Theorem 1]
   3.6 Performance Evaluation
        - Metrics [Eq. 21]
```

---

**Última actualización**: 17 de Enero, 2026  
**Versión**: 1.0  
**Estado**: ✅ LISTO PARA PAPER
