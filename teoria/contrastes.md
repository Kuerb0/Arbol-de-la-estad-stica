# Contrastes de igualdad: qué test usar

Ampliación de una nota propia sobre contrastes de igualdad. En código: `contrastes.elegir_contraste` aplica este árbol.

## Árbol de decisión (grupos independientes)

```
¿Qué variable comparo entre grupos?
├── Categórica (o 0/1)  ──►  tabla de contingencia
│        ├── 2x2 y alguna frecuencia esperada < 5  ──►  Fisher exacto
│        └── resto  ──►  Chi-cuadrado de independencia   (efecto: V de Cramér)
└── Numérica
         ¿Normal en cada grupo? (n ≥ 30 por grupo: aceptable por TCL; si no, Shapiro-Wilk)
         ├── SÍ ── ¿varianzas iguales? (Levene)
         │      ├── 2 grupos ── sí: t de Student  |  no: t de Welch          (efecto: d de Cohen)
         │      └── >2 grupos ─ sí: ANOVA         |  no: ANOVA de Welch      (efecto: eta²)
         │                       └─ si es significativo: post-hoc Tukey HSD
         └── NO ──  2 grupos: U de Mann-Whitney       (efecto: r biserial de rangos)
                    >2 grupos: Kruskal-Wallis          (efecto: eta² de H)
```

Para muestras apareadas (antes/después, mismos individuos): t pareada (normal), Wilcoxon (no normal),
Friedman (>2 medidas). *Pendiente de implementar en el árbol.*

## Supuestos y cómo comprobarlos

| Supuesto | Cómo se comprueba | Qué hacer si falla |
|---|---|---|
| Independencia | Diseño del estudio (no se testea) | Modelos mixtos / clustering de errores |
| Normalidad | Shapiro-Wilk (n pequeño); QQ-plot; con n ≥ 30 por grupo el TCL lo suaviza | No paramétrico (Mann-Whitney, Kruskal) |
| Homocedasticidad | Levene (centrado en la mediana = Brown-Forsythe, robusto) | Welch (t o ANOVA) |
| Frecuencias esperadas (chi²) | Todas ≥ 5 (o ≤ 20 % por debajo de 5) | Agrupar niveles o Fisher |

## Reglas de Gauss (cómo interpretar)

1. **p-valor ≠ importancia.** Con n grande (decenas de miles) cualquier diferencia mínima sale significativa.
   Reporta siempre el tamaño del efecto: d de Cohen (0.2 pequeño, 0.5 medio, 0.8 grande), V de Cramér (≈0.1/0.3/0.5), eta² (≈0.01/0.06/0.14).
2. **No rechazar H0 no es probar igualdad.** Para demostrar equivalencia hace falta un test de equivalencia (TOST).
3. **Comparaciones múltiples:** con k grupos hay k(k-1)/2 pares; usa Tukey (o Bonferroni/Holm) en vez de varios t-tests sueltos.
4. **Elegir el test mirando los datos** (normalidad, varianzas) puede inflar el error tipo I; documenta la regla de decisión antes de ver los resultados.
5. **Con OLS/GLM**, los p-valores de coeficientes asumen homocedasticidad: usar errores robustos (HC3) si hay heterocedasticidad.

## Cómo reportar

> "Los clientes del cluster 3 pagan en media 120 € más que los del cluster 1 (t de Welch, p < 0.001, d = 0.45, efecto medio)."

Siempre: test usado y por qué, estadístico y p, tamaño del efecto en lenguaje de negocio, y limitaciones.
