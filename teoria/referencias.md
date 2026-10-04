# Dónde leer más (tus manuales)

Mapa de qué manual de tu colección (carpeta de manuales PDF en OneDrive UAH) consultar para cada rama del árbol. Para localizar el tema exacto, busca por el nombre del tema dentro del PDF; solo cito capítulos cuando estoy seguro de la numeración.

| Rama / tema | Dónde mirar |
|---|---|
| Regresión logística, clasificación | **ISLR** (James, Witten, Hastie, Tibshirani), cap. 4; **ESL** (Hastie et al.) |
| Selección de variables y regularización (Ridge/Lasso) | **ISLR** cap. 6; **Harrell, Regression Modeling Strategies** (peligros del stepwise, sobreajuste) |
| Validación, calibración, sobreajuste | **Harrell** (validación con remuestreo, calibración); **Kuhn & Johnson, Applied Predictive Modeling** (evaluación de modelos) |
| Clases desbalanceadas | **Kuhn & Johnson** cap. 16; **Practical Statistics for Data Scientists** (copia incompleta) |
| PCA y K-means | **ISLR** cap. 12; **Johnson & Wichern** y **Härdle**, *Applied Multivariate Statistical Analysis* |
| Contrastes de hipótesis, t, chi², ANOVA, no paramétricos | **Rice**, *Mathematical Statistics and Data Analysis*; **Field**, *Discovering Statistics Using SPSS* (supuestos y post-hoc, muy didáctico) |
| Estimación por máxima verosimilitud, IRLS, teoría de GLM | **Rice** (estimación y verosimilitud); **ESL** |
| Remuestreo, bootstrap, validación cruzada | **ISLR** cap. 5; **Practical Statistics for Data Scientists** |

## Material propio relacionado

- Notas propias de contrastes y de GLM de trabajos de consultoría (resumidas en `contrastes.md` y `glm.md`).

## Máster de Ciencias Actuariales y Financieras (UAH)

Los conceptos de cada asignatura están en `conceptos/catalogo.json` (visor: rama «Conceptos y temario»). Bibliografía que citan las guías docentes y que conviene tener a mano:

| Tema | Referencia de la guía docente |
|---|---|
| Simulación, Monte Carlo, bootstrap | Efron & Tibshirani, *An introduction to the Bootstrap*; Robert & Casella, *Monte Carlo Statistical Methods*; Gentle, *Random Number Generation and Monte Carlo Methods* |
| Reservas por chain-ladder con bootstrap | England & Verrall (1999, 2002); Renshaw & Verrall (1994) |
| Regresión, GLM y aprendizaje estadístico para actuarios | Denuit & Trufin, *Effective statistical learning methods for actuaries*; Frees, *Regression modeling with actuarial and financial applications*; de Jong & Heller, *GLM for insurance data*; Wüthrich & Merz, *Statistical foundations of actuarial learning*; Hastie, Tibshirani & Friedman (ESL); James et al. (ISLR) |
| Recurso abierto | Huang (2026), *Actuarial Data Science: Open Learning Resource* (Zenodo, doi 10.5281/zenodo.20718647); Wüthrich et al. (2026), *AI Tools for Actuaries* (SSRN 5162304) |
| Previsión social | Anderson, *Pension Mathematics for Actuaries*; McGill, *Fundamentals of Private Pensions*; Micocci et al., *Pension Fund Risk Management* |

## Very Normal (canal de estadística)

Canal de Christian P. (doctorando en bioestadística): <https://www.youtube.com/@very-normal>, con boletín en <https://verynormal.substack.com>. Lo que se ha podido verificar y está en el catálogo de conceptos:

- Vídeo «The most important ideas in modern statistics» (<https://www.youtube.com/watch?v=nCyGhqQWj2g>): 26 conceptos (significación, contrafactuales, bootstrap, chequeos predictivos, regularización, modelos jerárquicos y de efectos mixtos, ensayos N-of-1 y de cesta, metaanálisis, Metropolis, conjugadas, análisis de decisión adaptativo, robustez, mediana, colas pesadas, normal, EDA, contrastes).
- Vídeo «Why does the number 30 appear so much in statistics?» (<https://www.youtube.com/watch?v=ixfy7BzNIHc>) — solo el título.
- Posts: «Understanding The Normal Distribution Intuitively», «Question 3: Benefits of Maximum Likelihood Estimation», «Question 4: Calculating Response Rates», «Question 6: Simulating Power», «Question 7: Dealing With Switchers», «Why study statistics? (Stats From Scratch, ch. 1)».
- Libros que el autor menciona en su FAQ: Rosner (*Fundamentals of Biostatistics*), Casella & Berger (*Statistical Inference*), McElreath (*Statistical Rethinking*), Shao, van der Vaart, Kutner, Agresti (*Categorical Data Analysis*), Diggle, Cox, ISLR y Friedman & Piantadosi (*Fundamentals of Clinical Trials*).

**Limitación:** YouTube no deja listar los vídeos del canal sin navegador (la página se construye con JavaScript y el RSS está bloqueado), así que **la lista completa de vídeos está pendiente**. Para completarla: pegar los títulos (o la lista de reproducción) y añadirlos a `conceptos/catalogo.json`.
