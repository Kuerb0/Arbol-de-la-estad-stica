"""Rama ACTUARIAL: biometría y tablas de vida, matemática de vida, no vida (reservas, agregada, ruina, primas,
credibilidad) y demografía."""
from .demografia import (estandarizar_tasas, exposicion_por_edad, indicadores_fecundidad, proyeccion_leslie,
                         tasas_especificas)
from .no_vida import (bootstrap_chain_ladder, chain_ladder, credibilidad_buhlmann, prima_por_principios,
                      probabilidad_ruina, recursion_panjer, simular_siniestralidad_agregada)
from .tablas_vida import (ajustar_ley_mortalidad, decrementos_multiples, probabilidad_supervivencia, qx_ley,
                          tabla_conjunta, tabla_mortalidad, vida_futura)
from .vida import (conmutados, prima_neta, prima_tarifa, provision_matematica, renta_actuarial, seguro_vida,
                   sensibilidad_longevidad)

__all__ = ["ajustar_ley_mortalidad", "bootstrap_chain_ladder", "chain_ladder", "conmutados", "credibilidad_buhlmann",
           "decrementos_multiples", "estandarizar_tasas", "exposicion_por_edad", "indicadores_fecundidad", "prima_neta",
           "prima_por_principios", "prima_tarifa", "probabilidad_ruina", "probabilidad_supervivencia", "provision_matematica",
           "proyeccion_leslie", "qx_ley", "recursion_panjer", "renta_actuarial", "seguro_vida", "sensibilidad_longevidad",
           "simular_siniestralidad_agregada", "tabla_conjunta", "tabla_mortalidad", "tasas_especificas", "vida_futura"]
