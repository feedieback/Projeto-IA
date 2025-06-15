import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Entradas
prob_olho_esq = ctrl.Antecedent(np.linspace(0, 1, 101), 'Prob_Olho_Esq')
prob_olho_dir = ctrl.Antecedent(np.linspace(0, 1, 101), 'Prob_Olho_Dir')
ear = ctrl.Antecedent(np.linspace(0, 0.4, 101), 'EAR')

# Saída
nivel_fadiga = ctrl.Consequent(np.linspace(0, 100, 101), 'Nivel_Fadiga')

# Funções de pertinência refinadas
for var in [prob_olho_esq, prob_olho_dir]:
    var['baixa'] = fuzz.trapmf(var.universe, [0.0, 0.0, 0.25, 0.45])   # Aumentado para lidar com olhos naturalmente semicerrados
    var['media'] = fuzz.trimf(var.universe, [0.35, 0.55, 0.75])
    var['alta']  = fuzz.trapmf(var.universe, [0.65, 0.80, 1.0, 1.0])

# EAR mais sensível a olhos semiabertos
ear['baixo'] = fuzz.trapmf(ear.universe, [0.00, 0.00, 0.17, 0.24])
ear['medio'] = fuzz.trimf(ear.universe, [0.22, 0.28, 0.34])
ear['alto']  = fuzz.trapmf(ear.universe, [0.32, 0.36, 0.40, 0.40])

# Níveis de fadiga com transições mais suaves
nivel_fadiga['nulo']     = fuzz.trimf(nivel_fadiga.universe, [0, 0, 15])
nivel_fadiga['baixo']    = fuzz.trimf(nivel_fadiga.universe, [10, 25, 40])
nivel_fadiga['moderado'] = fuzz.trimf(nivel_fadiga.universe, [35, 50, 65])
nivel_fadiga['alto']     = fuzz.trimf(nivel_fadiga.universe, [60, 75, 90])
nivel_fadiga['critico']  = fuzz.trimf(nivel_fadiga.universe, [85, 100, 100])

# Regras fuzzy atualizadas
rules = [
    ctrl.Rule(prob_olho_esq['baixa'] & prob_olho_dir['baixa'] & ear['baixo'], nivel_fadiga['critico']),
    ctrl.Rule(prob_olho_esq['baixa'] & prob_olho_dir['baixa'] & ear['medio'], nivel_fadiga['alto']),
    ctrl.Rule(prob_olho_esq['baixa'] & prob_olho_dir['baixa'] & ear['alto'], nivel_fadiga['moderado']),

    # Novas regras para casos intermediários
    ctrl.Rule(prob_olho_esq['media'] & prob_olho_dir['baixa'] & ear['baixo'], nivel_fadiga['alto']),
    ctrl.Rule(prob_olho_esq['baixa'] & prob_olho_dir['media'] & ear['baixo'], nivel_fadiga['alto']),
    ctrl.Rule(prob_olho_esq['media'] | prob_olho_dir['media'], nivel_fadiga['moderado']),

    ctrl.Rule(prob_olho_esq['alta'] & prob_olho_dir['alta'] & ear['alto'], nivel_fadiga['nulo']),
    ctrl.Rule(prob_olho_esq['alta'] & prob_olho_dir['alta'] & ear['medio'], nivel_fadiga['baixo']),
    ctrl.Rule(prob_olho_esq['media'] & prob_olho_dir['media'] & ear['medio'], nivel_fadiga['moderado']),

    # Casos assimétricos considerados neutros
    ctrl.Rule(prob_olho_esq['alta'] & prob_olho_dir['baixa'] & ear['alto'], nivel_fadiga['baixo']),
    ctrl.Rule(prob_olho_esq['baixa'] & prob_olho_dir['alta'] & ear['alto'], nivel_fadiga['baixo']),
]

# Sistema de controle
fadiga_ctrl = ctrl.ControlSystem(rules)

# Função de inferência com debug opcional
def calcular_nivel_fadiga(prob_esq, prob_dir, ear_val, debug=False):
    sim = ctrl.ControlSystemSimulation(fadiga_ctrl)

    sim.input['Prob_Olho_Esq'] = np.clip(prob_esq, 0, 1)
    sim.input['Prob_Olho_Dir'] = np.clip(prob_dir, 0, 1)
    sim.input['EAR'] = np.clip(ear_val, 0, 0.4)

    try:
        sim.compute()
        resultado = sim.output['Nivel_Fadiga']
        if debug:
            print(f"[DEBUG] Esq: {prob_esq:.2f} Dir: {prob_dir:.2f} EAR: {ear_val:.3f} => Fadiga: {resultado:.2f}")
        return resultado
    except Exception as e:
        print("Erro no sistema fuzzy:", e)
        return 0
