import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import wavepacket as wp
import json
import os


# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================
print("Задайте начальные условия системы")

def get_number(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Ошибка. Ведите число")

x_min = get_number("Координата минимума оси X:")
x_max = get_number("Координата максимума оси X:")
mass = get_number("Масса частицы:")

N = 256

# Начальная волновая функция
x0 = get_number("Координата точки начала движения волны:")
sigma = 0.8
k0 = get_number("Кинетическая энергия волны:")

# Потенциальный барьер
V0 = get_number("Потенциальная энергия барьера:")
barrier_width = get_number("Ширина барьера по оси Х:")

# Время расчёта
dt = 0.01
num_steps = 1200

# Каждый 10-й шаг = кадр GIF
frame_step = 10


# ============================================================
# 2. СЕТКА
# ============================================================

dof = wp.grid.PlaneWaveDof(
    x_min,
    x_max,
    N
)

grid = wp.grid.Grid(dof)

dx = (x_max - x_min) / N

x = x_min + dx * np.arange(N)


# ============================================================
# 3. ПОТЕНЦИАЛЬНЫЙ БАРЬЕР
# ============================================================

def potential(x):
    return np.where(
        np.abs(x) <= barrier_width / 2,
        V0,
        0.0
    )


V = potential(x)


# ============================================================
# 4. ГАМИЛЬТОНИАН
# ============================================================

kinetic = wp.operator.CartesianKineticEnergy(
    grid,
    0,
    mass=mass
)

potential_operator = wp.operator.Potential1D(
    grid,
    0,
    potential
)

hamiltonian = kinetic + potential_operator


# ============================================================
# 5. НАЧАЛЬНАЯ ВОЛНОВАЯ ФУНКЦИЯ
# ============================================================

rms = sigma / np.sqrt(2.0)

gaussian = wp.special.Gaussian(
    x0,
    rms=rms,
    p=k0
)

psi0 = wp.builder.product_wave_function(
    grid,
    gaussian
)

psi0 = wp.normalize(psi0)


# ============================================================
# 6. УРАВНЕНИЕ ШРЁДИНГЕРА
# ============================================================

equation = wp.expression.SchroedingerEquation(
    hamiltonian
)

solver = wp.solver.OdeSolver(
    equation,
    dt=dt,
    rtol=1e-7,
    atol=1e-9
)


# ============================================================
# 7. РАССЧЁТ ЭВОЛЮЦИИ
# ============================================================

print("Расчёт эволюции...")

# propagate() возвращает:
#
#     (time, State)
#
# Поэтому сразу разделяем время и состояния.

propagation = list(
    solver.propagate(
        psi0,
        t0=0.0,
        num_steps=num_steps
    )
)

times_all = np.array([
    item[0]
    for item in propagation
])

states_all = [
    item[1]
    for item in propagation
]

print(
    f"Расчёт завершён. "
    f"Получено состояний: {len(states_all)}"
)


# ============================================================
# 8. ВЫБИРАЕМ КАДРЫ ДЛЯ GIF
# ============================================================

frame_indices = np.arange(
    0,
    len(states_all),
    frame_step
)

states = [
    states_all[i]
    for i in frame_indices
]

times = times_all[frame_indices]

print(
    f"Кадров GIF: {len(states)}"
)


# ============================================================
# 9. РАССЧИТЫВАЕМ |psi|^2
# ============================================================

print("Расчёт |psi|²...")

densities = np.array([
    wp.dvr_density(state)
    for state in states
])

print("|psi|² рассчитано.")


# ============================================================
# 10. СОЗДАЁМ ГРАФИК
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.set_xlim(
    x_min,
    x_max
)

density_max = np.max(densities)

ax.set_ylim(
    0,
    density_max * 1.25
)

ax.set_xlabel("x")

ax.set_ylabel(
    r"$|\psi(x,t)|^2$"
)

ax.set_title(
    r"Квантовое туннелирование"
)

ax.grid(
    alpha=0.2
)


# ============================================================
# 11. |psi|^2
# ============================================================

line_density, = ax.plot(
    x,
    densities[0],
    linewidth=2,
    label=r"$|\psi|^2$"
)


# ============================================================
# ПОТЕНЦИАЛЬНЫЙ БАРЬЕР
# ============================================================

ax_V = ax.twinx()

ax_V.set_ylim(
    0,
    V0 * 1.25
)

ax_V.set_ylabel(
    r"$V(x)$"
)

# Полупрозрачный красный барьер
barrier = ax_V.fill_between(
    x,
    0,
    V,
    where=(V > 0),
    color="red",
    alpha=0.25
)

# Красный контур
ax_V.plot(
    x,
    V,
    color="red",
    linewidth=1.5,
    alpha=0.7
)

# Линия V0
ax_V.axhline(
    V0,
    linestyle=":",
    linewidth=1,
    alpha=0.6
)

# Подпись V0
ax_V.text(
    barrier_width / 2 + 0.3,
    V0,
    rf"$V_0={V0}$",
    va="bottom"
)

# ============================================================
# 13. ПОДПИСЬ V0
# ============================================================

ax_V.text(
    barrier_width / 2 + 0.3,
    V0,
    rf"$V_0 = {V0}$",
    va="bottom"
)


# ============================================================
# 14. ВРЕМЯ
# ============================================================

time_text = ax.text(
    0.02,
    0.95,
    f"t = {times[0]:.2f}",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top"
)


# ============================================================
# 15. UPDATE
# ============================================================

def update(frame):

    line_density.set_ydata(
        densities[frame]
    )

    time_text.set_text(
        f"t = {times[frame]:.2f}"
    )

    return (
        line_density,
        time_text
    )

# ============================================================
# 16. СОЗДАНИЕ АНИМАЦИИ
# ============================================================

print("Создание GIF...")

animation = FuncAnimation(
    fig,
    update,
    frames=len(densities),
    interval=40,
    blit=False,
    repeat=True
)


# ============================================================
# 17. СОХРАНЕНИЕ GIF
# ============================================================

base_name = "quantum_tunneling"

def get_unique_filename(base_name, ext=".gif"):
    """
    Возвращает имя файла, которое ещё не существует.
    Если 'quantum_tunneling.gif' занят — пробует
    'quantum_tunneling (1).gif', 'quantum_tunneling (2).gif' и т.д.
    """
    filename = f"{base_name}{ext}"
    if not os.path.exists(filename):
        return filename

    counter = 1
    while True:
        filename = f"{base_name} ({counter}){ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1

output_file = get_unique_filename(base_name, ext=".gif")
print(f"Файл будет сохранён как: {output_file}")

print(f"Сохранение в {output_file}...")

animation.save(
    output_file,
    writer=PillowWriter(fps=25),
    dpi=100
)

plt.close(fig)

print()
print("===================================")
print("GIF успешно создан!")
print(f"Файл: {output_file}")
print("===================================")

def save_params_to_json(params, filename="params.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
    print(f"Параметры сохранены в {filename}")

json_file = os.path.splitext(output_file)[0] + ".json"

save_params_to_json({
    "x_min": x_min,
    "x_max": x_max,
    "mass": mass,
    "x0": x0,
    "sigma": sigma,
    "k0": k0,
    "V0": V0,
    "barrier_width": barrier_width,
    "dt": dt,
    "num_steps": num_steps,
    "frame_step": frame_step,
    "output_file": output_file,
}, json_file)