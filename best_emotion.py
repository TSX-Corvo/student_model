emotions = ["Ira", "Sorpresa", "Disgusto", "Disfrute", "Miedo", "Tristeza"]

categories = [
    "Ortografía y Gramática",
    "Elementos Narrativos",
    "Gramática y Sintaxis",
    "Comunicación y Lenguaje",
    "Figuras Literarias",
]


def indexfloat(l: list, needle: float, tol=1e-2) -> int:
    for idx, val in enumerate(l):
        if abs(val - needle) < tol:
            return idx
    raise ValueError(f"{needle} not found in list")


qtable = [
    [3.82853782, 4.33134865, -1.29737343, 4.17070399, 1.61471472],
    [0.68014021, 4.82130955, -0.89662077, 4.69724627, 4.099784],
    [0.83423914, 4.19247866, -0.86187273, 1.19621262, 0.0],
    [2.96585827, 1.6741, -0.79239631, 1.62362727, 3.31844102],
    [1.9981, 1.09, -0.8232638, 2.46414049, -0.5],
    [-0.47188862, 6.79346521, 0.0, 0.13721262, 0.26471472],
]

for idx, category in enumerate(qtable):
    print(
        f"La mejor categoría para la emoción '{emotions[idx]}' es {categories[indexfloat(category, max(category))]}"
    )
