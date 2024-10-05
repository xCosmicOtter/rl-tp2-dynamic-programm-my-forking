"""
Partie 2 - Programmation dynamique
==================================

Rappel: la programmation dynamique est une technique algorithmique qui
permet de résoudre des problèmes en les décomposant en sous-problèmes
plus petits, et en mémorisant les solutions de ces sous-problèmes pour
éviter de les recalculer plusieurs fois.
"""

# Exercice 1: Fibonacci
# ----------------------
# La suite de Fibonacci est définie par:
#   F(0) = 0
#   F(1) = 1
#   F(n) = F(n-1) + F(n-2) pour n >= 2
#
# Ecrire une fonction qui calcule F(n) pour un n donné.
# Indice: la fonction doit être récursive.


def fibonacci(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci.
    """
    if n == 0 or n == 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


# Exercice 2: Fibonacci avec mémorisation
# ---------------------------------------
# Ecrire une fonction qui calcule F(n) pour un n donné, en mémorisant
# les résultats intermédiaires pour éviter de les recalculer plusieurs
# fois.
# Indice: la fonction doit être récursive.


def fibonacci_memo(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci, en mémorisant les
    résultats intermédiaires.
    """

    if n == 0 or n == 1:
        return n
    f_0 = 0
    f_1 = 1
    for i in range(n-1):
        tmp = f_1
        f_1 = f_0 + f_1
        f_0 = tmp
    return f_1
