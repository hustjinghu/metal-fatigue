# coding: utf-8


def rambosg(s, E, K, n):
    """Calculates the strain from a given stress value according to Ramberg & Osgoods material law (3 parameters requiered).

    Args:
        s (array_like): stress
        E (array_like): Young's modulus
        K (array_like): cyclic strain hardening coefficient
        n (array_like): cyclic strain hardening exponent

    Returns:
        array_like: strain according to Ramberg & Osgood's material law, see [1]

    Notes:
        [1] W. Ramberg, W. R. Osgood: Description of stress-strain curves by three parameters. In: NACA Technical note. Band 902, No. 902, 1943, S. 1–28
    """
    eps = s / E + (s / K)**(1 / n)
    return eps


def masing(s, matlaw, **kwargs):
    """Calculates the change in strain or the stress from a given change in stress or strain according to Masings law and a given material law. 
    The material law must be a explicit function of the strain or stress.

    Args:
        s (array_like): stress/strain
        matlaw (function): initial material law (must be formulated explicitly as a function of s or eps)
        **kwargs (optional): optional Parameters handed to the material law

    Returns:
        array_like: strain/stress according to Masings law [1] and a given material law

    Notes:
        A material behavior according to masing is given, when by a given re-loading the doubled stress and strain changes are applied to the inital material law.

        [1] G. Masing: Eigenspannungen und Verfestigung beim Messing. In: Proc. 2nd Int. Cong. of Appl. Mech. Zürich 1926, S. 332–335
    """
    eps = 1. / 2 * matlaw(2 * s, **kwargs)
    return eps


def paris(K, C, m):
    """Calculates the crack growth rate from a given stress intensity factor range accodring to Paris. 

    Args:
        K (array_like): stress intensity factor
        C (array_like): crack growth coefficient
        m (array_like): crack growth exponent

    Returns:
        array_like: crack growth rate according to Paris [1]

    Notes:
        [1] P.C. Paris, M.P. Gomez, W.E. Anderson: A Rational Analytic Theory of Fatigue. The Trend in Engineering Vo.13 No.1 (1961), S. 9–14
    """
    dadN = C * K**m
    return dadN
