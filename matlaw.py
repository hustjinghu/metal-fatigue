def rambosg(s, E, K, n):
    """Calculates the strain from a given stress value according to Ramberg & Osgood's material law (3 parameters requiered).
    
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
    return s / E + (s / K)**n