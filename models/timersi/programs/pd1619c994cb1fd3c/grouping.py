import numpy as np
MAIN_PREFIX = ('H', 'M')
AUX_PREFIX = ('L',)

def _family(name):
    token = str(name).upper()
    if len(token) >= 4 and token[1] == 'U' and (token[2] in ('F', 'L')) and (token[3] == 'L'):
        return token[0]
    return None

def voltage_family_groups(names, min_group=2):
    fams = [_family(n) for n in names]
    if any((f is None for f in fams if f is not None)) or all((f is None for f in fams)):
        return [list(range(len(names)))]
    main, aux, other = ([], [], [])
    for i, (n, f) in enumerate(zip(names, fams)):
        if f in MAIN_PREFIX:
            main.append(i)
        elif f in AUX_PREFIX:
            aux.append(i)
        else:
            other.append(i)
    main = main + other
    groups = [g for g in (main, aux) if len(g) >= min_group]
    covered = {i for g in groups for i in g}
    rest = [i for i in range(len(names)) if i not in covered]
    if rest:
        if groups:
            groups[0] = sorted(groups[0] + rest)
        else:
            groups = [list(range(len(names)))]
    return [sorted(g) for g in groups]
JOINT = True

def target_groups(names, min_group=2):
    if JOINT:
        return [list(range(len(names)))]
    return voltage_family_groups(names, min_group=min_group)
