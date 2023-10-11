"""@thnhan"""

import numpy as np


def prot_to_token(Vocal_W1, proteins, protlen):
    def to_token(prot):
        token = np.array([25] * protlen)  # 25 la index cua pad
        if len(prot) <= protlen:
            token[:len(prot)] = [vocalmap[aa] for aa in prot]
        else:
            token[:len(prot)] = [vocalmap[aa] for aa in prot[:protlen]]
        return token

    vocalmap = Vocal_W1['vocal']
    vocalmap['_'] = len(vocalmap) - 1  # them ky tu "_" lam pad
    tokens = np.array(list(map(to_token, proteins)))

    return tokens
