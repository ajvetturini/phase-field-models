import jax
import math

# dH0: kcal / mol
# dS0: eu = cal / (K mol)
# Below is from the SantaLucia 1998 PNAS paper
NN_TABLE_SL1998 = {
    "AA": (-7.9, -22.2),
    "TT": (-7.9, -22.2),
    "AT": (-7.2, -20.4),
    "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7),
    "TG": (-8.5, -22.7),
    "GT": (-8.4, -22.4),
    "AC": (-8.4, -22.4),
    "CT": (-7.8, -21.0),
    "AG": (-7.8, -21.0),
    "GA": (-8.2, -22.2),
    "TC": (-8.2, -22.2),
    "CG": (-10.6, -27.2),
    "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9),
    "CC": (-8.0, -19.9),
    # flags
    "has_init": True,
    "has_sym": True,
    "init_gc": (0.1, -2.8),
    "init_at": (2.3, 4.1),
    "sym": (0.0, -1.4),
}

def get_dimers(seq):
    """Extract NN dimers from sequence.

    Args:
        seq (str): nucleic acid sequence.

    Returns:
        list: NN dimers.
    """
    return [seq[i : (i + 2)] for i in range(len(seq) - 1)]

def salt_corrected_entropy(s, seq, Na_conc):
    n_bp = len(seq) - 1  # number of stacking interactions
    correction = 0.368 * n_bp * math.log(Na_conc)
    return s + correction


def get_dH_and_dS(seq, couples, thermodynamic_table_dict, Na_conc=1.0):
    # Base nearest-neighbor sum
    h = sum([thermodynamic_table_dict[c][0] for c in couples])
    s = sum([thermodynamic_table_dict[c][1] for c in couples])


    # unified parameters omit terminal corrections
    if thermodynamic_table_dict.get("has_init", False):
        first = seq[0]
        last = seq[-1]

        # Get the appropriate initiation parameters
        init_h_gc = thermodynamic_table_dict["init_gc"][0]
        init_s_gc = thermodynamic_table_dict["init_gc"][1]
        init_h_at = thermodynamic_table_dict["init_at"][0]
        init_s_at = thermodynamic_table_dict["init_at"][1]

        # Check terminal base pair types
        first_is_gc = first in "GC"
        last_is_gc = last in "GC"

        if first_is_gc and last_is_gc:
            # Both ends are G-C
            h += init_h_gc
            s += init_s_gc
        elif not first_is_gc and not last_is_gc:
            # Both ends are A-T
            h += init_h_at
            s += init_s_at
        else:
            # One G-C end and one A-T end: average the parameters
            h += (init_h_gc + init_h_at) / 2.0
            s += (init_s_gc + init_s_at) / 2.0

    # Convert dH to cal/mol
    h_cal = h * 1000.0
    s_corr = salt_corrected_entropy(s, seq, Na_conc)

    return h_cal, s_corr


if __name__ == "__main__":
    seq1 = "CGATCG"  # Reportedly gives a dH of -42200, dS_salt of 1.84log([Na+]), dS_nosalt of -119.1
    seq2 = "GAGCTC"  # Reportedly gives a dH of -42400, dS_salt of 1.84log([Na+]), dS_nosalt of -120.6
    salt = 0.5
    seq1_couples = get_dimers(seq1)
    seq2_couples = get_dimers(seq2)
    seq1_h, seq1_s = get_dH_and_dS(seq1, seq1_couples, NN_TABLE_SL1998, 0.5)
    seq2_h, seq2_s = get_dH_and_dS(seq2, seq2_couples, NN_TABLE_SL1998, 0.5)

    print('Sequence 1')
    print(f'REPORTED: dH = -42200 and dS = -119.1 | MY CALCULATIONS: dH = {seq1_h:.2f} and dS = {seq1_s:.2f}')

    print('Sequence 2')
    print(f'REPORTED: dH = -42400 and dS = -120.6 | MY CALCULATIONS: dH = {seq2_h:.2f} and dS = {seq2_s:.2f}')