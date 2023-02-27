import numpy as np

def reduced_echelon_form(
    system: np.array,
    tol: float = 1e-8
) -> np.array:
    system = system.astype(float)
    row_len, col_len = system.shape
    row = 0
    for col in range(col_len):
        # choose pivot
        pivot_idx = np.argmax(abs(system[row:row_len, col]))
        pivot_idx += row
        # Edge case
        # transform almost zero to zero values
        if abs(system[pivot_idx, col]) <= tol:
            system[row:row_len, col] = np.zeros(row_len-row)
        else:
            # swap rows if pivot row is not current row
            if pivot_idx != row:
                system[[pivot_idx, row]] = system[[row,pivot_idx]]
            # pivot_row = system[row, :]
            # normalize pivot row
            system[row, :] = system[row, :] / system[row, col]
            pivot_row = system[row, :]
            # reduce rows above row with pivot if current row > 0
            if row > 0:
                rows_above = np.arange(row)
                system[rows_above, col:col_len] -= (
                    np.outer(system[rows_above, col], pivot_row[col:col_len])
                )
            # reduce rows below row with pivot if current row < len(row) - 1
            if row < row_len - 1:
                rows_below = np.arange(row+1, row_len)
                system[rows_below, col:col_len] -= (
                    np.outer(system[rows_below, col], pivot_row[col:col_len])
                )
            row += 1
        if row >= row_len:
            break
    return system
