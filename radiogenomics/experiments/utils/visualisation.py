"""
Author: Ibrahim Almakky
Date: 01/04/2021
Institution: MBZUAI
"""

# Copied from "https://gist.github.com/zachguo/10296432"
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(str(x)) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    cm_str = "    " + fst_empty_cell + " "
    # End CHANGES

    for label in labels:
        cm_str += "%{0}s".format(columnwidth) % label + " "

    # print()
    cm_str += "\n"
    # Print rows
    for i, label1 in enumerate(labels):
        # print("    %{0}s".format(columnwidth) % label1, end=" ")
        cm_str += "    %{0}s".format(columnwidth) % label1 + " "
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            # print(cell, end=" ")
            cm_str += cell + " "
        # print()
        cm_str += "\n"
    return cm_str
