#!/usr/bin/env python3

# Generate valid IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT lines for TensorImpl.cc

# The maximal tensor dimensionality
maxdim = 4


def is_valid(n_contr_idcs, dima, dimb, dimout):
    return dima > 0 and dimb > 0 and \
        dima >= n_contr_idcs and dimb >= n_contr_idcs and dimout > 0 and \
        dimout + n_contr_idcs + n_contr_idcs == dima + dimb


valid_combinations = []
for n_contr_idcs in range(maxdim + 1):
    for dima in range(maxdim + 1):
        for dimb in range(maxdim + 1):
            for dimout in range(maxdim + 1):
                if is_valid(n_contr_idcs, dima, dimb, dimout):
                    valid_combinations.append(
                        (n_contr_idcs, dima, dimb, dimout)
                    )

print("//")
print("// Instantiation generated from TensorImpl/instantiate_valid.py")
print("//")
for n_contr_idcs, dima, dimb, dimout in valid_combinations:
    if n_contr_idcs == 0:
        continue
    print(f"IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT({n_contr_idcs}, {dima}, {dimb}) //")  # noqa: E501

print()
print()
print("//")
print("// Instantiation generated from TensorImpl/instantiate_valid.py")
print("//")
for n_contr_idcs, dima, dimb, dimout in valid_combinations:
    if n_contr_idcs != 0:
        continue
    print(f"IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD({dima}, {dimb}) //")
