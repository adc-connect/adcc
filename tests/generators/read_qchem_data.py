from adcc.hdf5io import _extract_dataset
import numpy as np
import h5py


_dims_pref = "dims/"

mp_context_paths = {
    # MP1 data
    "mp1/df_o1v1": "mp1/df_o1v1",
    "mp1/t_o1o1v1v1": "mp1/t_o1o1v1v1"
    # MP2 data
}

adc_context_paths = {
}


def read_qchem_data(context_file: str, **kwargs: str) -> dict:
    """
    Read the desired data from the dumped libctx context.
    The data can be defined through the keyword args in the form
    "mp1/df_o1v1" = "mp1/df",
    where "mp1/df_o1v1" defines the path in the adcman context, while "mp1/df"
    the key under which the imported object is placed in the returned dict.
    If a object can not be found in the context, an exception is raised.
    Since arrays are exported as flattened vectors, a dimension object
    "dims/mp1/df_o1v1" is required for the correct import.
    """
    context = h5py.File(context_file, "r")

    data = {}
    for context_path, key in kwargs.items():
        # load the value from the context
        if context_path not in context:
            raise KeyError(f"Missing required context entry: {context_path}.")
        raw_value = context[context_path]
        # import the object
        assert isinstance(raw_value, h5py.Dataset)
        value = _extract_dataset(raw_value)
        # for numpy arrays we might have to reshape when we find a matching
        # dimension object
        dim_path = _dims_pref + context_path
        if isinstance(value, np.ndarray) and dim_path in context:
            # import the dimensions object
            dims = _extract_dataset(dim_path)
            value = value.reshape(dims)
        data[key] = value
    return data
