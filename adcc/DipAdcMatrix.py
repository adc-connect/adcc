from .AdcMatrix import AdcMatrix
from .adc_dip import matrix as dipmatrix

class DipAdcMatrix(AdcMatrix):
    default_block_orders = {
        "adc0":  dict(hh=0, hh_ph=None, hh_ph=None, hhhh_ph=None),  # noqa: E501
        "adc1":  dict(hh=1, hh_ph=None, hh_ph=None, hhhh_ph=None),  # noqa: E501
        "adc2":  dict(hh=2, hh_ph=1,    hh_ph=1,    hhhh_ph=0),     # noqa: E501
    }

    def __generate_matrix_block(self, *args, **kwargs):
        return dipmatrix.block(*args, **kwargs)
