from .AdcMatrix import AdcMatrix
from .adc_dip import matrix as dipmatrix

class DipAdcMatrix(AdcMatrix):
    def __generate_matrix_block(self, *args, **kwargs):
        return dipmatrix.block(*args, **kwargs)
