# Performing calculations with `adcc`

Manual for ADC methods and `adcc` features and how to invoke them.
This is about the high-level user perspective.
To checkout the full `adcc` `python` API, see [`adcc` API reference](reference.md)

Mostly introduce the functionality by examples.

Some examples can also be found in the `examples` folder
of the [`adcc` code repository](https://adc-connect.org/examples).


## Supported features
Currently `adcc` supports all ADC(n) variants up to level 3,
that is ADC(0), ADC(1), ADC(2), ADC(2)-x and ADC(3).
Additionally:
- Core-valence variants:
	- For all ADC(n) methods excluding ADC(3), the core-valence approximation
	  can be applied
- Spin-flip variants:
	- For a black-box computation of low-multiplicity multi-reference problems
	  spin-flip ADC variants can be employed.
- Properties:
	- One-particle properties such as dipole moments and oscillator strengths
	  are available.
- Both restricted as well as unrestricted references are supported.


## Starting and controlling ADC calculations
Discuss `run_adc` and variants.
