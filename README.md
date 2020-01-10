# ModelUtils.jl

## Vision

Information for every layer:

- name of the layer struct or function
- docstring
- layer hyperparameters
- number, size and (optionally name) of parameters
- children layers if they exist

Additional functionality:
- select subsections of the model easily
- filter parameters by layer type and name
- build parameter groups
- map gradients to parameters
- calculate output sizes from input size
- initialize layers selectively
- get number of parameters in model
- measure model throughput