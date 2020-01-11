# ModelUtils.jl documentation
```@meta
CurrentModule = ModelUtils
```
ModelUtils.jl provides utilities for working with Flux.jl models.

## Features

- initialization
- hooks
- iteration 

The most important type it defines is [`Model`](@ref Model), a tree data
structure that wraps your Flux model and is itself a Flux layer.

`ModelUtils.jl` should work with all models that work with Flux,
but for some custom types you might have to extend a few of
`ModelUtils.jl`s methods, see [Custom models](@ref) for more
information.

## Manual

```@contents
Pages = ["man/basic.md", "man/hooks.md", "man/custom.md", "man/initialization.md"]
```

## Reference

```@contents
Pages = ["lib/public.md"]
```