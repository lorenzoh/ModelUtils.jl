# Custom models

```@meta
CurrentModule = ModelUtils
```

When wrapping a layer with `Model`, the layer struct's `fieldnames` are stored inside `Model` and are automatically classified into one of 3 categories:

- `model.paramattrs`

    Fields with trainable parameters.
    By default all fields of type `AbstractArray{T,<:AbstractFloat}` are treated as trainable parameters

- `model.childrenattrs`
    Fields that contain children layers. By default fields that are a Flux layer are treated as children.

- `model.settingattrs` 
    Fields with hyperparameters that define the layer structure, e.g. padding on a convolution. Default field category.

While this should work well for even many custom layers, in some cases you have to define what category a field is yourself.

The classification happens with dispatch on the function [`attrtype`](@ref).

```@docs
AttrType
attrtype
```

## Defining a custom [`attrtype`](@ref) method

Let's say we create a custom layer that's like `Chain`:

```@example customchain
using Flux, ModelUtils # hide
struct MyChain
    layers
    MyChain(layers...) = new(layers)
end
(m::MyChain)(x) = foldl((x, layer) -> layer(x), m.layers, init = x)
```

Now we want `MyChain.layers` to be a `ChildAttr` but we can check that it is not classified as such:

```@example customchain
mychain = MyChain(Dense(10, 10), Dense(10, 10), softmax)
ModelUtils.attrtype(mychain, Val(:layers))
```

So we can define a custom `attrtype` method:

```@example customchain
import ModelUtils: attrtype
ModelUtils.attrtype(::MyChain, ::Val{:layers}) = ModelUtils.ChildAttr
``` 

And now we get the result we wanted:

```@example customchain
ModelUtils.attrtype(mychain, Val(:layers))
```
