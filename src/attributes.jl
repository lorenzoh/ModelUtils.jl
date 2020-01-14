using Flux

const FluxLayer = Union{
    Chain, Dense, Maxout, typeof(RNN), typeof(LSTM), typeof(GRU), Conv, ConvTranspose, MaxPool, MeanPool,
    DepthwiseConv, Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
    Hook
}

"""
    AttrType

Enum for layer field types.

- `ParamAttr`

    Field with trainable parameters, e.g. `Dense.W`
- `ChildAttr`

    Field that contains children layers, e.g. `Chain.layers`
- `SettingAttr`

    Fields with hyperparameters that define the layer structure, e.g. `Conv.pad`
"""
@enum AttrType begin
    ParamAttr
    SettingAttr
    ChildAttr
end

"""
    attrtype(layer::L, name::Val)::AttrType

Dispatch on a `layer`, and a field name (wrapped in a Val)
to determine what `AttrType` the field is.

If there is no custom method defined, for a layer type and field name,
`attrtype` will additionally dispatch on the field's value to give a
reasonable default.

By default the layers will be classified as following:

- `ParamAttr`

    Fields of type `AbstractArray{T,<:AbstractFloat}` are treated as trainable
    parameters

- `ChildAttr`

    Fields that are a Flux layer are treated as children.

- `SettingAttr`

    Any other

To customize for your type, define a method like

```julia
attrtype(::MyLayer, ::Val{:myfieldname}) = ChildAttr
```
"""
attrtype(::T, ::Val, ::Any) where T = SettingAttr
attrtype(::T, ::Val, ::AbstractArray) where T = ParamAttr
attrtype(::T, ::Val, ::FluxLayer) where T = ChildAttr
attrtype(::T, ::Val, ::NTuple{N, FluxLayer}) where {N, T} = ChildAttr

# dispatch on
attrtype(layer::Any, field::Val) = attrtype(layer, field, getfield(layer, getval(field)))
attrtype(::Chain, ::Val{:layers}) = ChildAttr
attrtype(::Hook, ::Val{:layer}) = ChildAttr


getval(::Val{T}) where T = T
