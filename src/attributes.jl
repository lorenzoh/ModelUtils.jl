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
    attrtype(layer, name::Val, value)::AttrType

Dispatch on a `layer`, a field name (wrapped in a Val), and the value of the field
to determine what `AttrType` it is.

By default the layers will be classified as following:

- `ParamAttr`

    Fields of type `AbstractArray{T,<:AbstractFloat}` are treated as trainable
    parameters

- `ChildAttr`

    Fields that are a Flux layer are treated as children.

- `SettingAttr`

    Default `AttrType`

Example
"""
attrtype(::T, ::Val, ::Any) where T = SettingAttr
attrtype(::T, ::Val, ::AbstractArray) where T = ParamAttr
attrtype(::T, ::Val, ::FluxLayer) where T = ChildAttr
attrtype(::T, ::Val, ::NTuple{N, FluxLayer}) where {N, T} = ChildAttr


attrtype(::Chain, ::Val{:layers}, ::Tuple) = ChildAttr
attrtype(::Chain, ::Val{:layers}, ::NTuple{N, FluxLayer}) where N = ChildAttr
