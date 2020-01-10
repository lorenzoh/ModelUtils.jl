using Flux

const FluxLayer = Union{
    Chain, Dense, Maxout, typeof(RNN), typeof(LSTM), typeof(GRU), Conv, Flux.CrossCor, ConvTranspose, MaxPool, MeanPool,
    DepthwiseConv, Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
    SkipConnection, Hook
}

@enum AttrType begin
    ParamAttr
    SettingAttr
    ChildAttr
    StateAttr
end

attrtype(layer::T, name::Val, value) where T = SettingAttr
attrtype(layer::T, name::Val, value::AbstractArray) where T = ParamAttr
attrtype(layer::T, name::Val, value::FluxLayer) where T = ChildAttr
attrtype(layer::T, name::Val, value::NTuple{N, FluxLayer}) where {N, T} = ChildAttr


attrtype(layer::Chain, name::Val{:layers}, value::Tuple) = ChildAttr
attrtype(layer::Chain, name::Val{:layers}, value::NTuple{N, FluxLayer}) where N = ChildAttr
attrtype(layer::BatchNorm, name::Val{:μ}, value::AbstractArray) = StateAttr
attrtype(layer::BatchNorm, name::Val{:σ²}, value::AbstractArray) = StateAttr
