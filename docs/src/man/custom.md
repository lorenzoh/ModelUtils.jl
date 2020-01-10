# Custom models

When wrapping a layer with `Model`, the layer struct's `fieldnames` are stored inside `Model` and are automatically classified into one of 3 categories:

- `model.paramattrs`

    Fields with trainable parameters.
    By default all fields of type `AbstractArray{T,<:AbstractFloat}` are treated as trainable parameters

- `model.childrenattrs`
    Fields that contain children layers. By default fields that are a Flux layer are treated as children.

- `model.settingattrs` 
    Fields with hyperparameters that define the layer structure, e.g. padding on a convolution. Default field category.

While this should work well for even many custom layers, in some cases you have to define what category a field is yourself.