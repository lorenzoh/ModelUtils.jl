
using Flux: params
using Zygote: Grads, gradient

"""
    gradientsonebatch(model, lossfn, batch) -> Grads

Calculate the forward and backward passes once and return
the gradients.

Useful for running hooks once or inspecting gradients without
taking an optimizer step.
"""
function gradientsonebatch(model, lossfn, batch)::Grads
    x, y = batch
    ps = params(model)
    gs = gradient(ps) do
        return lossfn(model(x), y)
    end
    return gs
end
