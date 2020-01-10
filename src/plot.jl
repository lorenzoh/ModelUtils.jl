using StatsPlots


function plotdistributions(names, xs; kwargs...)
    names_ = [string(i, ": ", name) for (i, name) in enumerate(names)]
    labels = map((names, g) -> fill(names, size(g)), names, xs);
    violin(labels, xs; lw=0, xrotation=20, alpha=.8, legend=false, kwargs...)
    boxplot!(labels, xs, alpha=.25)
end


function plotlayers(model, lossfn, batch; filter_fn = l -> true, sz = (900, 600), kwargs...)
    names, acts, grads = layerstats(model, lossfn, batch, filter_fn = filter_fn)
    gradsplot = plotdistributions(names, grads, title = "gradients")
    actsplot = plotdistributions(names, acts, title = "activations")
    l = @layout [a; b]
    plot(actsplot, gradsplot, layout = l, size = sz, kwargs...)
end
