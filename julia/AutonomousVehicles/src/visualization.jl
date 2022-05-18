using Plots, Plots.PlotMeasures
using LaTeXStrings
gr()

function barrier_plot(system_or_folder, postfix, barrier, lim; num_samples = 200)
    # Level sets
    x1 = LinRange(-lim, lim, num_samples)
    x2 = LinRange(-lim, lim, num_samples)

    # While this wrapping looks odd, it is necessary to allow parametric plots with AbstractPolynomialLike.
    f(x1, x2) = barrier(x1, x2)

    subfolder = folder(system_or_folder)
    data = contour(x1, x2, f,  
                    lw=1.2,
                    linecolor=:blue,
                    contour_labels=true,
                    colorbar_entry=false,
                    levels=[1, 5, 25, 125])
    p = plot(data, xlims=(-lim,lim), ylims=(-lim,lim), aspect_ratio=:equal,
                   title="B(x) level sets - " * subfolder, xlabel=L"x_1", ylabel=L"x_2", legend=:outertopright)

    # Initial set
    r0 = 1
    x0(t) = r0 * cos(t)
    y0(t) = r0 * sin(t)

    plot!(p, x0, y0, 0, 2π, linestyle=:dash, linecolor=:green, label="Initial set")

    # Safe set
    rs = 2
    xs(t) = rs * cos(t)
    ys(t) = rs * sin(t)

    t = LinRange(0, 2π, num_samples)
    xs_bbox, ys_bbox = vcat(-lim, rs, xs.(t), rs, -lim), vcat(-lim, -lim, ys.(t), lim, lim)
    plot!(p, xs_bbox, ys_bbox, fill=(0, :red), fillalpha=0.2, linealpha=0, label="")
    plot!(p, vcat(rs, lim, lim, rs), vcat(-lim, -lim, lim, lim), fill=(0, :red), fillalpha=0.2, linealpha=0, label="")
    plot!(p, xs, ys, 0, 2π, linecolor=:red, label="Safe set")

    display_and_save(system_or_folder, p, "$(subfolder)_barrier_level_sets_$postfix.pdf")
end

function display_and_save(system_or_folder, p, file_name)
    display(p)
    subfolder = folder(system_or_folder)
    savefig(p, joinpath(figure_path(subfolder), file_name))
end

folder(::BarrierFunctions.System{V, D, P}) where {V, D, P} = "autonomous_system"
folder(s::AbstractString) = s

function figure_path(subfolder)
    path = joinpath(@__DIR__, "figure", subfolder)
    mkpath(path)

    return path
end