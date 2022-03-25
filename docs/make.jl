using Documenter
using EigenDerivatives

makedocs(
    sitename = "EigenDerivatives",
    format = Documenter.HTML(),
    modules = [EigenDerivatives]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
