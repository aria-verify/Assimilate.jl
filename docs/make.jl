using Varia
using Documenter

DocMeta.setdocmeta!(Varia, :DocTestSetup, :(using Varia); recursive=true)

makedocs(;
    modules=[Varia],
    authors="Matt Graham <m.graham@ucl.ac.uk> and contributors",
    sitename="Varia.jl",
    format=Documenter.HTML(;
        canonical="https://aria-verify.github.io/Varia.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aria-verify/Varia.jl",
    devbranch="main",
)
