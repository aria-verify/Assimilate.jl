using Assimilate
using Documenter

DocMeta.setdocmeta!(Assimilate, :DocTestSetup, :(using Assimilate); recursive=true)

makedocs(;
    modules=[Assimilate],
    authors="Matt Graham <m.graham@ucl.ac.uk> and contributors",
    sitename="Assimilate.jl",
    format=Documenter.HTML(;
        canonical="https://aria-verify.github.io/Assimilate.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aria-verify/Assimilate.jl",
    devbranch="main",
)
