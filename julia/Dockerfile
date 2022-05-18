FROM julia:1.7.2-buster


RUN apt-get update && apt-get install -y bzip2 unzip

WORKDIR /app/AutonomousVehicles/

# ADD Project.toml /app/Project.toml
# ADD examples/Project.toml /app/examples/Project.toml

ADD . /app/

RUN ["julia", "--project", "-e", "using Pkg; Pkg.develop(path=\"../BarrierFunctions\"); Pkg.instantiate(); Pkg.precompile();"]
RUN julia --project -e "using Pkg; Pkg.status()"

ENTRYPOINT ["julia", "--project", "src/main.jl"]