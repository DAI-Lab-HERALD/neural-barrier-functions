import JSON 

macro name(x)
    string(x)
end

function save_barrier(B, file_path)
    data = [
        Dict("coefficient" => coefficient(term), "exponents" => exponents(term)) for term in terms(B)
    ]


    open(file_path, "w") do f
        JSON.print(f, data)
    end    
end