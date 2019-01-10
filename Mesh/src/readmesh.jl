"""
    read_mesh(filename) -> mesh::Mesh
"""
function read_mesh(fn::AbstractString)
    open(fn, "r") do io
        read_mesh(io)
    end
end

"""
    read_mesh(iostream) -> mesh::Mesh

Reads the mesh nodes, edges and elements stored in the input .mesh file 

Returns an object `mesh` of type `Mesh`, comprising both vector arrays.
"""
function read_mesh(io)

    thisLine = io |> readline |> strip
    while thisLine != "Dimension"
        thisLine = io |> readline |> strip
    end

	thisLine = io |> readline |> strip
    dim = parse(Int, thisLine)
#	println("Dimension = ", dim)
	if dim != 2
		error("Only 2 dimensional problem are considered so far!")
	end
	
	thisLine = io |> readline |> strip
    while thisLine != "Vertices"
        thisLine = io |> readline |> strip
    end

	thisLine = io |> readline |> strip
    NV = parse(Int, thisLine)

#    P = SVector{dim+1,Float64}
	vex = Array{Float64}(dim+1,NV)
    for i in 1:NV
        thisLine = io |> readline |>  strip
        d = readdlm(IOBuffer(thisLine), Float64)
		vex[:,i] = d
    end

	thisLine = io |> readline |> strip
    while thisLine != "Edges"
        thisLine = io |> readline |> strip
    end

	thisLine = io |> readline |> strip
    NE = parse(Int, thisLine)

#    P = SVector{dim+1,Int64}
#	edg = Array{P}(NE)
	edg = Array{Int64}(dim+1, NE)
    for i in 1:NE
        thisLine = io |> readline |>  strip
        d = readdlm(IOBuffer(thisLine), Int64)
		edg[:,i] = d
    end

	thisLine = io |> readline |> strip
    while thisLine != "Triangles"
        thisLine = io |> readline |> strip
    end

	thisLine = io |> readline |> strip
    NT = parse(Int, thisLine)

#    P = SVector{dim+2,Int64}
	tri = Array{Int64}(dim+2,NT)
    for i in 1:NT
        thisLine = io |> readline |>  strip
        d = readdlm(IOBuffer(thisLine), Int64)
		tri[:,i] = d
    end
	return (vex,edg,tri)
end
