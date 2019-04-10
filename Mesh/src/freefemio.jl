##########################
# title: freefemio
# date: 2019-03-28
# tags: julia, freefem, io
######################### 

"""
	readfreefem(filename)
Reads a txt file from freefem++ ofstream
"""
function readfreefem(filename)
	D = zeros(Float64, 0)
	n = 0
	open(filename) do f
		thisLine = f |> readline |> strip

		# read the number of components
		n = parse(Int, thisLine)
		while size(D,1) < n
			thisLine = f |> readline |> strip
			d = readdlm(IOBuffer(thisLine), Float64)
			if size(d,2) > 1
				d = d'
			end
			D = [D; d]
		end
	end
		
	if size(D,1) != n
		error("freefemio.jl line 28; vector length dismatch") 
	end
	return(D)
end

"""
	writefreefem(filename)
Write a txt file to be read by freefem++ ifstream
! The comtent u must be a vector so far!
"""
function writefreefem(filename, u)
	dl = "\t"
	nu = length(u)
	nloop = Int64(floor(nu/5))
	open(filename, "w") do f
		println(f, nu, dl)
		for i = 1:nloop
			n = (i-1)*5
			println(f, dl, u[n+1], dl, u[n+2], dl, u[n+3], dl, u[n+4], dl, u[n+5])
		end
		str = "\t"
		for i = (5*nloop+1):nu
			str = str*string(u[i])*dl
		end
		println(f, str)
	end
end
