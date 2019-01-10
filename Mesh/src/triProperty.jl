##################################################
# Mingjie Liao: mliao0320@sina.com
# 09-Jan-2019
# computing triangle properties  with given X and T
#	: incenter
#	: area
#	: maxh
##################################################


type TriIterator
    Vertex
    Triangle
end

element(Vertex, Triangle) = TriIterator(Vertex, Triangle)
Base.start(Trit::TriIterator) = 0
Base.done(Trit::TriIterator, idx::Int64) = (idx == size(Trit.Triangle, 1))
Base.next(Trit::TriIterator, idx::Int64) = compute(Trit.Vertex, Trit.Triangle[idx+1, :]), idx+1

compute(Vertex, tri) = incenter_maxh_area(Vertex, tri) 
function incenter_maxh_area(X, t)
    xt = X[1:2, t]
    r = diff([xt xt[:,1]],2)
    r = sqrt(sumabs2(r,1))
	h = maximum(r)
    r = r[[2 3 1]]
    x = dot(r,xt[1,:])/sum(r)
    y = dot(r,xt[2,:])/sum(r)

	ax = xt[1,:]
	ay = xt[2,:]

	ay1 = ay[[2 3 1]]
	ay2 = ay[[3 1 2]]

	s = (dot(ax,ay1) - dot(ax,ay2))/2
    return (x,y,h,s)
end

function mesh_incenter_maxh_area(Vex, Tri)
	nT = size(Tri,1)
	Incenter = zeros(Float64, 2, nT)
	MaxH = zeros(Float64, nT)
	Area = zeros(Float64, nT)
	n = 0
	for (x,y,h,s) in element(Vex, Tri)
		n += 1
		Incenter[:,n] = [x y]
		MaxH[n] = h
		Area[n] = s
	end
	return Incenter, MaxH, Area
end
