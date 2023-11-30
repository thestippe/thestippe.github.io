let containerDisc = d3.select("#discriminability") 

height = 100
width = 1125

var svg = containerDisc.append("svg")
        .attr("id", 'myid')
        .attr("width", width)
        .attr("height", height)
        .style("background-color", "lightgray")

s = []
var n = 30

var r = width/(2*n)

for (var i=0; i<n; i++){
        s.push(i/n)
        }

var myColor = d3.scaleLinear().domain([0, 1]).range(["#ffffff", "ff0000"])

var x = d3.scaleLinear().domain([0, 1-1/n]).range([r, width-r])


svg.append("g").selectAll("points")
.data(s).enter().append("circle").attr('r', r)
.attr("cy", 50).attr("cx", d => x(d))
.attr("fill", d => d3.interpolateReds(d))


