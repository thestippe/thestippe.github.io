let containerCont = d3.select("#continuity") 
height = 600
width = 600

var svg = containerCont.append("svg")
        .attr("id", 'myid')
        .attr("width", width)
        .attr("height", height)

var pi = Math.PI
var R = 200
var r = 20
var x0 = 300
var y0 = 300
var n = 15

x_coords = []
y_coords = []

for(var i=0;i<n; i++){
x_coords.push(x0 + R*Math.sin(2.0*pi*i/n))
y_coords.push(y0 + R*Math.cos(2.0*pi*i/n))
}


for(var i=0;i<n;i++){
        svg.append("circle").attr('cx', x_coords[i]).attr('cy', y_coords[i]).attr('r', r).attr('fill', 'none')
                        .attr("stroke", "steelblue").style("stroke-dasharray", ("3, 3")).style('stroke-width', 3)
        }

