let proxContainer = d3.select('#proximityLabel')
height = 300
width =  500
var svg = proxContainer.append('svg')
.attr('height', height).attr('width', width)
.attr('id', 'proxSvg')

var x = d3.scaleLinear().range([0, 400]).domain([0, 5])
var y = d3.scaleLinear().range([300, 0]).domain([0, 15])

dt1 = [
{'x': 0, 'y': 0},
{'x': 1, 'y': 1},
{'x': 2, 'y': 2},
{'x': 3, 'y': 3},
{'x': 4, 'y': 4},
{'x': 5, 'y': 6},
]

dt2 = [
{'x': 0, 'y': 0},
{'x': 1, 'y': 2},
{'x': 2, 'y': 4},
{'x': 3, 'y': 5},
{'x': 4, 'y': 8},
{'x': 5, 'y': 9},
]

dt3 = [
{'x': 0, 'y': 0},
{'x': 1, 'y': 3},
{'x': 2, 'y': 6},
{'x': 3, 'y': 9},
{'x': 4, 'y': 12},
{'x': 5, 'y': 12},
]

dt = [dt1, dt2, dt3]

var k = 1
for(data of dt){
svg.append('path').datum(data).attr('fill', 'none').attr('stroke', 'grey').attr('stroke-width', 3).attr('d', d3.line().x(d => x(d.x)).y(d => y(d.y)))
svg.append('text').attr('x', x(5)).attr('y', y(data[5].y)).text('Curve '+k).attr('fill', 'grey')
k+=1
}

