let containerProxy = d3.select("#proximity") 
height = 210
width = 1300

var svg = containerProxy.append("svg")
        .attr("id", 'myid')
        .attr("width", width)
        .attr("height", height)

var dx = 10
var dy = 10

pos_arr = []

for(var i=1;i<21; i++){
pos_arr.push(i)
}

pos_arr1 = []

for(var i=1;i<11; i++){
pos_arr1.push(i)
}

var r = 4

for(x of pos_arr1){
        for(y of pos_arr1){

        svg.append("circle").attr('cx', 2*dx*x).attr('cy', 2*dy*y).attr('r', r).attr('fill', 'steelblue')
                        .attr("stroke", "steelblue")
        }
}

shift = 400

for(x of pos_arr1){
        for(y of pos_arr){

        svg.append("circle")
                        .attr('cx', shift + 2*dx*x).attr('cy', dy*y).attr('r', r).attr('fill', 'steelblue')
                        .attr("stroke", "steelblue")
        }
}


for(x of pos_arr){
        for(y of pos_arr1){

        svg.append("circle")
                        .attr('cx', shift*2+dx*x).attr('cy', 2*dy*y).attr('r', r).attr('fill', 'steelblue')
                        .attr("stroke", "steelblue")
        }
}
