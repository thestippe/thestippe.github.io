let containerSimilar = d3.select("#similarity") 
height = 210
width = 1300

var svg = containerSimilar.append("svg")
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

var r = 6

function f(x){
        if(x % 2 == 0){
                return 'steelblue'
        }
        else{return 'crimson'}
}

for(x of pos_arr1){
        for(y of pos_arr1){

        svg.append("circle").attr('cx', 2*dx*x).attr('cy', 2*dy*y).attr('r', r).attr('fill', f(x))
                        .attr("stroke", f(x))
        }
}

