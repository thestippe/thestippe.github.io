let paColContainer = d3.select("#combined_chart") 

var svg = paColContainer.append("svg")
        .attr("id", 'myidCombined')
        .attr("width", width)
        .attr("height", height)

console.log('times')
console.log(times)
console.log(timesSize)

dt = d3.selectAll('#preattentive_color_list ul')
console.log(dt)

