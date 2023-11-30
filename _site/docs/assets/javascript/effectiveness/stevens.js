let container = d3.select("#stevens") 

height = 900
width = 1000

sideBorder = 200
centralBorder = 150

var svg = container.append("svg")
        .attr("id", 'myid')
        .attr("width", width)
        .attr("height", height)

x = []

for (var i=0; i<300; i++){
        x.push(i/100)
}

console.log(x)

let powers = {
        // "depth": 0.67,
        // "lightness": 0.7,
        "brightness": 0.5,
        "area": 0.7,
        "length": 1,
        "saturation": 1.7,
}

var myColor = d3.scaleOrdinal().domain(Object.keys(powers)).range(d3.schemeTableau10)

let y = []


for(s of x){
        var dt = {}
        dt['x'] = s
        for(channel of Object.keys(powers)){
                dt[channel] = s**powers[channel]
        }
        y.push(dt)
}


console.log(y)

var xAxis = d3.scaleLinear()
        .range([ sideBorder, +width - sideBorder ])
        .domain([0, 3])

var yAxis = d3.scaleLinear()
        .range([ centralBorder, +height-centralBorder ])
        .domain([8, 0])

svg.append("g")
        .call(d3.axisLeft(yAxis).ticks(3)).attr("transform", "translate("+sideBorder+", 0)")

svg.append("g")
        .call(d3.axisBottom(xAxis).ticks(3)).attr("transform", "translate(0, "+(+height-centralBorder)+")")

svg.append("text")
        .text("Stimulus")
        .attr("x", 470)
        .attr("y", height-130)

svg.append("text")
        .text("Perceived intensity")
        .attr("x", 100)
        .attr("y", 130)

for(channel of Object.keys(powers)){
        svg.append("path")
                .datum(y)
                .attr("fill", "none")
                .attr("stroke", myColor(channel))
                .attr("stroke-width", 3)
                .attr("d", d3.line()
                        .x(function(d) { return xAxis(d.x) })
                        .y(function(d) { return yAxis(d[channel]) })
                )
        svg.append("text")
                .text(channel+ " (a="+powers[channel]+")")
                .attr("x", xAxis(3))
                .attr("y", yAxis(y[299][channel]))
        .style('fill', myColor(channel))
}


