
var width = 950
var height = 500

var n = 20

var xMin = 50
var xMax = 450

var xDelta = xMax - xMin

var yMin = 50
var yMax = 450

var yDelta = yMax - yMin

var x = d3.scaleLinear().
        domain([0, 1]). //Warning: it is reversed: in svg y goes from top to bottom
        range([xMin, xMax])

var y = d3.scaleLinear().
        domain([1, 0]). //Warning: it is reversed: in svg y goes from top to bottom
        range([yMin, yMax])

var colorscheme = d3.scaleOrdinal().range(['blue', 'red']).domain([0, 1])

let paColContainer = d3.select("#preattentive_color") 
var ul = d3.select('#preattentive_color_list')
        .append('ul').attr('style','columns: 3')

var svg = paColContainer.append("svg")
        .attr("id", 'myid')
        .attr("width", width)
        .attr("height", height)



xVal = Array.from({length: n}, () => Math.random())

yVal = Array.from({length: n}, () => Math.random())

data = []

for(var i=1; i<n; i++){
        elem = {
                'x': xVal[i],
                'y': yVal[i],
                'col': 'blue',
                'z': 0
        }
        data.push(elem)
}

xn = {
        'x': xVal[0],
        'y': yVal[0],
        'col': 'red',
        'z': 1
}

data.push(xn)

console.log(data)

var pts =   svg.append("g").selectAll("points")
        .data(data)

pts.enter()
        .append("circle")
        .attr("r", 10)
        .attr("fill", (d) => colorscheme(d.z))
        .attr("stroke", (d) => colorscheme(d.z))
        .attr("stroke-width", 1.5)
        .attr("cx", function(d) { return x(d.x) })
        .attr("cy", function(d) { return y(d.y) })
        .on("click", (event) => updateData(event.z))

var oldTime = 0
var newTime = 0
var maxTime = 1000000000000

var times = []

function updateData(z){
        if(z==1){

                newxVal = Array.from({length: n}, () => Math.random())
                newyVal = Array.from({length: n}, () => Math.random())
                newData = []

                for(var i=1; i<n; i++){
                        elem = {
                                'x': newxVal[i],
                                'y': newyVal[i],
                                'col': 'blue',
                                'z': 0
                        }
                        newData.push(elem)
                }
                xn = {
                        'x': newxVal[0],
                        'y': newyVal[0],
                        'col': 'red',
                        'z': 1
                }

                newData.push(xn)

                pts.exit().remove()
                svg.selectAll('*').remove()

                var newpts =   svg.append("g").selectAll("points")
                        .data(newData)

                newpts.enter()
                        .append("circle")
                        .attr("r", 10)
                        .attr("fill", (d) => colorscheme(d.z))
                        .attr("stroke", (d) => colorscheme(d.z))
                        .attr("stroke-width", 1.5)
                        .attr("cx", function(d) { return x(d.x) })
                        .attr("cy", function(d) { return y(d.y) })
                        .on("click", (event) => updateData(event.z))

                oldTime = newTime
                newTime = Date.now()
                var timeDelta = newTime - oldTime
                if(timeDelta>0.0001 && timeDelta < maxTime){
                        times.push(timeDelta)
                        var dt = timeDelta/1000
                        var mean = d3.mean(times)/1000
                        var std = d3.deviation(times)/1000
                        var extn = d3.extent(times)
                        var minTime = d3.min(times)/1000
                        var numTimes = times.length

                        svg.append("text")
                                .text("        N:    "+ numTimes)
                                .attr("x", 500)      
                                .attr("y", 280)      

                        svg.append("text")           
                                .text("Last time:    "+ dt.toFixed(2) +" s")
                                .attr("x", 500)      
                                .attr("y", 300)      

                        svg.append("text")           
                                .text("Mean time:    "+ mean.toFixed(2) +" s")
                                .attr("x", 500)      
                                .attr("y", 320)      

                        svg.append("text")           
                                .text(" time std:    "+ std.toFixed(2) +" s")
                                .attr("x", 500)      
                                .attr("y", 340)      

                        svg.append("text")           
                                .text(" min time:    "+ minTime.toFixed(2) +" s")
                                .attr("x", 500)
                                .attr("y", 360)

                        var xp = d3.scaleLinear()
                                .domain(extn)
                                .range([500, 900])

                        svg.selectAll("timePoints").data(times).enter()
                                .append('circle')
                                .attr('cx', (t) => xp(t))
                                .attr('cy', 400)
                                .attr('r', 4)
                                .attr('stroke', 'steelblue')
                                .attr('fill', 'steelblue')

                        svg.append("g")
                                .call(d3.axisBottom(xp).ticks(4)).attr("transform", "translate(0, 420)")
                        ul.selectAll('li').data(times).enter().append('li').text((d) => d/1000)
                        }


        }

}

pts.exit().remove()

