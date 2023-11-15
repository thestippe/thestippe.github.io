
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
var sizef = d3.scaleOrdinal().range([7, 11]).domain([0, 1])

let paSizeContainer = d3.select("#preattentive_size") 

var ulSize = d3.select('#preattentive_size_list')
        .append('ul').attr('style','columns: 3')

var svg1 = paSizeContainer.append("svg")
        .attr("id", 'myid1')
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

var pts1 =   svg1.append("g").selectAll("points1")
        .data(data)


pts1.enter()
        .append("circle")
        .attr("r", 10)
        .attr("fill", "blue")
        .attr("stroke", "blue")
        .attr("stroke-width", 1.5)
        .attr("cx", function(d) { return x(d.x) })
        .attr("cy", function(d) { return y(d.y) })

svg1.append('rect')
        .attr("x", function(d) { return x(
                data[data.length-1].x)-10 })
        .attr("y", function(d) { return y(
                data[data.length-1].y)-10 })
        .attr('width', 20)
        .attr('height', 20)
        .attr("fill", "blue")
        .attr("stroke", "blue")
        .attr("stroke-width", 1.5)
        .on("click", (event) => updateDataSize(1))

var oldTime = 0
var newTimeSize = 0
var maxTimeSize = 1000000000000

var timesSize = []

function updateDataSize(z){
        if(z==1){

                newxVal = Array.from({length: n}, () => Math.random())
                newyVal = Array.from({length: n}, () => Math.random())
                newDataSize = []

                for(var i=1; i<n; i++){
                        elem = {
                                'x': newxVal[i],
                                'y': newyVal[i],
                                'col': 'blue',
                                'z': 0
                        }
                        newDataSize.push(elem)
                }
                xn = {
                        'x': newxVal[0],
                        'y': newyVal[0],
                        'col': 'red',
                        'z': 1
                }

                newDataSize.push(xn)

                pts1.exit().remove()
                svg1.selectAll('*').remove()

                var newpts1 =   svg1.append("g").selectAll("points1")
                        .data(newDataSize)

                newpts1.enter()
                        .append("circle")
                        .attr("r", 10)
                        .attr("fill", "blue")
                        .attr("stroke", "blue")
                        .attr("stroke-width", 1.5)
                        .attr("cx", function(d) { return x(d.x) })
                        .attr("cy", function(d) { return y(d.y) })

                svg1.append('rect')
                        .attr("x", function(d) { return x(
                                newDataSize[newDataSize.length-1].x)-10 })
                        .attr("y", function(d) { return y(
                                newDataSize[newDataSize.length-1].y)-10 })
                        .attr('width', 20)
                        .attr('height', 20)
                        .attr("fill", "blue")
                        .attr("stroke", "blue")
                        .attr("stroke-width", 1.5)
                        .on("click", (event) => updateDataSize(1))


                oldTimeSize = newTimeSize
                newTimeSize = Date.now()
                var timeDeltaSize = newTimeSize - oldTimeSize
                if(timeDeltaSize>0.0001 && timeDeltaSize < maxTime){
                        timesSize.push(timeDeltaSize)
                        var dt = timeDeltaSize/1000
                        var mean = d3.mean(timesSize)/1000
                        var std = d3.deviation(timesSize)/1000
                        var extn = d3.extent(timesSize)
                        var minTime = d3.min(timesSize)/1000
                        var numTimes = timesSize.length

                        svg1.append("text")
                                .text("        N:    "+ numTimes)
                                .attr("x", 500)      
                                .attr("y", 280)      

                        svg1.append("text")           
                                .text("Last time:    "+ dt.toFixed(2) +" s")
                                .attr("x", 500)      
                                .attr("y", 300)      

                        svg1.append("text")           
                                .text("Mean time:    "+ mean.toFixed(2) +" s")
                                .attr("x", 500)      
                                .attr("y", 320)      

                        svg1.append("text")           
                                .text(" time std:    "+ std.toFixed(2) +" s")
                                .attr("x", 500)      
                                .attr("y", 340)      

                        svg1.append("text")           
                                .text(" min time:    "+ minTime.toFixed(2) +" s")
                                .attr("x", 500)
                                .attr("y", 360)

                        var xp = d3.scaleLinear()
                                .domain(extn)
                                .range([500, 900])

                        svg1.selectAll("timePoints").data(timesSize).enter()
                                .append('circle')
                                .attr('cx', (t) => xp(t))
                                .attr('cy', 400)
                                .attr('r', 4)
                                .attr('stroke', 'steelblue')
                                .attr('fill', 'steelblue')


                        svg1.append("g")
                                .call(d3.axisBottom(xp).ticks(4)).attr("transform", "translate(0, 420)")

                        ulSize.selectAll('li').data(timesSize).enter().append('li').text((d) => d/1000)
                        }


        }

}

pts1.exit().remove()

