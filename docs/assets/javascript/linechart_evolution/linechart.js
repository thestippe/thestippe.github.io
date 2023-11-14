let lineContainer = d3.select("#linechart") 
d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/temperatures.csv",
        d3.autoType).then(plotLinechart) // do not rely on default data types!

function plotLinechart(data){
        var maxval = 1.1*d3.max(data, d=> d.Greece) // find the appropriate scale, with some margin


        // We first create an empty svg with the appropriate dimensions
        var svg = lineContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 1200)
                .attr("height", 550)



        var x = d3.scaleTime()
                .range([ 100, 800 ])
                .domain([0, 1363])

        svg.append("g")
                .call(d3.axisBottom(x)
    .tickFormat(d3.format('d'))).attr("transform", "translate(0, 500)")

        var y = d3.scaleLinear().
                domain([maxval, 0]). //Warning: it is reversed: in svg y goes from top to bottom
                range([0, 500])

        svg.append("g")
                .call(d3.axisLeft(y)).attr("transform", "translate(100, 0)")

        svg.append("text")
                .text("Greece monthly temperature")
                .attr("x", 400)
                .attr("y", 20)

        svg.append("text")
                .text("C")
                .attr("x", 0)
                .attr("y", 200)

        svg.append("text")
                .text("index")
                .attr("x", 450)
                .attr("y", 540)

    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 3)
      .attr("d", d3.line()
        .x(function(d) { return x(d.id) })
        .y(function(d) { return y(d.Greece) }))
        }


