let lineContainer = d3.select("#linechart") 
d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/gdp_per_capita_filtered.csv",
        d3.autoType).then(plotLinechart) // do not rely on default data types!

function plotLinechart(data){
        var maxval = 1.1*d3.max(data, d=> d.Greece) // find the appropriate scale, with some margin
        var minval = 0.95*d3.min(data, d=> d.Greece) // find the appropriate scale, with some margin


        // We first create an empty svg with the appropriate dimensions
        var svg = lineContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 800)
                .attr("height", 550)



        var x = d3.scaleTime()
                .range([ 100, 700 ])
                .domain([1970, 2017])

        svg.append("g")
                .call(d3.axisBottom(x)
    .tickFormat(d3.format('d')).ticks(5)).attr("transform", "translate(0, 500)")

        var y = d3.scaleLinear().
                domain([maxval, minval]). //Warning: it is reversed: in svg y goes from top to bottom
                range([0, 500])

        svg.append("g")
                .call(d3.axisLeft(y).ticks(5)).attr("transform", "translate(100, 0)")

        svg.append("text")
                .text("Greece yearly average temperature")
                .attr("x", 300)
                .attr("y", 20)

        svg.append("text")
                .text("C")
                .attr("x", 0)
                .attr("y", 200)

        svg.append("text")
                .text("Year")
                .attr("x", 350)
                .attr("y", 540)

    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 3)
      .attr("d", d3.line()
        .x(function(d) { return x(d.year) })
        .y(function(d) { return y(d["Italy"]) }))
        }


