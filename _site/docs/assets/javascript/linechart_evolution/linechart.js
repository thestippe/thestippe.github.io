let lineContainer = d3.select("#linechart") 
d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/gdp_per_capita_filtered.csv",
        d3.autoType).then(plotLinechart) // do not rely on default data types!

function plotLinechart(data){

        console.log(data)
        var maxval = 40000 //d3.max(data, d=> d.Italy) // find the appropriate scale, with some margin
        var minval = 0 // 0.95*d3.min(data, d=> d.Italy) // find the appropriate scale, with some margin


        // We first create an empty svg with the appropriate dimensions
        var svg = lineContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 1200)
                .attr("height", 950)



        var x = d3.scaleTime()
                .range([ 100, 1050 ])
                .domain([1970, 2017])

        svg.append("g")
                .call(d3.axisBottom(x)
    .tickFormat(d3.format('d')).ticks(5)).attr("transform", "translate(0, 850)")

        var y = d3.scaleLinear().
                domain([maxval, minval]). //Warning: it is reversed: in svg y goes from top to bottom
                range([50, 850])

        svg.append("g")
                .call(d3.axisLeft(y).ticks(5)).attr("transform", "translate(100, 0)")

        svg.append("text")
                .text("Italy adjusted GDP per capita")
                .attr("x", 450)
                .attr("y", 20)

        svg.append("text")
                .text("USD")
                .attr("x", 0)
                .attr("y", 500)

        svg.append("text")
                .text("Year")
                .attr("x", 550)
                .attr("y", 900)

    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 3)
      .attr("d", d3.line()
        .x(function(d) { return x(d.year) })
        .y(function(d) { return y(d["Italy"]) }))
        }


