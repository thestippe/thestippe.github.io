let multiplelineContainer = d3.select("#multiple_linechart") 
d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/yearly_mean_temperatures.csv",
        d3.autoType).then(plotLinechart) // do not rely on default data types!

function plotLinechart(data){

        // We first create an empty svg with the appropriate dimensions
        var svg = multiplelineContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 800)
                .attr("height", 550)

        const countries = ["Italy", "Spain", "Greece", "Portugal"]
        var maxval = 19
        var minval = 10

        var x = d3.scaleTime()
                .range([ 100, 700 ])
                .domain([1900, 2013])

        svg.append("g")
                .call(d3.axisBottom(x)
    .tickFormat(d3.format('d')).ticks(5)).attr("transform", "translate(0, 500)")

        var y = d3.scaleLinear().
                domain([maxval, minval]). //Warning: it is reversed: in svg y goes from top to bottom
                range([0, 500])

        var colorscheme = d3.scaleOrdinal().range(
        [ "#7fc97f", "#beaed4", "#fdc086", "#386cb0" ])
                .domain(countries)

        svg.append("g")
                .call(d3.axisLeft(y).ticks(5)).attr("transform", "translate(100, 0)")

        svg.append("text")
                .text("Yearly average temperature")
                .attr("x", 400)
                .attr("y", 20)

        svg.append("text")
                .text("C")
                .attr("x", 0)
                .attr("y", 200)

        svg.append("text")
                .text("Year")
                .attr("x", 450)
                .attr("y", 540)

        for(country of countries){

        svg.append("text").text(country)
                .attr("x", 720)
                .attr("y",y(data[data.length-1][country]))
      .style("fill", function(){ return colorscheme(country) })

    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", function(){ return colorscheme(country) })
      .attr("stroke-width", 3)
      .attr("d", d3.line()
        .x(function(d) { return x(d.year) })
        .y(function(d) { return y(d[country]) }))
        }
}


