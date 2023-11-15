let multiplelineContainer = d3.select("#multiple_linechart") 

d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/gdp_per_capita_filtered.csv",
        d3.autoType).then(plotLinechart) // do not rely on default data types!

function plotLinechart(data){

        // We first create an empty svg with the appropriate dimensions
        var svg = multiplelineContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 1200)
                .attr("height", 900)

        const countries = ["Italy", "Spain", "Greece", "Portugal"]
        var maxval = 40500
        var minval = 0

        var x = d3.scaleTime()
                .range([ 100, 1050 ])
                .domain([1970, 2017])

        svg.append("g")
                .call(d3.axisBottom(x)
    .tickFormat(d3.format('d')).ticks(5)).attr("transform", "translate(0, 800)")

        var y = d3.scaleLinear().
                domain([maxval, minval]). //Warning: it is reversed: in svg y goes from top to bottom
                range([0, 800])

        var colorscheme = d3.scaleOrdinal().range(
        [ "#7fc97f", "#beaed4", "#fdc086", "#386cb0" ])
                .domain(countries)

        svg.append("g")
                .call(d3.axisLeft(y).ticks(5)).attr("transform", "translate(100, 0)")

        svg.append("text")
                .text("Adjusted GDP/capita")
                .attr("x", 500)
                .attr("y", 20)

        svg.append("text")
                .text("USD")
                .attr("x", 0)
                .attr("y", 320)

        svg.append("text")
                .text("Year")
                .attr("x", 650)
                .attr("y", 840)

        y0p = +y(data[data.length-1][countries[0]])

        for(country of countries){

        yn = Math.max(+1.05*y0p, +y(data[data.length-1][country]))


        console.log(yn)
        y0p = +yn

        svg.append("text").text(country)
                .attr("x", 1060)
                .attr("y", yn)
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


