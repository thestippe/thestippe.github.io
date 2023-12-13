let smContainer = d3.select("#sm_linechart") 

d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/gdp_per_capita_filtered.csv",
        d3.autoType).then(plotLinechart) // do not rely on default data types!

function plotLinechart(data){

        // We first create an empty svg with the appropriate dimensions
        var svg = smContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 1200)
                .attr("height",300)

        const countries = ["Italy", "Spain", "Greece", "Portugal"]
        var maxval = 50000
        var minval = 0

        const step = 200
        const dy = 250
        delta = 200
        var i = 0

        for(country of countries){
        x0 = 50 + i*250
        x1 = x0 + 200
        xt = x0 + 100
        var x = d3.scaleTime()
                .range([ x0, x1 ])
                .domain([1970, 2017])
        var k = i*dy+delta

        svg.append("g")
                .call(d3.axisBottom(x)
    .tickFormat(d3.format('d')).ticks(5)).attr("transform", "translate(0, 200)")

        var y = d3.scaleLinear().
                domain([maxval, minval]). //Warning: it is reversed: in svg y goes from top to bottom
                range([10, delta])

        svg.append("g")
                .call(d3.axisLeft(y).ticks(5)).attr("transform", "translate("+ x0 +", 0)")

        svg.append("text").text(country).attr("x", xt).attr("y", 20)

        // svg.append("text")
        //         .text("Yearly average temperature")
        //         .attr("x", 400)
        //         .attr("y", 20)

        // svg.append("text")
        //         .text("C")
        //         .attr("x", 0)
        //         .attr("y", 200)

        // svg.append("text")
        //         .text("Year")
        //         .attr("x", 450)
        //         .attr("y", 540)


    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 3)
      .attr("d", d3.line()
        .x(function(d) { return x(d.year) })
        .y(function(d) { return y(d[country]) }))

        i += 1;
        }
}


