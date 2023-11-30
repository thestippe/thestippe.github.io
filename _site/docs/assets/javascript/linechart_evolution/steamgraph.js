let steamgraphContainer = d3.select("#steamgraph") 

d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/gdp_per_capita_filtered.csv",
        d3.autoType).then(plotSteamGraph) // do not rely on default data types!

function plotSteamGraph(data){


        // We first create an empty svg with the appropriate dimensions
        var svg = steamgraphContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 1250)
                .attr("height", 650)

        const countries = ["Italy", "Spain", "Greece", "Portugal"]
        var maxval = 130000
        var minval = 0

        var x = d3.scaleTime()
                .range([ 100, 1100 ])
                .domain([1970, 2017])

        var y = d3.scaleLinear().
                domain([maxval, minval]). //Warning: it is reversed: in svg y goes from top to bottom
                range([0, 500])

        var colorscheme = d3.scaleOrdinal().range(
                [ "#7fc97f", "#beaed4", "#fdc086", "#386cb0" ])
                .domain(countries)

        var stackGen = d3.stack()
        .offset(d3.stackOffsetSilhouette)
                .keys(countries);

        var stackedSeries = stackGen(data);

        var areaGen = d3.area()
                .x((d) => x(d.data.year))
                .y0((d) => y(d[0]))
                .y1((d) => y(d[1]));

        svg.append("g")
        .attr("transform", "translate(0, 600)")
                .call(d3.axisBottom(x)
                        .tickFormat(d3.format('d')).ticks(5).tickSize(-550))
        .select(".domain").remove()

        svg.append("text")
                .text("Adjusted GDP/capita")
                .attr("x", 500)
                .attr("y", 20)


        svg.append("text")
                .text("Year")
                .attr("x", 1000)
                .attr("y", 600)
  // Customization
  svg.selectAll(".tick line").attr("stroke", "#b8b8b8")



        svg.selectAll('aa')
                .data(stackedSeries)
                .enter()
                .append("path")
                .attr("d", areaGen)
                .attr("fill", (d) => colorscheme(d.key))
        .attr("transform", "translate(0, -200)")

        for(var k=0; k<4; k++){
                ypos = y(stackedSeries[k][stackedSeries[k].length-1][0]) - 50
        svg.append("text")
                .text(countries[k])
                .attr("x", 1000)
                .attr("y", ypos)
        .attr("transform", "translate(0, -200)")

        }

}


