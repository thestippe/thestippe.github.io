let scatterplotContainer = d3.select("#my_scatterplot") 
d3.csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        d3.autoType).then(plotLinechart) // do not rely on default data types!

function plotLinechart(data){
        var maxvalX = 9.5 // find the appropriate scale, with some margin
        var maxvalY = 8 // find the appropriate scale, with some margin


        // We first create an empty svg with the appropriate dimensions
        var svg = scatterplotContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", 800)
                .attr("height", 550)



        var x = d3.scaleLinear()
                .range([ 120, 800 ])
                .domain([0, maxvalX])

        svg.append("g")
      .attr("class", "axis")
                .call(d3.axisBottom(x)
    .tickFormat(d3.format('d'))).attr("transform", "translate(0, 500)")

        var y = d3.scaleLinear().
                domain([maxvalY, 0]). //Warning: it is reversed: in svg y goes from top to bottom
                range([0, 450])

        svg.append("g")
      .attr("class", "axis")
                .call(d3.axisLeft(y).tickFormat(d3.format('d'))).attr("transform", "translate(120, 50)")

        // svg.append("text")
        //         .text("Gold price")
        //         .attr("x", 400)
        //         .attr("y", 20)

        svg.append("text")
      .style("font", "20px times")
                .html("Sepal length")
                .attr("x", 0)
                .attr("y", 200)

        svg.append("text")
      .style("font", "20px times")
                .html("[mm]")
                .attr("x", 25)
                .attr("y", 225)

        svg.append("text")
      .style("font", "20px times")
                .text("Sepal width [mm]")
                .attr("x", 450)
                .attr("y", 540)

        svg.append("g").selectAll("points")
                .data(data).enter()
                .append("circle")
                .attr("r", 2.5)
                .attr("fill", "none")
                .attr("stroke", "steelblue")
                .attr("stroke-width", 1.5)
                .attr("cx", function(d) { return x(d.sepal_length) })
                .attr("cy", function(d) { return y(d.sepal_width) })
}



