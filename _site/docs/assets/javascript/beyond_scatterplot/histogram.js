
d3.csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        d3.autoType).then(plotHistogram) // do not rely on default data types!

function plotHistogram(data){

        let histogramContainer = d3.select("#hist") 
        var height = 600
        var width = 800

        var maxvalX = 9.5 // find the appropriate scale, with some margin

        var svg = histogramContainer.append('svg').attr('height', height).attr('width', width).attr('id', 'histsvg')

        var x = d3.scaleLinear().domain([0, maxvalX]).range([100, 700])
        var histogram = d3.histogram().value( d=>d.sepal_length ).domain(x.domain()).thresholds(x.ticks(30))

          svg.append("g")
      .attr("transform", "translate(0," + 500 + ")")
      .call(d3.axisBottom(x));

        var bins = histogram(data)

          // Y axis: scale and draw:
  var y = d3.scaleLinear()
      .range([500, 0]);
      y.domain([0, d3.max(bins, function(d) { return d.length; })]);

          svg.append("g").attr('transform', 'translate(100, 0)')
      .call(d3.axisLeft(y));


          // append the bar rectangles to the svg element
          svg.selectAll("rect")
              .data(bins)
              .enter()
              .append("rect")
                .attr("x", 1)
                .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; })
                .attr("width", function(d) { return x(d.x1) - x(d.x0) -1 ; })
                .attr("height", function(d) { return 500 - y(d.length); })
                .style("fill", "steelblue")

        svg.append("text")
                .text("Sepal width [mm]")
                .attr("x", 350).attr('font-size', '16px')
                .attr("y", 540)
}

