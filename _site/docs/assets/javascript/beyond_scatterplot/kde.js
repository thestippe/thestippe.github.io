
d3.csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        d3.autoType).then(plotKDE) // do not rely on default data types!

function plotKDE(data){

        let kdeContainer = d3.select("#kde1d") 
        var height = 600
        var width = 800

        var maxvalX = 9.5 // find the appropriate scale, with some margin

        var svg = kdeContainer.append('svg').attr('height', height).attr('width', width).attr('id', 'histsvg')

        var x = d3.scaleLinear().domain([0, maxvalX]).range([100, 750])
        var y = d3.scaleLinear().domain([0.4, 0]).range([100, 500])

        svg.append("g")
                .attr("transform", "translate(0," + 500 + ")")
                .call(d3.axisBottom(x));


        svg.append("g").attr('transform', 'translate(100, 0)')
                .call(d3.axisLeft(y));

        // Compute kernel density estimation
        var kde = kernelDensityEstimator(kernelEpanechnikov(0.5), x.ticks(500))
        var density =  kde( data.map(function(d){  return d.sepal_length; }) )

        // Plot the area
        svg.append("path")
                .attr("class", "mypath")
                .datum(density)
                .attr("fill", "steelblue")
                .attr("opacity", ".8")
                .attr("stroke", "#000")
                .attr("stroke-width", 1)
                .attr("stroke-linejoin", "round")
                .attr("d",  d3.line()
                        .curve(d3.curveBasis)
                        .x(function(d) { return x(d[0]); })
                        .y(function(d) { return y(d[1]); })
                );

        svg.append("text")
                .text("Sepal width [mm]")
                .attr("x", 350).attr('font-size', '16px')
                .attr("y", 540)

        svg.append("text")
                .html("KDE with Gaussian kernel, &sigma;=0.5")
                .attr("x", 300).attr('font-size', '16px')
                .attr("y", 120)


}

// Function to compute density
function kernelDensityEstimator(kernel, X) {
  return function(V) {
    return X.map(function(x) {
      return [x, d3.mean(V, function(v) { return kernel(x - v); })];
    });
  };
}
function kernelEpanechnikov(k) {
  return function(v) {
    //return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
          return Math.exp(-v*v/(k*k))/Math.sqrt(2.0*Math.PI*k*k)
  };
}
