        let container = d3.select("#barchart") 
        d3.csv("https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Tokyo-Olympics/Medals.csv",
                d3.autoType).then(showData) // do not rely on default data types!

        function showData(data){
                var maxval = 2 + d3.max(data, d=> d.Gold) // find the appropriate scale, with some margin


                // We first create an empty svg with the appropriate dimensions
                var svg = container.append("svg")
                        .attr("id", 'myid')
                        .attr("width", 1000)
                        .attr("height", 400)

                // Before plotting the data we sort them
                // This must be done before any computation (y and barLength)!
                data.sort(function(b, a) {
                        return a.Gold - b.Gold
                })

                data = data.filter(d => d.Gold > 7)


                // This function associates to each country the appropriate y
                var y = d3.scaleBand()
                        .range([ 0, 300 ])
                        .domain(data.map(function(d) { return d.Team }))
                        .padding(0.2)


                // We now compute the length of each bar by using a linear scaling
                var barLength = d3.scaleLinear().
                        domain([0, maxval]).
                        range([0, 800])


                // We now plot the left axis. We will leave 100px for the legend
                svg.append("g")
                        .call(d3.axisLeft(y))
                        .attr("transform", "translate(150, 0)")

                // We now plot the bottom axis just below 300px (the heigth of the graph)
                 svg.append("g")
                        .call(d3.axisBottom(barLength))
                        .attr("transform", "translate(150, 300)")

                // Adding some random label to the x axis
                svg.append("text")
                        .text("Tokyo 2020 gold medals")
                        .attr("x", 500)
                        .attr("y", 350)

                // We finally plot the result as a vertical bar chart
                svg.selectAll("mybars") // We select "myBars", which does not exist - it doesn't matter, we create it
                        .data(data).enter() // Performing the operation for each new element
                        .append('rect') // For each data in element we create a rectangle
                        .attr("x", 150)  // Each element will be positioned at x=0
                        .attr("y", function(d) { return y(d.Team) })  // We put each bar at the corresponding y
                        .attr("width", function(d) { return barLength(d.Gold) }) // We used the computed length
                        .attr("height", 20) // Each bar will have the sape width
                        .attr('stroke', 'black')
                        .attr('fill', 'steelblue')
        }


