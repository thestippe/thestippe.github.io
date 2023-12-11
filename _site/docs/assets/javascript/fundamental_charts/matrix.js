        let matrixContainer = d3.select("#my_matrix_chart") 
        d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/six_nations_2016_2023_scoretable.csv", d3.autoType).then(showData) // do not rely on default data types!

        function showData(data){

                var maxval = d3.max(data, d=> d.total_points) // find the appropriate scale, with some margin

                // We first create an empty svg with the appropriate dimensions
                var svg = matrixContainer.append("svg")
                        .attr("id", 'myid')
                        .attr("width", 720)
                        .attr("height", 600)


                var domain_x = d3.set(data.map(function(d) { return d.team })).values()

                var domain_y = d3.set(data.map(function(d) { return d.team })).values()


                // X scale
                var x = d3.scalePoint()
                        .range([150, 550])
                        .domain(domain_x)

                svg.append("g")
              .attr("class", "axis")
                        .call(d3.axisTop(x))
            .attr("transform", "translate(0, 50)")

                svg.append("text")
                        .html("Team")
                        .attr("x", 350)
                        .attr("y", 20)

                // Y scale
                var y = d3.scalePoint()
                        .range([100, 550])
                        .domain(domain_y)

                svg.append("text")
                        .html("VS")
                        .attr("x", 0)
                        .attr("y", 300)

                svg.append("g")
                        .attr("class", "axis")
                        .call(d3.axisLeft(y))
                        .attr("transform", "translate(100,0)")

                var size = d3.scaleSqrt()
                        .domain([0, maxval])
                        .range([0, 40]);

                cor = svg.selectAll("cor")
                    .data(data)
                    .enter()
                    .append("circle")
                .attr('r', function(d){ return size(d.total_points) })
                .attr("fill", "steelblue")
                .attr("stroke", "steelblue")
                .attr("cx", function(d) { return x(d.team) })
                .attr("cy", function(d) { return y(d.opponent) })

                // build the legend

                svg.append('line')
                    .style("stroke", "black")
                    .style("stroke-width", 1)
                    .attr("x1", 600)
                    .attr("y1", 50)
                    .attr("x2", 600)
                    .attr("y2", 580);
                let cnt = 0
                const dims = [100, 200, 300, 400]
                for(let sz of dims){
                        svg.append('circle')
                                .attr('cx', 660)
                                .attr('cy', 100+cnt)
                                .attr('r', size(sz))
                                .attr("fill", "steelblue")
                                .attr("stroke", "steelblue")

                        svg.append('text')
                                .attr('x', 645)
                                .attr('y', 100+cnt+5)
                        .text(sz)
                        cnt += 100
                }


}
