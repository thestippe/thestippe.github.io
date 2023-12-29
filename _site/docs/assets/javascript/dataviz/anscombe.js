let anscombeContainer = d3.select("#anscombe") 

d3.csv("https://gist.githubusercontent.com/alansmithy/4a863fed88f346e89921454dae3ab8f9/raw/d169e523ae4f9aa925145c4f6ed10421fe6f8f28/anscombe.csv",
        d3.autoType).then(plotAnscombe) // do not rely on default data types!

function plotAnscombe(data){
        let hMargin = 100
        let vMargin = 100

        let hSep = 10
        let vSep = 10

        let width = 1100
        let height = 900

        let graphWidth = +(width-3*vMargin)/2.0
        let graphHeight = +(height-3*hMargin)/2.0

        var svg = anscombeContainer.append("svg")
                .attr("id", 'myid')
                .attr("width", width+200)
                .attr("height",height+200)

        let xMin = 0
        let xMax = 20

        let yMin = 0
        let yMax = 15

        var k = 0

        cols = ["#48acf7","#ea81bc","#cc9b50","#33bb8c"]

        for(let i=0;i<2;i++){
                for(let j=0;j<2;j++){
                        k += 1

                        var x0 = vMargin + i*(vMargin+graphWidth)
                        var x1 = x0 + graphWidth

                        var x = d3.scaleLinear()
                                .range([x0, x1 ])
                                .domain([xMin, xMax])
                        xa = x0

                        var y0 = hMargin + j*(hMargin+graphHeight)
                        var y1 = y0 + graphHeight

                        var x = d3.scaleLinear()
                                .range([x0, x1 ])
                                .domain([xMin, xMax])

                        var y = d3.scaleLinear()
                                .range([y0, y1 ])
                                .domain([yMax, yMin])

                        const regression = d3.regressionLinear()
                          .x(d => d.x)
                          .y(d => d.y)
                          .domain([0, 15]);

                        linear = regression(data)
                        // linearRegressionLine = ss.linearRegressionLine(linearRegression)
                        // linearRegression = d3.regressionLinear(data)
                        //   .x(d => d.x)
                        //   .y(d => d.y)
                        //   .domain([0, 25]);

                        svg.append("g")
                                .call(d3.axisBottom(x).ticks(4)).attr('transform',
                                        'translate('+0+','+y1+')')

                        svg.append("g")
                                .call(d3.axisLeft(y).ticks(3)).attr('transform',
                                        'translate('+x0+','+0+')')

                        svg.append('text').text(k).attr('x', x(0)+20).attr('y', y(15))
                        svg.append('text').text('x').attr('x', x(12)).attr('y', y(0)+20)
                        svg.append('text').text('y').attr('x', x(0)-15).attr('y', y(8))

                        svg.selectAll('points').data(data)
                                .enter().append("circle")
                                .filter(function(d) { return +d.seriesname.split(' ')[1] == k })
                                .style("fill", cols[k-1])
                                .attr("r", 3.5)
                                .attr("cx", function(d) { return x(d.x); })
                                .attr("cy", function(d) { return y(d.y); });

                        svg.append('line').attr('x1', x(linear[0][0])).attr('x2', x(linear[1][0])).attr('y1', y(linear[0][1])).attr('y2', y(linear[1][1])).attr('fill', 'none')
                                .attr('style', 'stroke:'+cols[k-1]+";stroke-width:2")

                }
        }


}

