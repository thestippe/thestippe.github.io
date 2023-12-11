
d3.csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        d3.autoType).then(plotBoxplot) // do not rely on default data types!

function plotBoxplot(data){
        let boxplotContainer = d3.select("#boxplotdiv")

        var height = 100
        var width = 600
        var horizontalMargin = 100
        var verticalMargin = 50

        var boxHeight = 50
        var boxY = 50
        var xRange = d3.extent(data, d=>d.sepal_length)

        var x = d3.scaleLinear().domain([0, 9.5]).range([horizontalMargin, horizontalMargin+width])

        dt = []
        for(elem of data){
                dt.push(elem.sepal_length)
        }

        sorted_data = dt.sort()
        console.log('DT')
        console.log(dt)
        console.log(sorted_data)


        var dataMean = d3.mean(data, d => d.sepal_length)
        var dataMedian = d3.median(data, d => d.sepal_length)
        var dataStd = d3.deviation(data, d => d.sepal_length)
        var dataQuantile5 = d3.quantile(sorted_data, 0.05 )
        var dataQuantile95 = d3.quantile(sorted_data, 0.95)

        var dataQuantile25 = d3.quantile(sorted_data, 0.25)
        var dataQuantile75 = d3.quantile(sorted_data, 0.75)

        var deltaY = Math.abs(dataQuantile75-dataQuantile25)
        console.log(dataQuantile5)
        console.log(dataQuantile25)
        console.log(dataQuantile75)
        console.log(dataQuantile95)


        var svg = boxplotContainer.append('svg').attr('height', +(height+2*verticalMargin)).attr('width', width+2*verticalMargin).attr('id', 'boxplotsvg')

        svg.append("g")
                .attr("transform", "translate(0," + (height+verticalMargin) + ")")
                .call(d3.axisBottom(x));



        svg.append('line')
                .attr('x1', x(dataQuantile5))
                .attr('x2', x(dataQuantile95))
        .attr('y1', verticalMargin+boxY)
        .attr('y2', verticalMargin+boxY)
        .attr('stroke', 'black')

        svg.append('rect')
                .attr('x', x(dataQuantile25))
        .attr('y', verticalMargin+boxY/2)
        .attr('width', x(dataQuantile75)-x(dataQuantile25))
        .attr('height', boxHeight)
        .attr('fill', 'steelblue')
        .attr('stroke', 'black')

        svg.append('line')
                .attr('x1', x(dataMedian))
                .attr('x2', x(dataMedian))
        .attr('y1', verticalMargin+boxY-boxHeight/2)
        .attr('y2', verticalMargin+boxY+boxHeight/2)
        .attr('stroke', 'black')

        svg.append('line')
                .attr('x1', x(dataQuantile5))
                .attr('x2', x(dataQuantile5))
        .attr('y1', verticalMargin+boxY-boxHeight/2)
        .attr('y2', verticalMargin+boxY+boxHeight/2)
        .attr('stroke', 'black')

        svg.append('line')
                .attr('x1', x(dataQuantile95))
                .attr('x2', x(dataQuantile95))
        .attr('y1', verticalMargin+boxY-boxHeight/2)
        .attr('y2', verticalMargin+boxY+boxHeight/2)
        .attr('stroke', 'black')

        svg.append("text")
                .text("Sepal width [mm]")
                .attr("x", 350).attr('font-size', '16px')
                .attr("y", 185)


}
