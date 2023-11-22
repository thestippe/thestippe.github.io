let containerColor = d3.select("#gender_color_first") 


d3.csv("https://raw.githubusercontent.com/thestippe/thestippe.github.io/main/data/Economist_women-research.csv", d3.autoType).then(showData) // do not rely on default data types!

function showData(data){
        fields = [
                "Health sciences",
                "Physical sciences",
                "Engineering",
                "Computer science and maths",
                "% of women inventores"
        ]

        fields_names = {
                "Health sciences": [ "Health", "sciences"],
                "Physical sciences": [ "Physical", "sciences"],
                "Engineering": [" ", "Engineering"],
                "Computer science and maths": [ "Comp. sci.", "and maths"],
                "% of women inventores": [ "% of", " inventores"]
        }


        weight = {
                "Health sciences": "normal",
                "Physical sciences": "normal",
                "Engineering": "normal",
                "Computer science and maths": "normal",
                "% of women inventores": "bold"
        }

        fig_height = 600
        fig_width = 600
        top_border = 90
        bottom_border = 40
        left_border = 150
        right_border = 250

        var colorMen = 'steelblue'
        var colorWomen = 'LightSeaGreen'

        square_length = 30

        tShift = 15

        lAxisShift = left_border - 5

        height = fig_height-top_border-bottom_border
        width = fig_width-left_border-right_border

        var domain_country = d3.set(
                data.map( 
        function(d){ return d.Country })).values()

        var x = d3.scalePoint()
                .range([left_border, left_border + width]).domain(fields)

        var y = d3.scalePoint()
                .range([ top_border, top_border+height ])
                .domain(domain_country)

        var myColor = d3.scaleSequential().domain([1, 0])
  .interpolator(d3.interpolateBlues);

        var l = d3.scaleSqrt().range([0, square_length]).domain([0, 1])

        var shift = l(1)/2

        var svg = containerColor.append("svg")
                .attr("id", 'myid')
                .attr("width", fig_width)
                .attr("height", fig_height)

        let yAxis = svg.append("g")
                .attr("class", "axis")
                .call(d3.axisLeft(y).tickSize(0))
                
        yAxis.attr("transform", "translate("+lAxisShift+","+shift+")")
                .select(".domain").remove().style('font-family', 'DejaVu Serif')


        matrix = svg.selectAll('squares').data(data)

        // svg.append('rect')
        //         .attr('x', x("% of women inventores")-10)
        //         .attr('y',top_border)
        //         .attr('height', +(height+top_border-60))
        //         .attr('width',50)
        //         .attr("fill", "darkgrey")

        for(field of fields){
        var k = 0
        for (fname of fields_names[field]){
        yt = 75 + k
        var s = +(x(field)+l(0.5) )
        svg.append("text")
                .html(fname)
                .attr("x", s)
                .attr("y", yt)
                  .attr("font-family", "DejaVu Sans")
                .style("font-size", "0.7rem")
                .style("font-weight", weight[field])
                .attr("transform", "rotate(-45 "+s+" "+yt+")")
                 k += tShift
        }

        matrix.enter()
                .append('rect')
                .attr('x', x(field))
                .attr('y', function(d){ return y(d.Country) })
                .attr('height', l(1))
                .attr('width', l(1))
                .attr("fill", function(d){return myColor(d[field])})



        }

        legArray = Array.from(Array(50).keys())
        for(ll of legArray){
                svg.append('line')
                        .attr('x1', +20+2*ll)
                        .attr('x2', +20+2*ll)
                        .attr('y1', 60)
                        .attr('y2', 70).style("stroke", myColor(ll/50)).style("stroke-width", 2)
}

        svg.append("text")
                .text("0%")
                .attr("x", 20)
                .attr("y", 55)
                  .attr("font-family", "DejaVu Sans")
                .style("font-size", "0.7rem")

        svg.append("text")
                .text("100%")
                .attr("x", 120)
                .attr("y", 55)
                  .attr("font-family", "DejaVu Sans")
                .style("font-size", "0.7rem")

        svg.append("text")
                .text("Women %")
                .attr("x", 50)
                .attr("y", 45)
                  .attr("font-family", "DejaVu Sans")
                .style("font-size", "0.7rem")


}
