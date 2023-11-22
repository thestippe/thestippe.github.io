let container = d3.select("#gender_area") 


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

        console.log(data[0].Engineering)
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

        var l = d3.scaleSqrt().range([0, square_length]).domain([0, 1])

        var shift = l(1)/2

        var svg = container.append("svg")
                .attr("id", 'myid')
                .attr("width", fig_width)
                .attr("height", fig_height)

        let yAxis = svg.append("g")
                .attr("class", "axis")
                .call(d3.axisLeft(y).tickSize(0))
                
        yAxis.attr("transform", "translate("+lAxisShift+","+shift+")")
                .select(".domain").remove().style('font-family', 'DejaVu Serif')


        matrix = svg.selectAll('squares').data(data)

        svg.append('rect')
                .attr('x', x("% of women inventores")-10)
                .attr('y',top_border)
                .attr('height', +(height+top_border-60))
                .attr('width',50)
                .attr("fill", "darkgrey")

        var w = + x("% of women inventores")-10
        s0 = top_border
        svg.append('rect')
                .attr('x', w)
                .attr('y',0)
                .attr('height', s0)
                .attr('width',35)
                .attr("fill", "darkgrey")
         .attr('transform', "rotate(45  "+ (w) +" "+s0+")")


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
                .attr("fill", colorMen)

        matrix.enter()
                .append('rect')
                .attr('x', x(field))
                .attr('y', function(d){ return y(d.Country) })
                .attr('height', function(d){ return l(d[field]) })
                .attr('width', function(d){ return l(d[field]) })
                .attr("fill", colorWomen)


        }

        svg.append('rect').attr('x', 60).attr('y', 50).attr('height', l(0.3)).attr('width', l(0.3)).attr('fill', colorWomen)
        svg.append('text').text('women %').attr('x', 80).attr('y', 58)
                  .attr("font-family", "DejaVu Sans")
                .style("font-size", "0.7rem")

        svg.append('rect').attr('x', 60).attr('y', 70).attr('height', l(0.3)).attr('width', l(0.3)).attr('fill', colorMen)
        svg.append('text').text('men %').attr('x', 80).attr('y', 78)
                  .attr("font-family", "DejaVu Sans")
                .style("font-size", "0.7rem")

        console.log(l(1))
        console.log(l(0.5))

}
