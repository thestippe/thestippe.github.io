var catWidth = 950
var catHeight = 150

catPalette = d3.select('#categorical_example')

var svg = catPalette.append("svg")
        .attr("id", 'myid')
        .attr("width", catWidth)
        .attr("height", catHeight)

for(let i=0; i<5; i++){
col = d3.hcl(80+i*80, 40, 70)
svg.append('rect').attr('x', 140*i).attr('y', 50)
.attr('height', 100).attr('width', 140)
.attr('fill', col.rgb().toString())
}

svg.append('text').attr('x', 0).attr('y', 32)
.text('An example of fixed-chroma and fixed-luminance categorical color map')

