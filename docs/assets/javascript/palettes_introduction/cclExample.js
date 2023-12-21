var catWidth = 950
var catHeight = 150

cclPalette = d3.select('#cyclic_example')

var svg3 = cclPalette.append("svg")
        .attr("id", 'myid')
        .attr("width", catWidth)
        .attr("height", catHeight)

for(let i=0; i<12; i++){
col = d3.hcl(220+i*30, 30, 75)
         if(!col.displayable()){col = d3.color("black")}
svg3.append('rect').attr('x', 60*i).attr('y', 50)
.attr('height', 100).attr('width', 60)
.attr('fill', col.rgb().toString())
}

svg3.append('text').attr('x', 0).attr('y', 32)
.text('An example of fixed-chroma and fixed-luminance cyclic color map')

