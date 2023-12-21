var catWidth = 950
var catHeight = 150

quanPalette = d3.select('#quantitative_example')

var svg1 = quanPalette.append("svg")
        .attr("id", 'myid')
        .attr("width", catWidth)
        .attr("height", catHeight)

for(let i=0; i<100; i++){
col = d3.hcl(-175+2.2*i, 45, 91-i*0.69)
        if(!col.displayable()){col = d3.color("black")}
svg1.append('rect').attr('x', 7*i).attr('y', 50)
.attr('height', 100).attr('width', 7)
.attr('fill', col.rgb().toString())
}

svg1.append('text').attr('x', 0).attr('y', 32)
.text('An example of fixed-chroma quantitative color map')
