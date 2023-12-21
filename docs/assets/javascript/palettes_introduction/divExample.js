var catWidth = 950
var catHeight = 150

divPalette = d3.select('#diverging_example')

var svg2 = divPalette.append("svg")
        .attr("id", 'myid')
        .attr("width", catWidth)
        .attr("height", catHeight)

for(let i=0; i<50; i++){
col = d3.hcl(-175-2.5*i, 27, 90-1.5*i)
         if(!col.displayable()){col = d3.color("black")}
svg2.append('rect').attr('x', 7*(50-i)).attr('y', 50)
.attr('height', 100).attr('width', 7)
.attr('fill', col.rgb().toString())

col1 = d3.hcl(-175+2.5*i, 27, 90-1.5*i)
         if(!col1.displayable()){col1 = d3.color("black")}
svg2.append('rect').attr('x', 7*(50+i)).attr('y', 50)
.attr('height', 100).attr('width', 7)
.attr('fill', col1.rgb().toString())

// col1 = d3.hcl(-175-2.2*i, 20, 90-i)
//         if(!col.displayable()){col1 = d3.color("black")}
// svg2.append('rect').attr('x', 50*6+6*i).attr('y', 50)
// .attr('height', 100).attr('width', 150)
// .attr('fill', col1.rgb().toString())
}

svg2.append('text').attr('x', 0).attr('y', 32)
.text('An example of fixed-chroma diverging color map')

