const height = 100
const width = 400

let surfHeight = 600
let surfWidth = 600

let nx = 300
let ny = 150

dH = 720/nx
dL = 100/ny
dX = surfWidth/nx
dY = surfHeight/ny

document.getElementById('CategoricalminHue').defaultValue = -70
document.getElementById('CategoricalmaxHue').defaultValue = 100
document.getElementById('CategoricalChroma').defaultValue = 45
document.getElementById('CategoricalValueMin').defaultValue = 15
document.getElementById('CategoricalValueMax').defaultValue = 95
document.getElementById('CategoricalNumClasses').defaultValue = 20

var minHue = document.getElementById('CategoricalminHue').value
var maxHue = document.getElementById('CategoricalmaxHue').value
var chroma = document.getElementById('CategoricalChroma').value
var valueMin = document.getElementById('CategoricalValueMin').value 
var valueMax = document.getElementById('CategoricalValueMax').value 
var numClasses = document.getElementById('CategoricalNumClasses').value


catContainer = d3.select('#catPalette')
catSurface = d3.select('#catSurface')

catContainer.selectAll("svg").remove()
catSurface.selectAll("svg").remove()

var svgCat = catContainer.append('svg').attr('height', height).attr('width', width).attr('id', 'catSvg')


var ulCat = catContainer.append('ul').attr('id', 'palette').attr('hidden', 'hidden')



var svgSurf = catSurface.append('svg')
        .attr('height', surfHeight).attr('width', surfWidth)
        .attr('id', 'catSurf')


for(let i=0;i<nx;i++){
        for(let j=0;j<ny;j++){
        col = d3.hcl(+360+dH*i, chroma, +dL*j)
        if(!col.displayable()){col = d3.color("black")}

svgSurf.append('rect').attr('x', +i*dX).attr('y', +j*dY).attr('height', +dY).attr('width',+dX).attr('fill', col.rgb().toString())
        }
}

let dx = +width/numClasses
let dHue = +(maxHue - minHue)/(numClasses-1)
let dValue = +(valueMax - valueMin)/(numClasses-1)
for(let k=0;k<numClasses;k++){
        col = d3.hcl(+minHue+k*dHue, chroma, +valueMin+k*dValue)
        if(!col.displayable()){col = d3.color("black")}
        svgCat.append('rect').attr('x', dx*k).attr('y', 50).attr('height', height).attr('width', dx).attr('fill', col.rgb().toString())

        textCol = col.formatHex()
        ulCat.append('li').text(textCol)
}

var minHue0 = +minHue + 360
var maxHue0 = +maxHue + 360


svgLine = svgSurf.append('line').attr('x1', +minHue0/720*surfWidth)
.attr('x2', +maxHue0/720*surfWidth)
.attr('y1', +valueMin/100*surfHeight)
.attr('y2', +valueMax/100*surfHeight)
.attr('stroke', 'lightgrey')
        .attr('stroke-width', '2px')
.attr('id', 'lineDrawn')

        d3.select('#catSurf').append('circle').attr('cx', +minHue0/720*surfWidth)
        .attr('cy', +valueMin/100*surfHeight).attr('id', 'minCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMin")

        d3.select('#catSurf').append('circle').attr('cx', +maxHue0/720*surfWidth)
        .attr('cy', +valueMax/100*surfHeight).attr('id', 'maxCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMax")



function drawPalette(){

var minHue = document.getElementById('CategoricalminHue').value
var maxHue = document.getElementById('CategoricalmaxHue').value
var chroma = document.getElementById('CategoricalChroma').value
var valueMin = document.getElementById('CategoricalValueMin').value 
var valueMax = document.getElementById('CategoricalValueMax').value 
var numClasses = document.getElementById('CategoricalNumClasses').value


catContainer = d3.select('#catPalette')
catContainer.selectAll("svg").remove()




catContainer.select("#palette").remove()
var svgCat = catContainer.append('svg').attr('height', height).attr('width', width).attr('id', 'catSvg')



var ulCat = catContainer.append('ul').attr('id', 'palette').attr('hidden', 'hidden')

let dx = +width/numClasses
let dHue = +(maxHue - minHue)/(numClasses-1)
let dValue = +(valueMax - valueMin)/(numClasses-1)
for(let k=0;k<numClasses;k++){
        col = d3.hcl(+minHue+k*dHue, chroma, +valueMin+k*dValue)
        if(!col.displayable()){col = d3.color("black")}
        svgCat.append('rect').attr('x', dx*k).attr('y', 50).attr('height', height).attr('width', dx).attr('fill', col.rgb().toString())
        textCol = col.formatHex()
        ulCat.append('li').text(textCol)
        console.log(textCol)
}

var minHue0 = +minHue + 360
var maxHue0 = +maxHue + 360


}

function updateMinHue(val) {
          document.getElementById('minHueInput').innerHTML=val; 
        d3.select('#lineDrawn').remove()
        d3.selectAll('#minCircle').remove()
        d3.selectAll('#maxCircle').remove()

        var minHue = document.getElementById('CategoricalminHue').value
        var maxHue = document.getElementById('CategoricalmaxHue').value
        var valueMin = document.getElementById('CategoricalValueMin').value 
        var valueMax = document.getElementById('CategoricalValueMax').value 

        var minHue0 = +minHue + 360
        var maxHue0 = +maxHue + 360

        d3.select('#catSurf')
        .append('line').attr('x1', +minHue0/720*surfWidth)
        .attr('x2', +maxHue0/720*surfWidth)
        .attr('y1', +valueMin/100*surfHeight)
        .attr('y2', +valueMax/100*surfHeight)
        .attr('stroke', 'lightgrey')
        .attr('stroke-width', '2px')
        .attr('id', 'lineDrawn')

        d3.select('#catSurf').append('circle').attr('cx', +minHue0/720*surfWidth)
        .attr('cy', +valueMin/100*surfHeight).attr('id', 'minCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMin")

        d3.select('#catSurf').append('circle').attr('cx', +maxHue0/720*surfWidth)
        .attr('cy', +valueMax/100*surfHeight).attr('id', 'maxCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMax")

var minDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x1', d3.event.x).attr('y1', d3.event.y)

            document.getElementById('CategoricalValueMin').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalminHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('minHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)
          document.getElementById('minValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)

    });

minDragHandler(svgSurf.select("#minCircle"));

var maxDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x2', d3.event.x).attr('y2', d3.event.y)

            document.getElementById('CategoricalValueMax').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalmaxHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('maxHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)

          document.getElementById('maxValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)
    })

maxDragHandler(svgSurf.select("#maxCircle"));
        }

function updateMaxHue(val) {
          document.getElementById('maxHueInput').innerHTML=val; 
        d3.select('#lineDrawn').remove()
        d3.selectAll('#minCircle').remove()
        d3.selectAll('#maxCircle').remove()

        var minHue = document.getElementById('CategoricalminHue').value
        var maxHue = document.getElementById('CategoricalmaxHue').value
        var valueMin = document.getElementById('CategoricalValueMin').value 
        var valueMax = document.getElementById('CategoricalValueMax').value 

        var minHue0 = +minHue + 360
        var maxHue0 = +maxHue + 360

        d3.select('#catSurf')
        .append('line').attr('x1', +minHue0/720*surfWidth)
        .attr('x2', +maxHue0/720*surfWidth)
        .attr('y1', +valueMin/100*surfHeight)
        .attr('y2', +valueMax/100*surfHeight)
        .attr('stroke', 'lightgrey')
        .attr('stroke-width', '2px')
        .attr('id', 'lineDrawn')

        d3.select('#catSurf').append('circle').attr('cx', +minHue0/720*surfWidth)
        .attr('cy', +valueMin/100*surfHeight).attr('id', 'minCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMin")

        d3.select('#catSurf').append('circle').attr('cx', +maxHue0/720*surfWidth)
        .attr('cy', +valueMax/100*surfHeight).attr('id', 'maxCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMax")

var minDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x1', d3.event.x).attr('y1', d3.event.y)

            document.getElementById('CategoricalValueMin').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalminHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('minHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)
          document.getElementById('minValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)

    });

minDragHandler(svgSurf.select("#minCircle"));

var maxDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x2', d3.event.x).attr('y2', d3.event.y)

            document.getElementById('CategoricalValueMax').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalmaxHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('maxHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)

          document.getElementById('maxValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)
    })

maxDragHandler(svgSurf.select("#maxCircle"));

        }
function updateChroma(val) {
          document.getElementById('chromaInput').innerHTML=val; 
        d3.selectAll('#minCircle').remove()
        d3.selectAll('#maxCircle').remove()

        var chroma = document.getElementById('CategoricalChroma').value

        catSurface = d3.select('#catSurface')
        catSurface.selectAll("svg").remove()

var svgSurf = catSurface.append('svg').attr('height', surfHeight).attr('width', surfWidth).attr('id', 'catSurf')


for(let i=0;i<nx;i++){
        for(let j=0;j<ny;j++){
        col = d3.hcl(+360+dH*i, chroma, +dL*j)
        if(!col.displayable()){col = d3.color("black")}

svgSurf.append('rect').attr('x', +i*dX).attr('y', +j*dY).attr('height', +dY).attr('width',+dX).attr('fill', col.rgb().toString())
        }
}

        var minHue = document.getElementById('CategoricalminHue').value
        var maxHue = document.getElementById('CategoricalmaxHue').value
        var valueMin = document.getElementById('CategoricalValueMin').value 
        var valueMax = document.getElementById('CategoricalValueMax').value 

        var minHue0 = +minHue + 360
        var maxHue0 = +maxHue + 360

        d3.select('#catSurf')
        .append('line').attr('x1', +minHue0/720*surfWidth)
        .attr('x2', +maxHue0/720*surfWidth)
        .attr('y1', +valueMin/100*surfHeight)
        .attr('y2', +valueMax/100*surfHeight)
        .attr('stroke', 'lightgrey')
        .attr('stroke-width', '2px')
        .attr('id', 'lineDrawn')


        d3.select('#catSurf').append('circle').attr('cx', +minHue0/720*surfWidth)
        .attr('cy', +valueMin/100*surfHeight).attr('id', 'minCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMin")

        d3.select('#catSurf').append('circle').attr('cx', +maxHue0/720*surfWidth)
        .attr('cy', +valueMax/100*surfHeight).attr('id', 'maxCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMax")


var minDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x1', d3.event.x).attr('y1', d3.event.y)

            document.getElementById('CategoricalValueMin').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalminHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('minHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)
          document.getElementById('minValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)

    });

minDragHandler(svgSurf.select("#minCircle"));

var maxDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x2', d3.event.x).attr('y2', d3.event.y)

            document.getElementById('CategoricalValueMax').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalmaxHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('maxHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)

          document.getElementById('maxValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)
    })

maxDragHandler(svgSurf.select("#maxCircle"));



        }
function updateMinValue(val) {
          document.getElementById('minValueInput').innerHTML=val; 
        d3.select('#lineDrawn').remove()
        d3.selectAll('#minCircle').remove()
        d3.selectAll('#maxCircle').remove()

        var minHue = document.getElementById('CategoricalminHue').value
        var maxHue = document.getElementById('CategoricalmaxHue').value
        var valueMin = document.getElementById('CategoricalValueMin').value 
        var valueMax = document.getElementById('CategoricalValueMax').value 

        var minHue0 = +minHue + 360
        var maxHue0 = +maxHue + 360

        d3.select('#catSurf')
        .append('line').attr('x1', +minHue0/720*surfWidth)
        .attr('x2', +maxHue0/720*surfWidth)
        .attr('y1', +valueMin/100*surfHeight)
        .attr('y2', +valueMax/100*surfHeight)
        .attr('stroke', 'lightgrey')
        .attr('stroke-width', '2px')
        .attr('id', 'lineDrawn')

        d3.select('#catSurf').append('circle').attr('cx', +minHue0/720*surfWidth)
        .attr('cy', +valueMin/100*surfHeight).attr('id', 'minCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMin")

        d3.select('#catSurf').append('circle').attr('cx', +maxHue0/720*surfWidth)
        .attr('cy', +valueMax/100*surfHeight).attr('id', 'maxCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMax")

var minDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x1', d3.event.x).attr('y1', d3.event.y)

            document.getElementById('CategoricalValueMin').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalminHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('minHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)
          document.getElementById('minValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)

    });

minDragHandler(svgSurf.select("#minCircle"));

var maxDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x2', d3.event.x).attr('y2', d3.event.y)

            document.getElementById('CategoricalValueMax').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalmaxHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('maxHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)

          document.getElementById('maxValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)
    })

maxDragHandler(svgSurf.select("#maxCircle"));
        }

function updateMaxValue(val) {
          document.getElementById('maxValueInput').innerHTML=val; 
        d3.select('#lineDrawn').remove()
        d3.selectAll('#minCircle').remove()
        d3.selectAll('#maxCircle').remove()

        var minHue = document.getElementById('CategoricalminHue').value
        var maxHue = document.getElementById('CategoricalmaxHue').value
        var valueMin = document.getElementById('CategoricalValueMin').value 
        var valueMax = document.getElementById('CategoricalValueMax').value 

        var minHue0 = +minHue + 360
        var maxHue0 = +maxHue + 360

        d3.select('#catSurf')
        .append('line').attr('x1', +minHue0/720*surfWidth)
        .attr('x2', +maxHue0/720*surfWidth)
        .attr('y1', +valueMin/100*surfHeight)
        .attr('y2', +valueMax/100*surfHeight)
        .attr('stroke', 'lightgrey')
        .attr('stroke-width', '2px')
        .attr('id', 'lineDrawn')

        d3.select('#catSurf').append('circle').attr('cx', +minHue0/720*surfWidth)
        .attr('cy', +valueMin/100*surfHeight).attr('id', 'minCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMin")

        d3.select('#catSurf').append('circle').attr('cx', +maxHue0/720*surfWidth)
        .attr('cy', +valueMax/100*surfHeight).attr('id', 'maxCircle').attr('r', 5)
                .attr('style', 'fill:lightgrey').attr("href", "#pointerMax")

var minDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x1', d3.event.x).attr('y1', d3.event.y)

            document.getElementById('CategoricalValueMin').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalminHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('minHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)
          document.getElementById('minValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)

    });

minDragHandler(svgSurf.select("#minCircle"));

var maxDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x2', d3.event.x).attr('y2', d3.event.y)

            document.getElementById('CategoricalValueMax').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalmaxHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('maxHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)

          document.getElementById('maxValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)
    })

maxDragHandler(svgSurf.select("#maxCircle"));

        }

function createFile(){
  //create or obtain the file's content

  var dt = document.getElementById('palette').querySelectorAll('li')
  var content = ""
  for (li of dt){content += (li.innerHTML+",")}

  //create a file and put the content, name and type
  var file = new File(["\ufeff"+content.substr(0, content.length-1)], 'colormap_data_perspectives.txt', {type: "text/plain:charset=UTF-8"});

  //create a ObjectURL in order to download the created file
  url = window.URL.createObjectURL(file);

  //create a hidden link and set the href and click it
  var a = document.createElement("a");
  a.style = "display: none";
  a.href = url;
  a.download = file.name;
  a.click();
  window.URL.revokeObjectURL(url);
}

var minDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x1', d3.event.x).attr('y1', d3.event.y)

            document.getElementById('CategoricalValueMin').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalminHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('minHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)
          document.getElementById('minValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)

    });

minDragHandler(svgSurf.select("#minCircle"));

var maxDragHandler = d3.drag()
    .on("drag", function () {
        d3.select(this)
            .attr("cx", d3.event.x)
            .attr("cy", d3.event.y);

        d3.select('#lineDrawn').attr('x2', d3.event.x).attr('y2', d3.event.y)

            document.getElementById('CategoricalValueMax').value = d3.event.y*100/surfHeight
            document.getElementById('CategoricalmaxHue').value = d3.event.x*720/surfWidth-360

          document.getElementById('maxHueInput').innerHTML =  parseInt(d3.event.x*720/surfWidth-360)

          document.getElementById('maxValueInput').innerHTML =  parseInt(d3.event.y*100/surfHeight)
    })

maxDragHandler(svgSurf.select("#maxCircle"));

function drawExternalPalette(){
        svgHeight = 100
        svgWidth = 800

        var cmapData = document.getElementById('inputPalette').value
        console.log(cmapData)
        dataList = cmapData.split(',')
        n = dataList.length
        console.log(n)
        let dx = svgWidth/n

        extContainer = d3.select('#externalPalette')
        extContainer.selectAll("svg").remove()

        svgExt = extContainer.append('svg').attr('height', svgHeight).attr('width', svgWidth)
        for(let s=0;s<n;s++){
                svgExt.append('rect').attr('x',+s*dx).attr('width',+dx).attr('y',0).attr('height',svgHeight).attr('fill', dataList[s])
        }
}
