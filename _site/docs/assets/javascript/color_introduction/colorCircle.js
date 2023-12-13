

var sliderLuminance = document.getElementById("luminanceRange");
var circleContainer = d3.select('#colCircle')

var outputLuminance = document.getElementById("luminanceValue");

outputLuminance.innerHTML = 50

var svg = circleContainer.append('svg').attr('height', 500).attr('width', 500)

var cx0 = 250
var cy0 = 250

luminance = 50

for(let r=3; r<200; r+=5.5){
        for(let theta=0; theta<360; theta += 5){
                col = d3.hcl(theta, r, luminance)
                v = d3.color(col).rgb().toString()
                console.log(v)
                cx = cx0 + r*Math.cos(2.0*Math.PI/360*theta)
                cy = cy0 + r*Math.sin(2.0*Math.PI/360*theta)
        svg.append('circle').attr('r', 9).attr('cx', cx).attr('cy', cy).attr('fill', v)
        }
}

sliderLuminance.oninput = function(){

outputLuminance.innerHTML = this.value

var luminance = this.value

var cx0 = 250
var cy0 = 250

for(let r=3; r<200; r+=5.5){
        for(let theta=0; theta<360; theta += 5){
                col = d3.hcl(theta, r, luminance)
                v = d3.color(col).rgb().toString()
                console.log(v)
                cx = cx0 + r*Math.cos(2.0*Math.PI/360*theta)
                cy = cy0 + r*Math.sin(2.0*Math.PI/360*theta)
        svg.append('circle').attr('r', 9).attr('cx', cx).attr('cy', cy).attr('fill', v)
        }
}
}
