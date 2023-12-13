

var sliderLuminance = document.getElementById("luminanceRange");
var circleContainer = d3.select('#colCircle')

var outputLuminance = document.getElementById("luminanceValue");


outputLuminance.innerHTML = 50

var svg = circleContainer.append('svg').attr('height', 700).attr('width', 700)

var cx0 = 350
var cy0 = 350

luminance = 50


var outputColor = document.getElementById("correspondingColor")
function mouseDown(){
        outputColor.innerHTML = "RGB: "+this.style.fill
}

for(let r=3; r<250; r+=5){
        for(let theta=0; theta<360; theta += 4){
                col = d3.hcl(theta, r, luminance)
                v = d3.color(col).rgb().toString()
                cx = cx0 + r*Math.sin(2.0*Math.PI/360*theta)
                cy = cy0 + r*Math.cos(2.0*Math.PI/360*theta)
        svg.append('circle').attr('r', 9).attr('cx', cx).attr('cy', cy).attr('fill', v).attr('class', 'smallCircle').attr('id', 'circle_'+cx+'_'+cy)

                var crc = document.getElementById('circle_'+cx+'_'+cy)
                crc.style.fill = v

                crc.addEventListener("mousedown", mouseDown)


        }
}

sliderLuminance.oninput = function(){

outputLuminance.innerHTML = this.value

var luminance = this.value

for(let r=3; r<250; r+=5){
        for(let theta=0; theta<360; theta += 4){
                var col = d3.hcl(theta, r, luminance)
                var v = d3.color(col).rgb().toString()
                var cx = cx0 + r*Math.sin(2.0*Math.PI/360*theta)
                var cy = cy0 + r*Math.cos(2.0*Math.PI/360*theta)
                var crc = document.getElementById('circle_'+cx+'_'+cy)
                crc.style.fill = v

                var outputColor = document.getElementById("correspondingColor")

                crc.addEventListener("mousedown", mouseDown)


        }
}

}





