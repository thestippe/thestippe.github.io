redValue  =0
blueValue =0
greenValue=0

var svgCircle = document.getElementById("rgbCircle")

var sliderRed = document.getElementById("redRange");
var outputRed = document.getElementById("redValue");
outputRed.innerHTML = sliderRed.value;

sliderRed.oninput = function() {
  outputRed.innerHTML = this.value;
  redValue = this.value
  svgCircle.setAttribute("fill", "rgb("+redValue+","+greenValue+","+blueValue+")")
}

var sliderGreen = document.getElementById("greenRange");
var outputGreen = document.getElementById("greenValue");
outputGreen.innerHTML = sliderRed.value;

sliderGreen.oninput = function() {
  outputGreen.innerHTML = this.value;
  greenValue = this.value
  svgCircle.setAttribute("fill", "rgb("+redValue+","+greenValue+","+blueValue+")")
}

var sliderBlue = document.getElementById("blueRange");
var outputBlue = document.getElementById("blueValue");
outputBlue.innerHTML = sliderRed.value;

sliderBlue.oninput = function() {
  outputBlue.innerHTML = this.value;
  blueValue = this.value
  svgCircle.setAttribute("fill", "rgb("+redValue+","+greenValue+","+blueValue+")")
}

