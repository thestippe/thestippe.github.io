
var timer = 1500
var hasRun = 0
var maxNumIter=6
var height = 300
var width = 300

function startTimer() {

if(hasRun==0){
hasRun = 1

numIter=0

var numBlueBalls = 10
minRedBalls = 3
maxRedBalls = 6
var disIter = 5


var demo = d3.select("#demoPerception") 
demo.selectAll('*').remove()

var svg = demo.append('svg')
        .attr('id', 'demoId')
        .attr('height', height)
        .attr('width', width)


for(var k=0; k<disIter; k++){
                svg.append('circle')
                        .attr('id', 'disappearing_'+k)
                        .attr('r', 10)
                        .attr('cx', width*Math.random())
                        .attr('cy', height*Math.random())
                        .attr('fill', 'black')
        }
        for(var k=0; k<numBlueBalls-1; k++){

        svg.append('circle')
                .attr('id', 'blueCircle')
                .attr('r', 10)
                .attr('cx', width*Math.random())
                .attr('cy', height*Math.random())
                .attr('fill', 'steelblue')
        }

numRedBalls = Math.floor(
Math.random() * (
maxRedBalls - minRedBalls + 1) + minRedBalls
)
for(var k=0;k<numRedBalls;k++){

svg.append('circle')
        .attr('r', 10)
        .attr('id', 'redCircles')
        .attr('cx', width*Math.random())
        .attr('cy', height*Math.random())
        .attr('fill', 'crimson')
}
var countRedBalls = numRedBalls

function runAnimation(){
        numIter += 1
        // opacity = (maxNumIter-numIter)/maxNumIter
        opacity = 0
        if(numIter<disIter+1){
        d3.select('#disappearing_'+numIter).attr('fill-opacity', opacity)
        }
        numRedBalls = Math.floor(
        Math.random() * (
        maxRedBalls - minRedBalls + 1) + minRedBalls
        )
        countRedBalls += numRedBalls
        d3.selectAll('#redCircles').remove()
        for(var k=0;k<numRedBalls;k++){

        svg.append('circle')
                .attr('r', 10)
                .attr('id', 'redCircles')
                .attr('cx', width*Math.random())
                .attr('cy', height*Math.random())
                .attr('fill', 'crimson')
        }



        if(numIter>maxNumIter){
                stopAnimation(myVar)
        }
}

function stopAnimation(vr){
        clearInterval(vr)

demo.selectAll('*').remove()

var demo1 = d3.select("#demoPerception") 

var svg1 = demo1.append('svg')
        .attr('height', height)
        .attr('width', width)

svg1.append('text').attr('x', 100).attr('y', 100)
        .text('How many red circles did you counted?')
 
setTimeout(showSurprise, 3000)

        function showSurprise(){

svg1.append('text').attr('x', 100).attr('y', 130)
        .text('They were '+countRedBalls)

svg1.append('text').attr('x', 100).attr('y', 160)
        .text('Did you also noticed the '+disIter+' blue circles disappearing?')

        }
}

var myVar = setInterval(runAnimation, timer)
}

hasRun = 0
}
