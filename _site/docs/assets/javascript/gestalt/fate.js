let containerFate = d3.select("#fate") 
height = 600
width = height
var r = 5
var n = 20
var modV = 0.00

var svg = containerFate.append("svg")
        .attr("id", 'myid')
        .attr("width", width)
        .attr("height", height)

// id = setInterval(frame, 5);

cx = []
cy = []
vx = []
vy = []


for(var i=0;i<n;i++){
        cx.push(width*Math.random())
        cy.push(height*Math.random())
        s = Math.random()
        s1 = Math.random()
        console.log(s)
        if(s<0.5){
                vx.push(s1*modV)
        }
        else{
                vx.push(-s1*modV)
        }

        s = Math.random()
        s1 = Math.random()
        console.log(s)
        if(s<0.5){
                vy.push(s1*modV)
        }
        else{
                vy.push(-s1*modV)
        }
        svg.append("circle")
                .attr('cx', cx[i]).attr('cy', cy[i]).attr('r', r)
                .attr('fill', 'steelblue') 
                .attr("stroke", "steelblue").attr('id', 'myCircle'+i)
}


let myVar = setInterval(myTimer, 50);

function barrier(cx){
        c0 = -100
        var l = cx - width/2
        var x0 = - width/2
        var x1 = width/2
        pot0 = 1/(l-x0)**2
        pot1 = 1/(l+x0)**2
        dpot = -2/(l-x0)**3 -2/(l+x0)**3
        return c0*dpot
}

function harmonic(cx, i){
        c0 = -0.01
        l0 = 0
        var rs = cx[i] - d3.mean(cx) + l0
        return c0*rs
}

function hspeed(vx, i){
        c0 = -0.2
        var vxn = d3.mean(vx)
        return c0*vxn
}

function myTimer() {
        var cxOld = cx
        var vxOld = vx

        var cyOld = cy
        var vyOld = vy
        for(var i=0;i<n;i++){

        dvx = barrier(cx[i]) + harmonic(cxOld, i) + 0.1*vyOld[i]
        vx[i] = vx[i] + dvx/2

        dvy = barrier(cy[i]) + harmonic(cyOld, i) - 0.1*vxOld[i]
        vy[i] = vy[i] + dvy/2
        }
        vxOld = vx
        vyOld = vy

        for(var i=0;i<n;i++){
        cx[i] += vx[i]
                if(cx[i]<0){
                        cx[i] = -cx[i]
                        vx[i] = - vx[i]
                }
                else if(cx[i]>width){
                        cx[i] = (cx[i] -width)
                        vx[i] = - vx[i]
                }
                cx[i] = cx[i] % width

        cy[i] += vy[i]

                if(cy[i]<0){
                        cy[i] = -cy[i]
                        vy[i] = - vy[i]
                }
                else if(cy[i]>height){
                        cy[i] = cy[i] - height
                        vy[i] = - vy[i]
                }
                cy[i] = cy[i] % height
        }
        var cxOld = cx
        var cyOld = cy

        for(var i=0;i<n;i++){
        dvx = barrier(cx[i]) + harmonic(cxOld, i) + 0.1*vyOld[i]
        vx[i] = vx[i] + dvx/2
                if(cx[i]<0){
                        cx[i] = -cx[i]
                        vx[i] = - vx[i]
                }
                if(cx[i]>width){
                        cx[i] = (cx[i] -width)
                        vx[i] = - vx[i]
                }
                cx[i] = cx[i] % width

        dvy = barrier(cy[i]) + harmonic(cyOld, i) - 0.1*vxOld[i]
        vy[i] = vy[i] + dvy/2
                if(cy[i]<0){
                        cy[i] = -cy[i]
                        vy[i] = - vy[i]
                }
                if(cy[i]>height){
                        cy[i] = cy[i] - height
                        vy[i] = - vy[i]
                }
                cy[i] = cy[i] % height
                d3.select('#myCircle'+i).attr('cx', cx[i]).attr('cy', cy[i])
        }
        

}
