let mapContainer = d3.select('#my_symbol_chart')

// Create data for circles:
var markers = [
  {long: 13.62, lat: 45.94, time: 18}, // Gorizia
  {long: 11.34, lat: 44.49, time: 7}, // Bologna
  {long: 10.97, lat: 44.16, time: 0.5}, // Porretta
  {long: 9.19, lat: 45.47, time: 0.5}, // Milano
  {long: 7.68, lat: 45.07, time: 5}, // Torino
];


// Load external data and boot

Promise.all([
d3.json("https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson")
]).then(showData)


function showData([map]){

var width = 3000
var height = 3000

var svg = mapContainer.append("svg")
        .attr("id", 'myid')
        .attr("width", width)
        .attr("height", height)

var size = d3.scaleSqrt()
        .domain([0, 20])
        .range([0, 20]);

// Map and projection
var projection = d3.geoMercator().center([47, 13])
    .scale(width  / 1.5)
    .translate([width / 2, height / 2])

    // Draw the map
    svg.append("g")
        .selectAll("path")
        .data(map.features)
        .enter().append("path")
            .attr("fill", "lightgray")
            .attr("d", d3.geoPath()
                .projection(projection)
            )
            .style("stroke", "#fff")


    svg
      .selectAll("myCircles")
      .data(markers)
      .enter()
      .append("circle")
        .attr("cx", function(d){ return projection([d.long, d.lat])[0] })
        .attr("cy", function(d){ return projection([d.long, d.lat])[1] })

        .attr('r', function(d){ return size(d.time) })
        .style("fill", "steelblue")
        .attr("stroke-width", 3)
        .attr("fill-opacity", .8)
}


