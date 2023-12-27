document.addEventListener("DOMContentLoaded", function(event) { 

  // get all of the elements with the 'card' class.
  const scrollListEven = document.querySelectorAll(".cardEven")

  const callback = (entries, observer) => {
    entries.forEach((entry) => {

      if (entry.isIntersecting) {

        entry.target.classList.add("scrolled-in");

      }
      })
  }
  
  const options = {}
  
  const myObserver = new IntersectionObserver(callback, options)

  scrollListEven.forEach(scrollItem => {
    myObserver.observe(scrollItem)
  })

});



document.addEventListener("DOMContentLoaded", function(event) { 

  // get all of the elements with the 'card' class.
  const scrollListOdd = document.querySelectorAll(".cardOdd")

  const callback = (entries, observer) => {
    entries.forEach((entry) => {

      if (entry.isIntersecting) {

        entry.target.classList.add("scrolled-in");

      }
      })
  }
  
  const options = {}
  
  const myObserver = new IntersectionObserver(callback, options)

  scrollListOdd.forEach(scrollItem => {
    myObserver.observe(scrollItem)
  })

});



