document.addEventListener("DOMContentLoaded", function(event) { 

  // get all of the elements with the 'card' class.
  const scrollList = document.querySelectorAll(".card")

  const callback = (entries, observer) => {
    entries.forEach((entry) => {

      if (entry.isIntersecting) {

        entry.target.classList.add("scrolled-in");

      }
      })
  }
  
  const options = {}
  
  const myObserver = new IntersectionObserver(callback, options)

  scrollList.forEach(scrollItem => {
    myObserver.observe(scrollItem)
  })

});



