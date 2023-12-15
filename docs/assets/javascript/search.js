
function searchText(){
        var event = window.event
        if(event.keyCode == 13){
                openFocus()
        }
        else if(event.keyCode == 40){
                focusNext()
        }
        else if(event.keyCode == 38){
                focusBefore()
        }
        else{
        resetActive()

        closeList()
        data = document.getElementById('postList').querySelectorAll("li")
        searchBar = document.getElementById('search_0')

        dtList = []
        for(elem of data){
                val = elem.innerHTML.split(';')
                if(val[0].toLowerCase().includes(searchBar.value)){

                        dtList.push({"title":val[0], "tags": val[1]})
                }
        }
        if(dtList.length>0 && searchBar.value.length>0){
                let suggestions = document.createElement('div')
                ind = 1
                for(elem of dtList){
                        // out += "<div id='search_"+ind+"' class='searchResults'><a href='"+elem.tags.replace("_","-")+"'"+">"+elem.title+"</a></div>"
                        newDiv = document.createElement('div')
                        newDiv.setAttribute('id', 'search_'+ind)
                        newDiv.setAttribute('class', 'searchResults')
                        href = document.createElement('a')
                        href.innerHTML = elem.title
                        href.setAttribute('href', elem.tags.replace('_', '-'))
                        newDiv.appendChild(href)
                        suggestions.appendChild(newDiv)
                        ind += 1

                }

                suggestions.setAttribute('id', 'suggestions')

                suggestions.setAttribute('class', 'suggestionsResults')
                searchBar.parentNode.appendChild(suggestions)
        }
        else{closeList()}
}
}

    function closeList() {
        let suggestions = document.getElementById('suggestions');
        if (suggestions)
            suggestions.parentNode.removeChild(suggestions);
    }

function focusNext(){
        let allResults =  document.querySelectorAll(".searchResults");
        let resultNumber = allResults.length
        const currInput = document.activeElement;
        activeField = document.getElementById('search_focus')
        let elemVal = +activeField.innerHTML
        if(elemVal<resultNumber){
                let nextId = +elemVal + 1
                activeField.innerHTML = nextId
                let currString = 'search_'+elemVal
                let nextString = 'search_'+nextId

                const nextInput = document.getElementById(nextString)
                const currentInput = document.getElementById(currString)


                nextInput.style.background = 'lightgray'
                currentInput.style.background = '#fffcfa'

                nextInput.classList.add('active')
        }
}

function focusBefore(){
        let allResults =  document.querySelectorAll(".searchResults");
        let resultNumber = allResults.length
        const currInput = document.activeElement;
        activeField = document.getElementById('search_focus')
        let elemVal = +activeField.innerHTML
        if(elemVal>0){
                let nextId = +elemVal - 1
                activeField.innerHTML = nextId
                let currString = 'search_'+elemVal
                let nextString = 'search_'+nextId


                const nextInput = document.getElementById(nextString)
                const currentInput = document.getElementById(currString)

                nextInput.classList.add('active')

                if(nextId>0){
                nextInput.style.background = 'lightgray'
}
                currentInput.style.background = '#fffcfa'

        }
}

function openFocus(){

        activeField = document.getElementById('search_focus')
        if(+activeField.innerHTML>0){

                let searchResult = document.getElementById('search_'+activeField.innerHTML)
                window.open(searchResult.firstChild.href, '_self')
        }else{
        }
}

function resetActive(){
        let currentActive = document.getElementById('search_focus')
        let activeElement = document.getElementById('search_'+currentActive.innerHTML) // restyle
        activeElement.style.background = '#fffcfa'

        currentActive.innerHTML = 0

}
