
function searchText(){
        closeList()
        data = document.getElementById('postList').querySelectorAll("li")
        searchBar = document.getElementById('searchBar')
        console.log(searchBar.value)

        dtList = []
        for(elem of data){
                val = elem.innerHTML.split(';')
                if(val[0].toLowerCase().includes(searchBar.value)){

                        dtList.push({"title":val[0], "tags": val[1]})
                }
        }
        if(dtList.length>0 && searchBar.value.length>0){
                suggestions = document.createElement('div')
                out = ""
                for(elem of dtList){
                        out += "<div class='searchResults'><a href='"+elem.tags.replace("_","-")+"'"+">"+elem.title+"</a></div>"

                }

                suggestions.innerHTML = out
                suggestions.setAttribute('id', 'suggestions')

                suggestions.setAttribute('class', 'suggestionsResults')
                searchBar.parentNode.appendChild(suggestions)
        }
        else{closeList()}
}

    function closeList() {
        let suggestions = document.getElementById('suggestions');
        if (suggestions)
            suggestions.parentNode.removeChild(suggestions);
    }

