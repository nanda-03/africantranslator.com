
var selectedImages = [];
var button = document.getElementById("button-submission");
button.innerText = 'Mettre les elements dans le pannier';
button.disabled = true;

for(checkbox of document.getElementsByTagName('checkbox')){
    if(checkbox.checked){
        button.innerText = 'Sauvegarder';
        button.disabled = false;
        break;
    }
}

function images_select(checkbox){
    a = checkbox.id.split('-')[1]
    if(checkbox.checked){
        selectedImages.push(a);
    }
    else{
        selectedImages.splice(selectedImages.indexOf(a));
    }
    console.log(selectedImages);

    if(selectedImages.length!=0){
        button.disabled = false;
        button.innerText = 'Sauvegarder';
    }else{
        button.disabled = true;
        button.innerText = 'Mettre les elements dans le pannier';
    }
}

function submission(){
    console.log(selectedImages);
    //window.location.reload(true);
    //window.location.replace('http://127.0.0.1:8000/log')
    //window.location.assign('http://127.0.0.1:8000/log');
    $.ajax({
        type: "GET",
        url: "validation",
        dataType: "json",
        traditional: true,
        //headers: { 'api-key':'myKey' },
        data: {'list_article': JSON.stringify(selectedImages)},
        success: function(data) {
                console.log(data);
        },
        error:function(data) {
            console.log('error', data);
        }
    });
}

function modifier(){
    console.log('modifier');
    
    document.getElementById('username').disabled = false;
    document.getElementById('password').disabled = false;
    document.getElementById('modal-link').disabled = true;
}