import requests


main_path = "/content/drive/My Drive/datasets/YourVersion/"
checkpoint_dir ="/content/drive/My Drive/datasets/YourVersion/checkpoint_dir"








def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)



class Download:

    def __init__(self, destination = "models/"):
        super(Download, self).__init__()
        self.destination = destination

    def bulk_dowload(self , file_map = {}):
        for key in file_map.keys():
            download_file_from_google_drive(key, self.destination + file_map[key])


if __name__ == "__main__":
    Download().bulk_dowload({'0B6BV8NMiOnLqdFh2YUtGNnF6b3BkWUlraWhTV1pFdFloa3NV' : "Algo.java"})



    

    