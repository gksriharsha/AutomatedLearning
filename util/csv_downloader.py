import csv,os
import requests
import uuid

def download_csv(link):
    CSV_URL = link

    id = uuid.uuid4()
    with requests.Session() as s:
        download = s.get(CSV_URL)

        decoded_content = download.content.decode('utf-8')
        with open(f'datasets/{id}.csv','w',newline='') as f:
            cw = csv.writer(f)
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            for row in my_list:
                cw.writerow(row)
            
    return os.path.abspath(f'datasets/{id}.csv')