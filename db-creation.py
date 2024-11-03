import requests
import csv
from tqdm import tqdm

base_url = "https://api.artic.edu/api/v1/artworks"

params = {
    "page": 1,
    "limit": 100
}

artworks = []

desired_count = 10000

with tqdm(total=desired_count, desc="Carregando obras de arte") as pbar:
    while len(artworks) < desired_count:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            artworks.extend(data['data'])
            pbar.update(len(data['data']))
            
            params["page"] += 1
        else:
            print(f"Erro na requisição: {response.status_code}")
            break

artworks = artworks[:desired_count]

columns = ["id", "title", "artist_title", "description", "publication_history", "medium_display", "date_display"]

with open("art-db.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    
    for artwork in artworks:
        writer.writerow({col: artwork.get(col, "") for col in columns})

print(f"Salvo {len(artworks)} obras de arte no arquivo 'art-db.csv'.")