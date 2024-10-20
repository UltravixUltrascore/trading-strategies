import os
import platform
import requests
import zipfile
import io

# URL di base per ChromeDriver
chrome_driver_base_url = "https://chromedriver.storage.googleapis.com/"

def get_chrome_driver_version():
    # Ottieni la versione corrente di Google Chrome installata
    version_url = "https://chromedriver.storage.googleapis.com/LATEST_RELEASE"
    response = requests.get(version_url)
    if response.status_code == 200:
        chrome_version = response.text.strip()
        print(f"Versione di ChromeDriver trovata: {chrome_version}")
        return chrome_version
    else:
        print(f"Errore nel richiedere la versione di ChromeDriver: {response.status_code}")
        return None

def get_chrome_driver():
    chrome_version = get_chrome_driver_version()
    if chrome_version:
        # Determina il sistema operativo corrente
        system = platform.system().lower()
        if system == "darwin":
            os_type = "mac64"
        elif system == "windows":
            os_type = "win32"
        else:
            os_type = "linux64"

        # Costruisci l'URL per il download del ChromeDriver
        download_url = f"{chrome_driver_base_url}{chrome_version}/chromedriver_{os_type}.zip"
        print(f"Scaricando ChromeDriver da: {download_url}")
        download_and_extract(download_url)
    else:
        print("Errore nel determinare la versione di ChromeDriver.")

def download_and_extract(url):
    response = requests.get(url)
    if response.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall()  # Estrai i file nella directory corrente
        print("ChromeDriver scaricato ed estratto con successo!")
    else:
        print(f"Errore nel download di ChromeDriver: {response.status_code}")

if __name__ == "__main__":
    get_chrome_driver()
