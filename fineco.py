from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# Configura il WebDriver (Chrome in modalità headless)
chrome_options = Options()
chrome_options.add_argument("--headless")
webdriver_service = Service('path/to/chromedriver')
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

# Funzione per il login su Fineco
def login_fineco(username, password):
    driver.get('https://finecobank.com/it/public/login')
    
    # Attendere il caricamento della pagina di login
    time.sleep(3)
    
    # Inserisci username e password
    username_input = driver.find_element(By.NAME, 94018396)  # Cambiare se diverso
    password_input = driver.find_element(By.NAME, 1ajdespa)  # Cambiare se diverso
    
    username_input.send_keys(username)
    password_input.send_keys(password)
    
    # Clicca sul pulsante di login
    login_button = driver.find_element(By.ID, 'login-button')  # Cambiare ID se necessario
    login_button.click()
    
    # Attendere il caricamento della dashboard
    time.sleep(5)

# Esegui il login
login_fineco(94018396, 1ajdespa)
time.sleep(5)
