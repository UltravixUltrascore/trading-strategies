import requests
from bs4 import BeautifulSoup as bs

# Funzione per ottenere il punteggio e la raccomandazione
def get_stock_rating_and_recommendation(ticker):
    url = f"http://www.marketbeat.com/stocks/NASDAQ/{ticker}/"
    response = requests.get(url)
    soup = bs(response.text, 'html.parser')

    score = soup.find('span', class_='mr-score')
    recommendation = soup.find('span', class_='mr-stat c-green')

    if score and recommendation:
        score_text = score.text.strip().split()[0]
        if score_text == 'N/A':
            return None, recommendation.text.strip()
        return float(score_text), recommendation.text.strip()
    else:
        return None, None

# Lista dei ticker
tickers = 'MMM, ABT, ABBV, ABVC, ACN, ACON, ADBE, AFRM, AIG, ABNB, ALK, ALB, AA, BABA, ALGN, GOOGL, ALT, MO, AMZN, AMC, AMD, AAL, AXP, AMP, AMGN, ADI, ANSS, AAPL, AMAT, ADM, ACHR, ADX, ARBK, ANET, ARM, ARRY, ASML, ASTS, T, TEAM, ADSK, BIDU, BKR, BAC, GOLD, BRK, BYND, BBAI, BIIB, BTBT, BITF, BB, BLNK, SQ, BE, BA, BKNG, BMY, BTI, AVGO, AI, CCCC, CDNS, CCJ, CAN, CSIQ, CGC, CCL, CVNA, CAT, CELH, CVX, CIFR, CSCO, C, CLSK, NET, CNHI, KO, COIN, CMCSA, COP, WISH, CRBP, COST, CRSP, CROX, CRWD, CVS, DQ, DDOG, DE, DELL, DAL, DXCM, DWAC, DOCU, DKNG, EBAY, ELV, ELF, LLY, ENPH, ENSC, ESPR, EL, EXLS, XOM, FSLY, FDX, RACE, AG, FSLR, FVRR, FL, F, FTNT, FCEL, GME, GCT, GE, GM, GILD, GTLB, GS, HL, HIVE, HD, HPQ, HPE, HUM, HUT, IBM, ILMN, INTU, ISRG, IOVA, IREN, IRBT, JD, JKS, JNJ, JPM, KLAC, LRCX, LMND, LXRX, LI, LMT, LOGI, LBPH, LCID, LULU, MARA, MRVL, MA, MTCH, MCD, MDT, MELI, MRK, META, MCHP, MU, MSFT, MSTR, MBLY, MRNA, MNDY, MDB, MNST, MS, NTES, NFLX, NEM, NEE, NKE, NKLA, NIO, NOK, NVO, NCLH, NVAX, NU, NVDA, NXPI, ORLY, OXY, OCGN, ONON, ON, ONMD, OPEN, ORCL, PACB, PGY, PAGS, PLTR, PANW, PARA, PYPL, PTON, PEP, PFE, PHUN, PDD, PINS, PLUG, PSNY, PG, QCOM, QS, RVSN, REGN, RIOT, RIVN, HOOD, RBLX, RKLB, ROKU, ROMA, RTX, RUM, RCL, CRM, LAES, S, NOW, SHOP, SLB, SNAP, SNOW, SOFI, SEDG, SAVE, SPOT, SBUX, STM, STNE, SDIG, SYK, RUN, SMCI, SYM, SNPS, TMUS, TSM, TGT, TDOC, TME, WULF, TSLA, TEVA, TXN, KHC, TTD, TMO, TLRY, TWLO, UBER, PATH, UAL, UNH, U, UPST, UPS, VLO, VZ, SPCE, V, WBA, WMT, DIS, WBD, WFC, WDC, XPEV, ZIM, ZTS, ZM, ZS'.split(', ')

# Lista per memorizzare i risultati
results = []

# Ciclo per raccogliere i dati
for ticker in tickers:
    score, recommendation = get_stock_rating_and_recommendation(ticker)
    if score is not None and recommendation is not None:
        results.append((ticker, score, recommendation))

# Funzione per ordinare in base alla raccomandazione e poi al punteggio
def sort_key(item):
    recommendation_order = {
        'Buy': 1,
        'Moderate Buy': 2,
        'Hold': 3,
        'Healthy': 4,
        'Weak': 5
    }
    return recommendation_order.get(item[2], 6), -item[1]

# Ordina i risultati
sorted_results = sorted(results, key=sort_key)

# Stampa i risultati ordinati
for ticker, score, recommendation in sorted_results:
    print(f"{ticker} Score: {score}, Recommendation: {recommendation}")
