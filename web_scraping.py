import requests
from bs4 import BeautifulSoup
import pandas as pd

def web_scraping(excel=True):
  """
  Parameters
  ----------
  excel: boolean
  Outputs the scraped data in xlsx format.
  """
  df=pd.DataFrame(columns={'question','answer'})
  subpage_list=['bills_statement','email_statement_delivery','online_stock_login','phishing_scam_faqs','m2u_pin_password','m2u_enhanced_security',
                'online_banking_for_business_customer','transcation_authorisation_code_tac','paytransfer_faq','reload_faq','online_settings_faq']
  for subpage in subpage_list:
    URL = 'https://www.maybank2u.com.my/maybank2u/malaysia/en/personal/faq/online_banking/'+subpage+'.page?'
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find(id="iw_comp1505554980909")
    faq = results.find_all('div', class_='question-block')
    for q in faq:
        question = q.find('h3')
        answer = q.find('p')
        if None in (question,answer):
            continue
        df.loc[len(df)]=[question.text.strip(),answer.text.strip()]
  if excel:
      df.to_excel ('Database.xlsx', index = False, header=True)
  return df

if __name__ == '__main__':
    web_scraping()
