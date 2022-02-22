import pandas as pd

def read_database():
    df1 = pd.read_excel(r"Database.xlsx", sheet_name = 'GENERAL_INTENTS', index_col= 0).to_dict(orient='index')
    df2 = pd.read_excel(r"Database.xlsx", sheet_name = 'FAQS', index_col= 0).to_dict(orient='index')
    df3 = pd.read_excel(r"Database.xlsx", sheet_name = 'DEFAULT_REPLY', index_col= 0).to_dict(orient='index')
    df4 = pd.read_excel(r"Database.xlsx", sheet_name = 'BANK_BALANCE', index_col= 0).to_dict(orient='index')

    return df1, df2, df3, df4

if __name__ == '__main__':
    df1, df2, df3, df4 = read_database()
    print(f"1: {df1} \n2: {df2} \n3: {df3} \n4: {df4}")
