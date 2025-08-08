import pandas as pd

excel_file = input("Enter the excel file: ")

df = pd.read_excel(excel_file)
df.to_html("output.html")
