import pandas as pd

def save_to_csv(df, filename):
    df.to_csv(f'{filename}.csv', encoding='utf-8-sig', index=False)
    print(f"File saved to {filename}")
    
def save_to_excel(df, filename):
    df_limited = df.iloc[:, :255]
    with pd.ExcelWriter(f'{filename}.xlsx', engine='xlsxwriter') as writer:
        df_limited.to_excel(writer, index=False)
    print(f"File saved to {filename}")
    
    
def save_to_excel_2(df, filename):
    with pd.ExcelWriter(f'{filename}.xlsx', engine='xlsxwriter') as writer:
        columns_per_sheet = 10 #16384
        num_sheets = df.shape[1] // columns_per_sheet + (1 if df.shape[1] % columns_per_sheet != 0 else 0)
        
        for i in range(num_sheets):
            start_col = i * columns_per_sheet
            end_col = (i + 1) * columns_per_sheet if i != num_sheets - 1 else df.shape[1]
            subset = df.iloc[:, start_col:end_col]
            subset.to_excel(writer, sheet_name=f'sheet_{i + 1}', index=False)

    print(f"File saved to {filename}.xlsx")
