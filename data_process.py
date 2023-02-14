import numpy as np
import pandas as pd

def main(file):
    data_raw = pd.read_csv(file)
    
    processed_data = []
    for i, row in data_raw.iterrows():
        processed_row = []
        for j, item in enumerate(row):
            if j == 0:
                continue
            elif pd.isnull(item):
                processed_row.append(-1)
            elif item == True:
                processed_row.append(1)
            elif item == False:
                processed_row.append(0)
            else:
                processed_row.append(item)
        processed_data.append(processed_row)

    pd_data = pd.DataFrame(processed_data)
    pd_data.columns = pd.read_csv(file, nrows=1).columns.tolist()[1:]
    processed_data_name = "./data/processed_data_" + file[16:] 
    pd.DataFrame(pd_data).to_csv(processed_data_name, index=False)
    return processed_data_name
    

if __name__ == "__main__":
    main()