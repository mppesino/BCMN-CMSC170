'''
data.py 
Holds functions for data-related operations
'''

import csv

class Data:
    
    # load_csv()
        # Simple csv reader, does not do cleaning (use CSV loader in preprocess.py if you need to clean CSVs)
        # However, ideally, we don't have to do cleaning anymore as the data is already preprocessed
    # returns: 2D array representing csv file rows and columns
    def load_csv(self, filename: str):
        with open(filename, newline='') as csvfile:
            data=[]
            pokemon = csv.reader(csvfile, delimiter=',')
            for row in pokemon:
                new_row = []
                for value in row:
                    new_row.append(value)
                data.append(new_row)
            return data

    def save_csv(self, data: list, outfile: str):
        with open(outfile, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    # build_header_lookup()
        # given a 2D array, takes the first row and generates an index lookup table
    # returns: dict with indices of each column header label ex. {0: "pokemon_name", 1: "type_1"...}
    def build_header_lookup(self, data:list):
        header = {}
        for j, row in enumerate(data[0]):
                header[j] = row
        return header

    def build_index_lookup(self, data: list):
        # returns {"pokemon_name": 0, "type_1": 1, ...}
        return {name: i for i, name in enumerate(data[0])}