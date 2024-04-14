import json
import csv
import pandas as pd

# Load JSON data from the file
with open('ipc.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)


# Create a CSV file and write header
csv_file = open('output.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['section', 'section_title', 'section_desc'])

# Write data to CSV
for entry in data:
    csv_writer.writerow([entry.get('Section', ''),
                        entry.get('section_title', ''),
                        entry.get('section_desc', '')])

# Close the CSV file
csv_file.close()

# Read CSV file into a DataFrame
csv_file = 'output.csv'
df = pd.read_csv(csv_file)

# Convert DataFrame to Excel and save it
excel_file = 'output.xlsx'
df.to_excel(excel_file, index=False)

print(f'Conversion successful. Excel file saved as {excel_file}')

