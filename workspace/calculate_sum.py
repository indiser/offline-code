import csv

def sum_gross(filename):
    total_gross = 0
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader)
            for row in reader:
                # Assuming gross is the 4th column (index 3) based on the preview
                if len(row) > 3:
                    try:
                        # Attempt to clean and convert to integer/float
                        gross_str = row[3].replace('$', '').replace(',', '').strip()
                        total_gross += int(gross_str)
                    except ValueError:
                        # Handle cases where conversion fails
                        continue
        return total_gross
    except FileNotFoundError:
        return f"Error: File {filename} not found."

if __name__ == "__main__":
    print(sum_gross('films.csv'))
