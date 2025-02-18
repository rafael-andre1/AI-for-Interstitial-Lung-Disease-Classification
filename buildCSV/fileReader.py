import os

root_dir = 'X:\RafaelAndre\MedGIFT\ILD_DB_txtROIs'

fibrosis_list = []

# Quick function for obtaining patient id from directory
def getID(dir):
    res = ""
    for c in reversed(dir):
        if c in ('/','\\'): break

        # Add to left due to being reversed
        res = c + res
    return res


for dirpath, dirnames, filenames in os.walk(root_dir):
    # Open MedGIFT / txtROI
    print(f"Current directory: {dirpath}")

    # For each folder create id (ex patientID = folder number)
    for file in filenames:
        # Iterate through files in order to find .txt
        if 'txt' in file:
            fileID = f"{file}"
            print(f"Selected: {fileID}")

            # Define path for access
            file_path = os.path.join(dirpath,file)
            with open(file_path, 'r') as file:
                content = file.read()

                # Open txt (ex folder 84), if it contains keyword "fibrosis" set label to 1, otherwise 0 
                # Excludes cases where detects folder inside of folder   
                if "fibrosis" in content and '-' not in dirpath:
                    fibrosis_list.append(getID(dirpath))

print(fibrosis_list)





# Export dataframe as .csv if only ID -> Label, .pkl otherwise

