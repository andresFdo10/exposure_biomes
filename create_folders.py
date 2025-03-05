import os
import zipfile
import pandas as pd

def run():
    # Define the list of the ecoregion names.
    ecoregions = pd.read_csv('./outputs/csv/ecoregion_list.csv')
    ecoregions = list(ecoregions['ECO_NAME'])
    # print(ecoregion_list)

    # ecoregions = [
    #     'Bahia coastal forests', 
    #     'Cauca valley montane forests', 
    #     'Cordillera de Merida páramo',
    #     'Cordillera Oriental montane forests',
    #     'Magdalena Valley montane forests',
    #     'Guianan lowland moist forests',
    #     'Uatumã-Trombetas moist forests',
    #     'Tapajós-Xingu moist forests',
    #     'Xingu-Tocantins-Araguaia moist forests',
    #     'Guianan Highlands moist forests',
    #     'Ucayali moist forests',
    #     'Iquitos várzea',
    #     'Guianan savanna',
    #     'Magdalena-Urabá moist forests',
    #     'Chocó-Darién moist forests',
    #     'Apure-Villavicencio dry forests',
    #     'Monte Alegre várzea',
    #     'Guajira-Barranquilla xeric scrub',
    #     'Maracaibo dry forests',
    #     'Catatumbo moist forests',
    #     'Magdalena Valley dry forests',
    #     'Lara-Falcón dry forests',
    #     'Paraguaná xeric scrub',
    #     'Cordillera La Costa montane forests',
    #     'Cauca Valley dry forests',
    #     'Eastern Panamanian montane forests',
    #     'Patía valley dry forests',
    #     'Venezuelan Andes montane forests'
    #     ]

    # Define the subfolders to be created inside each ecoregion folder.
    subfolders = ['treecover_30', 'treecover_50', 'treecover_75']

    # Base directory where folders will be created.
    base_dir = 'inputs/04_csv'

    # Create the base directory if it doesn't exist.
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create the ecoregion folders.
    for ecoregion in ecoregions:
        # Create the ecoregion folder.
        folder_path = os.path.join(base_dir, ecoregion)
        os.makedirs(folder_path, exist_ok=True)
        for subfolder in subfolders:
            folder_path = os.path.join(base_dir, ecoregion, subfolder)
            os.makedirs(folder_path, exist_ok=True)

    print("Folders and subfolders created successfully.")

    # Base directory where the ecoregion folders are located
    # base_dir = "./inputs/04_csv/"

    # Loop through each ecoregion folder
    for ecoregion in os.listdir(base_dir):
        ecoregion_path = os.path.join(base_dir, ecoregion)
        
        # Ensure it's a directory
        if os.path.isdir(ecoregion_path):
            # Loop through each subfolder (treecover_30, treecover_50, treecover_75)
            for subfolder in os.listdir(ecoregion_path):
                subfolder_path = os.path.join(ecoregion_path, subfolder)

                # Ensure it's a directory
                if os.path.isdir(subfolder_path):
                    # Loop through files in the subfolder
                    for file in os.listdir(subfolder_path):
                        if file.endswith(".zip"):
                            zip_path = os.path.join(subfolder_path, file)
                            extract_to = subfolder_path  # Extract in the same directory

                            # Unzip the file
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_to)
                            
                            print(f"Extracted: {file} -> {extract_to}")

    print("All zip files have been extracted successfully!")

if __name__ == "__main__":
    run()