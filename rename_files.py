import os

def rename_files(brighter_folder, celebA_folder):
    # List all files in the brighter folder
    brighter_files = os.listdir(brighter_folder)

    for file in brighter_files:
        # Split the file name to get the base and unique identifier
        parts = file.split('_')
        base_name = parts[0] + '_' + parts[1]
        unique_id = parts[2]

        # Construct the new file name for the corresponding file in celebA folder
        new_filename = base_name + '_' + unique_id

        # Original file name in celebA folder
        original_filename = base_name + '.png'
        print(parts[3])
        # Check if the original file exists in celebA folder
        if original_filename in os.listdir(celebA_folder):
            # Rename the file
            #os.rename(os.path.join(celebA_folder, original_filename),os.path.join(celebA_folder, new_filename))
            print(f"Renamed {original_filename} to {new_filename}")

# Example usage
brighter_folder = 'recovered_images_brighter'
celebA_folder = 'recovered_images_celebA_IR152'
rename_files(brighter_folder, celebA_folder)




#recovered_image_00_549.png

#recovered_image_00.png