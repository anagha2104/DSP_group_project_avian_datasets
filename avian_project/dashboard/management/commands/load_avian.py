import pandas as pd
from django.core.management.base import BaseCommand
from dashboard.models import BirdTrait

class Command(BaseCommand):
    help = 'Loads data from the AVONET CSV file into the BirdTrait database'

    def handle(self, *args, **kwargs):
        # 1. Define the path to your CSV file
        # Make sure this path points to where your teammate put the file
        csv_path = 'C:/Users/praty/DSP_Project/DSP_group_project_avian_datasets/data/raw/core/AVONET1_BirdLife.csv' 
        
        try:
            self.stdout.write(f"Reading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # 2. Django databases hate NaN values, they prefer Python's None
            # This replaces all Pandas missing values with None
            df = df.where(pd.notnull(df), None)
            
            # 3. Keep track of how many we add
            count = 0

            # 4. Loop through the dataframe and create database entries
            for index, row in df.iterrows():
                # update_or_create is great because if you run this script twice, 
                # it will just update existing birds instead of causing duplication crash errors
                bird, created = BirdTrait.objects.update_or_create(
                    species_name=row['Species1'],
                    defaults={
                        'mass': row['Mass'],
                        'wing_length': row['Wing.Length'],
                        'beak_length': row['Beak.Length_Culmen'],
                        'tarsus_length': row['Tarsus.Length'],
                        'migration': row.get('Migration', None),
                        'habitat': row.get('Habitat', None),
                        'diet': row.get('Diet', None),
                        'trophic_level': row.get('Trophic.Level', None),
                        'trophic_niche': row.get('Trophic.Niche', None),
                        'primary_lifestyle': row.get('Primary.Lifestyle', None),
                        'centroid_latitude': row.get('Centroid.Latitude', None),
                        'centroid_longitude': row.get('Centroid.Longitude', None),
                        'range_size': row.get('Range.Size', None),
                    }
                )
                
                if created:
                    count += 1
                
                # Print a progress update every 1000 birds so you know it hasn't frozen
                if count % 1000 == 0 and count > 0:
                    self.stdout.write(f"Loaded {count} birds...")

            self.stdout.write(self.style.SUCCESS(f'Successfully loaded/updated {count} birds into the database!'))

        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"Could not find the file at {csv_path}. Please check the path."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred: {e}"))