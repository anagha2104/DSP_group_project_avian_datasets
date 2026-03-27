from django.db import models

class BirdTrait(models.Model):
    # Taxonomic Identifiers
    species_name = models.CharField(max_length=255, unique=True, help_text="Species1 from AVONET")

    # Morphological Descriptors (Numeric Traits)
    # Using null=True, blank=True [One way to handle missing data]
    mass = models.FloatField(null=True, blank=True, verbose_name="Body Mass (g)")
    wing_length = models.FloatField(null=True, blank=True, verbose_name="Wing Length (mm)")
    beak_length = models.FloatField(null=True, blank=True, verbose_name="Beak Length (Culmen, mm)")
    tarsus_length = models.FloatField(null=True, blank=True, verbose_name="Tarsus Length (mm)")

    # Ecological Properties (Categorical Factors)
    migration = models.CharField(max_length=100, null=True, blank=True)
    habitat = models.CharField(max_length=100, null=True, blank=True)
    diet = models.CharField(max_length=100, null=True, blank=True)
    trophic_level = models.CharField(max_length=100, null=True, blank=True)
    trophic_niche = models.CharField(max_length=100, null=True, blank=True)
    primary_lifestyle = models.CharField(max_length=100, null=True, blank=True)

    # Biogeographical & Spatial Data
    centroid_latitude = models.FloatField(null=True, blank=True)
    centroid_longitude = models.FloatField(null=True, blank=True)
    range_size = models.FloatField(null=True, blank=True)

    #Returns the bird's name not BirdTrait
    def __str__(self):
        return self.species_name