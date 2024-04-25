# ITRF Coordinates

The **satkit** package contains a special class, `itrfcoord`, for representing coordinates in the International Terrestrial Reference Frame (ITRF).  This class allows for easy conversion between Cartesian and geodetic (latitude, longitude, and altitude) coordinates, and also provides convenience quaternions for maping relative locations into the North-East-Down (NED) or East-North-Up (ENU) frames.

## Construction

The `itrfcoord` class can be constructed by specifying either the Cartesian coordinates in the ITRF frame (units are meters) or by specifying the geodetic coordinates.

Example is below:

```python
import satkit as sk

# Specify a coordinate via Geodetic latitude & longitude (if altitude is left out, default is 0
boston = sk.itrfcoord(latitude_deg=42.14, longitude_deg=-71.15)

# Denver, CO is at a higher altitude; altitude is specified in meters
denver = sk.itrfcoord(latitude_deg=32.7392, longitude_deg=-104.99, altitude=1600)
print(f'Denver = {denver}')

# Get the Cartesian vector for Denver
print(f'Denver vector coordinate is {denver.vector}')

# Create a duplicate of the Denver coordinate, using vector as input
denver2 = sk.itrfcoord(denver.vector)
print(f'Denver created from Cartesian is {denver2}')

# Output is:
# Denver = ITRFCoord(lat:  32.7392 deg, lon: -104.9900 deg, hae:  1.60 km)
# Denver vector coordinate is [-1389345.60167272 -5188730.62268842  3430531.07678629]
# Denver created from cartesian is ITRFCoord(lat:  32.7392 deg, lon: -104.9900 deg, hae:  1.60 km)
```

## Geodesic Distances

```python
import satkit as sk

newyork = sk.itrfcoord(latitude_deg=40.7128, longitude_deg=-74.0060)
destinations = [
    {'name': 'London', 'latitude_deg': 51.5072, 'longitude_deg': -0.1276},
    {'name': 'Paris', 'latitude_deg': 48.8566, 'longitude_deg': 2.3522},
    {'name': 'Toronto', 'latitude_deg': 43.6532, 'longitude_deg': -79.3832},
    {'name': 'Tokyo', 'latitude_deg': 35.6763, 'longitude_deg': 139.65},
    {'name': 'Sau Paulo', 'latitude_deg': -23.5558, 'longitude_deg': -46.6396}
]
for dest in destinations:
    destcoord = sk.itrfcoord(latitude_deg=dest['latitude_deg'], longitude_deg=dest['longitude_deg'])
    # Distance, start, and end heading along great circle to destination coordinate
    dist, _heading_start, _heading_end = newyork.geodesic_distance(destcoord)
    print(f'New York to {dest['name']} distance is {dist/1e3:.0f} km')
# Results:
# New York to London distance is 5585 km
# New York to Paris distance is 5853 km
# New York to Toronto distance is 551 km
# New York to Tokyo distance is 10876 km
# New York to Sau Paulo distance is 7658 km
```
