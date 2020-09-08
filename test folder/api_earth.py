import requests

start_time = input('Enter the start time like "2017-01-1" ')
end_time = input('Enter the end time like "2019-01-1" ')
latitude = input('Enter the latitude ')
longitude = input('Enter the longitude ')
maxradius = input('Enter the max radius in km ')
minmag = input('Enter the min magnitud ')




url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?'

response = requests.get(url, headers={'Accept': 'application/json'}, params={
    'format': 'geojson',
    'starttime': start_time,
    'endtime': end_time,
    'latitude': latitude,
    'longitude': longitude,
    'maxradiuskm': maxradius,
    'minmagnitude': minmag

})

data = response.json()

n = len(data['features'])

for i in range(n):
    print('Place:',data['features'][i]['properties']['place'],'Magnitude:',data['features'][i]['properties']['mag'])
# print('Place:',data['features'][0]['properties']['place'],'Magnitude:',data['features'][0]['properties']['mag'])



# 2019-01-01
# 2019-05-01
# 51.51
# -0.12
# 2000
# 2