# Given a set of inputs, return a list of urls to download mp3 files
# Tracks provides a nested dictionary of the meta-data
# download_urls provides a list of the urls for download

import requests
import json
import ast

version = 'v3.0'
entity = 'artists'
subentity = 'tracks'
clientid = '40c6d044'
format = 'json'
order = 'joindate_desc'

# Inputs
name = 'we+are+fm'
album_datebetween = '0000-00-00_2012-01-01'
limit = 'all'
# End Inputs

client_id_pm = 'client_id=' + clientid
format_id_pm = 'format=' + format
order_id_pm = 'order=' + order
name_id_pm = 'name=' + name
album_datebetween_id_pm = 'album_datebetween=' + album_datebetween
limit_pm = 'limit=' + limit


parameterslist = [format_id_pm, order_id_pm, name_id_pm, album_datebetween_id_pm, limit_pm]


api_parameters = client_id_pm

for pm in parameterslist:
	
	api_parameters += '&' + pm

url = 'https://api.jamendo.com/' + version + '/' + entity + '/' + subentity + '/?' + api_parameters

r = requests.get(url)

json_file = r.json()
tracks = json_file['results'][0]['tracks']

#print tracks

download_urls = []

for i in xrange(len(tracks)):
	
	download_urls.append(tracks[i]['audiodownload'])

print download_urls
