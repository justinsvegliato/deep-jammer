import requests
import json
import ast



version = 'v3.0'
entity = 'artists'
subentity = 'musicinfo'
clientid = '40c6d044'
format = 'json'
order = 'joindate_desc'

# Inputs
artists_per_run = 200
total_artists = 1000
# End Inputs

client_id_pm = 'client_id=' + clientid
format_id_pm = 'format=' + format
order_id_pm = 'order=' + order
limit_id_pm = 'limit=' + str(artists_per_run)

artist_names = []

for i in xrange(total_artists/artists_per_run):
	offset = i * artists_per_run
	offset_id_pm = 'offset=' + str(offset)

	parameterslist = [format_id_pm, order_id_pm, offset_id_pm, limit_id_pm]

	api_parameters = client_id_pm

	for pm in parameterslist:
	
		api_parameters += '&' + pm

	url = 'https://api.jamendo.com/' + version + '/' + entity + '/' + subentity + '/?' + api_parameters
	r = requests.get(url)

	json_file = r.json()

	results = json_file['results']

	for i in xrange(len(results)):
	
		artist_names.append(results[i]['name'])

print artist_names