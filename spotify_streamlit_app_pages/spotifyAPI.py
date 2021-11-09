import requests
import base64
import datetime
import getpass

from urllib.parse import urlencode
import webbrowser
import time

class Spotify(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    expired = True
    client_id = None
    client_secret = None
    token_url = 'https://accounts.spotify.com/api/token'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id, self.client_secret = self.getCreds()

    def getCreds(self):
        print('Client ID:')
        client_id = getpass.getpass()

        print('Client Secret:')
        client_secret = getpass.getpass()

        return client_id, client_secret

    def get_client_credentials(self):
        '''
        Return a base-64 encoded string
        '''

        if self.client_secret == None or self.client_id == None:
            raise Exception("You must set client_id and client_secret.")


        client_creds = f"{self.client_id}:{self.client_secret}"

        client_creds_base64 = base64.b64encode(client_creds.encode())

        return client_creds_base64.decode()

    def get_token_headers(self):

        client_creds_base64 = self.get_client_credentials()

        token_headers = {
    "Authorization":f"Basic {client_creds_base64}", #base 64 encoded string
}

        return token_headers

    def get_token_data(self):

        token_data = {
    "grant_type":"client_credentials"
}

        return token_data

    def perform_auth(self):

        r = requests.post(self.token_url, data = self.get_token_data(), headers = self.get_token_headers())
        token_response_data = r.json()

        if r.status_code not in range(200, 299):
            return False

        now = datetime.datetime.now()
        self.access_token = token_response_data['access_token']
        expires_in = token_response_data['expires_in']
        expires = now + datetime.timedelta(seconds = expires_in)
        self.access_token_expires = expires
        self.expired = expires < now
        return True

    def get_token(self):
        auth_done = self.perform_auth()
        if not auth_done:
            raise Exception("Authentication Failed")
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_token()
        return token

    def get_resource_header(self):
        headers = {
            "Authorization": f"Bearer {self.get_token()}"
        }
        return headers

    def get_resource(self, lookup_id, resource_type = 'albums', version = 'v1', tracks = False):
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}"
        if tracks:
            endpoint += '/tracks'
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers = headers)

        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def get_album(self, _id):
        return self.get_resource(_id, resource_type = 'albums', version = 'v1')

    def get_artist(self, _id):
        return self.get_resource(_id, resource_type = 'artists', version = 'v1')

    def get_song_features(self, _id):
        return self.get_resource(_id, resource_type = 'audio-features', version = 'v1')

    def get_song_id(self, query, search_type = 'track'):
        song_id = self.search(query = str(query) , search_type = str(search_type))['tracks']['items'][0]['id']
        return song_id

    def get_playlist_id(self, query = 'Top 50 - USA', search_type = 'playlist'):
        #default returns top 50 songs in the USA
        playlist_id = self.search(str(query), search_type = 'playlist')['playlists']['items'][0]['id']
        return playlist_id

    def get_playlist_items(self, query = 'Top 50 - USA', search_type = 'playlist'):
        playlist_id = self.get_playlist_id(str(query), str(search_type))
        itemsDict = self.get_resource(playlist_id, resource_type = 'playlists', version = 'v1', tracks = True)
        return [i['track']['name'] for i in itemsDict['items']]


    def get_song_link(self, query, search_type = 'track'):
        endpoint = self.get_song_features(self.get_song_id(str(query), str(search_type)))['uri']
        return endpoint

    def play_song(self, query, search_type = 'track'):
        import webbrowser
        webbrowser.open(str(self.get_song_link(query, search_type)))

    def get_playlist_features(self, query, search_type = 'playlist'):
        import time
        features = []

        items = self.get_playlist_items(str(query))

        for item in items:
            time.sleep(1)
            features.append(self.get_song_features(self.get_song_id(query = str(item), search_type = 'track')))

        return features

    def base_search(self, q):
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"

        lookup_url = f"{endpoint}?{q}"
        r = requests.get(lookup_url, headers = headers)

        if r.status_code not in range(200, 299):
            return {}

        return r.json()

    def search(self, query = None, operator = None, operator_query = None, search_type = 'artist'):

        if query == None:
            raise Exception("A query is required")

        if isinstance(query, dict):
            query = " ".join([f"{k}:{v}" for k,v in query.items()])
        if operator != None and operator_query != None:
            if operator.lower() == "or" or operator.lower() == "not":
                operator = operator.upper()
                if isinstance(operator_query, str):
                    query = f"{query} {operator} {operator_query}"

        query_params = urlencode({"q": str(query), "type": str(search_type.lower())})

        return self.base_search(query_params)
