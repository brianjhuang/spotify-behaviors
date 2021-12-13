import requests
import base64
import datetime
import getpass

from urllib.parse import urlencode
import webbrowser
import time

class Spotify(object):
    '''
    This class helps us authenticate and fetch data 
    from the Spotify API. 
    
    Parameters:
    access_token (string) - our access token to fetch data
    access_token_expires (datetime) - time until token expires
    expired (bookean) - a boolean value indicating whether our token is expired or not. defaults to true
    client_id (string) - our client id to fetch our token
    client_secret (string) - our client secret to fetch our token
    token_url (string) - where we can fetch our token
    
    '''
    access_token = None
    access_token_expires = datetime.datetime.now()
    expired = True
    client_id = '' #erase before pushing to github, fill in with your client_id and secret if using personally
    client_secret = ''
    token_url = 'https://accounts.spotify.com/api/token'

    def __init__(self, *args, **kwargs):
        '''
        Our constructor. Initalizes and collects our Client ID and Client secret.
        '''
        super().__init__(*args, **kwargs)
        if (self.client_id == None and self.client_secret == None):
            #if we haven't set a default client_id and client_secret
            self.client_id, self.client_secret = self.getCreds()
            #ask the user for their creds

    def getCreds(self):
        '''
        Ask the user for their client_id and client_secret.
        Using the getpass library allows us to do it without revealing sensitive information.
        
        Returns:
        client_id(string) - the client id
        client_secret(string) - the client secret
        '''
        print('Client ID:')
        client_id = getpass.getpass()
        #fetch id

        print('Client Secret:')
        client_secret = getpass.getpass()
        #fetch secret
        return client_id, client_secret

    def get_client_credentials(self):
        '''
        Return a base-64 encoded string
        '''

        if self.client_secret == None or self.client_id == None:
            raise Exception("You must set client_id and client_secret.")


        client_creds = f"{self.client_id}:{self.client_secret}"
        #turn our client_id and client_secret into a dictionary

        client_creds_base64 = base64.b64encode(client_creds.encode())
        #encode the client credentials into base64

        return client_creds_base64.decode()
        #return the decoded base64 client credentials 

    def get_token_headers(self):
        '''
        Get our token headers.
        
        Returns (dictionary) - Token hearders
        '''

        client_creds_base64 = self.get_client_credentials()

        token_headers = {
    "Authorization":f"Basic {client_creds_base64}", #base 64 encoded string
}

        return token_headers

    def get_token_data(self):
        '''
        Get token data.
        
        Returns (dictionary) - Token data
        '''

        token_data = {
    "grant_type":"client_credentials"
}

        return token_data

    def perform_auth(self):
        '''
        Performs our authentication for us.
        
        Returns:
        boolean - True or False value based on the status of our authentication.
        '''

        r = requests.post(self.token_url, data = self.get_token_data(), headers = self.get_token_headers())
        token_response_data = r.json()
        #set the request for the token

        if r.status_code not in range(200, 299):
            return False
        #if we don't fail

        now = datetime.datetime.now()
        #get current time
        self.access_token = token_response_data['access_token']
        expires_in = token_response_data['expires_in'] #get the time till it expires
        expires = now + datetime.timedelta(seconds = expires_in)
        self.access_token_expires = expires #set it to our instance variable so we can fetch
        self.expired = expires < now #set if it is expired or not
        return True

    def get_token(self):
        '''
        Performs authentication and returns the headers with our token
        
        Returns:
        headers (dictionary): our token with headers
        '''
        auth_done = self.perform_auth()
        if not auth_done:
            raise Exception("Authentication Failed")
        #if auth is true
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        #get token, expires date, and current time
        if expires < now:
            self.perform_auth()
            return self.get_token()
        return token

    def get_resource_header(self):
        '''
        Get resource headers.
        
        Returns:
        headers (dictionary) : our resource headers
        '''
        headers = {
            "Authorization": f"Bearer {self.get_token()}"
        }
        return headers

    def get_resource(self, lookup_id, resource_type = 'albums', version = 'v1', tracks = False):
        '''
        Get our resources.
        
        Params:
        lookup_id (string) : the id or item we want to look up (albums, tracks, artists)
        resource_type : the type of thing we want (tracks, albums, artists, playlists). default is albums
        version: the version of the API we want. default is v1
        tracks: if we want to get tracks. default is false.
        
        Returns:
        r.json() (json/dictionary): the resources we want for the item we're looking up
        '''
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}"
        if tracks:
            endpoint += '/tracks'
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers = headers)

        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def get_album(self, _id):
        '''
        Get the album we want.
        
        Params:
        _id (string) - the item we want
        
        Returns:
        album (json/dictionary) - the album we want
        '''
        return self.get_resource(_id, resource_type = 'albums', version = 'v1')

    def get_artist(self, _id):
        '''
        Get the artist we want.
        
        Params:
        _id (string) - the item we want
        
        Return:
        artist (json/dictionary) - the artist we want
        '''
        return self.get_resource(_id, resource_type = 'artists', version = 'v1')

    def get_song_features(self, _id):
        '''
        Get the song we want.
        
        Params:
        _id (string) - the item we want
        
        Returns:
        
        song_features (json/dictionary) - all the features of the song we want
        '''
        return self.get_resource(_id, resource_type = 'audio-features', version = 'v1')

    def get_song_id(self, query, search_type = 'track'):
        '''
        Get the song id we want.
        
        Params:
        query (string) - the item we want
        search_type - default is track, should not be changed
        
        Returns:
        song_id (string) - the song id we're looking for
        '''
        song_id = self.search(query = str(query) , search_type = str(search_type))['tracks']['items'][0]['id']
        return song_id

    def get_playlist_id(self, query = 'Top 50 - USA', search_type = 'playlist', desired_artist = 'Spotify'):
        '''
        Get the playlist id we want.
        
        Params:
        query (string) - the item we want
        search_type - default is playlist, should not be changed
        desired_artist = 'For duplicate playlist, we grab by name'
        
        Returns:
        
        playlist_id(string) - the id of the playlist we're looking for.
        '''
        #default returns top 50 songs in the USA
        items = self.search(str('Top 50 - USA'), search_type = 'playlist')['playlists']['items']
        for i in items:
            playlist_maker = i['owner']['display_name']
            if playlist_maker == desired_artist:
                playlist_id = i['id']
        
        return playlist_id

    def get_playlist_items(self, query = 'Top 50 - USA', search_type = 'playlist'):
        '''
        Get the playist items we want.
        
        Params:
        query (string) - the item we want
        search_type - default is playlist, should not be changed
        
        Returns:
        
        The items in the specified playlist.
        '''
        playlist_id = self.get_playlist_id(str(query), str(search_type))
        itemsDict = self.get_resource(playlist_id, resource_type = 'playlists', version = 'v1', tracks = True)
        return [i['track']['name'] for i in itemsDict['items']]


    def get_song_link(self, query, search_type = 'track'):
        '''
        Get the playist link we want.
        
        Params:
        query (string) - the item we want
        search_type - default is playlist, should not be changed
        
        Returns:
        endpoint (string) - the endpoint we want to get each link
        '''
        endpoint = self.get_song_features(self.get_song_id(str(query), str(search_type)))['uri']
        return endpoint

    def play_song(self, query, search_type = 'track'):
        '''
        This plays a song using the webbrowser library. 
        
        Params:
        query (string) - the song we want to play
        search_type - default is track, should not be changed
        '''
        webbrowser.open(str(self.get_song_link(query, search_type)))

    def get_playlist_features(self, query, search_type = 'playlist'):
        '''
        This gets the playlist features we want.
        
        Params:
        query (string) - the name or query for the playlist we want
        search_type (string) - default is playlistm should not be changed
        
        Returns:
        
        features (json/dictionary) - the features of the playlist
        '''
        features = []

        items = self.get_playlist_items(str(query))

        for item in items:
            time.sleep(1)
            features.append(self.get_song_features(self.get_song_id(query = str(item), search_type = 'track')))

        return features

    def base_search(self, q):
        '''
        The basic search for items in the API.
        
        Params:
        q (string) - the query or item we want
        
        Returns:
        A dictionary/json with the items we want
        '''
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"

        lookup_url = f"{endpoint}?{q}"
        r = requests.get(lookup_url, headers = headers)

        if r.status_code not in range(200, 299):
            return {}

        return r.json()

    def search(self, query = None, operator = None, operator_query = None, search_type = 'artist'):
        '''
        Advanced search/building on the base search function.
        
        Params:
        query (string) - the item/query we want to look for
        operator (string) - the operator for the query 
        operator_query (string) - the operator query
        search_type (string) - what we want to look for, default search_type is artist
        
        Returns:
        
        A json/dictionary of what we searched for.
        '''

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
