from django.http import HttpResponse

from projects.common import Project
from django.db import connection
import json
import numpy as np


class Marconi(Project):
    def fetch_song(self, cursor, song_id):
        cursor.execute("SELECT song_name, artist_name, vec from song2vec WHERE song_id = %s", (song_id,))
        return cursor.fetchone()

    def fill_dict(self, request, d):
        song1_id = request.GET.get('song1_id')
        song2_id = request.GET.get('song2_id')
        if song1_id:
            with connection.cursor() as cursor:
                song1_name, artist_name1, song1_vec = self.fetch_song(cursor, song1_id)
                if song2_id:
                    song2_name, artist_name2, song2_vec = self.fetch_song(cursor, song2_id)
                    mid_vec = [x1 + x2 for x1, x2 in zip(song1_vec, song2_vec)]
                    v_len = np.linalg.norm(mid_vec)
                    mid_vec = [x / v_len for x in mid_vec]
                else:
                    mid_vec = song1_vec
                    song2_name = ''

                play_list = []
                cursor.execute(
                    "SELECT song_id, song_name, artist_name, cube_distance(cube(vec), cube(%s)) as distance "
                    "FROM song2vec ORDER BY distance "
                    "LIMIT 10",
                    (mid_vec,),
                )
                for song_id, song_name, artist_name, distance in cursor:
                    play_list.append((song_id, song_name, artist_name))
            d['song1_id'] = song1_id
            d['song1'] = song1_name
            d['song2_id'] = song2_id
            d['song2'] = song2_name
            d['play_list'] = play_list
            d['song_ids'] = ','.join(x[0] for x in play_list)

    def handle_request(self, handler, request):
        if handler == 'autocomplete':
            term = request.GET.get('term', '')
            if len(term) < 3:
                return HttpResponse('[]')
            term = term.lower().strip()
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT song_id, song_name, artist_name "
                    "FROM song2vec WHERE lower(song_name) LIKE %s "
                    "ORDER BY song_count DESC LIMIT 7",
                    (term + '%',),
                )
                matches = [
                    {'value': song_id, 'label': '%s (%s)' % (song_name, artist_name)}
                    for song_id, song_name, artist_name in cursor
                ]
            return HttpResponse(json.dumps(matches))
