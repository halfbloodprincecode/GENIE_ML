import sqlite3
from libs.basicIO import pathBIO

class SqliteDBMS:
    def __init__(self, fullpath: str):
        self.fullpath = pathBIO(fullpath)
        if not self.fullpath.endswith('.db'):
            self.fullpath = self.fullpath + '.db' 
        self.con = sqlite3.connect(self.fullpath)
        self.cursor = self.con.cursor()

    def get_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        res = self.cursor.fetchall()
        return [r[0] for r in res]