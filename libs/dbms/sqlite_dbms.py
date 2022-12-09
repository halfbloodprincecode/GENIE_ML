import sqlite3
from loguru import logger
from libs.basicIO import pathBIO

class SqliteDBMS:
    def __init__(self, fullpath: str):
        self.fullpath = pathBIO(fullpath)
        if not self.fullpath.endswith('.db'):
            self.fullpath = self.fullpath + '.db' 
        self.con = sqlite3.connect(self.fullpath)
        self.cursor = self.con.cursor()

    def get_colnames(self, table_name):
        res = self.con.execute('select * from {}'.format(table_name))
        logger.critical('@@@@@@@@@@@@ ={}'.format(res))
        res = res.description
        logger.critical('@@@@@@@@@@@@2 ={}'.format(res))


    def get_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        res = self.cursor.fetchall()
        return [r[0] for r in res]