import os
import time
import sqlite3
import orm_sqlite
import pandas as pd

class Metrics():
    def __init__(self, db, fname: str, tableName: str, items: list):
        self.tableName = tableName.lower().strip().replace(' ', '_')
        db_path = os.path.join(
            os.path.dirname(os.path.realpath(db.__file__)),
            f'{fname}.db'
        )
        self.db_path = db_path
        self.DB = orm_sqlite.Database(db_path)
        
        Metric = type(
            self.tableName, #'Metric',
            (orm_sqlite.Model,),
            {
                'step': orm_sqlite.IntegerField(primary_key=True),
                **{item: orm_sqlite.FloatField() for item in items},
                'timestamp': orm_sqlite.StringField()
            },
        )

        Metric.objects.backend = self.DB
        self.Model = Metric

    def add(self, spec):
        assert 'step' in spec, 'metric record does not have "step" key.'
        spec['timestamp'] = str(time.time())
        statuscode = self.Model(spec).save()
        if statuscode == -1:
            pk = spec.get('step', None)
            obj = self.Model.objects.get(pk=pk)
            assert obj is not None, f'metric record with step={pk} does not exist.'
            for key in spec.keys():
                obj[key] = spec.get(key, None)
            statuscode = self.Model.objects.update(obj)
        return statuscode
    
    def delete(self, where='true'):
        where = f'step >= {where}' if isinstance(where, int) else where
        return self.DB.execute(f'delete from {self.tableName} where {where}', *[], autocommit=True)
    
    def select(self, sql: str):
        "Returns query results, a list of sqlite3.Row objects."
        return self.DB.select(sql, *[], size=None)

    def sql(self, sql: str):
        "Executes an SQL statement and returns rows affected."
        return self.DB.execute(sql, *[], autocommit=True)

    def to_csv(self, sql=None, dist=None):
        sql = sql if sql else f'SELECT * FROM {self.tableName}'
        dist = self.db_path.replace('.db', '.csv') if dist is None else dist
        conn = sqlite3.connect(self.db_path, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
        db_df = pd.read_sql_query(sql, conn)
        db_df.to_csv(dist, index=False)
        conn.close()

"""
Example:
    import database as my_db
    metrics = Metrics(my_db, 'my-db', ['loss', 'val_loss'])
    for s, m in zip([-3, 2, 21, 4, 232, 44, 23], [(6.12,3.21), (0.01,1.78), (6.3,1.5), (2.1,5.2), (12.36, 41.25), (54.58, 66.225), (22.456, 6.3)]):
        metrics.add({
            'step': s,
            'loss': m[0],
            'val_loss': m[1]
        })
    print(metrics.select('select * from metric'))
    print(metrics.delete())
    print(metrics.db_path)
    metrics.to_csv()
    metrics.to_csv(dist='/home/test.csv', sql='select step, loss from metric where step > 0')
"""