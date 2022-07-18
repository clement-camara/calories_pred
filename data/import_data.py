import psycopg2

conn = psycopg2.connect(
        user="clement",
        password="postgres",
        host="localhost",
        port="5432",
        database="diet"
    )

    cur = conn.cursor()
    sql = '''
UPDATE persons
SET column1 = value1,
    column2 = value2,
    ...
WHERE condition;

'''

cur.execute(sql)
conn.commit()
