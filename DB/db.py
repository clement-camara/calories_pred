import psycopg2

try:
    conn = psycopg2.connect(
        user="clement",
        password="postgres",
        host="localhost",
        port="5432",
        database="diet"
    )
    cur = conn.cursor()
    sql = '''
    SELECT * FROM persons; 
    '''

    cur.execute(sql)
    conn.commit()
    print("Table créée avec succès dans PostgreSQL")

    # fermeture de la connexion à la base de données
    cur.close()
    conn.close()
    print("La connexion PostgreSQL est fermée")
except (Exception, psycopg2.Error) as error:
    print("Error", error)



