import psycopg2

try:
    conn = psycopg2.connect(
        user="hjfcyeegymlkxs",
        password="ba02fd2a5f84813e5760ffb06498e68a7d84bce1a25b856bac7342ddbc095b0c",
        host="ec2-52-30-159-47.eu-west-1.compute.amazonaws.com",
        port="5432",
        database="d3jurkmim9r3gi"
    )
    cur = conn.cursor()
    sql = '''

CREATE TABLE persons
(
    user_id integer NOT NULL,
    gender character varying(50),
    age integer,
    height double precision,
    weight double precision,
    duration real,
    heart_rate real,
    body_temp real,
    CONSTRAINT persons_pkey PRIMARY KEY (user_id)
);

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