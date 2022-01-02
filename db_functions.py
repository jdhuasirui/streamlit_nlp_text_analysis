# import database management
import sqlite3
conn = sqlite3.connect("data.db")
c = conn.cursor()

def create_uploaded_filetable():
    c.execute('CREATE TABLE IF NOT EXISTS filestable(filename TEXT, filetype TEXT, filesize TEXT, uploadDate TIMESTAMP)')

def add_file_details(filename, filetype, filesize, uploadDate):
    c.execute('INSERT INTO filestable(filename, filetype, filesize, uploadDate) VALUES (?,?,?,?)',(filename, filetype, filesize, uploadDate))
    conn.commit()

def view_all_data():
    c.execute("SELECT * FROM filestable")
    data = c.fetchall()
    return data