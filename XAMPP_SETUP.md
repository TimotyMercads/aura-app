# Setting Up the Database with XAMPP

This guide gets MySQL running for AURA using XAMPP's control panel and
phpMyAdmin — almost everything here is clicking buttons, not typing
commands. Do this once before running the app for the first time.

---

## 1. Install XAMPP (skip if you already have it)

1. Go to **https://www.apachefriends.org/**
2. Download the Windows installer.
3. Run it and click through the installer with the default options
   (make sure **MySQL** and **phpMyAdmin** stay checked — they're
   selected by default).
4. Finish the install. It creates a folder like `C:\xampp`.

---

## 2. Start MySQL

1. Open the **XAMPP Control Panel** (search for it in the Start Menu).
2. You'll see a row for **MySQL** and a row for **Apache**.
3. Click **Start** on the **MySQL** row. It should turn green and show a
   port number (e.g. 3306).
4. Click **Start** on the **Apache** row too — you only need this one
   for the next step (viewing phpMyAdmin in your browser); the AURA app
   itself doesn't need Apache running.

If MySQL's Start button turns red immediately or shows an error, see
**Troubleshooting** at the bottom.

---

## 3. Create the database tables

1. With Apache running, click the **Admin** button on the Apache row
   (or just go to **http://localhost/phpmyadmin** in your browser).
2. In phpMyAdmin, click the **SQL** tab along the top (make sure you're
   at the main/server level, not inside a specific database — if you
   see a list of databases on the left, you're in the right place).
3. Open `schema.sql` (in the AURA project folder) in Notepad, select
   all the text (Ctrl+A), and copy it (Ctrl+C).
4. Paste it into the big text box on phpMyAdmin's SQL tab.
5. Click **Go** (bottom right).
6. You should see a success message. In the left sidebar, a new
   database called **aura_db** should now appear — click it and
   confirm you see 4 tables: `users`, `uploads`, `students`, and
   `intervention_log`.

That's it — the database is ready. You only need to do this once (unless
you want to wipe everything and start over — see Troubleshooting).

---

## 4. Run the app as usual

Nothing else changes from before:

```
cd path\to\flaskapp
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

`pip install -r requirements.txt` will now also install a few new
packages (Flask-Login, SQLAlchemy, PyMySQL, python-dotenv) needed for
login and the database — that's expected the first time.

Go to **http://127.0.0.1:5000** — it should send you to a login page.
Click **Sign up** to create your first teacher account, then log in.

---

## Do I need XAMPP running every time?

**Yes — MySQL specifically.** Before running `python app.py`, open the
XAMPP Control Panel and make sure **MySQL** shows green/Running. Apache
is only needed if you want to open phpMyAdmin again; the Flask app
itself doesn't use Apache at all.

If you forget and open the app anyway, you'll see a friendly "Can't
reach the database" page instead of a crash — just start MySQL in XAMPP
and refresh.

---

## Troubleshooting

**MySQL won't start / turns red immediately in the Control Panel**
This is almost always **port 3306 already being used** by something
else on your PC (sometimes another MySQL install, or Skype in older
versions). In the XAMPP Control Panel, click **Config** on the MySQL
row → **my.ini**, and look for `port=3306` — change it to something
like `3307` in both places it appears, save, and restart MySQL. Then
create a file named `.env` in the `flaskapp` folder (copy
`.env.example` and rename it) and set `DB_PORT=3307` to match.

**phpMyAdmin shows "Access denied" or asks for a password**
A fresh XAMPP install's MySQL root user has no password, which is what
AURA expects by default. If you previously set one (e.g. via the XAMPP
security wizard), create a `.env` file (copy `.env.example`) and set
`DB_PASSWORD=yourpassword`.

**I want to wipe all data and start completely fresh**
In phpMyAdmin, click the **aura_db** database in the left sidebar, then
**Operations** tab → **Drop the database**. Then repeat Step 3 above
(paste `schema.sql` into the SQL tab and click Go) to recreate it empty.

**"ModuleNotFoundError" mentioning pymysql, flask_login, or sqlalchemy**
Run `pip install -r requirements.txt` again inside your activated
virtual environment — these are new dependencies added for login and
the database.
